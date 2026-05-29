"""사후 가격 라벨링 — KOSPI 상대 + 절대 수익 자동 라벨.

PRD 모듈 #4 (LabelingScheduler). 미라벨 가설을 ``horizon_weeks`` 경과 후
KOSPI 상대 수익률로 자동 라벨링. LLM 호출 0 — 가격 fetch 만.

라벨링 임계값 (호라이즌별 스케일):
  - 2 주: ±2%
  - 4 주: ±3%
  - 8 주: ±5%
  - 그 외 정수: ``2 + (h - 2) * 0.5`` 선형 보간 (3→2.5, 5→3.5, 6→4, 7→4.5)

Verdict 판정:
  - ``bullish`` & relative ≥ +threshold → right
  - ``neutral`` & |relative| ≤ threshold/2 → right
  - ``bearish`` & relative ≤ -threshold → right
  - 그 외 → wrong

휴장일 처리: 목표 날짜가 휴장일이면 *직후 거래일* 가격 사용 (yfinance 가
주말·공휴일 제외해 close_series 반환).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd

from tradingagents.dataflows.market_history import fetch_close_series
from tradingagents.hermes.hypothesis_store import HypothesisStore


def horizon_threshold(horizon_weeks: int) -> float:
    """호라이즌(주) → KOSPI 상대 적중 임계값 (%).

    2 주=2, 4 주=3, 8 주=5 의 anchor 사이를 선형 보간.
    h=2: 2.0, h=3: 2.5, h=4: 3.0, h=5: 3.5, h=6: 4.0, h=7: 4.5, h=8: 5.0.
    """
    return 2.0 + (horizon_weeks - 2) * 0.5


def judge_verdict(
    direction: str,
    relative_return_pct: float,
    threshold: float,
) -> str:
    """direction + relative return + threshold → 'right' / 'wrong'."""
    if direction == "bullish":
        return "right" if relative_return_pct >= threshold else "wrong"
    if direction == "bearish":
        return "right" if relative_return_pct <= -threshold else "wrong"
    if direction == "neutral":
        return "right" if abs(relative_return_pct) <= threshold / 2 else "wrong"
    raise ValueError(f"unknown direction: {direction!r}")


def _close_at_or_after(series: pd.DataFrame, target_date: str) -> Optional[dict]:
    """``target_date`` 이후 첫 거래일 (휴장일 보정) 의 ``(date, close)``.

    series 는 ``date`` 오름차순. 없으면 None.
    """
    if series.empty:
        return None
    after = series[series["date"] >= target_date]
    if after.empty:
        return None
    row = after.iloc[0]
    return {"date": str(row["date"]), "close": float(row["close"])}


class LabelingScheduler:
    """미라벨 가설에 사후 가격 라벨 (KOSPI relative + absolute) 부여.

    ``HypothesisStore`` 의 ``add_label(label_kind="auto_relative", ...)`` 를
    사용해 한 라벨 row 에 ``actual_return_pct`` (절대) + ``actual_relative_
    return_pct`` (상대) 함께 저장. 자동 라벨링은 호라이즌당 *한 번만* —
    재호출해도 중복 추가 안 함.
    """

    def __init__(
        self,
        store: HypothesisStore,
        *,
        fetcher: Callable[[str, str, str], pd.DataFrame] = fetch_close_series,
    ):
        self.store = store
        # fetcher 주입 — 테스트에서 monkeypatch 없이 직접 fake 함수 전달.
        self._fetch = fetcher

    # === Public API ===

    def label_hypothesis(
        self,
        hypothesis_id: int,
        *,
        today: Optional[str] = None,
    ) -> dict:
        """단일 가설 라벨링. 반환 dict 는 다음 ``status`` 중 하나:

          - ``labeled``: 라벨 부여 성공 (verdict / returns 포함)
          - ``too_early``: 호라이즌 경과 전
          - ``already_labeled``: 이미 auto_relative 라벨 있음
          - ``no_target_price`` / ``no_kospi``: 가격 fetch 실패 (네트워크 등)
          - ``not_found``: hypothesis_id 없음
        """
        h = self.store.get(hypothesis_id)
        if h is None:
            return {"status": "not_found"}

        # 중복 방지: 이미 auto_relative 라벨이 붙어 있으면 skip.
        existing = self.store.get_labels(hypothesis_id)
        if any(lbl["label_kind"] == "auto_relative" for lbl in existing):
            return {"status": "already_labeled"}

        as_of = h["as_of_date"]
        horizon_w = int(h["horizon_weeks"])
        # 호라이즌 종료 캘린더 날짜 (휴장일이면 _close_at_or_after 가 직후 거래일로 보정).
        target_date = (
            datetime.strptime(as_of, "%Y-%m-%d") + timedelta(weeks=horizon_w)
        ).strftime("%Y-%m-%d")

        today = today or datetime.utcnow().strftime("%Y-%m-%d")
        if today < target_date:
            return {"status": "too_early", "target_date": target_date}

        # fetch 윈도우: as_of_date 부터 target_date + 7 일 (휴장일 보정 여유).
        end_window = (
            datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=7)
        ).strftime("%Y-%m-%d")

        ticker_series = self._fetch(h["ticker"], as_of, end_window)
        kospi_series = self._fetch("kospi", as_of, end_window)

        start_t = _close_at_or_after(ticker_series, as_of)
        end_t = _close_at_or_after(ticker_series, target_date)
        if start_t is None or end_t is None:
            return {"status": "no_target_price"}

        start_k = _close_at_or_after(kospi_series, as_of)
        end_k = _close_at_or_after(kospi_series, target_date)
        if start_k is None or end_k is None:
            return {"status": "no_kospi"}

        if start_t["close"] <= 0 or start_k["close"] <= 0:
            return {"status": "no_target_price"}

        target_ret = (end_t["close"] / start_t["close"] - 1) * 100
        kospi_ret = (end_k["close"] / start_k["close"] - 1) * 100
        relative = target_ret - kospi_ret

        threshold = horizon_threshold(horizon_w)
        verdict = judge_verdict(h["direction"], relative, threshold)

        self.store.add_label(
            hypothesis_id,
            "auto_relative",
            verdict=verdict,
            actual_return_pct=round(target_ret, 4),
            actual_relative_return_pct=round(relative, 4),
            labeled_at_date=end_t["date"],
        )
        return {
            "status": "labeled",
            "verdict": verdict,
            "actual_return_pct": round(target_ret, 4),
            "actual_relative_return_pct": round(relative, 4),
            "threshold_pct": threshold,
            "labeled_at_date": end_t["date"],
        }

    def run_pending(self, *, today: Optional[str] = None) -> dict:
        """모든 미라벨 가설에 대해 ``label_hypothesis`` 일괄 호출.

        반환 dict: ``{labeled, too_early, already_labeled, no_target_price,
        no_kospi}`` 각 카운트.
        """
        counts = {
            "labeled": 0, "too_early": 0, "already_labeled": 0,
            "no_target_price": 0, "no_kospi": 0,
        }
        for h in self.store.list():
            result = self.label_hypothesis(h["id"], today=today)
            status = result["status"]
            if status in counts:
                counts[status] += 1
        return counts
