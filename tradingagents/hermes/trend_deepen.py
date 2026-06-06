"""신규 진입 트렌드 종목 심층분석(C) — 모니터링 레이더의 '왜'를 채운다.

오늘 fade 와치리스트에 **새로** 진입한 종목(직전 N일엔 없던)을 골라, 한국 시장
분석가 5종(로컬 Ollama=무료)으로 수급·감성·시장·뉴스·재무를 읽고 mini LLM 으로
2~3줄 요약(관심 유입 배경)으로 종합해 ``trend_analyses`` 에 저장한다. 트렌드
내러티브(trend-analyzer 스킬)가 이 요약을 인용한다.

비용 통제: (1) 신규 진입만(매일 전수 아님), (2) ``top_n`` 캡, (3) **시간예산** —
종목당 분석가 5종이 로컬이라 느려 한 tick 에 다 못 끝내면 예산까지만 하고 나머지는
다음 tick(캐시라 한 번만), (4) ``(market,entity,asof)`` 캐시 hit 은 분석가·LLM 0,
(5) ``TREND_DEEPEN=0`` 으로 끌 수 있음. 분석가는 KIS(한국 수급) 기반이라 **KR 전용**.

순환 import 회피: fade_ranking·analyst_runner·LLM 팩토리는 함수 내부 지연 import.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..dataflows.trend_store import TrendStore

logger = logging.getLogger(__name__)

_ANALYSTS = ("supply_demand", "sentiment", "market", "news", "fundamentals")
_KST = timezone(timedelta(hours=9))
_DEFAULT_TIME_BUDGET = 200.0  # 초 — cron 540s 안(수집22s+예산200+내러티브150+slack)
                              # tick 은 추가로 timeout 백스톱으로 deepen 을 감싼다.
_DEFAULT_TOP_N = 5
_DEFAULT_LOOKBACK = 7


def _today() -> str:
    return datetime.now(_KST).date().isoformat()


def new_entrants(
    market: str,
    asof: Optional[str] = None,
    *,
    lookback_days: int = _DEFAULT_LOOKBACK,
    top_n: int = _DEFAULT_TOP_N,
    store: Optional[TrendStore] = None,
) -> list:
    """오늘 fade 통과 종목 중 **직전 lookback_days 일 fade 에 없던** 신규 진입(코드).

    fade_score 상위 ``top_n`` 만. 순수 쿼리(새 상태 불필요).
    """
    from .trend_rank import _resolve_min_sources, fade_ranking

    store = store or TrendStore()
    ms = _resolve_min_sources(market, store)
    today = fade_ranking(market, asof_date=asof, top_n=100, min_sources=ms, store=store)
    if today.empty:
        return []
    asof_actual = today["asof_date"].iloc[0]
    seen: set = set()
    for d in store.asof_dates(market, lt=asof_actual, lookback_days=lookback_days):
        prev = fade_ranking(market, asof_date=d, top_n=100, min_sources=ms, store=store)
        if not prev.empty:
            seen.update(prev["entity"].tolist())
    fresh = today[~today["entity"].isin(seen)].head(top_n)
    return list(fresh["entity"])


def _analyst_llm():
    """분석가 LLM(로컬 Ollama gemma 기본 — 비용 0). mcp_server 와 동일 env 규약."""
    from ..llm_clients.factory import create_llm_client

    provider = os.environ.get("HERMES_ANALYST_PROVIDER", "ollama")
    model = os.environ.get("HERMES_ANALYST_MODEL", "gemma4:26b-a4b")
    base_url = os.environ.get("HERMES_ANALYST_BASE_URL", "http://localhost:11434/v1")
    return create_llm_client(provider, model, base_url=base_url).get_llm()


def _synthesis_llm():
    """종합 LLM(gpt-5.4-mini — 저렴). 브리지와 동일 DEFAULT_CONFIG."""
    from ..default_config import DEFAULT_CONFIG
    from ..llm_clients.factory import create_llm_client

    cfg = DEFAULT_CONFIG
    return create_llm_client(
        cfg["llm_provider"], cfg["quick_think_llm"], cfg.get("backend_url")
    ).get_llm()


def _analyst_ticker(code: str, store: TrendStore) -> str:
    """6자리 코드 → 분석가용 ticker. KIS 메타에 코스닥 표시 있으면 .KQ, 아니면 .KS.

    supply_demand·news·fundamentals 는 6자리로 동작, market(yfinance)만 접미사
    필요 — 미상이면 .KS(대형주 다수)로 best-effort, 실패는 분석가가 graceful.
    """
    try:
        from ..dataflows.kis_history_store import KisHistoryStore

        meta = KisHistoryStore().get_ticker_metadata(code) or {}
        mk = str(meta.get("market") or meta.get("exchange") or "").upper()
        if "KOSDAQ" in mk or mk.endswith("KQ"):
            return f"{code}.KQ"
    except Exception:
        pass
    return f"{code}.KS"


_SYNTH_PROMPT = (
    "다음은 한국 종목 {name}({code})에 대한 분석가 5종 보고서다. 이 종목이 최근 "
    "투자자 관심을 끄는 **배경**을 2~3줄로 한국어 요약하라. 수급(외인/기관)·뉴스/"
    "재료·시장 위치 위주로, 보고서에 실제 있는 내용만(없으면 생략, 환각·매매추천 "
    "금지). '모니터링' 관점.\n\n{reports}"
)


def _fallback_summary(name: str, code: str, reports: dict) -> str:
    """종합 LLM 실패 시 폴백 — 분석가(무료) 결과 일부를 저장해 재실행을 막는다.

    수급·뉴스 보고서 앞부분을 짧게 결합. 비싼 분석가 작업을 캐시에 보존(B4)해
    다음 tick 이 같은 종목을 재분석하지 않게 한다.
    """
    parts = []
    for key in ("supply_demand", "news", "market"):
        rep = (reports.get(key) or "").strip()
        if rep and not rep.startswith("<"):  # <not applicable>/<unavailable> 제외
            parts.append(rep.replace("\n", " ")[:160])
        if len(parts) >= 2:
            break
    body = " / ".join(parts) if parts else "분석가 데이터 수집됨(종합 보류)"
    return f"(자동 종합 보류) {name}({code}) — {body}"


def _synthesize(code: str, name: str, reports: dict) -> Optional[str]:
    """분석가 보고서 dict → mini LLM 2~3줄 종합. 실패 시 None."""
    body = "\n\n".join(
        f"## {k}\n{(v or '').strip()[:1500]}" for k, v in reports.items()
    )
    try:
        llm = _synthesis_llm()
        out = llm.invoke(_SYNTH_PROMPT.format(name=name, code=code, reports=body))
        text = getattr(out, "content", None) or str(out)
        return text.strip() or None
    except Exception as exc:
        logger.warning("종합 LLM 실패 %s: %s", code, exc)
        return None


def deepen(
    market: str = "kr",
    asof: Optional[str] = None,
    *,
    top_n: int = _DEFAULT_TOP_N,
    lookback_days: int = _DEFAULT_LOOKBACK,
    time_budget_s: float = _DEFAULT_TIME_BUDGET,
    store: Optional[TrendStore] = None,
) -> dict:
    """KR 신규 진입 종목을 시간예산 안에서 심층분석·캐시. 반환=상태 dict.

    분석가는 로컬(무료), 종합은 mini(저렴). 캐시 hit 은 둘 다 0. 예산 초과분은
    다음 tick(캐시라 한 번만). ``TREND_DEEPEN=0`` 이면 비활성.
    """
    if os.environ.get("TREND_DEEPEN", "1") != "1":
        return {"status": "disabled", "analyzed": 0}
    if market != "kr":  # 분석가는 한국 수급 기반 — KR 전용
        return {"status": "skip_market", "market": market, "analyzed": 0}

    store = store or TrendStore()
    asof = asof or _today()
    codes = new_entrants(
        market, asof=asof, lookback_days=lookback_days, top_n=top_n, store=store
    )
    if not codes:
        return {"status": "ok", "analyzed": 0, "entrants": 0}

    # 적재된 실제 asof(오늘 데이터 없으면 최신) — 캐시 키 일관성
    from .trend_rank import _resolve_min_sources, fade_ranking
    fade = fade_ranking(
        market, asof_date=asof, top_n=100,
        min_sources=_resolve_min_sources(market, store), store=store,
    )
    asof_actual = fade["asof_date"].iloc[0] if not fade.empty else asof

    from .analyst_runner import run_analyst
    from .kr_peer_bridge import KrPeerCache
    names = KrPeerCache().reverse(codes)

    t0 = time.monotonic()
    analyzed = 0
    cached = 0
    deferred = 0
    for code in codes:
        if store.get_analysis(market, code, asof_actual) is not None:
            cached += 1
            continue
        if time.monotonic() - t0 > time_budget_s:
            deferred += 1
            continue  # 예산 초과 — 다음 tick(캐시라 한 번만)
        name = (names.get(code) or {}).get("name") or code
        ticker = _analyst_ticker(code, store)
        try:
            llm = _analyst_llm()
            reports = {
                a: run_analyst(ticker, asof_actual, a, llm=llm) for a in _ANALYSTS
            }
        except Exception as exc:
            logger.warning("분석가 실행 실패 %s: %s", code, exc)
            continue
        # 종합 실패해도 분석가(무료) 결과로 폴백 저장 → 캐시되어 재분석 안 함(B4).
        summary = _synthesize(code, name, reports) or _fallback_summary(name, code, reports)
        store.save_analysis(market, code, asof_actual, summary, name=name)
        analyzed += 1
    return {
        "status": "ok",
        "asof_date": asof_actual,
        "entrants": len(codes),
        "analyzed": analyzed,
        "cached": cached,
        "deferred": deferred,
    }


def main(argv: Optional[list] = None) -> int:
    """cron tick 진입점 — 신규 진입 심층분석 1회. 항상 exit 0(자가복구)."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="트렌드 신규 진입 심층분석(C)")
    ap.add_argument("--market", default="kr")
    ap.add_argument("--asof-date", default=None)
    ap.add_argument("--top-n", type=int, default=_DEFAULT_TOP_N)
    ap.add_argument("--time-budget", type=float, default=_DEFAULT_TIME_BUDGET)
    args = ap.parse_args(argv)
    try:
        res = deepen(
            args.market, asof=args.asof_date, top_n=args.top_n,
            time_budget_s=args.time_budget,
        )
    except Exception as exc:  # cron 이 죽지 않게
        res = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
