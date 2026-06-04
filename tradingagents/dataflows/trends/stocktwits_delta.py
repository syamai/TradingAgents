"""StockTwits — 소셜 메시지 볼륨 + 감성 방향.

기존 ``dataflows/stocktwits.py`` 의 collect_stocktwits_messages 를 재사용해 종목별
당일 메시지 볼륨과 bullish 비율을 수집한다. universe 는 그날 뜬 후보
(hot_candidates)로 한정(per-ticker 라 전수 조회 금지).

SV-Delta(자기 trailing baseline 대비 비정상 볼륨)는 일별 볼륨이 store 에 누적된
뒤에야 계산 가능하므로, Phase2 현재는 **당일 볼륨(raw)** 을 abnormal_value 로 쓰고
forward 로 baseline 을 쌓는다. 감성(bullish 비율)은 extra 로 raw_value 에 같이
싣지 않고 별도 metric 행으로 분리하지 않는다(Phase1 단순화) — 볼륨 신호로 등록해
fade_ranking 의 ≥2소스 동의를 강화하는 게 목적.

leadingness = C(coincident/contrarian, 소셜 buzz).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..stocktwits import collect_stocktwits_messages
from .base import COINCIDENT, SignalRow

_SOURCE = "stocktwits_delta"
_METRIC = "social_volume"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _bull_ratio(messages: list) -> Optional[float]:
    bull = bear = 0
    for m in messages:
        ent = (m.get("entities") or {}).get("sentiment") or {}
        basic = ent.get("basic") if isinstance(ent, dict) else None
        if basic == "Bullish":
            bull += 1
        elif basic == "Bearish":
            bear += 1
    return bull / (bull + bear) if (bull + bear) else None


def collect_stocktwits_delta(
    *,
    market: str = "us",
    asof_date: Optional[str] = None,
    universe: Optional[list] = None,
    timeout: float = 10.0,
    lookback_days: int = 1,
) -> tuple[list[SignalRow], bool]:
    """universe 종목의 당일 StockTwits 메시지 볼륨을 수집.

    universe 없으면 빈(첫 tick — 실패 아님). 종목별 실패는 건너뛴다.
    """
    asof = asof_date or _today()
    if not universe:
        return [], True
    rows: list[SignalRow] = []
    for tk in universe:
        msgs, ok = collect_stocktwits_messages(
            str(tk), lookback_days=lookback_days, max_pages=2, timeout=timeout
        )
        if not ok or not msgs:
            continue
        vol = float(len(msgs))
        rows.append(
            SignalRow(
                release_date=asof, asof_date=asof, market=market, entity=str(tk).upper(),
                source=_SOURCE, metric=_METRIC, leadingness=COINCIDENT,
                raw_value=vol, abnormal_value=vol, rank=None,
            )
        )
    if not rows:
        return [], True  # universe 는 있었으나 메시지 0 — 실패 아님
    rows.sort(key=lambda r: r.raw_value or 0.0, reverse=True)
    for rank, r in enumerate(rows, start=1):
        r.rank = rank
    return rows, True
