"""섹터 로테이션 — SPDR 섹터 ETF 상대강도(RRG 근사).

11개 SPDR 섹터 ETF 의 SPY 대비 상대강도(RS) 모멘텀으로 어느 섹터로 리더십이
회전 중인지 본다. RS-momentum 양수 = Improving→Leading 사분면(노트). 개별 buzz 와
달리 섹터 로테이션은 leading~coincident·월간 지속·저잡음이라 노트가 가장 높이
평가한 신호군이다.

yfinance 사용(US watchlist 전용 — KIS 백테스트 유니버스엔 들어가지 않음).
leadingness = L. entity_type = 'sector'.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .base import LEADING, SignalRow

logger = logging.getLogger(__name__)

_SOURCE = "sector_rotation"
_METRIC = "rs_momentum"
_BENCH = "SPY"

# SPDR 11 섹터 ETF → 섹터명
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "HealthCare",
    "XLI": "Industrials",
    "XLY": "ConsumerDiscretionary",
    "XLP": "ConsumerStaples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "RealEstate",
    "XLC": "CommunicationServices",
}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def collect_sector_rotation(
    *,
    market: str = "us",
    asof_date: Optional[str] = None,
    rs_window: int = 21,
    top_n: int = 11,
    lookback_days: int = 120,  # 시그니처 일관성용
    timeout: float = 30.0,  # 시그니처 일관성용(yfinance 자체 타임아웃 사용)
) -> tuple[list[SignalRow], bool]:
    """섹터별 SPY 대비 RS-momentum(%) 상위를 SignalRow 로 수집.

    반환 ``(rows, fetch_ok)``. 데이터 수집/계산 실패 시 ``([], False)``.
    """
    asof = asof_date or _today()
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance 미설치 — sector_rotation degrade")
        return [], False

    tickers = list(SECTOR_ETFS) + [_BENCH]
    try:
        raw = yf.download(tickers, period="6mo", progress=False)
        data = raw["Close"] if "Close" in raw else None
    except Exception as exc:  # yfinance 예외 타입 다양 → 광범위 캐치
        logger.warning("yfinance download 실패: %s", exc)
        return [], False
    if data is None or data.empty or _BENCH not in data.columns or len(data) <= rs_window:
        return [], False

    rs = data.div(data[_BENCH], axis=0)  # 각 섹터의 SPY 대비 상대강도
    try:
        mom = rs.iloc[-1] / rs.iloc[-1 - rs_window] - 1.0  # trailing RS-momentum
    except (IndexError, KeyError):
        return [], False

    scored: list[tuple[float, str]] = []
    for etf, name in SECTOR_ETFS.items():
        if etf not in mom.index:
            continue
        v = mom[etf]
        if v != v:  # NaN
            continue
        scored.append((float(v), name))
    if not scored:
        return [], False

    scored.sort(reverse=True)
    rows: list[SignalRow] = []
    for rank, (v, name) in enumerate(scored[:top_n], start=1):
        rows.append(
            SignalRow(
                release_date=asof,
                asof_date=asof,
                market=market,
                entity=name,
                entity_type="sector",
                source=_SOURCE,
                metric=_METRIC,
                leadingness=LEADING,
                raw_value=None,
                abnormal_value=round(v * 100, 3),  # RS-momentum %
                rank=rank,
            )
        )
    return rows, True
