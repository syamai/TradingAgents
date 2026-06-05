"""옵션 O/S — 옵션/주식 거래량 비율 (informed 관심 대리지표).

종목별 당일 **옵션 총거래량 / 주식 거래량**(O/S, Johnson-So). O/S 가 높으면
informed trading 가능성. 단 노트는 이 leadingness 를 "이 코퍼스에선 미검증,
보수적으로 C 취급"이라 평가 → leadingness=C.

yfinance 옵션체인은 *현재 스냅샷만*(과거 시계열 없음)이라 자기 baseline 대비
비정상 O/S 는 forward 누적 후에야 가능. 현재는 당일 O/S 비율을 raw/abnormal 로
적재한다. per-ticker × 만기 라 느려서 universe(hot_candidates) + 가까운 만기
``max_expiries`` 개로만 제한한다(단기 옵션이 거래량 대부분).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .base import COINCIDENT, SignalRow

logger = logging.getLogger(__name__)

_SOURCE = "options_os"
_METRIC = "os_ratio"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def collect_options_os(
    *,
    market: str = "us",
    asof_date: Optional[str] = None,
    universe: Optional[list] = None,
    max_expiries: int = 4,
    lookback_days: int = 1,  # 시그니처 일관성용
    timeout: float = 20.0,  # 시그니처 일관성용
) -> tuple[list[SignalRow], bool]:
    """universe 종목의 당일 O/S(옵션/주식 거래량)를 수집.

    universe 없으면 빈(첫 tick — 실패 아님). 종목별 실패는 건너뛴다.
    """
    asof = asof_date or _today()
    if not universe:
        return [], True
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance 미설치 — options_os degrade")
        return [], False

    rows: list[SignalRow] = []
    for tk in universe:
        try:
            t = yf.Ticker(str(tk))
            expiries = list(t.options or [])[:max_expiries]
            if not expiries:
                continue
            opt_vol = 0.0
            for exp in expiries:
                oc = t.option_chain(exp)
                opt_vol += float(oc.calls["volume"].fillna(0).sum())
                opt_vol += float(oc.puts["volume"].fillna(0).sum())
            hist = t.history(period="1d")
            if hist.empty:
                continue
            stk_vol = float(hist["Volume"].iloc[-1])
            if stk_vol <= 0 or opt_vol <= 0:
                continue
            os_ratio = opt_vol / stk_vol
            rows.append(
                SignalRow(
                    release_date=asof, asof_date=asof, market=market,
                    entity=str(tk).upper(), source=_SOURCE, metric=_METRIC,
                    leadingness=COINCIDENT, raw_value=round(opt_vol, 0),
                    abnormal_value=round(os_ratio, 5), rank=None,
                )
            )
        except Exception as exc:  # yfinance 예외 다양 → 종목 스킵
            logger.warning("options_os 실패 %s: %s", tk, exc)
            continue

    if not rows:
        return [], True
    rows.sort(key=lambda r: r.abnormal_value or 0.0, reverse=True)
    for rank, r in enumerate(rows, start=1):
        r.rank = rank
    return rows, True
