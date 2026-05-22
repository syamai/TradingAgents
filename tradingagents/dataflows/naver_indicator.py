"""네이버 OHLCV → stockstats 기반 기술적 지표 계산.

yfinance의 ``get_stock_stats_indicators_window`` 출력 포맷과 동일하게
``"## {indicator} values from {start} to {end}:\\n\\n{date: value}\\n..."``
형식으로 반환한다. LLM이 두 소스를 같은 스키마로 비교할 수 있게.

스코프: 12개 지표 (yfinance 어댑터와 동일 셋).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .naver_stock import fetch_naver_ohlcv_df


_SUPPORTED = {
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi", "boll", "boll_ub", "boll_lb", "atr", "vwma", "mfi",
}


def _format_value(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    return str(v)


def compute_indicator_from_naver(
    symbol: str, indicator: str, curr_date: str, look_back_days: int = 30
) -> str:
    """네이버 OHLCV를 받아 지표 시계열을 계산해 문자열로 반환."""
    if indicator not in _SUPPORTED:
        return f"<unavailable: indicator '{indicator}' not in supported set>"

    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    # 200 SMA 같은 장기 지표를 안정적으로 계산하려면 충분한 히스토리가 필요.
    history_start = end_dt - timedelta(days=max(look_back_days + 250, 300))

    df = fetch_naver_ohlcv_df(
        symbol,
        history_start.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )
    if df.empty:
        return f"<unavailable: no Naver OHLCV for {symbol}>"

    # stockstats는 lowercase 컬럼 + Date column을 선호. wrap()이 자체적으로
    # 대문자도 처리하지만 명시적으로 변환해 일관성 확보.
    df_for_stats = df.reset_index().rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })

    from stockstats import wrap
    sdf = wrap(df_for_stats)
    sdf[indicator]  # stockstats가 지연 계산하므로 트리거

    sdf["_date_str"] = pd.to_datetime(sdf["Date"]).dt.strftime("%Y-%m-%d")
    value_by_date = dict(zip(sdf["_date_str"], sdf[indicator]))

    window_start = end_dt - timedelta(days=look_back_days)
    lines = []
    cur = end_dt
    while cur >= window_start:
        ds = cur.strftime("%Y-%m-%d")
        if ds in value_by_date:
            lines.append(f"{ds}: {_format_value(value_by_date[ds])}")
        else:
            lines.append(f"{ds}: N/A: Not a trading day (weekend or holiday)")
        cur -= timedelta(days=1)

    return (
        f"## {indicator} values from {window_start.strftime('%Y-%m-%d')} to {curr_date} "
        f"(source: Naver Finance):\n\n"
        + "\n".join(lines)
        + "\n"
    )
