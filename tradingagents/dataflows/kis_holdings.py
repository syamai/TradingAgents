"""KIS 수급 데이터 → 일별 누적 보유량/보유비율(holdings) derived 테이블 계산.

raw ``investor`` 테이블의 일별 순매수(``*_qty``)에서 derived 컬럼:
  - ``*_cum_qty``: 첫 거래일부터 그 날짜까지의 ``*_qty`` 누적합(부호 보존)
  - ``*_pct``: ``|*_cum_qty| / sum(|10 sub *_cum_qty|) × 100`` — 영향력 비중

10 sub (가장 세분화된 단위, 시장 제로섬 만족):
  외국인: registered / unregistered
  기관: pension / private_equity / investment_trust / securities / bank / insurance
  개인: retail
  기타법인: other_corp

11 주체 (10 sub + 외국인 통합):
  ``foreign`` = ``foreign_registered`` + ``foreign_unregistered`` (정보용; 분모 X)

추가 컬럼:
  ``price_change_pct``: ``(close − prev_close) / prev_close × 100``

임의 구간(lookback) 보유 변화는 ``holdings_window`` 가 차분
``cum[end] − cum[start-1]`` 으로 계산. 비율은 차분 후 abs 합 재분배.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

SUBS_10: tuple[str, ...] = (
    "foreign_registered", "foreign_unregistered",
    "pension", "private_equity", "investment_trust", "securities",
    "bank", "insurance",
    "retail", "other_corp",
)

INFO_TOTALS: tuple[str, ...] = ("foreign",)
"""분모에 포함되지 않는 통합 카테고리. ``foreign`` 만 등록+비등록 합으로 도출."""

ALL_SUBJECTS: tuple[str, ...] = SUBS_10 + INFO_TOTALS


def _net_col(s: str) -> str:
    return f"{s}_net_qty"


def _cum_col(s: str) -> str:
    return f"{s}_cum_qty"


def _pct_col(s: str) -> str:
    return f"{s}_pct"


OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

HOLDINGS_COLUMNS: list[str] = (
    ["date"] + OHLCV_COLUMNS + ["price_change_pct"]
    + [_net_col(s) for s in ALL_SUBJECTS]
    + [_cum_col(s) for s in ALL_SUBJECTS]
    + [_pct_col(s) for s in ALL_SUBJECTS]
)


def compute_holdings(investor_df: pd.DataFrame) -> pd.DataFrame:
    """``investor`` raw 행 → holdings derived. 한 ticker 분량.

    입력 컬럼 가정: ``date``, ``close``, ``{sub}_qty`` for sub in SUBS_10.
    ``foreign`` 통합은 ``foreign_registered`` + ``foreign_unregistered`` 합으로 도출.

    분모 = 10 sub의 ``|cum_qty|`` 행별 합. 0 분모(모든 cum=0) → pct 0%.
    """
    if investor_df.empty:
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)

    df = investor_df.sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": df["date"]})
    # OHLCV — backfill 이전 종목은 close만 있을 수 있음. 누락은 NaN.
    for col in OHLCV_COLUMNS:
        out[col] = df[col] if col in df.columns else pd.NA

    # 가격 변동률 — 첫 행은 0 (이전 행 없음)
    prev = df["close"].shift(1)
    out["price_change_pct"] = (
        ((df["close"] - prev) / prev * 100).fillna(0).round(4)
    )

    # 10 sub: net + cumsum
    for s in SUBS_10:
        q = df[f"{s}_qty"]
        out[_net_col(s)] = q.astype("Int64")
        out[_cum_col(s)] = q.cumsum().astype("Int64")

    # 외국인 통합 — 등록 + 비등록 (raw raw foreign_qty와 별개로, sub 합으로 도출)
    out[_net_col("foreign")] = (
        out[_net_col("foreign_registered")] + out[_net_col("foreign_unregistered")]
    ).astype("Int64")
    out[_cum_col("foreign")] = (
        out[_cum_col("foreign_registered")] + out[_cum_col("foreign_unregistered")]
    ).astype("Int64")

    # 분모: 10 sub abs cum 합 (0 분모는 1로 치환 — 결과는 0%)
    abs_sum = sum(out[_cum_col(s)].abs() for s in SUBS_10)
    safe_denom = abs_sum.where(abs_sum != 0, other=1)
    for s in SUBS_10:
        out[_pct_col(s)] = (out[_cum_col(s)].abs() / safe_denom * 100).round(4)

    # 외국인 통합 pct = 등록 pct + 비등록 pct (sub 합과 자연 일치)
    out[_pct_col("foreign")] = (
        out[_pct_col("foreign_registered")] + out[_pct_col("foreign_unregistered")]
    ).round(4)

    return out[HOLDINGS_COLUMNS]


def holdings_window(
    holdings_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """``compute_holdings`` 결과 → 임의 [start, end] 구간 보유 변화.

    구간 cum_qty = ``cum[end] − cum[start-1]`` (start 직전 행의 누적을
    베이스라인). pct는 구간 cum_qty의 abs 합으로 재분배. ``foreign`` 통합도
    재계산.
    """
    if holdings_df.empty:
        return holdings_df

    df = holdings_df.sort_values("date").reset_index(drop=True)

    baseline: dict[str, int] = {}
    if start_date is not None:
        baseline_mask = df["date"] < start_date
        if baseline_mask.any():
            base_row = df.loc[baseline_mask].iloc[-1]
            baseline = {_cum_col(s): int(base_row[_cum_col(s)]) for s in ALL_SUBJECTS}
        else:
            baseline = {_cum_col(s): 0 for s in ALL_SUBJECTS}
        df = df.loc[~baseline_mask].reset_index(drop=True)
    else:
        baseline = {_cum_col(s): 0 for s in ALL_SUBJECTS}

    if end_date is not None:
        df = df.loc[df["date"] <= end_date].reset_index(drop=True)

    if df.empty:
        return df

    out = pd.DataFrame({"date": df["date"]})
    for col in OHLCV_COLUMNS:
        out[col] = df[col] if col in df.columns else pd.NA
    out["price_change_pct"] = df["price_change_pct"]

    for s in ALL_SUBJECTS:
        out[_net_col(s)] = df[_net_col(s)]
        out[_cum_col(s)] = (df[_cum_col(s)] - baseline[_cum_col(s)]).astype("Int64")

    # 10 sub abs 합으로 재분배 — foreign 통합은 그 후에 별도 sum
    abs_sum = sum(out[_cum_col(s)].abs() for s in SUBS_10)
    safe_denom = abs_sum.where(abs_sum != 0, other=1)
    for s in SUBS_10:
        out[_pct_col(s)] = (out[_cum_col(s)].abs() / safe_denom * 100).round(4)
    out[_pct_col("foreign")] = (
        out[_pct_col("foreign_registered")] + out[_pct_col("foreign_unregistered")]
    ).round(4)

    return out[HOLDINGS_COLUMNS]


def holdings_at(holdings_df: pd.DataFrame, date: str) -> Optional[dict]:
    """단일 date의 한 행 dict 반환. 없으면 None."""
    if holdings_df.empty:
        return None
    row = holdings_df.loc[holdings_df["date"] == date]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
