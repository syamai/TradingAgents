"""횡단면 팩터 랭킹 백테스트 — per-name 신호 엔진과 별개.

매 리밸런스 시점에 universe 전체를 특성(char)으로 랭크해 분위(quantile) 바스켓을
동일가중 long 보유한다. 진짜 횡단면 팩터(저변동성/모멘텀 등)는 시계열 임계 신호가
아니라 이 "줄 세워 상/하위 매수" 형태라야 학술 정의와 맞는다.

look-ahead 0 불변식: row r(종가확정)에서 랭크 → 진입가 close[r+1], 청산가 다음
리밸런스 close[r2+1]. 모든 char 는 trailing(rolling/shift). 게이트는 연도별 시장초과
IR(정보비율) — 모든 연도 IR>0 AND 중앙 IR>0.5 (per-name fair gate 와 동일 철학).
"""
from __future__ import annotations

import statistics as _st

import numpy as np
import pandas as pd

from tradingagents.hermes import backtest_engine as bt

TX_ONE_WAY = bt.TX_COST_ONE_WAY          # 편도 위탁수수료
RT_COST = bt.BUY_COST + bt.SELL_COST     # 왕복 거래비용(매수수수료 + 매도수수료+거래세)
GATE_EXCESS_MIN_IR = 0.5                  # per-name 게이트와 동일 임계


def _char_series(df: pd.DataFrame, kind: str, window: int) -> pd.Series:
    """종목 1개의 trailing 특성 시계열 (look-ahead 0)."""
    close = bt._num(df["close"])
    if kind == "vol":                                    # 저변동성: 낮을수록 매수
        return close.pct_change().rolling(window).std() * (252 ** 0.5)
    if kind == "mom":                                    # 모멘텀: 높을수록 매수
        return close / close.shift(window) - 1
    if kind == "short":                                  # 저공매도압력: 낮을수록 매수
        if "short_volume_ratio" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return bt._num(df["short_volume_ratio"]).rolling(window).mean()
    raise ValueError(f"unknown char kind: {kind!r}")


def _wide(holdings: dict, kind: str, window: int):
    """종목별 close/char 를 공통 날짜 캘린더(union) 위 wide 행렬로."""
    closes, chars = {}, {}
    for tk, df in holdings.items():
        idx = df["date"].astype(str).to_numpy()
        sc = pd.Series(bt._num(df["close"]).to_numpy(), index=idx)
        ch = pd.Series(_char_series(df, kind, window).to_numpy(), index=idx)
        closes[tk] = sc[~sc.index.duplicated()]
        chars[tk] = ch[~ch.index.duplicated()]
    close_m = pd.DataFrame(closes).sort_index()
    char_m = pd.DataFrame(chars).reindex(close_m.index)
    return close_m, char_m


# 채점 상한: 2025-06 이후 한국 증시 비정상 급등 구간은 OOS/검증에서 제외(과대평가 방지).
# 그 이후는 forward 관찰 전용(date_lo 로 별도 채점). (CLAUDE.md Conventions 정합)
SCORE_DATE_HI = "2025-06-30"


def _liq_wide(holdings: dict, window: int) -> pd.DataFrame:
    """종목별 trailing W일 평균 거래대금(close×volume) wide 행렬 (look-ahead 0)."""
    liqs = {}
    for tk, df in holdings.items():
        idx = df["date"].astype(str).to_numpy()
        tv = pd.Series((bt._num(df["close"]) * bt._num(df["volume"])).to_numpy(), index=idx)
        tv = tv[~tv.index.duplicated()].rolling(window).mean()
        liqs[tk] = tv
    return pd.DataFrame(liqs)


def run_xsec_rank_backtest(holdings: dict, kospi, *, kind: str, window: int,
                           quantile: float = 0.2, rebalance: int = 20,
                           direction: str = "low",
                           date_lo: str | None = None,
                           date_hi: str | None = SCORE_DATE_HI,
                           liq_window: int = 20, min_liq_pct: float = 0.0,
                           min_liq_krw: float = 0.0,
                           min_liq_median_mult: float = 0.0) -> dict:
    """횡단면 랭킹 백테스트. 반환: 메트릭 dict (excess IR 게이트 포함).

    채점 구간은 [date_lo, date_hi]. 기본 상한 2025-06-30(급등구간 제외). char 는
    full 시계열에서 trailing 계산 후 구간만 채점하므로 date_lo 근처도 look-ahead 0.
    forward 관찰은 date_lo=SCORE_DATE_HI, date_hi=None 로 호출.

    유동성 필터(전부 trailing, look-ahead 0):
      - min_liq_pct>0        : 그 시점 eligible 내 거래대금 백분위 하위 제외(상대).
      - min_liq_krw>0        : trailing 거래대금 절대 임계(원) 미만 제외(고정 절대 — 시기편향).
      - min_liq_median_mult>0: 그날 eligible 거래대금 중앙값의 배수 미만 제외
                               (절대 성격이되 시장 수준에 따라 자동 스케일 → 시기 일관).
    """
    if direction not in ("low", "high"):
        raise ValueError("direction must be low|high")
    close_m, char_m = _wide(holdings, kind, window)
    _use_liq = min_liq_pct > 0 or min_liq_krw > 0 or min_liq_median_mult > 0
    liq_m = _liq_wide(holdings, liq_window) if _use_liq else None
    if date_lo is not None:
        close_m, char_m = close_m[close_m.index >= date_lo], char_m[char_m.index >= date_lo]
    if date_hi is not None:
        close_m, char_m = close_m[close_m.index <= date_hi], char_m[char_m.index <= date_hi]
    dates = list(close_m.index)
    if liq_m is not None:
        liq_m = liq_m.reindex(dates)
    kseries = None
    if kospi is not None:
        kk = kospi.copy()
        ks = pd.Series(bt._num(kk["close"]).to_numpy(), index=kk["date"].astype(str).to_numpy())
        kseries = ks[~ks.index.duplicated()].reindex(dates)

    per_ret, per_ex, per_year, n_pick, pick_liq = [], [], [], [], []
    for r in range(window + 1, len(dates) - rebalance - 1, rebalance):
        r2 = r + rebalance
        char_row, entry, exit_ = char_m.iloc[r], close_m.iloc[r + 1], close_m.iloc[r2 + 1]
        elig = char_row.notna() & entry.notna() & exit_.notna() & (entry > 0)
        names = char_row[elig]
        liq_rank = None
        if liq_m is not None and len(names) >= 10:
            liq_row = liq_m.iloc[r][names.index]
            liq_rank = liq_row.rank(pct=True)                     # 그 시점 eligible 내 유동성 백분위
            keep = pd.Series(True, index=names.index)
            if min_liq_pct > 0:
                keep &= (liq_rank >= min_liq_pct).reindex(names.index).fillna(False)
            if min_liq_krw > 0:                                   # 고정 절대 거래대금(원) 임계
                keep &= (liq_row >= min_liq_krw).reindex(names.index).fillna(False)
            if min_liq_median_mult > 0:                           # 그날 중앙값 배수(시기 일관 절대)
                thr = min_liq_median_mult * float(liq_row.median())
                keep &= (liq_row >= thr).reindex(names.index).fillna(False)
            names = names[keep]
        if len(names) < 10:                              # 랭킹 의미 없는 얇은 날 제외
            continue
        k = max(1, int(round(len(names) * quantile)))
        picks = (names.nsmallest(k) if direction == "low" else names.nlargest(k)).index
        rets = (exit_[picks] / entry[picks] - 1.0) - RT_COST   # round-trip 비용(매도세 포함)
        pr = float(rets.mean())
        per_ret.append(pr)
        n_pick.append(len(picks))
        if liq_rank is not None:
            pick_liq.append(float(liq_rank.reindex(picks).mean()))
        if kseries is not None and pd.notna(kseries.iloc[r + 1]) and pd.notna(kseries.iloc[r2 + 1]):
            per_ex.append(pr - (float(kseries.iloc[r2 + 1]) / float(kseries.iloc[r + 1]) - 1.0))
        else:
            per_ex.append(float("nan"))
        per_year.append(dates[r + 1][:4])

    ppy = 252.0 / rebalance                              # 연 리밸런스 횟수
    n = len(per_ret)
    if n < 4:
        return {"gate_passed": False, "n_periods": n, "reason": "too few periods"}

    arr = np.array(per_ret, float)
    eq = np.cumprod(1 + arr)
    sharpe = float(arr.mean() / arr.std() * (ppy ** 0.5)) if arr.std() > 0 else None
    mdd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100)
    cum = float((eq[-1] - 1) * 100)

    ex = np.array([e for e in per_ex if not np.isnan(e)], float)
    excess_ir = float(ex.mean() / ex.std() * (ppy ** 0.5)) if len(ex) > 2 and ex.std() > 0 else None

    # 연도별 시장초과 IR (≥4 기간 있는 연도만) — 게이트용
    by_year = {}
    for y in sorted(set(per_year)):
        ys = np.array([per_ex[i] for i in range(n) if per_year[i] == y and not np.isnan(per_ex[i])], float)
        if len(ys) >= 4 and ys.std() > 0:
            by_year[y] = round(float(ys.mean() / ys.std() * (ppy ** 0.5)), 3)
    yr_irs = list(by_year.values())
    gate = bool(len(yr_irs) >= 3 and min(yr_irs) > 0
                and _st.median(yr_irs) > GATE_EXCESS_MIN_IR)

    return {
        "kind": kind, "window": window, "quantile": quantile, "rebalance": rebalance,
        "direction": direction, "n_periods": n, "avg_picks": round(float(np.mean(n_pick)), 1),
        "min_liq_pct": min_liq_pct, "min_liq_krw": min_liq_krw,
        "min_liq_median_mult": min_liq_median_mult,
        "avg_pick_liq_pct": round(float(np.mean(pick_liq)), 3) if pick_liq else None,
        "sharpe": None if sharpe is None else round(sharpe, 3),
        "mdd_pct": round(mdd, 2), "cum_return_pct": round(cum, 1),
        "excess_ir": None if excess_ir is None else round(excess_ir, 3),
        "by_year_excess_ir": by_year,
        "yr_ir_median": round(_st.median(yr_irs), 3) if yr_irs else None,
        "yr_ir_min": round(min(yr_irs), 3) if yr_irs else None,
        "gate_passed": gate,
    }
