"""백테스트 엔진 v2 — 신규 신호 평가 + 완화 게이트 (기존 엔진 무수정 확장).

검증 끝난 기존 자산을 그대로 import 재사용한다 (수정 0):
  ``_num`` / ``_attach_kospi`` / ``_metrics`` / ``_combine`` / ``split_of`` /
  ``TX_COST_ONE_WAY`` / ``TRADING_DAYS_PER_YEAR`` / ``_eval_signal``(5종) /
  ``GATE_MAX_DRAWDOWN_PCT`` / ``GATE_MIN_TRADES`` / ``GATE_MAX_WIN_RATE_GAP``.

신규:
  - ``_eval_signal_v2`` : 신규 4종(price_drop/trend_slope/rolling_corr/market_filter) 평가,
    기존 5종은 ``backtest_engine._eval_signal`` 위임.
  - ``_simulate_v2`` : 기존 ``_simulate`` 의 진입/청산/tx 로직을 그대로 따르되
    신호 결합만 ``_and_v2`` 사용 (look-ahead 불변식 동일 — 체결 close[i+1]).
  - 완화 게이트 : win_rate > 0.50, sharpe > 1.0 (사용자 지정). MDD/거래수/
    in-out 격차는 기존 임계 유지.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.strategy_spec_v2 import _BASE_SIGNALS, validate_spec_v2

# 완화 게이트 (사용자 지정 — strict 부등호).
GATE_V2_MIN_WIN_RATE = 0.50   # win_rate > 0.50
GATE_V2_MIN_SHARPE = 1.0      # sharpe   > 1.0


# === 신호 평가 (신규 3종 + 기존 위임, look-ahead 0) ===


def _eval_signal_v2(df: pd.DataFrame, sig: dict) -> pd.Series:
    """단일 신호 → row 별 boolean Series (row i 까지 데이터로만 평가)."""
    t = sig["signal"]
    if t in _BASE_SIGNALS:
        return bt._eval_signal(df, sig)

    if t == "price_drop":
        close = bt._num(df["close"])
        ret_w = close / close.shift(sig["window"]) - 1     # trailing
        return ret_w <= -sig["value"] / 100.0

    if t == "trend_slope":
        cum = bt._num(df[f"{sig['subject']}_net_qty"]).fillna(0).cumsum()
        change = cum - cum.shift(sig["window"])             # trailing 누적 변화
        return (change > 0) if sig["direction"] == "up" else (change < 0)

    if t == "rolling_corr":
        nq = bt._num(df[f"{sig['subject']}_net_qty"]).fillna(0)
        pchg = bt._num(df["price_change_pct"])
        r = nq.rolling(sig["lookback"]).corr(pchg)          # trailing Pearson r
        return r >= sig["min_r"]

    if t == "market_filter":
        if "market_close" not in df.columns:
            return pd.Series(False, index=df.index)
        close = bt._num(df.loc[:, "market_close"])
        ma = close.rolling(sig["window"]).mean()             # trailing KOSPI MA
        if sig["mode"] == "above_ma":
            return close > ma
        return close < ma

    if t == "flow_zscore":
        x = bt._num(df[f"{sig['subject']}_net_qty"]).fillna(0)
        roll = x.rolling(sig["window"]).sum()                # trailing W일 누적
        mu = roll.rolling(sig["lookback"]).mean()            # 자기 과거 N일 분포
        sd = roll.rolling(sig["lookback"]).std()
        z = (roll - mu) / sd.where(sd > 0)                   # sd<=0 → NaN → False
        return z >= sig["min_z"]

    if t == "flow_accel":
        x = bt._num(df[f"{sig['subject']}_net_qty"]).fillna(0)
        ma_s = x.rolling(sig["short"]).mean()                # 단기 평균
        ma_l = x.rolling(sig["long"]).mean()                 # 장기 평균
        return (ma_s - ma_l) > 0

    if t == "flow_divergence":
        smart = bt._num(df[f"{sig['subject']}_net_qty"]).fillna(0).rolling(sig["window"]).sum()
        retail = bt._num(df["retail_net_qty"]).fillna(0).rolling(sig["window"]).sum()
        return (smart > 0) & (retail < 0)

    raise ValueError(f"unknown signal type: {t!r}")


def _attach_market_columns(df: pd.DataFrame, kospi: Optional[pd.DataFrame]) -> pd.DataFrame:
    """종목 일자에 KOSPI 종가를 trailing 방식으로 결합한다.

    ``market_filter`` 전용 보조 컬럼이다. KOSPI 결손 시 원본 df 를 그대로 반환해
    해당 신호가 False 로 평가되게 한다. ``merge_asof`` 는 같은 날 또는 직전 거래일
    KOSPI 종가만 붙이므로 미래 시장 데이터를 보지 않는다.
    """
    if kospi is None or len(kospi) == 0 or "date" not in kospi.columns or "close" not in kospi.columns:
        return df
    left = df.copy()
    left["_orig_order"] = range(len(left))
    left["_date_dt"] = pd.to_datetime(left["date"])
    right = kospi.loc[:, ["date", "close"]].copy()
    right["_date_dt"] = pd.to_datetime(right["date"])
    right = right.sort_values("_date_dt").rename(columns={"close": "market_close"})
    merged = pd.merge_asof(
        left.sort_values(by="_date_dt"),
        right[["_date_dt", "market_close"]],
        on="_date_dt",
        direction="backward",
    )
    merged = merged.sort_values("_orig_order").drop(columns=["_orig_order", "_date_dt"])
    return merged.reset_index(drop=True)


def _and_v2(df: pd.DataFrame, signals: list) -> pd.Series:
    """신호 AND 결합. 빈 리스트 → 전부 False."""
    if not signals:
        return pd.Series(False, index=df.index)
    out = pd.Series(True, index=df.index)
    for s in signals:
        out = out & _eval_signal_v2(df, s).fillna(False)
    return out


# === 시뮬레이션 (기존 _simulate 와 동일 로직, 신호 결합만 _and_v2) ===


def _simulate_v2(spec: dict, df: pd.DataFrame, *, kospi: Optional[pd.DataFrame] = None):
    """spec + 단일 종목 holdings → (trades, daily_ret, active, clean_df).

    look-ahead 불변식은 ``backtest_engine._simulate`` 와 동일: 신호 row i →
    entry_price = close[i+1], 청산도 close[j+1], 단일 포지션, 끝 강제청산.
    """
    # close>0 + volume>0 (거래량 0 유령행 제거) — backtest_engine._simulate 와 동일.
    df = df[(bt._num(df["close"]) > 0) & (bt._num(df["volume"]) > 0)].reset_index(drop=True)
    df = _attach_market_columns(df, kospi)
    n = len(df)
    daily_ret = pd.Series(0.0, index=range(n))
    active = pd.Series(False, index=range(n))
    trades: list[dict] = []
    if n < 3:
        return trades, daily_ret, active, df

    entry_sig = _and_v2(df, spec["entry"]["all_of"]).to_numpy()
    exit_sig = _and_v2(df, spec["exit"].get("signal_all_of", [])).to_numpy()
    close = bt._num(df["close"]).to_numpy(dtype=float)
    bad_bar = bt._bad_bar_mask(close)    # 분할/감자 미조정 등 비정상 점프 봉 — 거래 제외
    dates = df["date"].astype(str).to_numpy()
    sl = spec["exit"].get("stop_loss_pct")
    tp = spec["exit"].get("take_profit_pct")
    mh = spec["exit"]["max_hold_days"]
    tx = bt.TX_COST_ONE_WAY

    i = 0
    while i < n - 2:
        if not entry_sig[i] or bad_bar[i + 1]:   # 진입 봉(i+1)이 비정상 점프면 보류
            i += 1
            continue
        e = i + 1
        entry_price = close[e]

        exit_idx: Optional[int] = None
        reason = None
        j = e
        while j < n:
            if bad_bar[j]:               # 비정상 점프 봉 — 트리거 평가·청산 보류
                j += 1
                continue
            held = j - e
            ret = close[j] / entry_price - 1
            trig = None
            if sl is not None and ret <= -sl / 100.0:
                trig = "stop_loss"
            elif tp is not None and ret >= tp / 100.0:
                trig = "take_profit"
            elif exit_sig[j]:
                trig = "signal"
            elif held >= mh:
                trig = "max_hold"
            if trig is not None:
                k = j + 1
                while k < n and bad_bar[k]:   # 청산 봉도 비정상 점프 회피
                    k += 1
                if k < n:
                    exit_idx, reason = k, trig
                else:
                    exit_idx, reason = j, "forced_eod"
                break
            j += 1
        if exit_idx is None:
            last = n - 1
            while last > e and bad_bar[last]:    # 강제청산도 정상 봉에서
                last -= 1
            exit_idx, reason = last, "forced_eod"

        exit_price = close[exit_idx]
        gross = exit_price / entry_price - 1
        net = (exit_price * (1 - tx)) / (entry_price * (1 + tx)) - 1
        trades.append({
            "entry_date": str(dates[e]),
            "exit_date": str(dates[exit_idx]),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_ret_pct": round(float(gross) * 100, 4),
            "net_ret_pct": round(float(net) * 100, 4),
            "hold_days": int(exit_idx - e),
            "exit_reason": reason,
        })

        for t in range(e + 1, exit_idx + 1):
            # 비정상 점프 봉(및 그 직후 복귀 봉)은 가짜 수익이라 미계상.
            daily_ret.iloc[t] = 0.0 if bad_bar[t] else close[t] / close[t - 1] - 1
            active.iloc[t] = True
        first = e + 1
        daily_ret.iloc[first] = (1 + daily_ret.iloc[first]) / (1 + tx) - 1
        daily_ret.iloc[exit_idx] = (1 + daily_ret.iloc[exit_idx]) * (1 - tx) - 1

        i = exit_idx + 1

    return trades, daily_ret, active, df


# === 완화 게이트 ===


def passes_single_gate_v2(m: dict) -> bool:
    return (
        m["n_trades"] >= bt.GATE_MIN_TRADES
        and m["win_rate"] is not None and m["win_rate"] > GATE_V2_MIN_WIN_RATE
        and m["sharpe"] is not None and m["sharpe"] > GATE_V2_MIN_SHARPE
        and m["mdd_pct"] >= bt.GATE_MAX_DRAWDOWN_PCT
    )


def passes_full_gate_v2(in_m: dict, out_m: dict) -> bool:
    if not (passes_single_gate_v2(in_m) and passes_single_gate_v2(out_m)):
        return False
    if in_m["win_rate"] is None or out_m["win_rate"] is None:
        return False
    return abs(in_m["win_rate"] - out_m["win_rate"]) <= bt.GATE_MAX_WIN_RATE_GAP


# === 공개 진입점 ===


def run_universe_backtest_v2(
    spec: dict,
    tickers: list[str],
    *,
    loader: Callable[[str], tuple] = load_holdings,
    kospi_fetcher: Optional[Callable[[str, str], pd.DataFrame]] = fetch_kospi,
) -> dict:
    """종목군 백테스트 v2 — in/out 분할 집계 + 완화 게이트.

    ``backtest_engine.run_universe_backtest`` 와 동일 구조, 차이는 (1) 검증
    ``validate_spec_v2``, (2) 시뮬 ``_simulate_v2``, (3) 게이트 ``*_v2``.
    """
    validate_spec_v2(spec)

    loaded: dict[str, pd.DataFrame] = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader(tk)
        if df is None or df.empty:
            continue
        loaded[tk] = df
        d0, d1 = str(df["date"].iloc[0]), str(df["date"].iloc[-1])
        min_d = d0 if (min_d is None or d0 < min_d) else min_d
        max_d = d1 if (max_d is None or d1 > max_d) else max_d

    kospi = None
    if loaded and kospi_fetcher is not None and min_d and max_d:
        try:
            kospi = kospi_fetcher(min_d, max_d)
        except Exception:        # noqa: BLE001
            kospi = None

    results: dict[str, dict] = {}
    for split in ("in", "out"):
        all_trades: list[dict] = []
        ret_frames: list[pd.Series] = []
        act_frames: list[pd.Series] = []
        for tk, df in loaded.items():
            if bt.split_of(tk) != split:
                continue
            trades, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
            bt._attach_kospi(trades, kospi)
            all_trades.extend(trades)
            idx = cdf["date"].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        port = bt._combine_korea_stock_portfolio(ret_frames, act_frames)
        excess = bt._excess_daily_korea(port, act_frames, kospi)
        results[split] = bt._metrics(all_trades, port, excess_daily=excess)

    in_m, out_m = results["in"], results["out"]
    return {
        "in_sample": in_m,
        "out_sample": out_m,
        "gate_passed": passes_full_gate_v2(in_m, out_m),
        "portfolio_policy": bt.KOREA_STOCK_PORTFOLIO_POLICY,
        "universe_size": len(loaded),
        "n_in": sum(1 for tk in loaded if bt.split_of(tk) == "in"),
        "n_out": sum(1 for tk in loaded if bt.split_of(tk) == "out"),
    }
