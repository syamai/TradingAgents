#!/usr/bin/env python3
"""Formal xsec/time/WF validation for #1446 with breadth + USD/KRW overlays.

This is intentionally a standalone formal runner rather than a saved strategy spec,
because the engine currently persists only KOSPI trailing-return market_overlay.  It
reuses the same simulator, portfolio combiner, cutoff loader, time split, and
walk-forward window definitions as strategy_validation.py, then applies candidate
post-portfolio exposure multipliers with 1-trading-day lag.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi, fetch_usdkrw
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _simulate_v2, passes_full_gate_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_validation import DEFAULT_EMBARGO_DAYS, GATE_EXCESS_MIN_IR

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
CUTOFF = os.environ.get("CUTOFF", "2025-06-30")
OUTDIR = Path("artifacts/market_overlay_experiments")
OUTDIR.mkdir(parents=True, exist_ok=True)
HI = "9999-12-31"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _universe_dates(holdings: dict[str, pd.DataFrame]) -> list[str]:
    dates: set[str] = set()
    for df in holdings.values():
        dates.update(df["date"].astype(str).tolist())
    return sorted(dates)


def _fetch_strategy_spec(strategy_id: int = 1446) -> tuple[str, dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT id,name,spec_json FROM strategies WHERE id=?", (strategy_id,)).fetchone()
    if row is None:
        raise SystemExit(f"strategy #{strategy_id} not found")
    return row["name"], json.loads(row["spec_json"])


def _load_cutoff_holdings() -> tuple[list[str], dict[str, pd.DataFrame], str, str]:
    raw_tickers = KisHistoryStore().list_tickers()
    tickers, loader0, _kf, _uf = _preload(raw_tickers)
    holdings: dict[str, pd.DataFrame] = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader0(tk)
        if df is None or df.empty:
            continue
        d = df["date"].astype(str)
        sub = df[d <= CUTOFF].reset_index(drop=True)
        sub = sub[(_num(sub["close"]) > 0) & (_num(sub["volume"]) > 0)].reset_index(drop=True)
        if len(sub) < 3:
            continue
        holdings[tk] = sub
        d0, d1 = str(sub["date"].iloc[0]), str(sub["date"].iloc[-1])
        min_d = d0 if min_d is None or d0 < min_d else min_d
        max_d = d1 if max_d is None or d1 > max_d else max_d
    if min_d is None or max_d is None:
        raise SystemExit("no cutoff holdings loaded")
    return tickers, holdings, min_d, max_d


def _overlay_context(holdings: dict[str, pd.DataFrame], kospi: pd.DataFrame, usdkrw: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    close_frames = []
    for tk, df in holdings.items():
        idx = df["date"].astype(str).to_numpy()
        close_frames.append(pd.Series(_num(df["close"]).to_numpy(dtype=float), index=idx, name=tk))
    C = pd.concat(close_frames, axis=1).sort_index()
    denom20 = C.rolling(20, min_periods=20).count().gt(0).sum(axis=1).replace(0, pd.NA)
    denom60 = C.rolling(60, min_periods=60).count().gt(0).sum(axis=1).replace(0, pd.NA)
    denom_pos60 = C.shift(60).notna().sum(axis=1).replace(0, pd.NA)
    breadth = pd.DataFrame({
        "above20": (C > C.rolling(20, min_periods=20).mean()).sum(axis=1) / denom20,
        "pos60": (C / C.shift(60) - 1.0 > 0).sum(axis=1) / denom_pos60,
        "above60": (C > C.rolling(60, min_periods=60).mean()).sum(axis=1) / denom60,
    }).sort_index().ffill()

    k = kospi[["date", "close"]].copy()
    k["date"] = k["date"].astype(str)
    kclose = pd.Series(_num(k["close"]).to_numpy(dtype=float), index=k["date"].to_numpy()).sort_index()

    fx = usdkrw[["date", "close"]].copy()
    fx["date"] = fx["date"].astype(str)
    fxclose = pd.Series(_num(fx["close"]).to_numpy(dtype=float), index=fx["date"].to_numpy()).sort_index()
    return {"breadth": breadth, "kclose": kclose, "fxclose": fxclose}


def _trail_ret(close: pd.Series, window: int) -> pd.Series:
    return close / close.shift(window) - 1.0


def _mult_for(index: pd.Index, variant: str, ctx: dict[str, pd.Series | pd.DataFrame]) -> pd.Series:
    m = pd.Series(1.0, index=index, dtype=float)
    if variant == "baseline":
        return m
    b = ctx["breadth"].reindex(index).ffill()  # type: ignore[union-attr]
    kclose = ctx["kclose"].reindex(index).ffill()  # type: ignore[union-attr]
    fxclose = ctx["fxclose"].reindex(index).ffill()  # type: ignore[union-attr]
    kospi80 = _trail_ret(kclose, 80) <= 0.0
    breadth_pos60 = b["pos60"] < 0.40
    breadth_above60 = b["above60"] < 0.40
    fx20_ge3 = _trail_ret(fxclose, 20) >= 0.03
    fx20_ge2_above60 = (_trail_ret(fxclose, 20) >= 0.02) & (fxclose > fxclose.rolling(60, min_periods=60).mean())

    specs: dict[str, tuple[pd.Series, float]] = {
        "kospi80_stock20": (kospi80, 20.0),
        "breadth_above60_lt40_stock30": (breadth_above60, 30.0),
        "breadth_pos60_lt40_stock30": (breadth_pos60, 30.0),
        "fx_ret20_ge2_above60ma_stock30": (fx20_ge2_above60, 30.0),
        "breadth_pos60_or_fx20_ge3_stock30": (breadth_pos60 | fx20_ge3, 30.0),
        "kospi80_or_breadth_pos60_or_fx20_ge3_stock20": (kospi80 | breadth_pos60 | fx20_ge3, 20.0),
    }
    if variant not in specs:
        raise ValueError(f"unknown variant {variant}")
    cond, stock_weight_pct = specs[variant]
    lagged = cond.reindex(index).fillna(False).shift(1, fill_value=False).astype(bool)
    # Engine policy stock_weight is a fraction (0.90). Convert requested 20%/30%
    # stock exposure to a multiplier on the 90%-stock portfolio return.
    return m.mask(lagged, stock_weight_pct / bt.KOREA_STOCK_PORTFOLIO_POLICY["stock_weight"] / 100.0)


def _window_metrics(spec: dict, holdings: dict[str, pd.DataFrame], kospi: pd.DataFrame, ctx: dict, variant: str, lo: str, hi: str, split_filter: Callable[[str], bool] | None = None) -> dict:
    ret_frames, act_frames, all_trades = [], [], []
    n_tk = 0
    for tk, df in holdings.items():
        if split_filter is not None and not split_filter(tk):
            continue
        d = df["date"].astype(str)
        sub = df[(d >= lo) & (d < hi)]
        if len(sub) < 3:
            continue
        sub = sub.reset_index(drop=True)
        trades, daily, active, cdf = _simulate_v2(spec, sub, kospi=kospi)
        bt._attach_kospi(trades, kospi)
        idx = cdf["date"].astype(str).to_numpy()
        ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
        act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        all_trades.extend(trades)
        n_tk += 1
    port = bt._combine_korea_stock_portfolio(ret_frames, act_frames)
    mult = _mult_for(port.index, variant, ctx)
    port2 = port * mult
    market = bt._market_daily_returns(kospi, port.index)
    invested = bt._invested_weight_korea(act_frames).reindex(port.index).fillna(0.0) * mult
    excess = port2 - invested * market
    m = bt._metrics(all_trades, port2, excess_daily=excess)
    m["_window"] = [lo, hi if hi != HI else "end"]
    m["_n_tickers"] = n_tk
    m["riskoff_days_pct"] = round(float((mult < 0.999).mean() * 100.0), 4) if len(mult) else 0.0
    m["avg_stock_weight_pct"] = round(float((mult * bt.KOREA_STOCK_PORTFOLIO_POLICY["stock_weight"] * 100.0).mean()), 4) if len(mult) else None
    return m


def _xsec(spec: dict, holdings: dict[str, pd.DataFrame], kospi: pd.DataFrame, ctx: dict, variant: str) -> dict:
    in_m = _window_metrics(spec, holdings, kospi, ctx, variant, "0000-01-01", HI, lambda tk: bt.split_of(tk) == "in")
    out_m = _window_metrics(spec, holdings, kospi, ctx, variant, "0000-01-01", HI, lambda tk: bt.split_of(tk) == "out")
    return {"in_sample": in_m, "out_sample": out_m, "gate_passed": passes_full_gate_v2(in_m, out_m)}


def _time(spec: dict, holdings: dict[str, pd.DataFrame], kospi: pd.DataFrame, ctx: dict, variant: str) -> dict:
    dates = _universe_dates(holdings)
    cut_idx = int(len(dates) * 70 / 100)
    cutoff = dates[cut_idx]
    oos_start = dates[min(cut_idx + DEFAULT_EMBARGO_DAYS, len(dates) - 1)]
    is_m = _window_metrics(spec, holdings, kospi, ctx, variant, dates[0], cutoff)
    oos_m = _window_metrics(spec, holdings, kospi, ctx, variant, oos_start, HI)
    return {"in_sample": is_m, "out_sample": oos_m, "gate_passed": passes_full_gate_v2(is_m, oos_m), "split": {"cutoff": cutoff, "oos_start": oos_start, "embargo_days": DEFAULT_EMBARGO_DAYS}}


def _wf(spec: dict, holdings: dict[str, pd.DataFrame], kospi: pd.DataFrame, ctx: dict, variant: str) -> dict:
    dates = _universe_dates(holdings)
    is_len = int(3.0 * bt.TRADING_DAYS_PER_YEAR)
    oos_len = int(1.0 * bt.TRADING_DAYS_PER_YEAR)
    step = oos_len
    windows = []
    is_start_idx = 0
    is_end_idx = is_len
    while is_end_idx + DEFAULT_EMBARGO_DAYS + oos_len <= len(dates):
        oos_lo = is_end_idx + DEFAULT_EMBARGO_DAYS
        oos_hi = oos_lo + oos_len
        windows.append((dates[is_start_idx], dates[is_end_idx], dates[oos_lo], dates[min(oos_hi, len(dates) - 1)]))
        is_start_idx += step
        is_end_idx += step
    rows = []
    for is_lo, is_hi, oos_lo, oos_hi in windows:
        is_m = _window_metrics(spec, holdings, kospi, ctx, variant, is_lo, is_hi)
        oos_m = _window_metrics(spec, holdings, kospi, ctx, variant, oos_lo, oos_hi)
        rows.append({"is_window": [is_lo, is_hi], "oos_window": [oos_lo, oos_hi], "in_sample": is_m, "out_sample": oos_m, "window_gate": passes_full_gate_v2(is_m, oos_m)})
    valid_ex = [r["out_sample"].get("excess_sharpe") for r in rows if r["out_sample"].get("excess_sharpe") is not None]
    valid_sh = [r["out_sample"].get("sharpe") for r in rows if r["out_sample"].get("sharpe") is not None]
    ex_med = round(st.median(valid_ex), 4) if valid_ex else None
    ex_min = round(min(valid_ex), 4) if valid_ex else None
    sh_med = round(st.median(valid_sh), 4) if valid_sh else None
    sh_min = round(min(valid_sh), 4) if valid_sh else None
    passed = bool(len(rows) >= 2 and len(valid_ex) == len(rows) and ex_min is not None and ex_min > 0 and ex_med is not None and ex_med > GATE_EXCESS_MIN_IR)
    return {"mode": "rolling", "n_windows": len(rows), "windows": rows, "oos_excess_ir_median": ex_med, "oos_excess_ir_min": ex_min, "oos_sharpe_median": sh_med, "oos_sharpe_min": sh_min, "gate_passed": passed, "gate_metric": "excess_ir", "gate_min_ir": GATE_EXCESS_MIN_IR}


def main() -> None:
    strategy_name, spec = _fetch_strategy_spec(1446)
    _tickers, holdings, min_d, max_d = _load_cutoff_holdings()
    kospi = fetch_kospi(min_d, max_d)
    usdkrw = fetch_usdkrw(min_d, max_d)
    ctx = _overlay_context(holdings, kospi, usdkrw)
    variants = [
        "baseline",
        "kospi80_stock20",
        "breadth_above60_lt40_stock30",
        "breadth_pos60_lt40_stock30",
        "fx_ret20_ge2_above60ma_stock30",
        "breadth_pos60_or_fx20_ge3_stock30",
        "kospi80_or_breadth_pos60_or_fx20_ge3_stock20",
    ]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "strategy_id": 1446, "strategy_name": strategy_name, "cutoff": CUTOFF, "data_span": [min_d, max_d], "universe_size": len(holdings), "fx_rows": len(usdkrw), "notes": "formal xsec/time/WF; post-portfolio overlay; all overlay signals lagged 1 trading day; stock_weight_pct converted by /90", "results": []}
    summary_rows = []
    for v in variants:
        print(f"[run] {v}", flush=True)
        xsec = _xsec(spec, holdings, kospi, ctx, v)
        time = _time(spec, holdings, kospi, ctx, v)
        wf = _wf(spec, holdings, kospi, ctx, v)
        rec = {"variant": v, "xsec": xsec, "time": time, "wf": wf}
        out["results"].append(rec)
        row = {
            "variant": v,
            "xsec_out_sharpe": xsec["out_sample"].get("sharpe"),
            "xsec_out_mdd_pct": xsec["out_sample"].get("mdd_pct"),
            "xsec_out_cum_return_pct": xsec["out_sample"].get("cum_return_pct"),
            "time_out_sharpe": time["out_sample"].get("sharpe"),
            "time_out_mdd_pct": time["out_sample"].get("mdd_pct"),
            "time_out_cum_return_pct": time["out_sample"].get("cum_return_pct"),
            "wf_excess_ir_median": wf.get("oos_excess_ir_median"),
            "wf_excess_ir_min": wf.get("oos_excess_ir_min"),
            "wf_raw_sharpe_median": wf.get("oos_sharpe_median"),
            "wf_raw_sharpe_min": wf.get("oos_sharpe_min"),
            "wf_gate": wf.get("gate_passed"),
            "xsec_riskoff_days_pct": xsec["out_sample"].get("riskoff_days_pct"),
            "time_riskoff_days_pct": time["out_sample"].get("riskoff_days_pct"),
        }
        summary_rows.append(row)
        print("[done]", json.dumps(row, ensure_ascii=False), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUTDIR / f"formal_1446_breadth_fx_overlay_{stamp}.json"
    out_csv = OUTDIR / f"formal_1446_breadth_fx_overlay_{stamp}.csv"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader(); wr.writerows(summary_rows)
    print(f"WROTE {out_json}")
    print(f"WROTE {out_csv}")


if __name__ == "__main__":
    main()
