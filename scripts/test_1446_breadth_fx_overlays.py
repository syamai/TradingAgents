#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi, fetch_usdkrw
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _simulate_v2
from tradingagents.hermes.strategy_research_v2 import _preload

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
CUTOFF = os.environ.get("CUTOFF", "2025-06-30")
OUTDIR = Path("artifacts/market_overlay_experiments")
OUTDIR.mkdir(parents=True, exist_ok=True)


def mdd(daily: pd.Series):
    eq = (1.0 + daily.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return float(dd.min()), str(peak), str(trough)


def perf(name: str, port: pd.Series, kospi_ret: pd.Series, mult: pd.Series) -> dict:
    port = port.fillna(0.0)
    n = len(port)
    wealth = float((1.0 + port).prod())
    years = n / 252.0
    dd, peak, trough = mdd(port)
    sd = statistics.stdev(port.tolist()) if n > 1 else 0.0
    mean = float(port.mean())
    kwealth = float((1.0 + kospi_ret.reindex(port.index).fillna(0.0)).prod())
    return {
        "name": name,
        "n_days": n,
        "final_wealth_krw": wealth * 100_000_000,
        "cum_return_pct": (wealth - 1.0) * 100.0,
        "annual_return_pct": ((wealth ** (1.0 / years) - 1.0) * 100.0) if wealth > 0 and years > 0 else None,
        "mdd_pct": dd * 100.0,
        "mdd_peak_date": peak,
        "mdd_trough_date": trough,
        "sharpe_daily": (mean / sd * math.sqrt(252.0)) if sd else 0.0,
        "avg_multiplier_pct": float(mult.reindex(port.index).fillna(1.0).mean() * 100.0),
        "riskoff_days_pct": float((mult.reindex(port.index).fillna(1.0) < 0.999).mean() * 100.0),
        "min_multiplier_pct": float(mult.reindex(port.index).fillna(1.0).min() * 100.0),
        "final_vs_kospi_krw": (wealth - kwealth) * 100_000_000,
    }


def trailing_return(s: pd.Series, window: int) -> pd.Series:
    return s / s.shift(window) - 1.0


def lag(cond: pd.Series) -> pd.Series:
    return cond.shift(1, fill_value=False).astype(bool)


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT spec_json FROM strategies WHERE id=1446").fetchone()
    spec = json.loads(row["spec_json"])

    raw_tickers = KisHistoryStore().list_tickers()
    tickers, loader0, _, _uf = _preload(raw_tickers)

    holdings = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader0(tk)
        if df is None or df.empty:
            continue
        d = df["date"].astype(str)
        df = df[d <= CUTOFF].reset_index(drop=True)
        if len(df) < 3:
            continue
        holdings[tk] = df
        d0, d1 = str(df["date"].iloc[0]), str(df["date"].iloc[-1])
        min_d = d0 if min_d is None or d0 < min_d else min_d
        max_d = d1 if max_d is None or d1 > max_d else max_d

    kospi = fetch_kospi(min_d, max_d)
    usdkrw = fetch_usdkrw(min_d, max_d)

    # Build #1446 portfolio daily returns using current simulator + Korea policy.
    ret_frames, act_frames = [], []
    close_frames = []
    for tk, df in holdings.items():
        trades, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
        idx = cdf["date"].astype(str).to_numpy()
        ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
        act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        close_frames.append(pd.Series(pd.to_numeric(cdf["close"], errors="coerce").to_numpy(), index=idx, name=tk))
    base_port = bt._combine_korea_stock_portfolio(ret_frames, act_frames)
    kret = bt._market_daily_returns(kospi, base_port.index)

    # Breadth matrix: use all available universe closes, not only active positions.
    C = pd.concat(close_frames, axis=1).sort_index()
    denom20 = C.rolling(20, min_periods=20).count().gt(0).sum(axis=1).replace(0, pd.NA)
    denom60 = C.rolling(60, min_periods=60).count().gt(0).sum(axis=1).replace(0, pd.NA)
    denom_pos60 = C.shift(60).notna().sum(axis=1).replace(0, pd.NA)
    above20 = (C > C.rolling(20, min_periods=20).mean()).sum(axis=1) / denom20
    pos60 = (C / C.shift(60) - 1.0 > 0).sum(axis=1) / denom_pos60
    above60 = (C > C.rolling(60, min_periods=60).mean()).sum(axis=1) / denom60
    breadth = pd.DataFrame({"above20": above20, "pos60": pos60, "above60": above60}).reindex(base_port.index).ffill()

    # USD/KRW series aligned to portfolio dates.
    fx = usdkrw[["date", "close"]].copy()
    fx["date"] = fx["date"].astype(str)
    fx_close = pd.Series(pd.to_numeric(fx["close"], errors="coerce").to_numpy(), index=fx["date"].to_numpy()).sort_index()
    fx_close = fx_close.reindex(base_port.index).ffill()
    fx_ret20 = trailing_return(fx_close, 20)
    fx_ret60 = trailing_return(fx_close, 60)
    fx_ma60 = fx_close.rolling(60, min_periods=60).mean()

    def mult_from(cond: pd.Series, stock_weight_pct: float) -> pd.Series:
        m = pd.Series(1.0, index=base_port.index)
        m = m.mask(lag(cond.reindex(base_port.index).fillna(False)), stock_weight_pct / 90.0)
        return m

    kclose = pd.Series(pd.to_numeric(kospi["close"], errors="coerce").to_numpy(), index=kospi["date"].astype(str).to_numpy()).sort_index().reindex(base_port.index).ffill()
    ret80_risk = trailing_return(kclose, 80) <= 0.0

    variants: dict[str, pd.Series] = {
        "baseline": pd.Series(1.0, index=base_port.index),
        "kospi_ret80_le0_stock20": mult_from(ret80_risk, 20.0),
        # Breadth filters.
        "breadth_above20_lt35_stock30": mult_from(breadth["above20"] < 0.35, 30.0),
        "breadth_above20_lt35_stock20": mult_from(breadth["above20"] < 0.35, 20.0),
        "breadth_pos60_lt40_stock30": mult_from(breadth["pos60"] < 0.40, 30.0),
        "breadth_pos60_lt40_stock20": mult_from(breadth["pos60"] < 0.40, 20.0),
        "breadth_above60_lt40_stock30": mult_from(breadth["above60"] < 0.40, 30.0),
        # FX filters.
        "fx_ret20_ge3_stock30": mult_from(fx_ret20 >= 0.03, 30.0),
        "fx_ret20_ge3_stock20": mult_from(fx_ret20 >= 0.03, 20.0),
        "fx_ret20_ge2_and_above60ma_stock30": mult_from((fx_ret20 >= 0.02) & (fx_close > fx_ma60), 30.0),
        "fx_ret60_ge5_stock30": mult_from(fx_ret60 >= 0.05, 30.0),
        # Combined breadth + FX risk-off.
        "breadth_above20_lt35_or_fx20_ge3_stock30": mult_from((breadth["above20"] < 0.35) | (fx_ret20 >= 0.03), 30.0),
        "breadth_pos60_lt40_or_fx20_ge3_stock30": mult_from((breadth["pos60"] < 0.40) | (fx_ret20 >= 0.03), 30.0),
        "kospi80_or_breadth_above60_lt40_stock20": mult_from(ret80_risk | (breadth["above60"] < 0.40), 20.0),
        "kospi80_or_breadth_pos60_lt40_or_fx20_ge3_stock20": mult_from(ret80_risk | (breadth["pos60"] < 0.40) | (fx_ret20 >= 0.03), 20.0),
        "breadth_pos60_lt40_or_fx20_ge3_stock20": mult_from((breadth["pos60"] < 0.40) | (fx_ret20 >= 0.03), 20.0),
    }

    results = []
    for name, mult in variants.items():
        results.append(perf(name, base_port * mult, kret, mult))
    results.sort(key=lambda r: (r["mdd_pct"], r["annual_return_pct"] or -999), reverse=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = OUTDIR / f"breadth_fx_1446_{stamp}.json"
    out_csv = OUTDIR / f"breadth_fx_1446_{stamp}.csv"
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": 1446,
        "cutoff": CUTOFF,
        "data_span": [min_d, max_d],
        "universe_size": len(holdings),
        "fx_rows": len(usdkrw),
        "notes": "post-portfolio exposure scaling; signals lagged 1 trading day; stock_weight_pct converted by /90",
        "results": results,
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader(); wr.writerows(results)
    print("WROTE", out_json)
    print("WROTE", out_csv)
    for r in results:
        print(r["name"], "ann", round(r["annual_return_pct"],2), "mdd", round(r["mdd_pct"],2), "final억", round(r["final_wealth_krw"]/1e8,3), "avgM", round(r["avg_multiplier_pct"],1), "riskoff", round(r["riskoff_days_pct"],1), "vsKospi억", round(r["final_vs_kospi_krw"]/1e8,3))


if __name__ == "__main__":
    main()
