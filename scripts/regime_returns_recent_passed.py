#!/usr/bin/env python3
"""Regime-separated realized returns for recent gate-passed v2 strategies.

- Uses current v2 simulator + Korea-stock portfolio policy.
- Restricts reporting window to date <= 2025-06-30 by default.
- KOSPI regime is based on trailing 60-trading-day KOSPI return:
  up >= +5%, down <= -5%, flat otherwise.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _simulate_v2
from tradingagents.hermes.strategy_research_v2 import _preload

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
CUTOFF = os.environ.get("CUTOFF", "2025-06-30")
IDS = [int(x) for x in os.environ.get("STRATEGY_IDS", "1473,1446,1401,1346,1345").split(",") if x.strip()]
REGIME_WINDOW = int(os.environ.get("REGIME_WINDOW", "60"))
UP_TH = float(os.environ.get("UP_TH", "0.05"))
DOWN_TH = float(os.environ.get("DOWN_TH", "-0.05"))


def ann_return(daily: pd.Series) -> float | None:
    daily = daily.dropna()
    n = len(daily)
    if n == 0:
        return None
    cum = float((1.0 + daily).prod() - 1.0)
    return float((1.0 + cum) ** (252.0 / n) - 1.0) if cum > -1 else -1.0


def cum_return(daily: pd.Series) -> float | None:
    daily = daily.dropna()
    if len(daily) == 0:
        return None
    return float((1.0 + daily).prod() - 1.0)


def max_drawdown(daily: pd.Series) -> float | None:
    daily = daily.dropna()
    if len(daily) == 0:
        return None
    eq = (1.0 + daily).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in IDS)
    rows = conn.execute(f"SELECT id,name,spec_json,wf_excess_ir_median,wf_excess_ir_min FROM strategies WHERE id IN ({placeholders})", IDS).fetchall()
    by_id = {int(r["id"]): r for r in rows}

    raw_tickers = KisHistoryStore().list_tickers()
    tickers, loader, _kf, _uf = _preload(raw_tickers)
    # load once
    holdings = {}
    min_d = max_d = None
    for tk in tickers:
        df, _ = loader(tk)
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
    k = kospi[["date", "close"]].copy()
    k["date"] = k["date"].astype(str)
    k = k.sort_values("date").drop_duplicates("date", keep="last")
    kclose = pd.Series(pd.to_numeric(k["close"], errors="coerce").to_numpy(), index=k["date"].to_numpy())
    kret_all = kclose.pct_change().replace([float("inf"), -float("inf")], pd.NA).fillna(0.0)
    ktrend = kclose / kclose.shift(REGIME_WINDOW) - 1.0
    regime = pd.Series("flat", index=kclose.index)
    regime[ktrend >= UP_TH] = "up"
    regime[ktrend <= DOWN_TH] = "down"
    regime = regime.rename("regime")

    output = {
        "cutoff": CUTOFF,
        "strategy_ids": IDS,
        "universe_size": len(holdings),
        "portfolio_policy": bt.KOREA_STOCK_PORTFOLIO_POLICY,
        "regime_definition": {
            "basis": f"KOSPI trailing {REGIME_WINDOW} trading-day return",
            "up": f">= {UP_TH*100:.1f}%",
            "down": f"<= {DOWN_TH*100:.1f}%",
            "flat": f"between {DOWN_TH*100:.1f}% and {UP_TH*100:.1f}%",
        },
        "data_span": [min_d, max_d],
        "results": [],
    }

    for sid in IDS:
        r = by_id.get(sid)
        if r is None:
            continue
        spec = json.loads(r["spec_json"])
        ret_frames, act_frames, trades = [], [], []
        for tk, df in holdings.items():
            tr, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
            trades.extend(tr)
            idx = cdf["date"].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        port = bt._combine_korea_stock_portfolio(ret_frames, act_frames)
        active_weight = bt._invested_weight_korea(act_frames).reindex(port.index).fillna(0.0)
        kret = bt._market_daily_returns(kospi, port.index)
        reg = regime.reindex(port.index).ffill().fillna("flat")

        rec = {
            "id": sid,
            "name": r["name"],
            "wf_excess_ir_median": r["wf_excess_ir_median"],
            "wf_excess_ir_min": r["wf_excess_ir_min"],
            "total_trade_count_pre_cutoff": len(trades),
            "overall_pre_cutoff": {
                "n_days": int(len(port)),
                "strategy_cum_return_pct": round(cum_return(port) * 100, 4),
                "strategy_annual_return_pct": round(ann_return(port) * 100, 4),
                "kospi_cum_return_pct": round(cum_return(kret) * 100, 4),
                "kospi_annual_return_pct": round(ann_return(kret) * 100, 4),
                "avg_active_weight_pct": round(float(active_weight.mean()) * 100, 4),
                "mdd_pct": round(max_drawdown(port) * 100, 4),
            },
            "by_regime": [],
        }
        for label, ko in [("up", "상승"), ("flat", "보합"), ("down", "하락")]:
            mask = (reg == label)
            pdaily = port[mask]
            kdaily = kret[mask]
            aw = active_weight[mask]
            rec["by_regime"].append({
                "regime": ko,
                "n_days": int(mask.sum()),
                "strategy_cum_return_pct": None if len(pdaily) == 0 else round(cum_return(pdaily) * 100, 4),
                "strategy_annual_return_pct": None if len(pdaily) == 0 else round(ann_return(pdaily) * 100, 4),
                "kospi_cum_return_pct": None if len(kdaily) == 0 else round(cum_return(kdaily) * 100, 4),
                "kospi_annual_return_pct": None if len(kdaily) == 0 else round(ann_return(kdaily) * 100, 4),
                "avg_active_weight_pct": None if len(aw) == 0 else round(float(aw.mean()) * 100, 4),
            })
        output["results"].append(rec)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
