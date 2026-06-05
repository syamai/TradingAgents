#!/usr/bin/env python3
"""Yearly contiguous KOSPI-regime segment returns for recent gate-passed v2 strategies.

For each strategy:
1) Simulate portfolio daily returns with current v2 engine/policy up to cutoff.
2) Classify KOSPI regime by trailing 60D return.
3) Split into *contiguous* regime segments, also breaking at calendar-year boundaries.
4) Emit segment cumulative returns and calendar-year returns.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _simulate_v2
from tradingagents.hermes.strategy_research_v2 import _preload

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
OUT_DIR = Path("/Users/selab/Source/trading-ai/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CUTOFF = os.environ.get("CUTOFF", "2025-06-30")
IDS = [int(x) for x in os.environ.get("STRATEGY_IDS", "1473,1446,1401,1346,1345").split(",") if x.strip()]
REGIME_WINDOW = int(os.environ.get("REGIME_WINDOW", "60"))
UP_TH = float(os.environ.get("UP_TH", "0.05"))
DOWN_TH = float(os.environ.get("DOWN_TH", "-0.05"))

SEG_CSV = OUT_DIR / "recent_passed_yearly_contiguous_regime_segments_20160602_20250630.csv"
YEAR_CSV = OUT_DIR / "recent_passed_calendar_year_returns_20160602_20250630.csv"
JSON_OUT = OUT_DIR / "recent_passed_yearly_regime_segments_20160602_20250630.json"


def cum_return(daily: pd.Series) -> float | None:
    daily = daily.dropna()
    if len(daily) == 0:
        return None
    return float((1.0 + daily).prod() - 1.0)


def annualize(cum: float | None, n_days: int) -> float | None:
    if cum is None or n_days <= 0:
        return None
    return float((1.0 + cum) ** (252.0 / n_days) - 1.0) if cum > -1 else -1.0


def pct(x: float | None) -> float | None:
    return None if x is None else round(x * 100.0, 4)


def build_year_regime_segments(reg: pd.Series) -> list[dict[str, Any]]:
    """Contiguous segments, split at calendar-year boundary too."""
    rows = []
    cur_reg = None
    start = None
    prev = None
    n = 0
    seg_no_by_year: dict[int, int] = {}
    for date, regime in reg.items():
        year = int(str(date)[:4])
        prev_year = int(str(prev)[:4]) if prev is not None else year
        boundary = cur_reg is None or regime != cur_reg or year != prev_year
        if boundary:
            if cur_reg is not None:
                y = int(str(start)[:4])
                seg_no_by_year[y] = seg_no_by_year.get(y, 0) + 1
                rows.append({
                    "year": y,
                    "segment_no_in_year": seg_no_by_year[y],
                    "regime": cur_reg,
                    "start": start,
                    "end": prev,
                    "trading_days": n,
                })
            cur_reg = regime
            start = date
            n = 1
        else:
            n += 1
        prev = date
    if cur_reg is not None:
        y = int(str(start)[:4])
        seg_no_by_year[y] = seg_no_by_year.get(y, 0) + 1
        rows.append({
            "year": y,
            "segment_no_in_year": seg_no_by_year[y],
            "regime": cur_reg,
            "start": start,
            "end": prev,
            "trading_days": n,
        })
    return rows


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in IDS)
    strategy_rows = conn.execute(
        f"SELECT id,name,spec_json,wf_excess_ir_median,wf_excess_ir_min FROM strategies WHERE id IN ({placeholders})",
        IDS,
    ).fetchall()

    raw_tickers = KisHistoryStore().list_tickers()
    tickers, loader, _, _uf = _preload(raw_tickers)
    holdings = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader(tk)
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
    if min_d is None or max_d is None:
        raise SystemExit("no holdings")

    kospi = fetch_kospi(min_d, max_d)
    k = kospi[["date", "close"]].copy()
    k["date"] = k["date"].astype(str)
    k = k.sort_values("date").drop_duplicates("date", keep="last")
    kclose = pd.Series(pd.to_numeric(k["close"], errors="coerce").to_numpy(), index=k["date"].to_numpy())
    trend = kclose / kclose.shift(REGIME_WINDOW) - 1.0
    regime = pd.Series("보합", index=kclose.index)
    regime[trend >= UP_TH] = "상승"
    regime[trend <= DOWN_TH] = "하락"
    segments = build_year_regime_segments(regime.loc[(regime.index >= min_d) & (regime.index <= max_d)])

    segment_rows = []
    year_rows = []
    payload = {
        "span": [min_d, max_d],
        "cutoff": CUTOFF,
        "regime_definition": {
            "basis": f"KOSPI trailing {REGIME_WINDOW} trading-day return",
            "up": f">= {UP_TH*100:.1f}%",
            "down": f"<= {DOWN_TH*100:.1f}%",
            "flat": f"between {DOWN_TH*100:.1f}% and {UP_TH*100:.1f}%",
        },
        "portfolio_policy": bt.KOREA_STOCK_PORTFOLIO_POLICY,
        "strategies": [],
    }

    for row in strategy_rows:
        sid = int(row["id"])
        spec = json.loads(row["spec_json"])
        ret_frames, act_frames, all_trades = [], [], []
        for tk, df in holdings.items():
            trades, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
            all_trades.extend(trades)
            idx = cdf["date"].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        port = bt._combine_korea_stock_portfolio(ret_frames, act_frames)
        kret = bt._market_daily_returns(kospi, port.index)
        active_weight = bt._invested_weight_korea(act_frames).reindex(port.index).fillna(0.0)

        # calendar year returns: chronology intact, all days in that year compounded.
        years = sorted({int(str(d)[:4]) for d in port.index})
        for year in years:
            ymask = pd.Series([str(d).startswith(str(year)) for d in port.index], index=port.index)
            p = port[ymask]
            kk = kret[ymask]
            aw = active_weight[ymask]
            pc = cum_return(p)
            kc = cum_return(kk)
            year_rows.append({
                "strategy_id": sid,
                "name": row["name"],
                "year": year,
                "period_start": p.index[0] if len(p) else None,
                "period_end": p.index[-1] if len(p) else None,
                "trading_days": len(p),
                "strategy_calendar_return_pct": pct(pc),
                "strategy_annualized_return_pct": pct(annualize(pc, len(p))),
                "kospi_calendar_return_pct": pct(kc),
                "kospi_annualized_return_pct": pct(annualize(kc, len(kk))),
                "avg_active_weight_pct": round(float(aw.mean()) * 100, 4) if len(aw) else None,
            })

        # contiguous regime segments, split by year.
        for seg in segments:
            mask = (port.index >= seg["start"]) & (port.index <= seg["end"])
            p = port[mask]
            kk = kret[mask]
            aw = active_weight[mask]
            pc = cum_return(p)
            kc = cum_return(kk)
            segment_rows.append({
                "strategy_id": sid,
                "name": row["name"],
                **seg,
                "strategy_segment_cum_return_pct": pct(pc),
                "strategy_segment_annualized_return_pct": pct(annualize(pc, len(p))),
                "kospi_segment_cum_return_pct": pct(kc),
                "kospi_segment_annualized_return_pct": pct(annualize(kc, len(kk))),
                "avg_active_weight_pct": round(float(aw.mean()) * 100, 4) if len(aw) else None,
            })

        payload["strategies"].append({
            "id": sid,
            "name": row["name"],
            "n_trades_pre_cutoff": len(all_trades),
        })

    with SEG_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(segment_rows[0].keys()))
        w.writeheader()
        w.writerows(segment_rows)
    with YEAR_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(year_rows[0].keys()))
        w.writeheader()
        w.writerows(year_rows)
    JSON_OUT.write_text(json.dumps({**payload, "segment_rows": segment_rows, "year_rows": year_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "segment_csv": str(SEG_CSV),
        "year_csv": str(YEAR_CSV),
        "json": str(JSON_OUT),
        "n_segment_rows": len(segment_rows),
        "n_year_rows": len(year_rows),
        "span": [min_d, max_d],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
