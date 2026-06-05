#!/usr/bin/env python3
"""Rerank saved v2 strategies after attaching USD/KRW risk-off overlay.

Candidate seed: saved DB robust 4-way Sharpe ranking.
Evaluation: current portfolio-aware engine, cutoff <= 2025-06-30, xsec + time split.
Ranking: gate desc, min(xsec in/out/time in/out Sharpe), avg Sharpe, worst MDD.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine import KOREA_STOCK_PORTFOLIO_POLICY
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_validation import run_time_split_validation

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
OUT_DIR = Path("/Users/selab/Source/trading-ai/artifacts/strategy_v2_portfolio_rerank")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
CUTOFF = os.environ.get("CUTOFF", "2025-06-30")
LIMIT = int(os.environ.get("RERANK_LIMIT", "30") or "30")
OFFSET = int(os.environ.get("RERANK_OFFSET", "0") or "0")
TAG = f"fx_overlay_top{OFFSET}_{LIMIT}_{TS}" if OFFSET else f"fx_overlay_top{LIMIT}_{TS}"
CSV_OUT = OUT_DIR / f"{TAG}.csv"
JSON_OUT = OUT_DIR / f"{TAG}.json"
ALL_JSONL = OUT_DIR / f"{TAG}.jsonl"

FX_OVERLAY = {
    "type": "usdkrw_trailing_return_ma_scale",
    "window": 20,
    "op": ">=",
    "threshold_pct": 2.0,
    "ma_window": 60,
    "risk_stock_weight_pct": 30.0,
}


def mval(m: dict, key: str, default: float = -999.0) -> float:
    v = m.get(key)
    return default if v is None else float(v)


def attach_overlay(spec: dict) -> dict:
    out = json.loads(json.dumps(spec, ensure_ascii=False))
    out["name"] = f"{out.get('name', 'strategy')}__fx20_ge2_ma60_stock30"
    out["market_overlay"] = dict(FX_OVERLAY)
    return out


def summarize(row: sqlite3.Row, spec: dict, xres: dict, tres: dict) -> dict:
    xi, xo = xres["in_sample"], xres["out_sample"]
    ti, to = tres["in_sample"], tres["out_sample"]
    sharpes = [mval(xi, "sharpe"), mval(xo, "sharpe"), mval(ti, "sharpe"), mval(to, "sharpe")]
    mdds = [mval(xi, "mdd_pct"), mval(xo, "mdd_pct"), mval(ti, "mdd_pct"), mval(to, "mdd_pct")]
    daily_excess_vols = [mval(xi, "daily_excess_vol_pct", 999), mval(xo, "daily_excess_vol_pct", 999), mval(ti, "daily_excess_vol_pct", 999), mval(to, "daily_excess_vol_pct", 999)]
    wins = [mval(xi, "win_rate", 0), mval(xo, "win_rate", 0), mval(ti, "win_rate", 0), mval(to, "win_rate", 0)]
    return {
        "base_id": int(row["id"]),
        "base_name": row["name"],
        "name": spec["name"],
        "gate": bool(xres.get("gate_passed") and tres.get("gate_passed")),
        "xsec_gate": bool(xres.get("gate_passed")),
        "time_gate": bool(tres.get("gate_passed")),
        "score_min_sharpe_4way": round(min(sharpes), 4),
        "score_avg_sharpe_4way": round(sum(sharpes) / 4, 4),
        "worst_mdd_pct_4way": round(min(mdds), 4),
        "max_daily_excess_vol_pct_4way": round(max(daily_excess_vols), 4),
        "avg_daily_excess_vol_pct_4way": round(sum(daily_excess_vols) / 4, 4),
        "avg_win_rate_4way": round(sum(wins) / 4, 4),
        "xsec_in_win": xi.get("win_rate"),
        "xsec_in_sharpe": xi.get("sharpe"),
        "xsec_in_daily_excess_vol_pct": xi.get("daily_excess_vol_pct"),
        "xsec_in_mdd": xi.get("mdd_pct"),
        "xsec_in_cum": xi.get("cum_return_pct"),
        "xsec_in_n": xi.get("n_trades"),
        "xsec_out_win": xo.get("win_rate"),
        "xsec_out_sharpe": xo.get("sharpe"),
        "xsec_out_daily_excess_vol_pct": xo.get("daily_excess_vol_pct"),
        "xsec_out_mdd": xo.get("mdd_pct"),
        "xsec_out_cum": xo.get("cum_return_pct"),
        "xsec_out_n": xo.get("n_trades"),
        "time_in_win": ti.get("win_rate"),
        "time_in_sharpe": ti.get("sharpe"),
        "time_in_daily_excess_vol_pct": ti.get("daily_excess_vol_pct"),
        "time_in_mdd": ti.get("mdd_pct"),
        "time_in_cum": ti.get("cum_return_pct"),
        "time_in_n": ti.get("n_trades"),
        "time_out_win": to.get("win_rate"),
        "time_out_sharpe": to.get("sharpe"),
        "time_out_daily_excess_vol_pct": to.get("daily_excess_vol_pct"),
        "time_out_mdd": to.get("mdd_pct"),
        "time_out_cum": to.get("cum_return_pct"),
        "time_out_n": to.get("n_trades"),
        "spec_hash": spec_hash(spec),
        "spec": spec,
    }


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT id, name, spec_json
        FROM strategies
        ORDER BY
          min(coalesce(in_sharpe,-999), coalesce(out_sharpe,-999),
              coalesce(time_in_sharpe,-999), coalesce(time_out_sharpe,-999)) DESC,
          ((coalesce(in_sharpe,0)+coalesce(out_sharpe,0)+coalesce(time_in_sharpe,0)+coalesce(time_out_sharpe,0))/4.0) DESC,
          id ASC
    """
    if LIMIT > 0:
        sql += f" LIMIT {LIMIT}"
    if OFFSET > 0:
        sql += f" OFFSET {OFFSET}"
    rows = conn.execute(sql).fetchall()
    print(f"[fx-rerank] loaded candidates={len(rows)} limit={LIMIT or 'all'} offset={OFFSET} cutoff={CUTOFF}", flush=True)

    raw_tickers = KisHistoryStore().list_tickers()
    tickers, loader0, kospi_fetcher, usdkrw_fetcher = _preload(raw_tickers)

    def cutoff_loader(tk: str):
        df, meta = loader0(tk)
        if df is None or df.empty:
            return df, meta
        d = df["date"].astype(str)
        return df[d <= CUTOFF].reset_index(drop=True), meta

    print(f"[fx-rerank] universe={len(tickers)} policy={KOREA_STOCK_PORTFOLIO_POLICY}", flush=True)
    results = []
    for i, row in enumerate(rows, 1):
        try:
            base_spec = json.loads(row["spec_json"])
            spec = attach_overlay(base_spec)
            xres = run_universe_backtest_v2(
                spec, tickers, loader=cutoff_loader,
                kospi_fetcher=kospi_fetcher, usdkrw_fetcher=usdkrw_fetcher)
            tres = run_time_split_validation(
                spec, tickers, loader=cutoff_loader,
                kospi_fetcher=kospi_fetcher, usdkrw_fetcher=usdkrw_fetcher)
            rec = summarize(row, spec, xres, tres)
            results.append(rec)
            with ALL_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[fx-rerank][ERR] #{row['id']} {row['name']}: {e}", file=sys.stderr, flush=True)
        if i % 5 == 0 or i == len(rows):
            best = max((r["score_min_sharpe_4way"] for r in results), default=None)
            print(f"[fx-rerank] {i}/{len(rows)} done best_min_sharpe={best}", flush=True)

    results.sort(key=lambda r: (
        r["gate"],
        r["score_min_sharpe_4way"],
        r["score_avg_sharpe_4way"],
        r["worst_mdd_pct_4way"],
        r["time_out_sharpe"] if r["time_out_sharpe"] is not None else -999,
    ), reverse=True)
    top = results[: min(30, len(results))]
    payload = {
        "generated_at": TS,
        "db": str(DB),
        "cutoff": CUTOFF,
        "evaluated": len(results),
        "candidate_seed_limit": LIMIT,
        "overlay": FX_OVERLAY,
        "portfolio_policy": KOREA_STOCK_PORTFOLIO_POLICY,
        "ranking": "gate desc, min(xsec_in/out/time_in/out sharpe) desc, avg_sharpe desc, worst_mdd desc",
        "top": top,
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if top:
        fieldnames = [k for k in top[0].keys() if k != "spec"] + ["spec_json"]
        with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in top:
                out = {k: v for k, v in r.items() if k != "spec"}
                out["spec_json"] = json.dumps(r["spec"], ensure_ascii=False)
                w.writerow(out)
    print(f"[fx-rerank] DONE evaluated={len(results)} json={JSON_OUT} csv={CSV_OUT} all={ALL_JSONL}", flush=True)
    for rank, r in enumerate(top[:10], 1):
        print(
            f"#{rank:02d} base_id={r['base_id']} minS={r['score_min_sharpe_4way']:.4f} "
            f"avgS={r['score_avg_sharpe_4way']:.4f} worstMDD={r['worst_mdd_pct_4way']:.2f} "
            f"xOOS={r['xsec_out_sharpe']} tOOS={r['time_out_sharpe']} gate={r['gate']} name={r['base_name']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
