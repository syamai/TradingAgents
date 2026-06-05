#!/usr/bin/env python3
"""Recompute saved v2 strategies under current portfolio-aware engine and emit top 30.

Ranking: primary = min of four Sharpes (xsec in/out + time IS/OOS), then average Sharpe,
then worst MDD. This favors strategies that are least-bad across both cross-sectional and
out-of-time validation, not only recent OOS winners.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.hermes.backtest_engine import KOREA_STOCK_PORTFOLIO_POLICY
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_validation import run_time_split_validation
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.dataflows.kis_history_store import KisHistoryStore

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
OUT_DIR = Path("/Users/selab/Source/trading-ai/artifacts/strategy_v2_portfolio_rerank")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
CSV_OUT = OUT_DIR / f"top30_portfolio_{TS}.csv"
JSON_OUT = OUT_DIR / f"top30_portfolio_{TS}.json"
ALL_JSONL = OUT_DIR / f"all_portfolio_{TS}.jsonl"


def mval(m: dict, key: str, default: float = -999.0) -> float:
    v = m.get(key)
    return default if v is None else float(v)


def gate_single(m: dict) -> bool:
    return (
        m.get("n_trades", 0) >= 50
        and m.get("win_rate") is not None and m["win_rate"] > 0.50
        and m.get("sharpe") is not None and m["sharpe"] > 1.0
        and m.get("mdd_pct") is not None and m["mdd_pct"] >= -20.0
    )


def summarize(row: sqlite3.Row, xres: dict, tres: dict) -> dict:
    xi, xo = xres["in_sample"], xres["out_sample"]
    ti, to = tres["in_sample"], tres["out_sample"]
    sharpes = [mval(xi, "sharpe"), mval(xo, "sharpe"), mval(ti, "sharpe"), mval(to, "sharpe")]
    mdds = [mval(xi, "mdd_pct"), mval(xo, "mdd_pct"), mval(ti, "mdd_pct"), mval(to, "mdd_pct")]
    daily_excess_vols = [mval(xi, "daily_excess_vol_pct", 999), mval(xo, "daily_excess_vol_pct", 999), mval(ti, "daily_excess_vol_pct", 999), mval(to, "daily_excess_vol_pct", 999)]
    wins = [mval(xi, "win_rate", 0), mval(xo, "win_rate", 0), mval(ti, "win_rate", 0), mval(to, "win_rate", 0)]
    time_gate = bool(tres.get("gate_passed"))
    xsec_gate = bool(xres.get("gate_passed"))
    spec = json.loads(row["spec_json"])
    return {
        "id": int(row["id"]),
        "name": row["name"],
        "gate": bool(xsec_gate and time_gate),
        "xsec_gate": xsec_gate,
        "time_gate": time_gate,
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
        "spec": spec,
        "spec_hash": spec_hash(spec),
    }


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    limit = int(os.environ.get("RERANK_LIMIT", "0") or "0")
    # Seed candidates from previously saved metrics by robust 4-way Sharpe, then
    # recompute those candidates under the current portfolio-aware engine.
    sql = """
        SELECT id, name, spec_json
        FROM strategies
        ORDER BY
          min(coalesce(in_sharpe,-999), coalesce(out_sharpe,-999),
              coalesce(time_in_sharpe,-999), coalesce(time_out_sharpe,-999)) DESC,
          ((coalesce(in_sharpe,0)+coalesce(out_sharpe,0)+coalesce(time_in_sharpe,0)+coalesce(time_out_sharpe,0))/4.0) DESC,
          id ASC
    """
    if limit > 0:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    print(f"[rerank] loaded candidate strategies={len(rows)} from {DB} (limit={limit or 'all'})", flush=True)

    raw_tickers = KisHistoryStore().list_tickers()
    print(f"[rerank] preloading tickers={len(raw_tickers)} ...", flush=True)
    tickers, loader, kospi_fetcher, usdkrw_fetcher = _preload(raw_tickers)
    print(f"[rerank] universe={len(tickers)} portfolio_policy={KOREA_STOCK_PORTFOLIO_POLICY}", flush=True)

    results = []
    for i, row in enumerate(rows, 1):
        spec = json.loads(row["spec_json"])
        try:
            xres = run_universe_backtest_v2(
                spec, tickers, loader=loader,
                kospi_fetcher=kospi_fetcher, usdkrw_fetcher=usdkrw_fetcher)
            tres = run_time_split_validation(
                spec, tickers, loader=loader,
                kospi_fetcher=kospi_fetcher, usdkrw_fetcher=usdkrw_fetcher)
            rec = summarize(row, xres, tres)
            results.append(rec)
            with ALL_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # keep batch going; print enough to diagnose
            print(f"[rerank][ERR] #{row['id']} {row['name']}: {e}", file=sys.stderr, flush=True)
        if i % 25 == 0 or i == len(rows):
            best = max((r["score_min_sharpe_4way"] for r in results), default=None)
            print(f"[rerank] {i}/{len(rows)} done best_min_sharpe={best}", flush=True)

    results.sort(key=lambda r: (
        r["gate"],
        r["score_min_sharpe_4way"],
        r["score_avg_sharpe_4way"],
        r["worst_mdd_pct_4way"],
        r["time_out_sharpe"] if r["time_out_sharpe"] is not None else -999,
    ), reverse=True)
    top30 = results[:30]
    payload = {
        "generated_at": TS,
        "db": str(DB),
        "evaluated": len(results),
        "portfolio_policy": KOREA_STOCK_PORTFOLIO_POLICY,
        "ranking": "gate desc, min(xsec_in/out/time_in/out sharpe) desc, avg_sharpe desc, worst_mdd desc",
        "top30": top30,
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [k for k in top30[0].keys() if k != "spec"] + ["spec_json"] if top30 else []
    if top30:
        with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in top30:
                row = {k: v for k, v in r.items() if k != "spec"}
                row["spec_json"] = json.dumps(r["spec"], ensure_ascii=False)
                w.writerow(row)

    print(f"[rerank] DONE evaluated={len(results)} top_json={JSON_OUT} top_csv={CSV_OUT} all_jsonl={ALL_JSONL}", flush=True)
    for rank, r in enumerate(top30, 1):
        print(
            f"#{rank:02d} id={r['id']} minS={r['score_min_sharpe_4way']:.4f} avgS={r['score_avg_sharpe_4way']:.4f} "
            f"worstMDD={r['worst_mdd_pct_4way']:.2f} gate={r['gate']} name={r['name']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
