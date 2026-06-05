#!/usr/bin/env python3
"""Parallel cutoff rerank for saved v2 strategies.

Recomputes xsec in/out and time in/out under the current portfolio-aware engine,
with all holdings rows capped at SCORE_DATE_HI (default 2025-06-30). This avoids
using the post-2025-06 Korea-market abnormal rally in validation.
"""
from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine import KOREA_STOCK_PORTFOLIO_POLICY, TX_COST_ONE_WAY
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_validation import run_time_split_validation

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
OUT_DIR = Path("/Users/selab/Source/trading-ai/artifacts/strategy_v2_portfolio_rerank")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DATE_HI = os.environ.get("SCORE_DATE_HI", "2025-06-30")
WORKERS = int(os.environ.get("RERANK_WORKERS", "4"))
LIMIT = int(os.environ.get("RERANK_LIMIT", "0") or "0")
OFFSET = int(os.environ.get("RERANK_OFFSET", "0") or "0")
GATE_PASSED_ONLY = os.environ.get("GATE_PASSED_ONLY", "0") == "1"
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_gate_suffix = "_gatepassed" if GATE_PASSED_ONLY else ""
PREFIX = f"cutoff_{SCORE_DATE_HI.replace('-', '')}{_gate_suffix}_offset{OFFSET}_limit{LIMIT or 'all'}_portfolio_{TS}" if OFFSET or LIMIT else f"cutoff_{SCORE_DATE_HI.replace('-', '')}{_gate_suffix}_all_portfolio_{TS}"
ALL_JSONL = OUT_DIR / f"{PREFIX}.jsonl"
TOP_CSV = OUT_DIR / f"{PREFIX}_top.csv"
TOP_JSON = OUT_DIR / f"{PREFIX}_top.json"

TICKERS = None
BASE_LOADER = None
KOSPI_FETCHER = None
USDKRW_FETCHER = None


def _ann(cum_pct: float | None, years: float | None) -> float | None:
    if cum_pct is None or years is None or years <= 0:
        return None
    base = 1.0 + float(cum_pct) / 100.0
    if base <= 0:
        return None
    return (base ** (1.0 / years) - 1.0) * 100.0


def _years(lo: str | None, hi: str | None) -> float | None:
    if not lo or not hi:
        return None
    return (date.fromisoformat(hi) - date.fromisoformat(lo)).days / 365.25


def _mval(m: dict[str, Any], key: str, default: float = -999.0) -> float:
    v = m.get(key)
    return default if v is None else float(v)


def _loader_cutoff(tk: str):
    df, meta = BASE_LOADER(tk)
    if df is None or df.empty:
        return df, meta
    d = df["date"].astype(str)
    return df[d <= SCORE_DATE_HI].reset_index(drop=True), meta


def _row_to_record(row_tuple):
    row_id, name, spec_json = row_tuple
    spec = json.loads(spec_json)
    xres = run_universe_backtest_v2(
        spec, TICKERS, loader=_loader_cutoff,
        kospi_fetcher=KOSPI_FETCHER, usdkrw_fetcher=USDKRW_FETCHER)
    tres = run_time_split_validation(
        spec, TICKERS, loader=_loader_cutoff,
        kospi_fetcher=KOSPI_FETCHER, usdkrw_fetcher=USDKRW_FETCHER)
    xi, xo = xres["in_sample"], xres["out_sample"]
    ti, to = tres["in_sample"], tres["out_sample"]
    sharpes = [_mval(xi, "sharpe"), _mval(xo, "sharpe"), _mval(ti, "sharpe"), _mval(to, "sharpe")]
    mdds = [_mval(xi, "mdd_pct"), _mval(xo, "mdd_pct"), _mval(ti, "mdd_pct"), _mval(to, "mdd_pct")]
    daily_excess_vols = [_mval(xi, "daily_excess_vol_pct", 999), _mval(xo, "daily_excess_vol_pct", 999), _mval(ti, "daily_excess_vol_pct", 999), _mval(to, "daily_excess_vol_pct", 999)]
    span = tres.get("split", {}).get("data_span", [None, SCORE_DATE_HI])
    xsec_years = _years(span[0], span[1])
    time_oos_start = tres.get("split", {}).get("oos_start")
    time_years = _years(time_oos_start, span[1])
    rec = {
        "id": int(row_id),
        "name": name,
        "cutoff": SCORE_DATE_HI,
        "portfolio_policy": KOREA_STOCK_PORTFOLIO_POLICY,
        "tx_cost_one_way_pct": round(TX_COST_ONE_WAY * 100.0, 6),
        "tx_cost_round_trip_pct": round(TX_COST_ONE_WAY * 200.0, 6),
        "xsec_gate": bool(xres.get("gate_passed")),
        "time_gate": bool(tres.get("gate_passed")),
        "gate": bool(xres.get("gate_passed") and tres.get("gate_passed")),
        "score_min_sharpe_4way": round(min(sharpes), 4),
        "score_avg_sharpe_4way": round(sum(sharpes) / 4.0, 4),
        "worst_mdd_pct_4way": round(min(mdds), 4),
        "max_daily_excess_vol_pct_4way": round(max(daily_excess_vols), 4),
        "avg_daily_excess_vol_pct_4way": round(sum(daily_excess_vols) / 4.0, 4),
        "xsec_years": round(xsec_years, 4) if xsec_years else None,
        "time_oos_years": round(time_years, 4) if time_years else None,
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
        "xsec_out_ann": round(_ann(xo.get("cum_return_pct"), xsec_years), 4) if _ann(xo.get("cum_return_pct"), xsec_years) is not None else None,
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
        "time_out_ann": round(_ann(to.get("cum_return_pct"), time_years), 4) if _ann(to.get("cum_return_pct"), time_years) is not None else None,
        "time_out_n": to.get("n_trades"),
        "time_split": tres.get("split"),
        "spec_hash": spec_hash(spec),
        "spec": spec,
    }
    # Economic validity: not a final trading filter, only for sorting top candidates.
    rec["economic_ok"] = bool(
        (rec["xsec_out_ann"] is not None and rec["xsec_out_ann"] > 0)
        and (rec["time_out_ann"] is not None and rec["time_out_ann"] > 0)
        and (rec["xsec_out_n"] or 0) >= 100
        and (rec["time_out_n"] or 0) >= 100
        and rec["worst_mdd_pct_4way"] >= -20.0
    )
    return rec


def _rank_key(r: dict[str, Any]):
    """안정성 우선 + 수익률 보조 고정 랭킹.

    사용자의 "가장 안정적이면서 수익률 높은" 기준을 재현 가능하게 고정한다.
    1) 경제성 기본 조건(economic_ok)
    2) 4-way 최저 Sharpe/평균 Sharpe: 네 검증구간 중 약한 구간 방지
    3) worst MDD: validation 낙폭 얕은 순
    4) max daily excess vol: 일별 초과수익 변동성 낮은 순
    5) time/xsec OOS 연환산 수익률: 안정성 통과 후보 사이 수익률 높은 순
    """
    def val(key: str, default: float = -999.0) -> float:
        v = r.get(key)
        return default if v is None else float(v)

    return (
        bool(r.get("economic_ok")),
        val("score_min_sharpe_4way"),
        val("score_avg_sharpe_4way"),
        val("worst_mdd_pct_4way"),
        -val("max_daily_excess_vol_pct_4way", 999.0),
        val("time_out_ann"),
        val("xsec_out_ann"),
    )


def main() -> int:
    global TICKERS, BASE_LOADER, KOSPI_FETCHER, USDKRW_FETCHER
    print(f"[cutoff-rerank] cutoff={SCORE_DATE_HI} workers={WORKERS} limit={LIMIT or 'all'} gate_passed_only={GATE_PASSED_ONLY}", flush=True)
    conn = sqlite3.connect(DB)
    where_clause = "WHERE gate_passed=1" if GATE_PASSED_ONLY else ""
    rows = conn.execute(
        f"""
        SELECT id, name, spec_json
        FROM strategies
        {where_clause}
        ORDER BY
          min(coalesce(in_sharpe,-999), coalesce(out_sharpe,-999),
              coalesce(time_in_sharpe,-999), coalesce(time_out_sharpe,-999)) DESC,
          ((coalesce(in_sharpe,0)+coalesce(out_sharpe,0)+coalesce(time_in_sharpe,0)+coalesce(time_out_sharpe,0))/4.0) DESC,
          id ASC
        """
    ).fetchall()

    if OFFSET > 0:
        rows = rows[OFFSET:]
    if LIMIT > 0:
        rows = rows[:LIMIT]
    print(f"[cutoff-rerank] loaded strategies={len(rows)} from {DB} (offset={OFFSET} limit={LIMIT or 'all'})", flush=True)
    raw = KisHistoryStore().list_tickers()
    print(f"[cutoff-rerank] preloading tickers={len(raw)} ...", flush=True)
    TICKERS, BASE_LOADER, KOSPI_FETCHER, USDKRW_FETCHER = _preload(raw)
    print(f"[cutoff-rerank] universe={len(TICKERS)} portfolio_policy={KOREA_STOCK_PORTFOLIO_POLICY}", flush=True)
    print(f"[cutoff-rerank] outputs all={ALL_JSONL} top_csv={TOP_CSV} top_json={TOP_JSON}", flush=True)

    results: list[dict[str, Any]] = []
    errors = 0
    # Use fork so the preloaded holdings cache is copy-on-write shared by workers.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=WORKERS) as pool, ALL_JSONL.open("w", encoding="utf-8") as f:
        for i, item in enumerate(pool.imap_unordered(_row_to_record, rows, chunksize=1), 1):
            results.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(rows):
                best = max((r["time_out_ann"] for r in results if r.get("time_out_ann") is not None), default=None)
                print(f"[cutoff-rerank] {i}/{len(rows)} done best_time_ann={best}", flush=True)

    results.sort(key=_rank_key, reverse=True)
    top = results[:50]
    payload = {
        "generated_at": TS,
        "cutoff": SCORE_DATE_HI,
        "db": str(DB),
        "evaluated": len(results),
        "errors": errors,
        "portfolio_policy": KOREA_STOCK_PORTFOLIO_POLICY,
        "transaction_cost": {
            "one_way_pct": round(TX_COST_ONE_WAY * 100.0, 6),
            "round_trip_pct": round(TX_COST_ONE_WAY * 200.0, 6),
        },
        "ranking": "economic_ok, 4-way min Sharpe, avg Sharpe, worst MDD, low max daily excess vol, time_out_ann, xsec_out_ann",
        "top": top,
    }
    TOP_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if top:
        fields = [k for k in top[0].keys() if k != "spec"] + ["spec_json"]
        with TOP_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in top:
                row = {k: v for k, v in r.items() if k != "spec"}
                row["spec_json"] = json.dumps(r["spec"], ensure_ascii=False)
                w.writerow(row)
    print(f"[cutoff-rerank] DONE evaluated={len(results)} top_json={TOP_JSON} top_csv={TOP_CSV}", flush=True)
    for rank, r in enumerate(top[:10], 1):
        print(
            f"#{rank:02d} id={r['id']} time_ann={r['time_out_ann']} xsec_ann={r['xsec_out_ann']} "
            f"minS={r['score_min_sharpe_4way']} worstMDD={r['worst_mdd_pct_4way']} econ={r['economic_ok']} name={r['name']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[cutoff-rerank] interrupted", file=sys.stderr, flush=True)
        raise
