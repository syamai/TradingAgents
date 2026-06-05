#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine import KOREA_STOCK_PORTFOLIO_POLICY, TX_COST_ONE_WAY
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_validation import run_time_split_validation

DB = Path.home() / ".tradingagents/hermes/strategies_v2.db"
OUT_DIR = Path("/Users/selab/Source/trading-ai/artifacts/market_overlay_experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CUTOFF = "2025-06-30"
IDS = [1104, 1035, 1081]
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
JSON_OUT = OUT_DIR / f"selected_1104_1035_1081_overlay_{TS}.json"
CSV_OUT = OUT_DIR / f"selected_1104_1035_1081_overlay_{TS}.csv"

OVERLAYS = {
    "baseline": None,
    "kospi80_stock20": {
        "type": "kospi_trailing_return_scale",
        "window": 80,
        "op": "<=",
        "threshold_pct": 0.0,
        "risk_stock_weight_pct": 20.0,
    },
    "fx20_ge2_ma60_stock30": {
        "type": "usdkrw_trailing_return_ma_scale",
        "window": 20,
        "op": ">=",
        "threshold_pct": 2.0,
        "ma_window": 60,
        "risk_stock_weight_pct": 30.0,
    },
}


def years(lo: str | None, hi: str | None) -> float | None:
    if not lo or not hi:
        return None
    return (date.fromisoformat(hi) - date.fromisoformat(lo)).days / 365.25


def ann(cum_pct: float | None, n_years: float | None) -> float | None:
    if cum_pct is None or not n_years or n_years <= 0:
        return None
    base = 1.0 + float(cum_pct) / 100.0
    if base <= 0:
        return None
    return (base ** (1 / n_years) - 1) * 100


def mval(m: dict, key: str, default: float = -999.0) -> float:
    v = m.get(key)
    return default if v is None else float(v)


def summarize(base_id: int, base_name: str, label: str, spec: dict, xres: dict, tres: dict) -> dict:
    xi, xo = xres["in_sample"], xres["out_sample"]
    ti, to = tres["in_sample"], tres["out_sample"]
    sharpes = [mval(xi, "sharpe"), mval(xo, "sharpe"), mval(ti, "sharpe"), mval(to, "sharpe")]
    mdds = [mval(xi, "mdd_pct"), mval(xo, "mdd_pct"), mval(ti, "mdd_pct"), mval(to, "mdd_pct")]
    vols = [mval(xi, "daily_excess_vol_pct", 999), mval(xo, "daily_excess_vol_pct", 999), mval(ti, "daily_excess_vol_pct", 999), mval(to, "daily_excess_vol_pct", 999)]
    span = tres.get("split", {}).get("data_span", [None, CUTOFF])
    x_years = years(span[0], span[1])
    t_years = years(tres.get("split", {}).get("oos_start"), span[1])
    return {
        "base_id": base_id,
        "base_name": base_name,
        "overlay": label,
        "gate": bool(xres.get("gate_passed") and tres.get("gate_passed")),
        "xsec_gate": bool(xres.get("gate_passed")),
        "time_gate": bool(tres.get("gate_passed")),
        "min_sharpe_4way": round(min(sharpes), 4),
        "avg_sharpe_4way": round(sum(sharpes) / 4, 4),
        "worst_mdd_pct_4way": round(min(mdds), 4),
        "max_daily_excess_vol_pct_4way": round(max(vols), 4),
        "xsec_out_sharpe": xo.get("sharpe"),
        "xsec_out_mdd": xo.get("mdd_pct"),
        "xsec_out_cum": xo.get("cum_return_pct"),
        "xsec_out_ann": round(ann(xo.get("cum_return_pct"), x_years), 4) if ann(xo.get("cum_return_pct"), x_years) is not None else None,
        "xsec_out_n": xo.get("n_trades"),
        "time_out_sharpe": to.get("sharpe"),
        "time_out_mdd": to.get("mdd_pct"),
        "time_out_cum": to.get("cum_return_pct"),
        "time_out_ann": round(ann(to.get("cum_return_pct"), t_years), 4) if ann(to.get("cum_return_pct"), t_years) is not None else None,
        "time_out_n": to.get("n_trades"),
        "spec": spec,
    }


def main() -> int:
    print(f"[selected-overlay] ids={IDS} cutoff={CUTOFF} policy={KOREA_STOCK_PORTFOLIO_POLICY} tx_one_way={TX_COST_ONE_WAY*100:.4f}%", flush=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"SELECT id,name,spec_json FROM strategies WHERE id IN ({','.join('?' for _ in IDS)})", IDS).fetchall()
    rows = sorted(rows, key=lambda r: IDS.index(int(r["id"])))
    raw = KisHistoryStore().list_tickers()
    tickers, loader0, kospi_fetcher, usdkrw_fetcher = _preload(raw)

    def cutoff_loader(tk: str):
        df, meta = loader0(tk)
        if df is None or df.empty:
            return df, meta
        d = df["date"].astype(str)
        return df[d <= CUTOFF].reset_index(drop=True), meta

    results = []
    for row in rows:
        base = json.loads(row["spec_json"])
        for label, overlay in OVERLAYS.items():
            spec = json.loads(json.dumps(base, ensure_ascii=False))
            if overlay is not None:
                spec["market_overlay"] = dict(overlay)
                spec["name"] = f"{spec.get('name', row['name'])}__{label}"
            else:
                spec.pop("market_overlay", None)
            print(f"[selected-overlay] run #{row['id']} {label}", flush=True)
            xres = run_universe_backtest_v2(spec, tickers, loader=cutoff_loader, kospi_fetcher=kospi_fetcher, usdkrw_fetcher=usdkrw_fetcher)
            tres = run_time_split_validation(spec, tickers, loader=cutoff_loader, kospi_fetcher=kospi_fetcher, usdkrw_fetcher=usdkrw_fetcher)
            rec = summarize(int(row["id"]), row["name"], label, spec, xres, tres)
            results.append(rec)
            print(f"  minS={rec['min_sharpe_4way']} MDD={rec['worst_mdd_pct_4way']} tAnn={rec['time_out_ann']} xAnn={rec['xsec_out_ann']}", flush=True)
    payload = {"generated_at": TS, "cutoff": CUTOFF, "ids": IDS, "overlays": OVERLAYS, "portfolio_policy": KOREA_STOCK_PORTFOLIO_POLICY, "tx_cost_one_way_pct": TX_COST_ONE_WAY*100, "results": results}
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [k for k in results[0].keys() if k != "spec"] + ["spec_json"]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            out = {k: v for k, v in r.items() if k != "spec"}
            out["spec_json"] = json.dumps(r["spec"], ensure_ascii=False)
            w.writerow(out)
    print(f"[selected-overlay] DONE json={JSON_OUT} csv={CSV_OUT}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
