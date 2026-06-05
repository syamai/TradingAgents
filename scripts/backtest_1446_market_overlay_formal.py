#!/usr/bin/env python3
"""Formal backtest for #1446 with market_overlay variants under cutoff loader."""
from __future__ import annotations

import copy
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_validation import run_time_split_validation, run_walk_forward_validation

DB = Path.home() / ".tradingagents" / "hermes" / "strategies_v2.db"
CUTOFF = os.environ.get("CUTOFF", "2025-06-30")
OUTDIR = Path("artifacts/market_overlay_experiments")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT id,name,spec_json FROM strategies WHERE id=?", (1446,)).fetchone()
    if row is None:
        raise SystemExit("strategy #1446 not found")
    base_spec = json.loads(row["spec_json"])

    raw_tickers = KisHistoryStore().list_tickers()
    tickers, loader0, _kf, _uf = _preload(raw_tickers)

    def cutoff_loader(tk: str):
        df, meta = loader0(tk)
        if df is None or df.empty:
            return df, meta
        d = df["date"].astype(str)
        return df[d <= CUTOFF].reset_index(drop=True), meta

    variants = [
        ("baseline", None),
        ("ret80_le_0_stock20", {
            "type": "kospi_trailing_return_scale",
            "window": 80,
            "op": "<=",
            "threshold_pct": 0.0,
            "risk_stock_weight_pct": 20.0,
        }),
        ("ret20_le_m8_stock30", {
            "type": "kospi_trailing_return_scale",
            "window": 20,
            "op": "<=",
            "threshold_pct": -8.0,
            "risk_stock_weight_pct": 30.0,
        }),
        ("ret60_le_0_stock30_cap20_m6_stock20", {
            "type": "kospi_trailing_return_scale",
            "window": 60,
            "op": "<=",
            "threshold_pct": 0.0,
            "risk_stock_weight_pct": 30.0,
            "shock_cap": {
                "window": 20,
                "op": "<=",
                "threshold_pct": -6.0,
                "cap_stock_weight_pct": 20.0,
            },
        }),
    ]

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_id": 1446,
        "strategy_name": row["name"],
        "cutoff": CUTOFF,
        "n_tickers_raw": len(raw_tickers),
        "n_tickers_preloaded": len(tickers),
        "results": [],
    }
    for name, overlay in variants:
        spec = copy.deepcopy(base_spec)
        spec["name"] = f"{base_spec['name']}__{name}"
        if overlay is None:
            spec.pop("market_overlay", None)
        else:
            spec["market_overlay"] = overlay
        print(f"[run] {name}", flush=True)
        xsec = run_universe_backtest_v2(spec, tickers, loader=cutoff_loader)
        time = run_time_split_validation(spec, tickers, loader=cutoff_loader)
        wf = run_walk_forward_validation(spec, tickers, loader=cutoff_loader)
        out["results"].append({"variant": name, "market_overlay": overlay, "xsec": xsec, "time": time, "wf": wf})
        print(
            f"[done] {name} "
            f"x_out_sharpe={xsec['out_sample'].get('sharpe')} x_out_mdd={xsec['out_sample'].get('mdd_pct')} "
            f"t_out_sharpe={time['out_sample'].get('sharpe')} t_out_mdd={time['out_sample'].get('mdd_pct')} "
            f"wf_ir_med={wf.get('oos_excess_ir_median')} wf_ir_min={wf.get('oos_excess_ir_min')}",
            flush=True,
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUTDIR / f"formal_1446_market_overlay_{stamp}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
