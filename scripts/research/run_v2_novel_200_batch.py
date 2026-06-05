#!/usr/bin/env python3
"""Evaluate 200 structurally novel v2 strategies, continuing after passes.

This is a one-shot research runner for post-#1518 exploration. It deliberately avoids
legacy rejected families: price_drop, price_return, realized_vol, short_ratio.
Families focus on technical state + flow confirmation:
- breakout/high-close confirmation + flow acceleration/z-score/divergence
- squeeze/range compression + volume expansion + smart-money consensus/dispersion
- distributed accumulation with KOSPI/FX overlays

Each candidate is validated, backtested with current portfolio-aware engine, walk-forward
validated, then saved to strategies_v2.db. Existing spec_hash rows are skipped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.forward_test import ETF_CODES
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_validation import engine_version, run_walk_forward_validation

MAX_EVAL = int(os.environ.get("V2_NOVEL_MAX", "200"))
OUT_JSONL = Path(os.environ.get("V2_NOVEL_OUT", "artifacts/strategy_v2_novel_200_batch.jsonl"))

ABBR = {
    "foreign_registered": "fr",
    "foreign_unregistered": "fu",
    "private_equity": "pe",
    "investment_trust": "it",
    "pension": "pn",
    "insurance": "ins",
    "bank": "bk",
    "other_corp": "oc",
}
SUBJECTS = ["foreign_unregistered", "private_equity", "investment_trust", "pension", "insurance", "bank", "other_corp"]
GROUPS = [("fast_money", 2), ("foreign_pair", 2), ("institution_defensive", 2), ("broad_smart", 3)]
BANNED = {"price_drop", "price_return", "realized_vol", "short_ratio"}


def overlay_kospi(window=80, threshold=0.0, risk=20.0):
    return {"type": "kospi_trailing_return_scale", "window": window, "op": "<=", "threshold_pct": threshold, "risk_stock_weight_pct": risk}


def overlay_fx(window=20, threshold=2.0, risk=30.0, ma=60):
    return {"type": "usdkrw_trailing_return_ma_scale", "window": window, "op": ">=", "threshold_pct": threshold, "ma_window": ma, "risk_stock_weight_pct": risk}


def base_exit(tp, sl, hold, unwind=3):
    return {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold), "signal_all_of": [{"signal": "fast_money_unwind", "window": unwind}]}


def add_overlay(spec: dict, overlay: dict | None) -> dict:
    if overlay:
        spec = json.loads(json.dumps(spec))
        spec["market_overlay"] = overlay
    return spec


def sig_types(spec: dict) -> set[str]:
    out = set()
    for side in ("entry", "exit"):
        obj = spec.get(side, {})
        for key in ("all_of", "signal_all_of"):
            for sig in obj.get(key, []) or []:
                if isinstance(sig, dict) and "signal" in sig:
                    out.add(sig["signal"])
    return out


def candidates() -> list[dict]:
    specs: list[dict] = []

    overlays = [("ko80", overlay_kospi(80, 0.0, 20.0)), ("ko120", overlay_kospi(120, 0.0, 30.0)), ("fx20", overlay_fx(20, 2.0, 30.0, 60)), ("none", None)]

    # N1: breakout + flow acceleration/zscore + liquidity, #1518 family expanded across actors/overlays.
    for subj in SUBJECTS:
        for bh_w, prox in [(20, 0.0), (20, 2.0), (60, 2.0), (120, 5.0)]:
            for flow in [
                {"signal": "flow_accel", "subject": subj, "short": 3, "long": 20},
                {"signal": "flow_accel", "subject": subj, "short": 5, "long": 20},
                {"signal": "flow_zscore", "subject": subj, "window": 5, "lookback": 120, "min_z": 1.5},
            ]:
                for ov_name, ov in overlays[:3]:
                    name = f"novel-break-{ABBR[subj]}-bh{bh_w}_{int(prox)}-{flow['signal'].replace('flow_','')}-{ov_name}-h20"
                    specs.append(add_overlay({
                        "spec_version": 2,
                        "name": name,
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "breakout_high", "window": bh_w, "proximity_pct": prox},
                            flow,
                            {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 5_000_000_000.0},
                        ]},
                        "exit": base_exit(10, 5, 20, 3),
                    }, ov))

    # N2: squeeze + volume surge + consensus. Not momentum; state transition after compression.
    for group, mb in GROUPS:
        for rc_w, rc_max in [(10, 10.0), (20, 10.0), (20, 15.0), (60, 20.0)]:
            for vs_s, vs_l, vs_r in [(3, 20, 1.5), (5, 60, 1.5), (5, 60, 2.0), (10, 60, 1.5)]:
                for ov_name, ov in overlays[:3]:
                    specs.append(add_overlay({
                        "spec_version": 2,
                        "name": f"novel-sqvol-cons-{group}-rc{rc_w}_{int(rc_max)}-vs{vs_s}_{vs_l}_{int(vs_r*10)}-{ov_name}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "range_compression", "window": rc_w, "max_pct": rc_max},
                            {"signal": "volume_surge", "short": vs_s, "long": vs_l, "min_ratio": vs_r},
                            {"signal": "flow_consensus", "group": group, "window": 5, "min_buyers": mb, "require_retail_sell": True},
                        ]},
                        "exit": base_exit(10, 5, 20, 5),
                    }, ov))

    # N3: close near day high + abnormal flow spike. Captures intraday absorption, not long price momentum.
    for subj in SUBJECTS:
        for min_pos in (0.6, 0.75, 0.9):
            for zwin, zlb, zmin in [(3, 60, 1.0), (5, 120, 1.5), (10, 250, 1.0), (20, 250, 2.0)]:
                for ov_name, ov in overlays[:3]:
                    specs.append(add_overlay({
                        "spec_version": 2,
                        "name": f"novel-closehi-z-{ABBR[subj]}-cl{int(min_pos*100)}-z{zwin}_{zlb}_{int(zmin*10)}-{ov_name}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "close_location", "min_pos": min_pos},
                            {"signal": "flow_zscore", "subject": subj, "window": zwin, "lookback": zlb, "min_z": zmin},
                            {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 5_000_000_000.0},
                        ]},
                        "exit": base_exit(8, 5, 10, 3),
                    }, ov))

    # N4: distributed accumulation + breakout/volume; no single actor dominance.
    for group, _mb in GROUPS:
        for share in (0.5, 0.6, 0.7):
            for bh_w, prox in [(20, 5.0), (60, 2.0), (120, 5.0)]:
                for vs_s, vs_l, vs_r in [(3, 20, 1.5), (5, 60, 1.5), (5, 60, 2.0)]:
                    specs.append(add_overlay({
                        "spec_version": 2,
                        "name": f"novel-disp-breakvol-{group}-s{int(share*100)}-bh{bh_w}_{int(prox)}-vs{vs_s}_{vs_l}_{int(vs_r*10)}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "flow_dispersion", "group": group, "window": 10, "max_share": share},
                            {"signal": "breakout_high", "window": bh_w, "proximity_pct": prox},
                            {"signal": "volume_surge", "short": vs_s, "long": vs_l, "min_ratio": vs_r},
                        ]},
                        "exit": base_exit(10, 5, 20, 5),
                    }, overlay_fx(20, 2.0, 30.0, 60)))

    # N5: quiet retail absorption: compression + smart/retail divergence + strong close.
    for subj in SUBJECTS:
        for rc_w, rc_max in [(10, 10.0), (20, 15.0), (60, 20.0)]:
            for min_pos in (0.6, 0.75):
                for ov_name, ov in overlays:
                    specs.append(add_overlay({
                        "spec_version": 2,
                        "name": f"novel-quietabs-{ABBR[subj]}-rc{rc_w}_{int(rc_max)}-cl{int(min_pos*100)}-{ov_name}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "range_compression", "window": rc_w, "max_pct": rc_max},
                            {"signal": "flow_divergence", "subject": subj, "window": 5},
                            {"signal": "close_location", "min_pos": min_pos},
                        ]},
                        "exit": base_exit(8, 5, 10, 3),
                    }, ov))

    seen, out = set(), []
    for s in specs:
        validate_spec_v2(s)
        st = sig_types(s)
        if st & BANNED:
            raise ValueError(f"banned signal in {s['name']}: {st & BANNED}")
        h = spec_hash(s)
        if h not in seen:
            seen.add(h)
            out.append(s)
    return out


def fmt(m: dict) -> str:
    def r(x):
        return "NA" if x is None else f"{x:.4g}" if isinstance(x, float) else str(x)
    return f"sh={r(m.get('sharpe'))} ex={r(m.get('excess_sharpe'))} mdd={r(m.get('mdd_pct'))} n={m.get('n_trades')} cum={r(m.get('cum_return_pct'))}"


def main() -> int:
    store = StrategyStoreV2()
    with store._conn() as c:
        before_total, before_pass, before_max = c.execute("SELECT count(*), coalesce(sum(gate_passed),0), coalesce(max(id),0) FROM strategies").fetchone()
        existing = {r[0] for r in c.execute("SELECT spec_hash FROM strategies")}
    raw = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in ETF_CODES]
    print(f"[novel-200] before total={before_total} max_id={before_max} passed={before_pass} raw={len(raw)} ETF-excluded", flush=True)
    tickers, loader, kf, uf = _preload(raw)
    specs = [s for s in candidates() if spec_hash(s) not in existing]
    print(f"[novel-200] universe={len(tickers)} candidates_new={len(specs)} max={MAX_EVAL} engine={engine_version()} banned={sorted(BANNED)}", flush=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    evaluated = 0
    passed = []
    best = None
    with OUT_JSONL.open("a", encoding="utf-8") as out:
        for spec in specs:
            if evaluated >= MAX_EVAL:
                break
            xsec = run_universe_backtest_v2(spec, tickers, loader=loader, kospi_fetcher=kf, usdkrw_fetcher=uf)
            wf = run_walk_forward_validation(spec, tickers, loader=loader, kospi_fetcher=kf, usdkrw_fetcher=uf)
            sid, is_new = store.save(spec, xsec, name=spec["name"], wf_result=wf, engine_version=engine_version())
            evaluated += 1
            row = {
                "strategy_id": sid,
                "is_new": is_new,
                "name": spec["name"],
                "gate_passed": bool(wf["gate_passed"]),
                "wf_excess_ir_median": wf.get("oos_excess_ir_median"),
                "wf_excess_ir_min": wf.get("oos_excess_ir_min"),
                "wf_oos_sharpe_median": wf.get("oos_sharpe_median"),
                "xsec_in": xsec["in_sample"],
                "xsec_out": xsec["out_sample"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            key = (row["wf_excess_ir_min"] if row["wf_excess_ir_min"] is not None else -999, row["wf_excess_ir_median"] if row["wf_excess_ir_median"] is not None else -999)
            if best is None or key > best[0]:
                best = (key, row)
            if row["gate_passed"]:
                passed.append(row)
            flag = "PASS" if row["gate_passed"] else "    "
            print(f"[{evaluated:03}/{MAX_EVAL}] {flag} #{sid} {spec['name'][:66]:66} wf_med={row['wf_excess_ir_median']} wf_min={row['wf_excess_ir_min']} | in {fmt(xsec['in_sample'])} out {fmt(xsec['out_sample'])}", flush=True)
    with store._conn() as c:
        after_total, after_pass, after_max = c.execute("SELECT count(*), coalesce(sum(gate_passed),0), coalesce(max(id),0) FROM strategies").fetchone()
    print(f"[novel-200] done evaluated={evaluated} new_rows={after_total-before_total} id={before_max+1}..{after_max} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[novel-200] PASSED_TOP " + json.dumps(passed[:10], ensure_ascii=False)[:8000], flush=True)
    if best:
        print("[novel-200] BEST " + json.dumps(best[1], ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
