#!/usr/bin/env python3
"""Run 100 structurally new v2 strategies using newly added technical state signals.

Concepts are intentionally not price_drop/short_ratio/price_return/realized_vol numeric variants:
- squeeze + participation expansion: range_compression + volume_surge + flow signal
- breakout with supply/demand confirmation: breakout_high + flow signal + liquidity
- retail capitulation absorption: flow_divergence + volume/range state
- macro/FX risk overlays as portfolio-level regime control
"""
from __future__ import annotations

import json
import os
from itertools import product

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.forward_test import ETF_CODES
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_validation import engine_version, run_walk_forward_validation

MAX_EVAL = int(os.environ.get("V2_NEW_TECH_MAX", "100"))
OUT_JSONL = os.environ.get("V2_NEW_TECH_OUT", "artifacts/strategy_v2_new_technical_batch.jsonl")

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


def overlay_kospi(window=80, threshold=0.0, risk=20.0):
    return {
        "type": "kospi_trailing_return_scale",
        "window": window,
        "op": "<=",
        "threshold_pct": threshold,
        "risk_stock_weight_pct": risk,
    }


def overlay_fx(window=60, threshold=5.0, risk=30.0, ma=120):
    return {
        "type": "usdkrw_trailing_return_ma_scale",
        "window": window,
        "op": ">=",
        "threshold_pct": threshold,
        "risk_stock_weight_pct": risk,
        "ma_window": ma,
    }


def base_exit(tp, sl, hold, exit_sig=None):
    out = {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold)}
    if exit_sig:
        out["signal_all_of"] = [exit_sig]
    return out


def candidates() -> list[dict]:
    specs: list[dict] = []

    # C1: volatility squeeze + volume expansion + smart-money consensus.
    for group, mb in [("fast_money", 2), ("foreign_pair", 2), ("institution_defensive", 2), ("broad_smart", 3)]:
        for rc_w, rc_max, vs_s, vs_l, vs_r in [(20, 10.0, 3, 20, 1.5), (20, 15.0, 5, 60, 2.0), (60, 20.0, 10, 60, 1.5)]:
            for ov_name, ov in [("ko80", overlay_kospi(80, 0.0, 20.0)), ("fx60", overlay_fx(60, 5.0, 30.0))]:
                specs.append({
                    "spec_version": 2,
                    "name": f"technew-squeeze-vol-cons-{group}-rc{rc_w}_{int(rc_max)}-vs{vs_s}_{vs_l}_{int(vs_r*10)}-{ov_name}-h20",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "range_compression", "window": rc_w, "max_pct": rc_max},
                        {"signal": "volume_surge", "short": vs_s, "long": vs_l, "min_ratio": vs_r},
                        {"signal": "flow_consensus", "group": group, "window": 5, "min_buyers": mb, "require_retail_sell": True},
                    ]},
                    "exit": base_exit(10, 5, 20, {"signal": "fast_money_unwind", "window": 5}),
                    "market_overlay": ov,
                })

    # C2: high breakout confirmed by flow acceleration/divergence, liquid names only.
    for subj in ["foreign_unregistered", "private_equity", "investment_trust", "pension", "other_corp"]:
        for bh_w, prox, flow_kind in [(20, 0.0, "accel"), (60, 2.0, "accel"), (120, 5.0, "div")]:
            for hold, tp, sl in [(20, 10, 5), (40, 15, 8)]:
                flow_sig = {"signal": "flow_accel", "subject": subj, "short": 5, "long": 20} if flow_kind == "accel" else {"signal": "flow_divergence", "subject": subj, "window": 10}
                specs.append({
                    "spec_version": 2,
                    "name": f"technew-breakout-{ABBR[subj]}-bh{bh_w}_{int(prox)}-{flow_kind}-liq50-tp{tp}sl{sl}h{hold}",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "breakout_high", "window": bh_w, "proximity_pct": prox},
                        flow_sig,
                        {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 5_000_000_000.0},
                    ]},
                    "exit": base_exit(tp, sl, hold, {"signal": "fast_money_unwind", "window": 3}),
                    "market_overlay": overlay_kospi(80, 0.0, 20.0),
                })

    # C3: retail capitulation absorption after quiet range, without price_drop trigger.
    for subj in ["foreign_registered", "foreign_unregistered", "private_equity", "investment_trust", "pension", "insurance", "bank"]:
        for rc_w, rc_max in [(10, 10.0), (20, 15.0), (60, 20.0)]:
            for ov_name, ov in [("none", None), ("ko120", overlay_kospi(120, 0.0, 30.0))]:
                spec = {
                    "spec_version": 2,
                    "name": f"technew-absorb-quiet-{ABBR[subj]}-rc{rc_w}_{int(rc_max)}-{ov_name}-h10",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "range_compression", "window": rc_w, "max_pct": rc_max},
                        {"signal": "flow_divergence", "subject": subj, "window": 5},
                        {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 10_000_000_000.0},
                    ]},
                    "exit": base_exit(8, 5, 10, {"signal": "fast_money_unwind", "window": 3}),
                }
                if ov:
                    spec["market_overlay"] = ov
                specs.append(spec)

    # C4: distributed accumulation + volume expansion; no single actor dominance.
    for group, share in [("broad_smart", 0.5), ("broad_smart", 0.6), ("institution_defensive", 0.6), ("fast_money", 0.7)]:
        for vs in [(3, 20, 1.5), (5, 60, 1.5), (5, 60, 2.0)]:
            for bh_w, prox in [(20, 5.0), (60, 2.0)]:
                specs.append({
                    "spec_version": 2,
                    "name": f"technew-distrib-vol-break-{group}-s{int(share*100)}-vs{vs[0]}_{vs[1]}_{int(vs[2]*10)}-bh{bh_w}_{int(prox)}",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "flow_dispersion", "group": group, "window": 10, "max_share": share},
                        {"signal": "volume_surge", "short": vs[0], "long": vs[1], "min_ratio": vs[2]},
                        {"signal": "breakout_high", "window": bh_w, "proximity_pct": prox},
                    ]},
                    "exit": base_exit(10, 5, 20, {"signal": "fast_money_unwind", "window": 5}),
                    "market_overlay": overlay_fx(60, 3.0, 45.0, 120),
                })

    # C5: strong close near day high + abnormal flow spike. This captures intraday absorption
    # without using price_drop or long price momentum.
    for subj in ["foreign_unregistered", "private_equity", "investment_trust", "pension", "insurance", "bank"]:
        for min_pos in (0.75, 0.9):
            for zwin, zlb, zmin in [(3, 60, 1.0), (5, 120, 1.5), (10, 250, 1.0)]:
                specs.append({
                    "spec_version": 2,
                    "name": f"technew-closehigh-zflow-{ABBR[subj]}-cl{int(min_pos*100)}-z{zwin}_{zlb}_{int(zmin*10)}-liq50",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "close_location", "min_pos": min_pos},
                        {"signal": "flow_zscore", "subject": subj, "window": zwin, "lookback": zlb, "min_z": zmin},
                        {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 5_000_000_000.0},
                    ]},
                    "exit": base_exit(8, 5, 10, {"signal": "fast_money_unwind", "window": 3}),
                    "market_overlay": overlay_kospi(80, 0.0, 20.0),
                })

    # Dedup by hash while preserving order.
    seen, out = set(), []
    for s in specs:
        validate_spec_v2(s)
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
        before_total, before_pass = c.execute("SELECT count(*), coalesce(sum(gate_passed),0) FROM strategies").fetchone()
        existing = {r[0] for r in c.execute("SELECT spec_hash FROM strategies")}
    raw = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in ETF_CODES]
    print(f"[new-tech] before total={before_total} passed={before_pass} raw={len(raw)} ETF-excluded", flush=True)
    tickers, loader, kf, uf = _preload(raw)
    specs = [s for s in candidates() if spec_hash(s) not in existing]
    print(f"[new-tech] universe={len(tickers)} candidates_new={len(specs)} max={MAX_EVAL} engine={engine_version()}", flush=True)
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    evaluated = 0
    passed = []
    best = None
    with open(OUT_JSONL, "a", encoding="utf-8") as out:
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
            flag = "PASS" if row["gate_passed"] else "    "
            print(f"[{evaluated:03}/{MAX_EVAL}] {flag} #{sid} {spec['name'][:58]:58} wf_med={row['wf_excess_ir_median']} wf_min={row['wf_excess_ir_min']} | in {fmt(xsec['in_sample'])} out {fmt(xsec['out_sample'])}", flush=True)
            if row["gate_passed"]:
                passed.append(row)
                break
    with store._conn() as c:
        after_total, after_pass = c.execute("SELECT count(*), coalesce(sum(gate_passed),0) FROM strategies").fetchone()
    print(f"[new-tech] done evaluated={evaluated} new_rows={after_total-before_total} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[new-tech] PASSED " + json.dumps(passed[0], ensure_ascii=False)[:4000], flush=True)
    elif best:
        print("[new-tech] BEST " + json.dumps(best[1], ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
