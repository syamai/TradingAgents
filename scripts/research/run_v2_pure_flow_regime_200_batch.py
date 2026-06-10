#!/usr/bin/env python3
"""Evaluate 200 v2 strategies in a pure flow/regime direction.

This runner is intentionally different from the #2160 / alttech absorption family:
- No range_compression, close_location, breakout_high, or volume_surge technical price-action triggers.
- No rejected price_drop, price_return, realized_vol, or short_ratio families.
- No flow_divergence retail-absorption signal.

Families focus on flow-regime evidence only: actor consensus, flow acceleration/z-score,
rolling flow/return synchronization, ownership-share rotation, net streak intensity,
liquidity filter, and market regime/overlay controls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.forward_test import ETF_CODES
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_spec_v2 import FLOW_GROUPS, validate_spec_v2
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_validation import engine_version, run_walk_forward_validation

MAX_EVAL = int(os.environ.get("V2_PURE_FLOW_MAX", "200"))
OUT_JSONL = Path(os.environ.get("V2_PURE_FLOW_OUT", "artifacts/strategy_v2_pure_flow_regime_200_batch.jsonl"))

SUBJECTS = [
    "foreign_registered",
    "foreign_unregistered",
    "private_equity",
    "investment_trust",
    "pension",
    "insurance",
    "bank",
    "securities",
    "other_corp",
]
ABBR = {
    "foreign_registered": "fr",
    "foreign_unregistered": "fu",
    "private_equity": "pe",
    "investment_trust": "it",
    "pension": "pn",
    "insurance": "ins",
    "bank": "bk",
    "securities": "sec",
    "other_corp": "oc",
}
GROUP_MIN = {"fast_money": 2, "foreign_pair": 2, "institution_defensive": 2, "broad_smart": 3}
BANNED = {
    "price_drop",
    "price_return",
    "realized_vol",
    "short_ratio",
    "range_compression",
    "close_location",
    "breakout_high",
    "volume_surge",
    "flow_divergence",
}


def overlay_kospi(window=80, threshold=0.0, risk=20.0):
    return {"type": "kospi_trailing_return_scale", "window": window, "op": "<=", "threshold_pct": threshold, "risk_stock_weight_pct": risk}


def overlay_fx(window=20, threshold=2.0, risk=30.0, ma=60):
    return {"type": "usdkrw_trailing_return_ma_scale", "window": window, "op": ">=", "threshold_pct": threshold, "ma_window": ma, "risk_stock_weight_pct": risk}


def add_overlay(spec: dict, overlay: dict | None) -> dict:
    if not overlay:
        return spec
    out = json.loads(json.dumps(spec))
    out["market_overlay"] = overlay
    return out


def base_exit(tp: float, sl: float, hold: int, unwind: int | None = 5, sell_subj: str | None = None) -> dict:
    sigs = []
    if unwind is not None:
        sigs.append({"signal": "fast_money_unwind", "window": unwind})
    if sell_subj is not None and len(sigs) < 2:
        sigs.append({"signal": "net_streak", "subject": sell_subj, "min_days": 3, "sign": "sell"})
    return {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold), "signal_all_of": sigs}


def sig_types(spec: dict) -> set[str]:
    out = set()
    for side, key in (("entry", "all_of"), ("exit", "signal_all_of")):
        for sig in spec.get(side, {}).get(key, []) or []:
            if isinstance(sig, dict):
                out.add(sig.get("signal"))
    return {x for x in out if x}


def group_subjects(group: str) -> tuple[str, ...]:
    return tuple(FLOW_GROUPS[group])


def candidates() -> list[dict]:
    specs: list[dict] = []
    overlays = [
        ("ko80r20", overlay_kospi(80, 0.0, 20.0)),
        ("ko120r30", overlay_kospi(120, 0.0, 30.0)),
        ("ko40m5r45", overlay_kospi(40, -5.0, 45.0)),
        ("fx20r30", overlay_fx(20, 2.0, 30.0, 60)),
        ("none", None),
    ]

    # P1: market-regime + flow surprise + flow/return synchronization.
    for subj in SUBJECTS:
        for zwin, zlb, zmin in [(3, 60, 1.0), (5, 120, 1.5), (10, 120, 1.0), (20, 250, 2.0)]:
            for corr_lb, min_r in [(20, 0.1), (60, 0.2), (120, 0.1), (120, 0.3)]:
                for mf_w in (20, 60, 120):
                    name = f"pureflow-zcorr-{ABBR[subj]}-z{zwin}_{zlb}_{int(zmin*10)}-r{corr_lb}_{int(min_r*10)}-mf{mf_w}"
                    specs.append({
                        "spec_version": 2,
                        "name": name,
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "flow_zscore", "subject": subj, "window": zwin, "lookback": zlb, "min_z": zmin},
                            {"signal": "rolling_corr", "subject": subj, "lookback": corr_lb, "min_r": min_r},
                            {"signal": "market_filter", "mode": "above_ma", "window": mf_w},
                        ]},
                        "exit": base_exit(10, 5, 20, 5, subj),
                    })

    # P2: group consensus + specific actor acceleration + liquidity; no price-action trigger.
    for group, mb in GROUP_MIN.items():
        for subj in group_subjects(group):
            for gw in (3, 5, 10, 20):
                for short, long in [(3, 20), (5, 20), (10, 60)]:
                    for liq in (1_000_000_000.0, 5_000_000_000.0, 10_000_000_000.0):
                        for ov_name, ov in overlays[:3]:
                            specs.append(add_overlay({
                                "spec_version": 2,
                                "name": f"pureflow-consacc-{group}-{ABBR[subj]}-w{gw}-a{short}_{long}-liq{int(liq/1e9)}-{ov_name}",
                                "direction": "long",
                                "entry": {"all_of": [
                                    {"signal": "flow_consensus", "group": group, "window": gw, "min_buyers": mb, "require_retail_sell": False},
                                    {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                                    {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": liq},
                                ]},
                                "exit": base_exit(8, 5, 10, 3, subj),
                            }, ov))

    # P3: ownership-share rotation away from retail into a named actor.
    for subj in [s for s in SUBJECTS if s != "retail"]:
        for w in (3, 5, 10, 20):
            for smart_delta in (2.0, 5.0, 10.0):
                for retail_delta in (2.0, 5.0, 10.0):
                    for mf_w in (20, 60):
                        specs.append({
                            "spec_version": 2,
                            "name": f"pureflow-pctrot-{ABBR[subj]}-w{w}-sd{int(smart_delta)}-rd{int(retail_delta)}-mf{mf_w}",
                            "direction": "long",
                            "entry": {"all_of": [
                                {"signal": "pct_delta", "subject": subj, "window": w, "op": ">=", "value": smart_delta},
                                {"signal": "pct_delta", "subject": "retail", "window": w, "op": "<=", "value": retail_delta},
                                {"signal": "market_filter", "mode": "above_ma", "window": mf_w},
                            ]},
                            "exit": base_exit(10, 8, 20, 5, subj),
                        })

    # P4: persistent buy streak with scale-free order-flow intensity.
    for subj in SUBJECTS:
        for days in (3, 4, 5, 7, 10):
            for vr_w, vr in [(3, 0.05), (5, 0.05), (5, 0.1), (10, 0.1), (20, 0.05)]:
                for liq_w, liq in [(20, 5_000_000_000.0), (60, 10_000_000_000.0)]:
                    for ov_name, ov in overlays[1:4]:
                        specs.append(add_overlay({
                            "spec_version": 2,
                            "name": f"pureflow-streakint-{ABBR[subj]}-d{days}-vr{vr_w}_{int(vr*100)}-liq{liq_w}_{int(liq/1e9)}-{ov_name}",
                            "direction": "long",
                            "entry": {"all_of": [
                                {"signal": "net_streak", "subject": subj, "min_days": days, "sign": "buy"},
                                {"signal": "net_vol_ratio", "subject": subj, "window": vr_w, "op": ">=", "value": vr},
                                {"signal": "liquidity_filter", "window": liq_w, "op": ">=", "value": liq},
                            ]},
                            "exit": base_exit(15, 8, 40, 5, subj),
                        }, ov))

    # P5: cumulative flow trend slope + synchronization + market regime.
    for subj in SUBJECTS:
        for tw in (10, 20, 60):
            for corr_lb, min_r in [(20, 0.1), (60, 0.1), (60, 0.2), (120, 0.2)]:
                for mf_w in (20, 60, 120):
                    specs.append({
                        "spec_version": 2,
                        "name": f"pureflow-sloper-{ABBR[subj]}-tw{tw}-r{corr_lb}_{int(min_r*10)}-mf{mf_w}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "trend_slope", "subject": subj, "window": tw, "direction": "up"},
                            {"signal": "rolling_corr", "subject": subj, "lookback": corr_lb, "min_r": min_r},
                            {"signal": "market_filter", "mode": "above_ma", "window": mf_w},
                        ]},
                        "exit": base_exit(10, 5, 20, 5, subj),
                    })

    seen, out = set(), []
    for spec in specs:
        validate_spec_v2(spec)
        st = sig_types(spec)
        if st & BANNED:
            raise ValueError(f"banned signal in {spec['name']}: {st & BANNED}")
        h = spec_hash(spec)
        if h not in seen:
            seen.add(h)
            out.append(spec)
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
    etf_codes = set(str(x)[:6] for x in ETF_CODES)
    raw = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in etf_codes]
    print(f"[pure-flow-200] before total={before_total} max_id={before_max} passed={before_pass} raw={len(raw)} ETF-excluded", flush=True)
    tickers, loader, kf, uf = _preload(raw)
    specs = [s for s in candidates() if spec_hash(s) not in existing]
    print(f"[pure-flow-200] universe={len(tickers)} candidates_new={len(specs)} max={MAX_EVAL} engine={engine_version()} banned={sorted(BANNED)}", flush=True)
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
            print(f"[{evaluated:03}/{MAX_EVAL}] {flag} #{sid} {spec['name'][:72]:72} wf_med={row['wf_excess_ir_median']} wf_min={row['wf_excess_ir_min']} | in {fmt(xsec['in_sample'])} out {fmt(xsec['out_sample'])}", flush=True)
    with store._conn() as c:
        after_total, after_pass, after_max = c.execute("SELECT count(*), coalesce(sum(gate_passed),0), coalesce(max(id),0) FROM strategies").fetchone()
    print(f"[pure-flow-200] done evaluated={evaluated} new_rows={after_total-before_total} id={before_max+1}..{after_max} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[pure-flow-200] PASSED_TOP " + json.dumps(passed[:10], ensure_ascii=False)[:8000], flush=True)
    if best:
        print("[pure-flow-200] BEST " + json.dumps(best[1], ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
