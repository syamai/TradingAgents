#!/usr/bin/env python3
"""Evaluate 100 v2 strategies in a direction deliberately different from recent breakout/volume/dispersion batches.

Focus: flow-regime / actor-rotation / retail-absorption signals only.
Deliberately excludes recent technical-breakout families and rejected mechanisms:
- no breakout_high, volume_surge, range_compression, close_location, flow_dispersion
- no price_drop, price_return, realized_vol, short_ratio

Every evaluated candidate is saved to strategies_v2.db and logged to JSONL.
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
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_validation import engine_version, run_walk_forward_validation

MAX_EVAL = int(os.environ.get("V2_FLOW_ROT_MAX", "100"))
OUT_JSONL = Path(os.environ.get("V2_FLOW_ROT_OUT", "artifacts/strategy_v2_flow_rotation_100_batch.jsonl"))

# Reject both historically failed families and the latest breakout/volume/dispersion direction.
BANNED = {
    "price_drop", "price_return", "realized_vol", "short_ratio",
    "breakout_high", "volume_surge", "range_compression", "close_location", "flow_dispersion",
}
SUBJECTS = ["pension", "insurance", "bank", "investment_trust", "private_equity", "foreign_registered", "foreign_unregistered"]
DEFENSIVE = ["pension", "insurance", "bank", "investment_trust"]
FAST = ["private_equity", "foreign_unregistered"]
ABBR = {
    "pension": "pn", "insurance": "ins", "bank": "bk", "investment_trust": "it",
    "private_equity": "pe", "foreign_registered": "fr", "foreign_unregistered": "fu",
}


def overlay_kospi(window=80, threshold=0.0, risk=20.0):
    return {"type": "kospi_trailing_return_scale", "window": window, "op": "<=", "threshold_pct": threshold, "risk_stock_weight_pct": risk}


def overlay_fx(window=20, threshold=2.0, risk=30.0, ma=60):
    return {"type": "usdkrw_trailing_return_ma_scale", "window": window, "op": ">=", "threshold_pct": threshold, "ma_window": ma, "risk_stock_weight_pct": risk}


def exit_rule(tp: int, sl: int, hold: int, unwind: int | None = 5):
    e = {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold), "signal_all_of": []}
    if unwind:
        e["signal_all_of"].append({"signal": "fast_money_unwind", "window": unwind})
    return e


def clone_with_overlay(spec: dict, ov: dict | None, tag: str) -> dict:
    spec = json.loads(json.dumps(spec))
    spec["name"] = f"{spec['name']}-{tag}"
    if ov:
        spec["market_overlay"] = ov
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


def base(name: str, entry: list[dict], exit_: dict) -> dict:
    return {"spec_version": 2, "name": name, "direction": "long", "entry": {"all_of": entry}, "exit": exit_}


def candidates() -> list[dict]:
    specs: list[dict] = []
    overlays = [("ko80", overlay_kospi(80, 0.0, 20.0)), ("ko120", overlay_kospi(120, 0.0, 30.0)), ("fx20", overlay_fx(20, 2.0, 30.0, 60)), ("none", None)]

    # F1: Defensive-institution rotation: defensive consensus + individual accumulation acceleration + retail absorption.
    for subj in DEFENSIVE:
        for win in (5, 10, 20):
            for short, long in ((3, 20), (5, 20), (10, 60)):
                s = base(
                    f"flowrot-defcons-{ABBR[subj]}-w{win}-a{short}_{long}",
                    [
                        {"signal": "flow_consensus", "group": "institution_defensive", "window": win, "min_buyers": 2, "require_retail_sell": True},
                        {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                        {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 5_000_000_000.0},
                    ],
                    exit_rule(10, 5, 20, 5),
                )
                for tag, ov in overlays[:3]:
                    specs.append(clone_with_overlay(s, ov, tag))

    # F2: Actor-specific flow spike with positive flow/return synchronization, no price breakout condition.
    for subj in SUBJECTS:
        for zwin, zlb, zmin in ((3, 60, 1.0), (5, 120, 1.5), (10, 120, 1.0), (20, 120, 1.5)):
            for corr_lb, min_r in ((20, 0.1), (60, 0.2), (120, 0.1)):
                s = base(
                    f"flowrot-zcorr-{ABBR[subj]}-z{zwin}_{zlb}_{int(zmin*10)}-r{corr_lb}_{int(min_r*10)}",
                    [
                        {"signal": "flow_zscore", "subject": subj, "window": zwin, "lookback": zlb, "min_z": zmin},
                        {"signal": "rolling_corr", "subject": subj, "lookback": corr_lb, "min_r": min_r},
                        {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 5_000_000_000.0},
                    ],
                    exit_rule(8, 5, 10, 3),
                )
                for tag, ov in overlays[1:4]:
                    specs.append(clone_with_overlay(s, ov, tag))

    # F3: Retail-absorption divergence: smart actor buys while retail sells; add subject trend slope instead of price state.
    for subj in SUBJECTS:
        if subj == "foreign_unregistered":
            continue
        for win in (5, 10, 20):
            for slope_w in (10, 20, 60):
                s = base(
                    f"flowrot-absorb-{ABBR[subj]}-d{win}-sl{slope_w}",
                    [
                        {"signal": "flow_divergence", "subject": subj, "window": win},
                        {"signal": "trend_slope", "subject": subj, "window": slope_w, "direction": "up"},
                        {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 10_000_000_000.0},
                    ],
                    exit_rule(10, 5, 20, 5),
                )
                for tag, ov in (("ko80", overlays[0][1]), ("fx20", overlays[2][1])):
                    specs.append(clone_with_overlay(s, ov, tag))

    # F4: Fast-money early rotation but constrained by foreign-pair/institutional consensus; short holding.
    for subj in FAST:
        for streak in (2, 3, 5):
            for group, min_buyers in (("fast_money", 2), ("foreign_pair", 2), ("broad_smart", 3)):
                for w in (3, 5, 10):
                    s = base(
                        f"flowrot-fastsync-{ABBR[subj]}-st{streak}-{group}-w{w}",
                        [
                            {"signal": "net_streak", "subject": subj, "min_days": streak, "sign": "buy"},
                            {"signal": "flow_consensus", "group": group, "window": w, "min_buyers": min_buyers, "require_retail_sell": True},
                            {"signal": "net_vol_ratio", "subject": subj, "window": w, "op": ">=", "value": 0.05},
                        ],
                        exit_rule(8, 5, 10, 3),
                    )
                    for tag, ov in (("fx20", overlays[2][1]), ("none", None)):
                        specs.append(clone_with_overlay(s, ov, tag))

    # F5: Ownership-share rotation: subject pct share rises while retail share falls. Pure flow composition shift.
    for subj in SUBJECTS:
        for w in (3, 5, 10, 20):
            for val in (2, 5):
                s = base(
                    f"flowrot-pctshift-{ABBR[subj]}-w{w}-v{val}",
                    [
                        {"signal": "pct_delta", "subject": subj, "window": w, "op": ">=", "value": val},
                        {"signal": "pct_delta", "subject": "retail", "window": w, "op": "<=", "value": val},
                        {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 5_000_000_000.0},
                    ],
                    exit_rule(10, 5, 20, None),
                )
                for tag, ov in (("ko80", overlays[0][1]), ("fx20", overlays[2][1]), ("none", None)):
                    specs.append(clone_with_overlay(s, ov, tag))

    seen, out = set(), []
    for s in specs:
        validate_spec_v2(s)
        st = sig_types(s)
        if st & BANNED:
            raise ValueError(f"banned/recent-direction signal in {s['name']}: {st & BANNED}")
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
    etf_codes = set(str(x)[:6] for x in ETF_CODES)
    raw = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in etf_codes]
    print(f"[flowrot-100] before total={before_total} max_id={before_max} passed={before_pass} raw={len(raw)} ETF-excluded", flush=True)
    tickers, loader, kf, uf = _preload(raw)
    specs = [s for s in candidates() if spec_hash(s) not in existing]
    print(f"[flowrot-100] universe={len(tickers)} candidates_new={len(specs)} max={MAX_EVAL} engine={engine_version()} banned={sorted(BANNED)}", flush=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    evaluated = 0
    passed: list[dict] = []
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
            print(f"[{evaluated:03}/{MAX_EVAL}] {flag} #{sid} {spec['name'][:70]:70} wf_med={row['wf_excess_ir_median']} wf_min={row['wf_excess_ir_min']} | in {fmt(xsec['in_sample'])} out {fmt(xsec['out_sample'])}", flush=True)
    with store._conn() as c:
        after_total, after_pass, after_max = c.execute("SELECT count(*), coalesce(sum(gate_passed),0), coalesce(max(id),0) FROM strategies").fetchone()
    print(f"[flowrot-100] done evaluated={evaluated} new_rows={after_total-before_total} id={before_max+1}..{after_max} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[flowrot-100] PASSED_TOP " + json.dumps(passed[:10], ensure_ascii=False)[:8000], flush=True)
    if best:
        print("[flowrot-100] BEST " + json.dumps(best[1], ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
