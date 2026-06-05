#!/usr/bin/env python3
"""Evaluate 100 additional v2 strategies in a different direction from the recent pure flow-rotation batch.

Direction: technical-state confirmation + flow quality.
- Uses close-location / compression / volume participation / breakout context with smart-flow confirmation.
- Excludes already rejected mechanisms: price_drop, price_return, realized_vol, short_ratio.
- Saves every evaluated candidate to strategies_v2.db and writes a JSONL artifact.
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

MAX_EVAL = int(os.environ.get("V2_ALT_TECH_MAX", "100"))
OUT_JSONL = Path(os.environ.get("V2_ALT_TECH_OUT", "artifacts/strategy_v2_alt_technical_100_batch.jsonl"))

BANNED = {"price_drop", "price_return", "realized_vol", "short_ratio"}
SUBJECTS = ["foreign_registered", "foreign_unregistered", "private_equity", "investment_trust", "pension", "insurance", "bank", "other_corp"]
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
    return {"type": "kospi_trailing_return_scale", "window": window, "op": "<=", "threshold_pct": threshold, "risk_stock_weight_pct": risk}


def overlay_fx(window=60, threshold=5.0, risk=30.0, ma=120):
    return {"type": "usdkrw_trailing_return_ma_scale", "window": window, "op": ">=", "threshold_pct": threshold, "ma_window": ma, "risk_stock_weight_pct": risk}


def exit_rule(tp: int, sl: int, hold: int, unwind: int | None = None):
    e = {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold)}
    if unwind is not None:
        e["signal_all_of"] = [{"signal": "fast_money_unwind", "window": unwind}]
    return e


def base(name: str, entry: list[dict], exit_: dict, overlay: dict | None = None) -> dict:
    s = {"spec_version": 2, "name": name, "direction": "long", "entry": {"all_of": entry}, "exit": exit_}
    if overlay is not None:
        s["market_overlay"] = overlay
    return s


def sig_types(spec: dict) -> set[str]:
    out: set[str] = set()
    for side in ("entry", "exit"):
        obj = spec.get(side, {})
        for key in ("all_of", "signal_all_of"):
            for sig in obj.get(key, []) or []:
                if isinstance(sig, dict) and "signal" in sig:
                    out.add(sig["signal"])
    return out


def candidates() -> list[dict]:
    specs: list[dict] = []
    overlays = [
        ("ko80r20", overlay_kospi(80, 0.0, 20.0)),
        ("ko120r30", overlay_kospi(120, 0.0, 30.0)),
        ("fx60r30", overlay_fx(60, 5.0, 30.0, 120)),
        ("none", None),
    ]

    # T1: strong close + smart-flow consensus + liquidity. Not a breakout trigger; focuses on intraday absorption quality.
    for group, min_buyers in [("fast_money", 2), ("foreign_pair", 2), ("institution_defensive", 2), ("broad_smart", 3)]:
        for win in (3, 5, 10, 20):
            for min_pos in (0.6, 0.75, 0.9):
                for tag, ov in overlays[:3]:
                    specs.append(base(
                        f"alttech-close-cons-{group}-w{win}-cl{int(min_pos*100)}-{tag}",
                        [
                            {"signal": "close_location", "min_pos": min_pos},
                            {"signal": "flow_consensus", "group": group, "window": win, "min_buyers": min_buyers, "require_retail_sell": True},
                            {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 5_000_000_000.0},
                        ],
                        exit_rule(8, 3, 10, 3),
                        ov,
                    ))

    # T2: quiet range + strong close + retail absorption by a specific actor.
    for subj in SUBJECTS:
        for rc_w, rc_max in [(10, 10.0), (20, 10.0), (20, 15.0), (60, 20.0)]:
            for min_pos in (0.75, 0.9):
                for tag, ov in [("ko80r20", overlays[0][1]), ("none", None)]:
                    specs.append(base(
                        f"alttech-quiet-close-div-{ABBR[subj]}-rc{rc_w}_{int(rc_max)}-cl{int(min_pos*100)}-{tag}",
                        [
                            {"signal": "range_compression", "window": rc_w, "max_pct": rc_max},
                            {"signal": "close_location", "min_pos": min_pos},
                            {"signal": "flow_divergence", "subject": subj, "window": 10},
                        ],
                        exit_rule(10, 5, 20, 5),
                        ov,
                    ))

    # T3: participation expansion + abnormal flow spike + flow/return synchronization.
    for subj in ["foreign_unregistered", "private_equity", "investment_trust", "pension", "insurance", "bank"]:
        for vs_s, vs_l, vs_r in [(3, 20, 1.5), (5, 20, 2.0), (5, 60, 1.5), (10, 60, 1.5)]:
            for zwin, zlb, zmin in [(3, 60, 1.0), (5, 120, 1.0), (10, 120, 1.5)]:
                specs.append(base(
                    f"alttech-vol-zcorr-{ABBR[subj]}-vs{vs_s}_{vs_l}_{int(vs_r*10)}-z{zwin}_{zlb}_{int(zmin*10)}",
                    [
                        {"signal": "volume_surge", "short": vs_s, "long": vs_l, "min_ratio": vs_r},
                        {"signal": "flow_zscore", "subject": subj, "window": zwin, "lookback": zlb, "min_z": zmin},
                        {"signal": "rolling_corr", "subject": subj, "lookback": 60, "min_r": 0.1},
                    ],
                    exit_rule(10, 5, 20, 3),
                    overlay_kospi(80, 0.0, 20.0),
                ))

    # T4: distributed accumulation near highs, emphasizing breadth of buyers rather than one hot actor.
    for group, share in [("broad_smart", 0.5), ("broad_smart", 0.6), ("institution_defensive", 0.5), ("institution_defensive", 0.6), ("fast_money", 0.7)]:
        for bh_w, prox in [(20, 2.0), (20, 5.0), (60, 5.0), (120, 5.0)]:
            for tag, ov in [("fx60r30", overlays[2][1]), ("ko120r30", overlays[1][1]), ("none", None)]:
                specs.append(base(
                    f"alttech-disp-nearhigh-{group}-s{int(share*100)}-bh{bh_w}_{int(prox)}-{tag}",
                    [
                        {"signal": "flow_dispersion", "group": group, "window": 10, "max_share": share},
                        {"signal": "breakout_high", "window": bh_w, "proximity_pct": prox},
                        {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 10_000_000_000.0},
                    ],
                    exit_rule(15, 8, 40, 5),
                    ov,
                ))

    # T5: compression + actor acceleration + positive close quality.
    for subj in SUBJECTS:
        for short, long in [(3, 20), (5, 20), (10, 60)]:
            for rc_w, rc_max in [(10, 10.0), (20, 15.0), (60, 20.0)]:
                specs.append(base(
                    f"alttech-squeeze-accel-close-{ABBR[subj]}-a{short}_{long}-rc{rc_w}_{int(rc_max)}",
                    [
                        {"signal": "range_compression", "window": rc_w, "max_pct": rc_max},
                        {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                        {"signal": "close_location", "min_pos": 0.75},
                    ],
                    exit_rule(8, 5, 10, 3),
                    overlay_fx(60, 5.0, 30.0, 120),
                ))

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
    etf_codes = set(str(x)[:6] for x in ETF_CODES)
    raw = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in etf_codes]
    print(f"[alttech-100] before total={before_total} max_id={before_max} passed={before_pass} raw={len(raw)} ETF-excluded", flush=True)
    tickers, loader, kf, uf = _preload(raw)
    specs = [s for s in candidates() if spec_hash(s) not in existing]
    print(f"[alttech-100] universe={len(tickers)} candidates_new={len(specs)} max={MAX_EVAL} engine={engine_version()} banned={sorted(BANNED)}", flush=True)
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
    print(f"[alttech-100] done evaluated={evaluated} new_rows={after_total-before_total} id={before_max+1}..{after_max} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[alttech-100] PASSED_TOP " + json.dumps(passed[:10], ensure_ascii=False)[:8000], flush=True)
    if best:
        print("[alttech-100] BEST " + json.dumps(best[1], ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
