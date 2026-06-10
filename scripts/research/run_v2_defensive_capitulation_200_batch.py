#!/usr/bin/env python3
"""Evaluate 200 v2 strategies in a defensive-capitulation direction.

New direction relative to the previous pure-flow batch:
- No flow_zscore / rolling_corr pair.
- No price-action triggers: range_compression, close_location, breakout_high, volume_surge.
- No rejected families: price_drop, price_return, realized_vol, short_ratio.
- No flow_divergence shortcut.

Economic idea: long candidates where retail is capitulating / losing ownership share while
stable or defensive institutions (pension, insurance, investment_trust, bank) accumulate,
often under weak-market filters or explicit KOSPI/FX risk overlays. This tests defensive
rotation / retail capitulation rather than breakout, absorption, or z-score synchronization.
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

MAX_EVAL = int(os.environ.get("V2_DEF_CAP_MAX", "200"))
OUT_JSONL = Path(os.environ.get("V2_DEF_CAP_OUT", "artifacts/strategy_v2_defensive_capitulation_200_batch.jsonl"))

DEFENSIVE = ["pension", "insurance", "investment_trust", "bank"]
SMART = ["pension", "insurance", "investment_trust", "bank", "foreign_registered"]
ABBR = {"pension": "pn", "insurance": "ins", "investment_trust": "it", "bank": "bk", "foreign_registered": "fr"}
BANNED = {
    "price_drop", "price_return", "realized_vol", "short_ratio",
    "range_compression", "close_location", "breakout_high", "volume_surge",
    "flow_divergence", "flow_zscore", "rolling_corr",
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


def exit_rule(tp: float, sl: float, hold: int, subj: str | None = None, unwind: int = 5) -> dict:
    sigs = [{"signal": "fast_money_unwind", "window": unwind}]
    if subj and len(sigs) < 2:
        sigs.append({"signal": "net_streak", "subject": subj, "min_days": 3, "sign": "sell"})
    return {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold), "signal_all_of": sigs}


def sig_types(spec: dict) -> set[str]:
    out = set()
    for side, key in (("entry", "all_of"), ("exit", "signal_all_of")):
        for sig in spec.get(side, {}).get(key, []) or []:
            if isinstance(sig, dict):
                out.add(sig.get("signal"))
    return {x for x in out if x}


def candidates() -> list[dict]:
    specs: list[dict] = []
    overlays = [
        ("ko80r20", overlay_kospi(80, 0.0, 20.0)),
        ("ko120r30", overlay_kospi(120, 0.0, 30.0)),
        ("ko40m5r45", overlay_kospi(40, -5.0, 45.0)),
        ("fx20r30", overlay_fx(20, 2.0, 30.0, 60)),
        ("none", None),
    ]

    # C1: retail capitulation + defensive consensus + liquidity under weak/neutral market.
    for gw in (3, 5, 10, 20):
        for retail_days in (3, 4, 5, 7, 10):
            for mf_mode, mf_w in [("below_ma", 20), ("below_ma", 60), ("above_ma", 20), ("above_ma", 60)]:
                for liq in (1_000_000_000.0, 5_000_000_000.0, 10_000_000_000.0):
                    for ov_name, ov in overlays[:3]:
                        specs.append(add_overlay({
                            "spec_version": 2,
                            "name": f"defcap-cons-retail-{gw}-rs{retail_days}-mf{mf_mode[:1]}{mf_w}-liq{int(liq/1e9)}-{ov_name}",
                            "direction": "long",
                            "entry": {"all_of": [
                                {"signal": "flow_consensus", "group": "institution_defensive", "window": gw, "min_buyers": 2, "require_retail_sell": True},
                                {"signal": "net_streak", "subject": "retail", "min_days": retail_days, "sign": "sell"},
                                {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": liq},
                            ]},
                            "exit": exit_rule(8, 5, 10, None, 3),
                        }, ov))
                        specs[-1]["entry"]["all_of"][2] = {"signal": "market_filter", "mode": mf_mode, "window": mf_w}

    # C2: a named defensive actor accelerates while retail ownership share falls.
    for subj in DEFENSIVE:
        for short, long in [(3, 20), (5, 20), (10, 60)]:
            for w in (3, 5, 10, 20):
                for rd in (2.0, 5.0, 10.0):
                    for ov_name, ov in overlays:
                        specs.append(add_overlay({
                            "spec_version": 2,
                            "name": f"defcap-acc-retailpct-{ABBR[subj]}-a{short}_{long}-w{w}-rd{int(rd)}-{ov_name}",
                            "direction": "long",
                            "entry": {"all_of": [
                                {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                                {"signal": "pct_delta", "subject": "retail", "window": w, "op": "<=", "value": rd},
                                {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 5_000_000_000.0},
                            ]},
                            "exit": exit_rule(10, 5, 20, subj, 5),
                        }, ov))

    # C3: smart ownership rotation: smart pct up + retail pct down + weak-market filter.
    for subj in SMART:
        for w in (3, 5, 10, 20):
            for sd in (2.0, 5.0, 10.0):
                for rd in (2.0, 5.0, 10.0):
                    for mf_w in (20, 60, 120):
                        specs.append({
                            "spec_version": 2,
                            "name": f"defcap-pctrot-weak-{ABBR[subj]}-w{w}-sd{int(sd)}-rd{int(rd)}-mf{mf_w}",
                            "direction": "long",
                            "entry": {"all_of": [
                                {"signal": "pct_delta", "subject": subj, "window": w, "op": ">=", "value": sd},
                                {"signal": "pct_delta", "subject": "retail", "window": w, "op": "<=", "value": rd},
                                {"signal": "market_filter", "mode": "below_ma", "window": mf_w},
                            ]},
                            "exit": exit_rule(15, 8, 40, subj, 5),
                        })

    # C4: stable buyer streak into retail selling pressure, scale-free net/volume confirmation.
    for subj in DEFENSIVE:
        for buy_days in (3, 4, 5, 7, 10):
            for sell_days in (3, 4, 5, 7):
                for vr_w, vr in [(3, 0.05), (5, 0.05), (5, 0.1), (10, 0.1), (20, 0.05)]:
                    for ov_name, ov in overlays[1:4]:
                        specs.append(add_overlay({
                            "spec_version": 2,
                            "name": f"defcap-streak-vr-{ABBR[subj]}-b{buy_days}-rs{sell_days}-vr{vr_w}_{int(vr*100)}-{ov_name}",
                            "direction": "long",
                            "entry": {"all_of": [
                                {"signal": "net_streak", "subject": subj, "min_days": buy_days, "sign": "buy"},
                                {"signal": "net_streak", "subject": "retail", "min_days": sell_days, "sign": "sell"},
                                {"signal": "net_vol_ratio", "subject": subj, "window": vr_w, "op": ">=", "value": vr},
                            ]},
                            "exit": exit_rule(10, 5, 20, subj, 5),
                        }, ov))

    # C5: defensive consensus plus fast-money unwind as entry avoidance via exit-only; longer hold.
    for gw in (5, 10, 20):
        for mb in (2, 3):
            for mf_mode, mf_w in [("below_ma", 20), ("below_ma", 60), ("above_ma", 20), ("above_ma", 60)]:
                for liq in (5_000_000_000.0, 10_000_000_000.0, 20_000_000_000.0):
                    for ov_name, ov in overlays:
                        specs.append(add_overlay({
                            "spec_version": 2,
                            "name": f"defcap-defcons-mktliq-w{gw}-mb{mb}-mf{mf_mode[:1]}{mf_w}-liq{int(liq/1e9)}-{ov_name}",
                            "direction": "long",
                            "entry": {"all_of": [
                                {"signal": "flow_consensus", "group": "institution_defensive", "window": gw, "min_buyers": mb, "require_retail_sell": True},
                                {"signal": "market_filter", "mode": mf_mode, "window": mf_w},
                                {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": liq},
                            ]},
                            "exit": exit_rule(15, 8, 40, None, 5),
                        }, ov))

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
    print(f"[def-cap-200] before total={before_total} max_id={before_max} passed={before_pass} raw={len(raw)} ETF-excluded", flush=True)
    tickers, loader, kf, uf = _preload(raw)
    specs = [s for s in candidates() if spec_hash(s) not in existing]
    print(f"[def-cap-200] universe={len(tickers)} candidates_new={len(specs)} max={MAX_EVAL} engine={engine_version()} banned={sorted(BANNED)}", flush=True)
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
    print(f"[def-cap-200] done evaluated={evaluated} new_rows={after_total-before_total} id={before_max+1}..{after_max} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[def-cap-200] PASSED_TOP " + json.dumps(passed[:10], ensure_ascii=False)[:8000], flush=True)
    if best:
        print("[def-cap-200] BEST " + json.dumps(best[1], ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
