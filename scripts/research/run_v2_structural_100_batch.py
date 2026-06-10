#!/usr/bin/env python3
"""Run 100 structurally different v2 strategy tests.

User-approved direction (2026-06-06):
- A 40: event/state-transition style: flow spike/base/re-acceleration sequences.
- B 30: risk-gated defensive rotation: market regime/overlay first, then defensive institution rotation.
- C 30: rank/relative-strength proxy: flow-quality score proxies using dispersion/consensus/zscore/liquidity.

This intentionally avoids simple parameter-only variants such as "retail sell + investment_trust buy".
Rejected families remain banned: price_drop, price_return, realized_vol, short_ratio.
Every evaluated strategy is saved to strategies_v2.db and emitted as JSONL.
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

MAX_EVAL = int(os.environ.get("V2_STRUCTURAL_MAX", "100"))
OUT_JSONL = Path(os.environ.get("V2_STRUCTURAL_OUT", "artifacts/strategy_v2_structural_100_batch.jsonl"))
BANNED = {"price_drop", "price_return", "realized_vol", "short_ratio"}

ABBR = {
    "foreign_registered": "fr",
    "foreign_unregistered": "fu",
    "private_equity": "pe",
    "investment_trust": "it",
    "pension": "pn",
    "insurance": "ins",
    "bank": "bk",
}


def overlay_kospi(window: int, threshold: float, risk: float) -> dict:
    return {"type": "kospi_trailing_return_scale", "window": window, "op": "<=", "threshold_pct": threshold, "risk_stock_weight_pct": risk}


def overlay_fx(window: int, threshold: float, risk: float, ma: int = 120) -> dict:
    return {"type": "usdkrw_trailing_return_ma_scale", "window": window, "op": ">=", "threshold_pct": threshold, "ma_window": ma, "risk_stock_weight_pct": risk}


def exit_rule(tp: int, sl: int, hold: int, unwind: int | None = None) -> dict:
    e = {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold)}
    if unwind is not None:
        e["signal_all_of"] = [{"signal": "fast_money_unwind", "window": unwind}]
    return e


def base(name: str, family: str, entry: list[dict], exit_: dict, overlay: dict | None = None, note: str = "") -> dict:
    s = {"spec_version": 2, "name": name, "direction": "long", "entry": {"all_of": entry}, "exit": exit_, "research_family": family}
    if note:
        s["research_note"] = note
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


def family_counts(specs: Iterable[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in specs:
        out[s.get("research_family", "unknown")] = out.get(s.get("research_family", "unknown"), 0) + 1
    return out


def event_transition_specs() -> list[dict]:
    """A: flow spike/base/re-acceleration sequences, not single actor threshold swaps."""
    specs: list[dict] = []
    subjects = ["foreign_registered", "foreign_unregistered", "private_equity", "investment_trust", "pension", "insurance", "bank"]
    # 28: spike + quiet base + acceleration. Economic idea: prior liquidity absorption followed by fresh actor-specific re-accumulation.
    for subj in subjects:
        for z_win, lookback, min_z, rc_w, rc_max, short, long, exit_ in [
            (3, 60, 1.5, 20, 10.0, 3, 20, exit_rule(10, 5, 20, 5)),
            (5, 120, 1.5, 20, 15.0, 5, 20, exit_rule(15, 5, 20, 5)),
            (10, 120, 1.0, 60, 20.0, 10, 60, exit_rule(15, 8, 40, 10)),
            (20, 250, 1.0, 60, 15.0, 5, 60, exit_rule(20, 8, 40, 10)),
        ]:
            specs.append(base(
                f"structA-spike-base-reaccel-{ABBR[subj]}-z{z_win}_{lookback}_{min_z}-rc{rc_w}_{int(rc_max)}-a{short}_{long}",
                "A_event_transition",
                [
                    {"signal": "flow_zscore", "subject": subj, "window": z_win, "lookback": lookback, "min_z": min_z},
                    {"signal": "range_compression", "window": rc_w, "max_pct": rc_max},
                    {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                ],
                exit_,
                None,
                "수급 spike 후 변동성 수축(base)과 동일 주체 재가속이 동시에 발생할 때만 진입",
            ))
    # 12: consensus/close-quality transition. Economic idea: actor consensus + strong close avoids weak spikes.
    for group, min_buyers in [("foreign_pair", 2), ("fast_money", 2), ("institution_defensive", 2), ("broad_smart", 3)]:
        for win, min_pos, vol_s, vol_l, vol_ratio in [
            (3, 0.75, 3, 20, 1.5),
            (5, 0.75, 5, 20, 1.5),
            (10, 0.9, 5, 60, 2.0),
        ]:
            specs.append(base(
                f"structA-consensus-strongclose-vol-{group}-w{win}-cl{int(min_pos*100)}-v{vol_s}_{vol_l}_{vol_ratio}",
                "A_event_transition",
                [
                    {"signal": "flow_consensus", "group": group, "window": win, "min_buyers": min_buyers, "require_retail_sell": True},
                    {"signal": "close_location", "min_pos": min_pos},
                    {"signal": "volume_surge", "short": vol_s, "long": vol_l, "min_ratio": vol_ratio},
                ],
                exit_rule(10, 5, 20, 5),
                None,
                "단일 주체가 아니라 합의 수급 + 종가 품질 + 참여 증가를 sequence proxy로 결합",
            ))
    return specs[:40]


def risk_gated_defensive_specs() -> list[dict]:
    """B: market/FX risk gate first, then defensive institution rotation."""
    specs: list[dict] = []
    overlays = [
        ("ko80r20", overlay_kospi(80, 0.0, 20.0)),
        ("ko120r30", overlay_kospi(120, 0.0, 30.0)),
        ("ko40m5r30", overlay_kospi(40, -5.0, 30.0)),
        ("fx60r30", overlay_fx(60, 5.0, 30.0, 120)),
        ("fx120r45", overlay_fx(120, 5.0, 45.0, 200)),
    ]
    # 18: defensive consensus + market_filter + liquidity, with entry-level risk gates.
    # Note: spec_hash may ignore market_overlay metadata, so uniqueness is built into entry/exit, not just overlay.
    for cons_w in [3, 5, 10, 20]:
        for mf_w in [20, 60, 120]:
            for liq in [5_000_000_000.0, 10_000_000_000.0, 20_000_000_000.0]:
                otag, ov = overlays[(len(specs) + cons_w + mf_w) % len(overlays)]
                specs.append(base(
                    f"structB-riskgate-defcons-liq-mf{mf_w}-cw{cons_w}-l{int(liq/1e9)}-{otag}",
                    "B_risk_gated_defensive",
                    [
                        {"signal": "market_filter", "mode": "above_ma", "window": mf_w},
                        {"signal": "flow_consensus", "group": "institution_defensive", "window": cons_w, "min_buyers": 2, "require_retail_sell": True},
                        {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": liq},
                    ],
                    exit_rule(10, 5, 20, 5),
                    ov,
                    "시장 레짐과 overlay를 먼저 걸고 방어기관 합의가 있는 거래가능 종목만 진입",
                ))
                if len(specs) >= 18:
                    break
            if len(specs) >= 18:
                break
        if len(specs) >= 18:
            break
    # 12: defensive dispersion + specific defensive acceleration; different from simple buy/sell thresholds.
    start_n = len(specs)
    for subj in ["pension", "insurance", "investment_trust", "bank"]:
        for disp_w, share, short, long, mf_w in [
            (5, 0.5, 3, 20, 60),
            (10, 0.6, 5, 20, 120),
            (20, 0.7, 10, 60, 120),
        ]:
            otag, ov = overlays[(len(specs) + disp_w) % len(overlays)]
            specs.append(base(
                f"structB-riskgate-disp-accel-{ABBR[subj]}-dw{disp_w}-s{int(share*100)}-a{short}_{long}-mf{mf_w}-{otag}",
                "B_risk_gated_defensive",
                [
                    {"signal": "market_filter", "mode": "above_ma", "window": mf_w},
                    {"signal": "flow_dispersion", "group": "institution_defensive", "window": disp_w, "max_share": share},
                    {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                ],
                exit_rule(15, 8, 40, 10),
                ov,
                "방어기관군 내부의 분산 매집 + 특정 방어 주체 가속으로 쏠림형 수급을 배제",
            ))
            if len(specs) - start_n >= 12:
                break
        if len(specs) - start_n >= 12:
            break
    return specs[:30]


def rank_proxy_specs() -> list[dict]:
    """C: cross-sectional rank/relative-strength proxy via flow-quality filters available in current binary engine."""
    specs: list[dict] = []
    # Current engine has no explicit top-N ranking signal in spec, so this uses quality-score proxies:
    # unusual flow z-score + group dispersion/consensus + liquidity/flow-price concordance.
    for group, min_buyers, share in [("broad_smart", 3, 0.6), ("foreign_pair", 2, 0.5), ("institution_defensive", 2, 0.6), ("fast_money", 2, 0.5), ("broad_smart", 4, 0.7)]:
        for subj in ["foreign_registered", "investment_trust", "pension"]:
            specs.append(base(
                f"structC-rankproxy-zdisp-{group}-{ABBR[subj]}-mb{min_buyers}-s{int(share*100)}",
                "C_rank_proxy",
                [
                    {"signal": "flow_zscore", "subject": subj, "window": 5, "lookback": 120, "min_z": 1.5},
                    {"signal": "flow_dispersion", "group": group, "window": 10, "max_share": share},
                    {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 20_000_000_000.0},
                ],
                exit_rule(10, 5, 20, 5),
                overlay_kospi(80, 0.0, 20.0),
                "절대조건 대신 수급 이례성·분산도·유동성을 결합한 상위 품질 proxy",
            ))
    for group, min_buyers in [("broad_smart", 3), ("foreign_pair", 2), ("institution_defensive", 2)]:
        for corr_subj, lookback, r in [
            ("foreign_registered", 60, 0.2),
            ("investment_trust", 60, 0.2),
            ("pension", 120, 0.1),
            ("insurance", 120, 0.1),
            ("bank", 20, 0.3),
        ]:
            specs.append(base(
                f"structC-rankproxy-cons-corr-{group}-{ABBR[corr_subj]}-lb{lookback}-r{r}",
                "C_rank_proxy",
                [
                    {"signal": "flow_consensus", "group": group, "window": 10, "min_buyers": min_buyers, "require_retail_sell": False},
                    {"signal": "rolling_corr", "subject": corr_subj, "lookback": lookback, "min_r": r},
                    {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 10_000_000_000.0},
                ],
                exit_rule(15, 5, 20, 5),
                overlay_kospi(120, 0.0, 30.0),
                "수급 합의가 가격과 동조되는 종목만 남기는 상대품질 proxy",
            ))
    return specs[:30]


def candidates() -> list[dict]:
    raw = event_transition_specs() + risk_gated_defensive_specs() + rank_proxy_specs()
    if len(raw) != 100:
        raise AssertionError(f"candidate split broken: {family_counts(raw)} total={len(raw)}")
    seen, out = set(), []
    for s in raw:
        validate_spec_v2(s)
        st = sig_types(s)
        if st & BANNED:
            raise ValueError(f"banned signal in {s['name']}: {st & BANNED}")
        h = spec_hash(s)
        if h not in seen:
            seen.add(h)
            out.append(s)
    if len(out) != 100:
        raise AssertionError(f"dedup reduced candidates to {len(out)}; counts={family_counts(out)}")
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
    raw_universe = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in etf_codes]
    print(f"[structural-100] before total={before_total} max_id={before_max} passed={before_pass} raw={len(raw_universe)} ETF-excluded", flush=True)
    tickers, loader, kf, uf = _preload(raw_universe)
    all_specs = candidates()
    specs = [s for s in all_specs if spec_hash(s) not in existing]
    print(f"[structural-100] approved_split={family_counts(all_specs)} candidates_new={len(specs)} max={MAX_EVAL} engine={engine_version()} banned={sorted(BANNED)}", flush=True)
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
                "family": spec.get("research_family"),
                "note": spec.get("research_note"),
                "gate_passed": bool(wf["gate_passed"]),
                "wf_excess_ir_median": wf.get("oos_excess_ir_median"),
                "wf_excess_ir_min": wf.get("oos_excess_ir_min"),
                "wf_oos_sharpe_median": wf.get("oos_sharpe_median"),
                "xsec_in": xsec["in_sample"],
                "xsec_out": xsec["out_sample"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            key = (
                row["wf_excess_ir_min"] if row["wf_excess_ir_min"] is not None else -999,
                row["wf_excess_ir_median"] if row["wf_excess_ir_median"] is not None else -999,
                row["xsec_out"].get("sharpe") if row["xsec_out"].get("sharpe") is not None else -999,
            )
            if best is None or key > best[0]:
                best = (key, row)
            if row["gate_passed"]:
                passed.append(row)
            flag = "PASS" if row["gate_passed"] else "    "
            print(f"[{evaluated:03}/{MAX_EVAL}] {flag} #{sid} {row['family']} {spec['name'][:64]:64} wf_med={row['wf_excess_ir_median']} wf_min={row['wf_excess_ir_min']} | in {fmt(xsec['in_sample'])} out {fmt(xsec['out_sample'])}", flush=True)
    with store._conn() as c:
        after_total, after_pass, after_max = c.execute("SELECT count(*), coalesce(sum(gate_passed),0), coalesce(max(id),0) FROM strategies").fetchone()
    print(f"[structural-100] done evaluated={evaluated} new_rows={after_total-before_total} id={before_max+1}..{after_max} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[structural-100] PASSED_TOP " + json.dumps(passed[:10], ensure_ascii=False)[:8000], flush=True)
    if best:
        print("[structural-100] BEST " + json.dumps(best[1], ensure_ascii=False)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
