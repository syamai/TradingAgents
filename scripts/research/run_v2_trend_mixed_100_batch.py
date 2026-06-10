#!/usr/bin/env python3
"""Run 100 mixed trend-oriented v2 strategies.

User-approved direction: D 혼합 100개
- A 35: 순수 가격추세 state (MA / breakout / close-location / volume) + flow confirmation.
- B 35: 수급추세 (flow_zscore / flow_accel / consensus / dispersion) + MA state.
- C 30: 레짐추세 (market_filter / overlay / liquidity) + individual trend quality.

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

MAX_EVAL = int(os.environ.get("V2_TREND_MIX_MAX", "100"))
OUT_JSONL = Path(os.environ.get("V2_TREND_MIX_OUT", "artifacts/strategy_v2_trend_mixed_100_batch.jsonl"))
SPECS_JSON = Path(os.environ.get("V2_TREND_MIX_SPECS", "research/strategy_candidates/trend_mixed_100_specs.json"))
BANNED = {"price_drop", "price_return", "realized_vol", "short_ratio"}
QUOTAS = {"A_price_trend": 35, "B_flow_trend": 35, "C_regime_trend": 30}

SUBJECTS = ["foreign_registered", "foreign_unregistered", "private_equity", "investment_trust", "pension", "insurance", "bank"]
ABBR = {
    "foreign_registered": "fr",
    "foreign_unregistered": "fu",
    "private_equity": "pe",
    "investment_trust": "it",
    "pension": "pn",
    "insurance": "ins",
    "bank": "bk",
}


def overlay_kospi(window: int = 80, threshold: float = 0.0, risk: float = 20.0) -> dict:
    return {"type": "kospi_trailing_return_scale", "window": window, "op": "<=", "threshold_pct": threshold, "risk_stock_weight_pct": risk}


def overlay_fx(window: int = 60, threshold: float = 5.0, risk: float = 30.0, ma: int = 120) -> dict:
    return {"type": "usdkrw_trailing_return_ma_scale", "window": window, "op": ">=", "threshold_pct": threshold, "ma_window": ma, "risk_stock_weight_pct": risk}


def exit_rule(tp: int, sl: int, hold: int, unwind: int | None = None) -> dict:
    e = {"take_profit_pct": float(tp), "stop_loss_pct": float(sl), "max_hold_days": int(hold)}
    if unwind is not None:
        e["signal_all_of"] = [{"signal": "fast_money_unwind", "window": unwind}]
    return e


def base(name: str, family: str, entry: list[dict], exit_: dict, overlay: dict | None = None, note: str = "") -> dict:
    s = {"spec_version": 2, "name": name, "direction": "long", "entry": {"all_of": entry}, "exit": exit_, "research_family": family}
    if overlay is not None:
        s["market_overlay"] = overlay
    if note:
        s["research_note"] = note
    return s


def sig_types(spec: dict) -> set[str]:
    out: set[str] = set()
    for side in ("entry", "exit"):
        obj = spec.get(side, {}) or {}
        for key in ("all_of", "signal_all_of"):
            for sig in obj.get(key, []) or []:
                if isinstance(sig, dict) and "signal" in sig:
                    out.add(sig["signal"])
    return out


def family_counts(specs: Iterable[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in specs:
        fam = s.get("research_family", "unknown")
        out[fam] = out.get(fam, 0) + 1
    return out


def price_trend_specs() -> list[dict]:
    """A: price trend state + participation/flow confirmation, no price_return."""
    specs: list[dict] = []
    overlays = [("none", None), ("ko80", overlay_kospi(80, 0.0, 20.0)), ("fx60", overlay_fx(60, 5.0, 30.0, 120))]
    # A1: MA uptrend + near/high breakout + smart money consensus.
    for ma in (20, 60):
        for bh_w, prox in ((20, 0.0), (20, 2.0), (60, 2.0), (60, 5.0), (120, 5.0)):
            for group, mb in (("foreign_pair", 2), ("fast_money", 2), ("broad_smart", 3), ("institution_defensive", 2)):
                tag, ov = overlays[(ma + bh_w + int(prox) + mb) % len(overlays)]
                specs.append(base(
                    f"trendmixA-ma{ma}-break{bh_w}_{int(prox)}-cons-{group}-{tag}",
                    "A_price_trend",
                    [
                        {"signal": "price_filter", "mode": "above_ma", "window": ma},
                        {"signal": "breakout_high", "window": bh_w, "proximity_pct": prox},
                        {"signal": "flow_consensus", "group": group, "window": 5, "min_buyers": mb, "require_retail_sell": True},
                    ],
                    exit_rule(10, 5, 20, 5),
                    ov,
                    "가격이 MA 위에서 전고점 접근/돌파하고 다수 스마트머니가 동의할 때만 진입",
                ))
    # A2: MA uptrend + strong close + volume expansion.
    for ma in (5, 20, 60):
        for min_pos in (0.75, 0.9):
            for vs_s, vs_l, ratio in ((3, 20, 1.5), (5, 20, 2.0), (5, 60, 1.5), (10, 60, 2.0)):
                specs.append(base(
                    f"trendmixA-ma{ma}-close{int(min_pos*100)}-vol{vs_s}_{vs_l}_{int(ratio*10)}",
                    "A_price_trend",
                    [
                        {"signal": "price_filter", "mode": "above_ma", "window": ma},
                        {"signal": "close_location", "min_pos": min_pos},
                        {"signal": "volume_surge", "short": vs_s, "long": vs_l, "min_ratio": ratio},
                    ],
                    exit_rule(8, 3, 10, 3),
                    overlay_kospi(80, 0.0, 20.0),
                    "상승 상태에서 종가 품질과 거래 참여가 동시에 개선되는 가격추세형",
                ))
    return specs


def flow_trend_specs() -> list[dict]:
    """B: flow trend/acceleration + stock MA state."""
    specs: list[dict] = []
    # B1: stock above MA + actor flow z-score + acceleration.
    for subj in SUBJECTS:
        for ma in (20, 60):
            for zwin, zlb, zmin, short, long in ((3, 60, 1.5, 3, 20), (5, 120, 1.5, 5, 20), (10, 120, 1.0, 10, 60)):
                specs.append(base(
                    f"trendmixB-ma{ma}-zaccel-{ABBR[subj]}-z{zwin}_{zlb}_{int(zmin*10)}-a{short}_{long}",
                    "B_flow_trend",
                    [
                        {"signal": "price_filter", "mode": "above_ma", "window": ma},
                        {"signal": "flow_zscore", "subject": subj, "window": zwin, "lookback": zlb, "min_z": zmin},
                        {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                    ],
                    exit_rule(15, 5, 20, 5),
                    None,
                    "가격 추세가 살아있는 종목에서 동일 주체 수급의 이례성과 가속을 동시에 요구",
                ))
    # B2: flow consensus/dispersion + MA state, avoiding single hot actor.
    for group, mb, share in (("foreign_pair", 2, 0.5), ("fast_money", 2, 0.6), ("institution_defensive", 2, 0.6), ("broad_smart", 3, 0.7)):
        for ma in (20, 60):
            for win in (5, 10, 20):
                specs.append(base(
                    f"trendmixB-ma{ma}-consdisp-{group}-w{win}-s{int(share*100)}",
                    "B_flow_trend",
                    [
                        {"signal": "price_filter", "mode": "above_ma", "window": ma},
                        {"signal": "flow_consensus", "group": group, "window": win, "min_buyers": mb, "require_retail_sell": True},
                        {"signal": "flow_dispersion", "group": group, "window": win, "max_share": share},
                    ],
                    exit_rule(10, 5, 20, 5),
                    overlay_fx(60, 5.0, 30.0, 120),
                    "다수 주체 합의와 분산 매집을 같이 요구해 단일 주체 쏠림을 배제",
                ))
    return specs


def regime_trend_specs() -> list[dict]:
    """C: market regime trend + individual trend quality/liquidity."""
    specs: list[dict] = []
    overlays = [("ko80r20", overlay_kospi(80, 0.0, 20.0)), ("ko120r30", overlay_kospi(120, 0.0, 30.0)), ("fx60r30", overlay_fx(60, 5.0, 30.0, 120))]
    # C1: KOSPI above MA + individual MA/breakout + liquidity.
    for mf in (20, 60, 120):
        for ma in (20, 60):
            for bh_w, prox in ((20, 2.0), (60, 5.0), (120, 5.0)):
                tag, ov = overlays[(mf + ma + bh_w) % len(overlays)]
                specs.append(base(
                    f"trendmixC-mkt{mf}-ma{ma}-break{bh_w}_{int(prox)}-liq-{tag}",
                    "C_regime_trend",
                    [
                        {"signal": "market_filter", "mode": "above_ma", "window": mf},
                        {"signal": "price_filter", "mode": "above_ma", "window": ma},
                        {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 10_000_000_000.0},
                    ],
                    exit_rule(10, 5, 20, 5),
                    ov,
                    "시장 레짐이 우호적일 때만 거래가능한 개별주 추세를 편입",
                ))
    # C2: market trend + defensive/broad flow confirmation + liquidity.
    for mf in (60, 120):
        for group, mb in (("institution_defensive", 2), ("broad_smart", 3), ("foreign_pair", 2)):
            for win, liq in ((5, 5_000_000_000.0), (10, 10_000_000_000.0), (20, 20_000_000_000.0)):
                tag, ov = overlays[(mf + win + mb) % len(overlays)]
                specs.append(base(
                    f"trendmixC-mkt{mf}-flow-{group}-w{win}-l{int(liq/1e9)}-{tag}",
                    "C_regime_trend",
                    [
                        {"signal": "market_filter", "mode": "above_ma", "window": mf},
                        {"signal": "flow_consensus", "group": group, "window": win, "min_buyers": mb, "require_retail_sell": True},
                        {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": liq},
                    ],
                    exit_rule(15, 8, 40, 10),
                    ov,
                    "시장 추세와 수급 추세가 동시에 맞는 거래가능 종목만 편입",
                ))
    # C3: market trend + individual price close quality + participation expansion.
    for mf in (20, 60, 120):
        for min_pos in (0.75, 0.9):
            for vs_s, vs_l, ratio in ((3, 20, 1.5), (5, 60, 1.5), (10, 60, 2.0)):
                tag, ov = overlays[(mf + int(min_pos * 100) + vs_s) % len(overlays)]
                specs.append(base(
                    f"trendmixC-mkt{mf}-close{int(min_pos*100)}-vol{vs_s}_{vs_l}_{int(ratio*10)}-{tag}",
                    "C_regime_trend",
                    [
                        {"signal": "market_filter", "mode": "above_ma", "window": mf},
                        {"signal": "close_location", "min_pos": min_pos},
                        {"signal": "volume_surge", "short": vs_s, "long": vs_l, "min_ratio": ratio},
                    ],
                    exit_rule(8, 5, 10, 3),
                    ov,
                    "시장 추세가 맞을 때 종가 품질과 거래 참여가 동시에 개선되는 개별주만 진입",
                ))
    return specs


def all_candidates() -> list[dict]:
    raw = price_trend_specs() + flow_trend_specs() + regime_trend_specs()
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
    return out


def select_new_by_quota(candidates: list[dict], existing_hashes: set[str]) -> list[dict]:
    selected: list[dict] = []
    for fam, quota in QUOTAS.items():
        fam_specs = [s for s in candidates if s.get("research_family") == fam and spec_hash(s) not in existing_hashes]
        if len(fam_specs) < quota:
            raise RuntimeError(f"not enough new specs for {fam}: have {len(fam_specs)} need {quota}")
        selected.extend(fam_specs[:quota])
    return selected


def fmt(m: dict) -> str:
    def r(x):
        return "NA" if x is None else f"{x:.4g}" if isinstance(x, float) else str(x)
    return f"sh={r(m.get('sharpe'))} ex={r(m.get('excess_sharpe'))} mdd={r(m.get('mdd_pct'))} n={m.get('n_trades')} cum={r(m.get('cum_return_pct'))}"


def main() -> int:
    store = StrategyStoreV2()
    with store._conn() as c:
        before_total, before_pass, before_max = c.execute("SELECT count(*), coalesce(sum(gate_passed),0), coalesce(max(id),0) FROM strategies").fetchone()
        existing = {r[0] for r in c.execute("SELECT spec_hash FROM strategies")}
    candidates = all_candidates()
    selected = select_new_by_quota(candidates, existing)[:MAX_EVAL]
    SPECS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPECS_JSON.write_text(json.dumps({"quotas": QUOTAS, "specs": selected}, ensure_ascii=False, indent=2), encoding="utf-8")

    etf_codes = set(str(x)[:6] for x in ETF_CODES)
    raw_universe = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in etf_codes]
    print(f"[trendmix-100] before total={before_total} max_id={before_max} passed={before_pass} raw={len(raw_universe)} ETF-excluded", flush=True)
    print(f"[trendmix-100] candidates={len(candidates)} selected={len(selected)} quotas={family_counts(selected)} specs={SPECS_JSON}", flush=True)
    tickers, loader, kf, uf = _preload(raw_universe)
    print(f"[trendmix-100] universe={len(tickers)} max={MAX_EVAL} engine={engine_version()} banned={sorted(BANNED)}", flush=True)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    evaluated = 0
    passed: list[dict] = []
    best = None
    with OUT_JSONL.open("a", encoding="utf-8") as out:
        for spec in selected:
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
                "research_family": spec.get("research_family"),
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
    print(f"[trendmix-100] done evaluated={evaluated} new_rows={after_total-before_total} id={before_max+1}..{after_max} passed_delta={after_pass-before_pass} out={OUT_JSONL}", flush=True)
    if passed:
        print("[trendmix-100] PASSED_TOP " + json.dumps(passed[:10], ensure_ascii=False)[:12000], flush=True)
    if best:
        print("[trendmix-100] BEST " + json.dumps(best[1], ensure_ascii=False)[:6000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
