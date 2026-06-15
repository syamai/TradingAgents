"""Stage3 forward 관찰 — Stage2 통과 newloop 전략을 2025-06-30 이후로 추적.

채점이 아니라 관찰이다: ledger 예산을 소비하지 않고(2025-06-30 이후는 채점 금지·
forward 전용, CLAUDE.md), 다중검정 보정도 적용하지 않는다(n_trials=1 고정). 같은
홀드아웃 100종목에서 IS(≤SCORE_DATE_HI) vs forward(>SCORE_DATE_HI)를 동일 전략으로
비교해 '엣지가 시간상 지속됐는가'만 격리한다(종목 변화 교란 제거).

register(): Stage2 통과 spec + IS 메트릭을 동결 기록(최초 등록 유지 — 재등록 무시).
observe(): forward 창 메트릭을 재계산(데이터가 쌓이면 재실행). 등록 후 MIN_FORWARD_WEEKS
경과해야 판정 자격(eligible) — 그 전엔 표본 부족이라 승급/폐기 보류.

왜 score() 를 그대로 못 쓰나: stage2_gate.score() 는 w1 을 SCORE_DATE_HI 로 강제
클램프해 forward 구간을 못 본다(채점 금지의 핵심). 그래서 forward 메트릭은
portfolio_daily→score_portfolio 를 직접 호출하되 ledger·다중검정을 절대 건드리지 않는다.

사용:
    uv run python -m tradingagents.newloop.forward \\
        --register --family <family> --spec-file <stage2 통과 spec.json> \\
        --out artifacts/newloop/forward_<family>.json
    uv run python -m tradingagents.newloop.forward   # 등록분 관찰만
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.newloop.stage1_gate import DEFAULT_WINDOW_END, DEFAULT_WINDOW_START, SCORE_DATE_HI
from tradingagents.newloop.stage2_gate import (
    GATE_VERSION,
    _max_hold_days,
    portfolio_daily,
    score,
    score_portfolio,
)

REGISTRY = Path.home() / ".tradingagents/newloop/forward_registry.json"
FWD_START = "2025-07-01"      # SCORE_DATE_HI(2025-06-30) 다음날 — forward 관찰 시작
MIN_FORWARD_WEEKS = 4         # 판정 자격 최소 관찰 주수 (#6: 연속 관찰 + 최소 4주 후 판정)

_KEEP = ("alpha_ann_pct", "beta", "t_alpha", "n_invested_days", "deployment",
         "status", "gate_passed")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read() -> list[dict]:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []


def _write(reg: list[dict]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def _load_all(specs):
    """홀드아웃 100종목 + KOSPI(전구간) + (필요 시) usdkrw 로드."""
    from tradingagents.hermes import backtest_engine as bt
    from tradingagents.hermes.backtest_engine_v2 import fetch_usdkrw
    from tradingagents.newloop.holdout import holdout_universe

    tickers, loader, _basket = holdout_universe()
    kospi = bt.fetch_kospi("2018-01-01", "2027-01-01")   # forward 까지 덮도록 전구간
    needs_fx = any((sp.get("market_overlay") or {}).get("type")
                   == "usdkrw_trailing_return_ma_scale" for _r, sp in specs)
    usdkrw = fetch_usdkrw("2018-01-01", "2027-01-01") if needs_fx else None
    return tickers, loader, kospi, usdkrw


def _fwd_metrics(spec: dict, tickers, loader, kospi, usdkrw) -> dict:
    """forward 창(>SCORE_DATE_HI) 관찰 메트릭. 채점 아님 — n_trials=1, ledger 비접촉."""
    r_p, r_m, w, k = portfolio_daily(spec, tickers, loader, kospi, usdkrw, FWD_START, None)
    if r_p is None:
        return {"status": "no_forward_data", "n_invested_days": 0}
    out = score_portfolio(r_p, r_m, w, n_trials=1, max_hold_days=_max_hold_days(spec), k=k)
    return {kk: out.get(kk) for kk in _KEEP}


def register(family: str, specs, registered_at: str,
             tickers=None, loader=None, kospi=None, usdkrw=None) -> int:
    """specs = [(ref, spec), ...]. IS(≤SCORE_DATE_HI) 메트릭 동결 기록. 신규만 추가."""
    if tickers is None:
        tickers, loader, kospi, usdkrw = _load_all(specs)
    reg = _read()
    seen = {(e["family"], e["strategy_ref"]) for e in reg}
    added = 0
    for ref, spec in specs:
        if (family, ref) in seen:           # 동결 유지 — 재등록은 무시(원래 등록일 보존)
            continue
        is_r = score(spec, tickers, loader, kospi, usdkrw,
                     DEFAULT_WINDOW_START, DEFAULT_WINDOW_END, n_trials=1)
        reg.append({"family": family, "strategy_ref": ref, "spec": spec,
                    "freeze_date": SCORE_DATE_HI, "registered_at": registered_at,
                    "gate_version": GATE_VERSION,
                    "is_metrics": {kk: is_r.get(kk) for kk in _KEEP}})
        added += 1
    _write(reg)
    return added


def observe(now_iso: str | None = None,
            tickers=None, loader=None, kospi=None, usdkrw=None) -> list[dict]:
    """등록 전략의 forward 메트릭 + 경과 주수·판정 자격을 계산. ledger 비접촉."""
    reg = _read()
    if not reg:
        return []
    if tickers is None:
        specs = [(e["strategy_ref"], e["spec"]) for e in reg]
        tickers, loader, kospi, usdkrw = _load_all(specs)
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now(timezone.utc).astimezone()
    rows = []
    for e in reg:
        fwd = _fwd_metrics(e["spec"], tickers, loader, kospi, usdkrw)
        weeks = (now - datetime.fromisoformat(e["registered_at"])).days / 7.0
        rows.append({"family": e["family"], "strategy_ref": e["strategy_ref"],
                     "registered_at": e["registered_at"],
                     "weeks_elapsed": round(weeks, 1),
                     "eligible_for_decision": weeks >= MIN_FORWARD_WEEKS,
                     "is": e["is_metrics"], "fwd": fwd})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage3 forward 관찰 — newloop 전략")
    ap.add_argument("--register", action="store_true",
                    help="--family + --spec-file 의 전략을 forward 등록(Stage2 통과분)")
    ap.add_argument("--family", default=None)
    ap.add_argument("--spec-file", default=None, help="등록할 spec JSON(단일 또는 배열)")
    ap.add_argument("--out", default=None, help="관찰 결과 JSON 저장 경로")
    args = ap.parse_args()

    if args.register:
        if not (args.family and args.spec_file):
            ap.error("--register 엔 --family 와 --spec-file 필요")
        with open(args.spec_file, encoding="utf-8") as f:
            loaded = json.load(f)
        specs = [(sp.get("name", i), sp)
                 for i, sp in enumerate(loaded if isinstance(loaded, list) else [loaded])]
        n = register(args.family, specs, _now())
        print(f"forward 등록: 신규 {n} 전략 → {REGISTRY}")

    rows = observe()
    print(f"\nforward 관찰 ({len(rows)} 전략 | 시작 {FWD_START} | 판정 최소 {MIN_FORWARD_WEEKS}주):")
    for r in rows:
        i, fw = r["is"], r["fwd"]
        elig = "판정가능" if r["eligible_for_decision"] else f"관찰중({r['weeks_elapsed']}주)"
        print(f"[{r['family']}/{r['strategy_ref']}] {elig}")
        print(f"   IS (≤{SCORE_DATE_HI}): α연율%={i.get('alpha_ann_pct')} β={i.get('beta')} "
              f"t(α)={i.get('t_alpha')} 판정={i.get('status')}")
        print(f"   FWD(>{SCORE_DATE_HI}): α연율%={fw.get('alpha_ann_pct')} β={fw.get('beta')} "
              f"t(α)={fw.get('t_alpha')} 투입일={fw.get('n_invested_days')} 판정={fw.get('status')}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"created_at": _now(), "forward_start": FWD_START,
             "min_weeks": MIN_FORWARD_WEEKS, "observations": rows},
            ensure_ascii=False, indent=2))
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
