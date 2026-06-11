"""Stage 1 게이트 — 시장 도움 제거 채점 (새 루프 1차 관문).

배경 (2026-06-11 실측): 백테스트 게이트·차감-바스켓 알파는 모두 베타 틸트에
속는다 — 출렁임 큰 종목을 고르면 상승장에서 그 초과분이 실력처럼 보이고,
레짐이 꺾이면 무너진다(SELECTED 14: 회귀 절편 전원 0, 모멘텀 β 1.2~2.5,
하락창 차감알파 유의 음수).

채점: 거래 단위 회귀  net = α + β·basket
  - α(절편)  = 같은 보유기간 바스켓이 0일 때의 기대 초과수익
               = **시장 도움을 뺀 종목선택 실력. 합격 근거는 오직 이것.**
  - β(기울기) = 시장 민감도 배율 — 보고만, 게이트하지 않음.
  - 하락창   = 바스켓이 음(-)이던 거래들의 차감알파 — 베타 위장의 직접 증상.

게이트 금지 지표(과거 오판정 봉인): 승률, 완전투자 벤치마크 대비 총수익,
차감-바스켓 알파 단독.

사용 (홀드아웃 채점 + 원장 기록):
    uv run python -m tradingagents.newloop.stage1_gate \\
        --family momfo-ride --hypothesis "외인추세+장기보유 모멘텀" \\
        --ids 3009,3010,3011 --out artifacts/newloop/stage1_xxx.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone

from tradingagents.newloop import ledger

# ── 게이트 사양 (2026-06-11 확정·동결) ──────────────────────────────────
# 결과를 보고 임계를 바꾸는 것을 막기 위해 버전으로 박는다.
# 변경 시 GATE_VERSION 을 올리고 사유를 여기 남긴다.
GATE_VERSION = "s1-v1"
GATE_MIN_TRADES = 30     # 미만이면 운/실력 구분 불가 → insufficient(판정 보류)
GATE_MIN_T_ALPHA = 3.0   # α 의 t 하한 — 수백 family 반복 시행의 다중비교 보정
GATE_DOWN_T_FLOOR = -2.0  # 하락창 차감알파가 유의하게 음수면 베타 위장 → 탈락

DEFAULT_WINDOW_START = "2025-07-01"
STRAT_DB = os.path.expanduser("~/.tradingagents/hermes/strategies_v2.db")  # 읽기전용 입력


def ols_alpha_beta(xs: list[float], ys: list[float]):
    """단순회귀 y = a + b·x → (a, t_a, b, t_b_vs1). n<3 또는 x 무변동이면 None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    a = my - b * mx
    rss = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    s2 = rss / (n - 2)
    se_a = math.sqrt(s2 * (1 / n + mx * mx / sxx))
    se_b = math.sqrt(s2 / sxx)
    return a, (a / se_a if se_a > 0 else 0.0), b, ((b - 1.0) / se_b if se_b > 0 else 0.0)


def _mean_t(xs: list[float]):
    """(n, 평균, t) — 1표본 t. 표본 부족/무분산이면 t=None."""
    if not xs:
        return 0, None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return 1, m, None
    sd = statistics.pstdev(xs)
    return len(xs), m, (m / (sd / math.sqrt(len(xs))) if sd > 0 else None)


def score_pairs(pairs: list[tuple[float, float]]) -> dict:
    """(net%, basket%) 쌍 목록 → 기계 판정. 순수 함수(엔진 비의존, 테스트 대상).

    판정 = 3조건 AND:
      ① n ≥ GATE_MIN_TRADES                      (미만: insufficient)
      ② t(α) ≥ GATE_MIN_T_ALPHA                  (시장 도움 뺀 실력이 유의)
      ③ 하락창 차감알파 t > GATE_DOWN_T_FLOOR     (베타 위장 증상 없음)
    하락창 표본이 없으면(일방향 장) ③ 검사 불가 → 합격 아닌 insufficient.
    """
    nets = [p[0] for p in pairs]
    baskets = [p[1] for p in pairs]
    out = {
        "gate_version": GATE_VERSION,
        "n_trades": len(pairs),
        "alpha_pct": None, "t_alpha": None, "beta": None, "t_beta_vs1": None,
        "down": {"n": 0, "mean_pct": None, "t": None},
        "up": {"n": 0, "mean_pct": None, "t": None},
        "checks": {"enough_trades": None, "alpha_significant": None, "down_not_broken": None},
        "status": "insufficient", "gate_passed": False,
    }
    if len(pairs) < GATE_MIN_TRADES:
        out["checks"]["enough_trades"] = False
        return out
    out["checks"]["enough_trades"] = True

    reg = ols_alpha_beta(baskets, nets)
    if reg is None:
        return out
    a, t_a, b, t_b = reg
    out.update(alpha_pct=round(a, 4), t_alpha=round(t_a, 3),
               beta=round(b, 4), t_beta_vs1=round(t_b, 3))

    down = [n - x for n, x in pairs if x < 0]
    up = [n - x for n, x in pairs if x > 0]
    for key, xs in (("down", down), ("up", up)):
        n_, m_, t_ = _mean_t(xs)
        out[key] = {"n": n_, "mean_pct": round(m_, 4) if m_ is not None else None,
                    "t": round(t_, 3) if t_ is not None else None}

    alpha_ok = t_a >= GATE_MIN_T_ALPHA
    out["checks"]["alpha_significant"] = bool(alpha_ok)
    if out["down"]["t"] is None:
        # 위장 여부를 못 본 채 합격시키는 것이 가장 위험 — 보류.
        return out
    down_ok = out["down"]["t"] > GATE_DOWN_T_FLOOR
    out["checks"]["down_not_broken"] = bool(down_ok)
    out["gate_passed"] = bool(alpha_ok and down_ok)
    out["status"] = "pass" if out["gate_passed"] else "fail"
    return out


def collect_pairs(spec: dict, tickers: list[str], loader, basket_ret,
                  w0: str, w1: str | None) -> list[tuple[float, float]]:
    """홀드아웃에서 실현 거래의 (net%, 같은 구간 바스켓%) 쌍 수집. forced_eod 제외."""
    from tradingagents.hermes.backtest_engine_v2 import _simulate_v2

    pairs: list[tuple[float, float]] = []
    for tk in tickers:
        df, _ = loader(tk)
        if df is None or df.empty:
            continue
        try:
            trades, _daily, _active, _cdf = _simulate_v2(spec, df)
        except Exception:
            continue
        for t in trades:
            ed = t["entry_date"]
            if ed < w0 or (w1 is not None and ed > w1):
                continue
            if t.get("exit_reason") == "forced_eod":
                continue
            br = basket_ret(ed, t["exit_date"])
            if br is None:
                continue
            pairs.append((float(t["net_ret_pct"]), float(br)))
    return pairs


def score(spec: dict, tickers: list[str], loader, basket_ret,
          w0: str = DEFAULT_WINDOW_START, w1: str | None = None) -> dict:
    out = score_pairs(collect_pairs(spec, tickers, loader, basket_ret, w0, w1))
    out["window"] = [w0, w1]
    return out


def _load_spec_from_db(sid: int) -> dict:
    con = sqlite3.connect(f"file:{STRAT_DB}?mode=ro", uri=True)
    row = con.execute("SELECT spec_json FROM strategies WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise SystemExit(f"전략 id {sid} 없음: {STRAT_DB}")
    return json.loads(row[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage1 게이트 — 시장 도움 제거 채점")
    ap.add_argument("--family", required=True, help="가설 계열 키 (원장 사전등록 단위)")
    ap.add_argument("--hypothesis", default=None, help="가설 한 줄 (family 최초 등록 시 필수)")
    ap.add_argument("--ids", default=None, help="기존 strategies_v2.db 의 전략 id 콤마 목록")
    ap.add_argument("--spec-file", default=None, help="spec JSON 파일(단일 또는 배열)")
    ap.add_argument("--window", default=f"{DEFAULT_WINDOW_START}:",
                    help="진입일 창 'YYYY-MM-DD:YYYY-MM-DD' (끝 생략 가능)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    if not args.ids and not args.spec_file:
        ap.error("--ids 또는 --spec-file 필요")

    # 사전등록 + 재응시 한도 확인 (채점보다 먼저 — 시험지 마모 방지)
    if args.hypothesis:
        ledger.register_family(args.family, args.hypothesis)
    ok, why = ledger.can_submit(args.family)
    if not ok:
        print(f"제출 거부 [{args.family}]: {why}")
        return 2

    specs: list[tuple[str, dict]] = []
    if args.ids:
        for s in args.ids.split(","):
            sid = int(s)
            specs.append((f"db:{sid}", _load_spec_from_db(sid)))
    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as f:
            loaded = json.load(f)
        # 채점 전 spec 전수 검증 — 깨진 spec이 0거래→insufficient로 제출 슬롯을
        # 낭비하지 않도록, 하나라도 무효면 원장 기록 없이 중단한다.
        from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
        for i, sp in enumerate(loaded if isinstance(loaded, list) else [loaded]):
            try:
                validate_spec_v2(sp)
            except Exception as e:
                print(f"spec 무효(제출 미기록): #{i} {sp.get('name', '?')} — {e}")
                return 2
            specs.append((f"file:{sp.get('name', i)}", sp))

    w0, _, w1 = args.window.partition(":")
    w1 = w1 or None

    from tradingagents.newloop.holdout import holdout_universe
    tickers, loader, basket_ret = holdout_universe()

    print(f"Stage1 {GATE_VERSION} | family={args.family} | 홀드아웃 {len(tickers)}종목 | 창 {w0}~{w1 or '끝'}")
    print(f"{'ref':>10} {'n':>5} {'α%':>7} {'t(α)':>6} {'β':>6} {'하락창α%':>9} {'t':>6}  판정")
    results = []
    for ref, sp in specs:
        r = score(sp, tickers, loader, basket_ret, w0, w1)
        results.append({"strategy_ref": ref, "gate_version": GATE_VERSION,
                        "status": r["status"], "result": r})
        dn = r["down"]
        print(f"{ref:>10} {r['n_trades']:>5} "
              f"{r['alpha_pct'] if r['alpha_pct'] is not None else '-':>7} "
              f"{r['t_alpha'] if r['t_alpha'] is not None else '-':>6} "
              f"{r['beta'] if r['beta'] is not None else '-':>6} "
              f"{dn['mean_pct'] if dn['mean_pct'] is not None else '-':>9} "
              f"{dn['t'] if dn['t'] is not None else '-':>6}  {r['status']}")

    sub = ledger.record_submission(args.family,  results)
    st = ledger.family_state(args.family)
    print(f"\n원장 기록: 제출 #{sub}/{st['max_submissions']} → family 상태: {st['status']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "gate_version": GATE_VERSION,
                "thresholds": {"min_trades": GATE_MIN_TRADES, "min_t_alpha": GATE_MIN_T_ALPHA,
                               "down_t_floor": GATE_DOWN_T_FLOOR},
                "family": args.family, "submission": sub,
                "window": [w0, w1], "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
