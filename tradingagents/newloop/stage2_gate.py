"""Stage 2 게이트 — 포트폴리오 시장중립 평가 (실제 배포 구성으로 굴린 자본곡선).

Stage1(거래 단위 픽 실력) 통과 전략이, 실제 배포 구성(목표 30종목·종목당 5%
상한·주식 90% 노출)으로 묶였을 때도 시장 도움을 벗기고 돈을 버는가.

왜 별도 단계인가 (2026-06-12, 사용자 지적):
  - 실제 거래는 단일 종목이 아니라 포트폴리오다 — 거래 단위 알파가 양수여도
    (a) 자리 30개 제약, (b) 동시보유 상관, (c) 시장이 올려준 몫 때문에
    포트폴리오가 돈 번다는 보장이 없다.

채점: **투입된 자본 기준** 포트폴리오 일수익을 시장에 회귀  r_p/w = α + β·r_m
  - 투입일(invested_weight>0)만 사용 — 현금일(수익 0)을 섞으면 (i) 시장 상승
    캡처가 절편으로 새 방향성 junk 가 가짜 통과하고, (ii) 진짜 실력이 현금에
    희석돼 가짜 탈락하며, (iii) 하락일 위장검사가 무력화된다(적대 감사 실증).
  - α(절편, 연율화) = 추정 베타를 벗긴 투입자본의 순수 실력. 합격 근거.
  - β = 실제 시장 민감도(가정 1 아님). 표준오차 = Newey-West(HAC), lag 는
    중첩보유 horizon(보유기간)에 맞춰 — 작으면 자기상관 미보정으로 t 폭증.
  - 합격선 t_crit = 소표본 HAC-t 의 두꺼운 꼬리를 wild block bootstrap 으로
    데이터에서 보정(정규분위는 재감사 실증 FWER 0.96 → 부트스트랩으로 봉쇄).

배포 충실도: 포트폴리오 결합·KOSPI(전구간)를 production 경로와 동일하게 재현.
market_overlay(디리스킹)는 포지션 사이징이라 y=r_p/w 비율 채점에서 상쇄됨 —
완전 risk-off(mult==0) 날만 회귀 제외(의도된 scale-invariance, 알파는 종목선택
실력이지 사이징이 아님).

미분산 경고: 새 종목에선 동시보유 종목수 k 가 적어(실측 #2884 투입일 53%가
단일종목) 채점이 단일종목 꼬리에 지배될 수 있다 → concentration 진단을 출력
(판정 게이트 아님, 해석용).

합격 기준(S2_*)은 사용자 확정(2026-06-15, s2-v2): 비용 인지 α 하한 3%/년.
채점 창은 2025-06-30 급등 컷오프 상한 — 그 이후는 forward 관찰 전용(newloop/forward.py).
ledger 기록·예산 공유: Stage2 채점도 '<family>-s2' 키로 gate_results 에 append 돼
홀드아웃 누적 N(HOLDOUT_TRIAL_BUDGET)을 Stage1 과 공유한다(다중검정 마모 반영).

사용:
    uv run python -m tradingagents.newloop.stage2_gate \\
        --ids 2884,2939 --out artifacts/newloop/stage2_xxx.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tradingagents.newloop import ledger
from tradingagents.newloop.stage1_gate import (
    DEFAULT_WINDOW_END,
    DEFAULT_WINDOW_START,
    SCORE_DATE_HI,
    STRAT_DB,
    _mean_t,
    deflated_t_threshold,
    wild_bootstrap_t_crit,
)

TRADING_DAYS = 252
GATE_VERSION = "s2-v2"   # s2-v1→v2: 합격 임계 확정(비용 인지 α 하한) + ledger 예산 공유 배선

# ── 합격 기준 (사용자 확정 2026-06-15 — s2-v2) ────────────────────────────────
S2_MIN_INVESTED_DAYS = 40   # 투입일(현금 제외) 최소 — 미만이면 판정 보류
S2_MIN_DOWN_DAYS = 15      # 투입 하락일 최소 — 베타 위장검사 표본
S2_ALPHA_ANN_MIN = 3.0    # 연율 시장중립 알파 하한(%). 비용 인지: 20일 보유·연 ~12 왕복 ×
# 왕복비용 ~0.3~0.5%(증권거래세 0.18%+수수료+슬리피지) ≈ 3~6% 드래그. 3% 는 보수적 하한 —
# t 유의성 AND 이 하한 = "진짜 AND 비용 후에도 경제적". 비용모델 기반이라 홀드아웃 적합이
# 아님(자기참조 시험지 마모 회피); forward 누적이 추가 검증.
S2_BETA_MAX = 1.6         # 시장 민감도 상한 — 베타 위장 차단
S2_DOWN_EXCESS_T_FLOOR = -2.0  # 투입 하락일 시장중립 초과수익 t 하한
S2_HAC_MIN_LAG = 10       # HAC 최소 lag (보유기간 모를 때 보수적 하한)


def _max_hold_days(spec: dict) -> int:
    try:
        return int(spec.get("exit", {}).get("max_hold_days") or 0)
    except (TypeError, ValueError):
        return 0


def hac_ols(y, x, maxlags: int):
    """y = a + b·x, Newey-West(HAC) 표준오차로 회귀. (a, t_a, b, t_b, n) 반환.

    HAC 커널 직접 구현의 미묘한 오류를 피하려 statsmodels 사용. 무분산·표본부족
    이면 None.
    """
    import statsmodels.api as sm

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n < 3 or float(np.std(x)) == 0.0:
        return None
    res = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True}
    )
    a, b = res.params
    t_a, t_b = res.tvalues
    return float(a), float(t_a), float(b), float(t_b), n


def score_portfolio(r_p, r_m, w, n_trials: int = 1, max_hold_days: int = 0, k=None) -> dict:
    """투입자본 기준 시장중립 채점 (순수 함수).

    r_p 포트폴리오 일수익, r_m 시장 일수익, w 일별 투입비중(현금일=0),
    k 일별 동시보유 종목수(미분산 진단용, 옵션).
    투입일(w>0)만 골라 자본효율 수익 y=r_p/w 를 r_m 에 회귀한다.
    n_trials = 누적 채점 수(다중검정 보정). 합격 판정은 PROVISIONAL.
    """
    r_p = np.asarray(r_p, dtype=float)
    r_m = np.asarray(r_m, dtype=float)
    w = np.asarray(w, dtype=float)
    n_total = len(r_p)
    inv = w > 0
    n_inv = int(inv.sum())
    t_crit_norm = deflated_t_threshold(n_trials)
    t_crit = t_crit_norm  # 회귀 성공 후 부트스트랩 임계로 교체
    out = {
        "stage": GATE_VERSION, "n_days": n_total, "n_invested_days": n_inv,
        "deployment": round(float(w.mean()), 4) if n_total else None,
        "alpha_ann_pct": None, "alpha_daily": None, "beta": None,
        "t_alpha": None, "t_beta": None, "maxlags": None,
        "concentration": None,
        "down": {"n": 0, "mean_excess_pct": None, "t": None},
        "multiplicity": {"n_trials": n_trials, "t_crit": round(t_crit, 3),
                         "t_crit_normal": round(t_crit_norm, 3), "t_crit_method": "normal"},
        "checks": {"enough_days": None, "alpha_significant": None,
                   "alpha_material": None, "beta_controlled": None,
                   "enough_down": None, "down_not_broken": None},
        "thresholds_provisional": False,
        "status": "insufficient", "gate_passed": False,
    }
    if n_inv < S2_MIN_INVESTED_DAYS:
        out["checks"]["enough_days"] = False
        return out
    out["checks"]["enough_days"] = True

    y = r_p[inv] / w[inv]   # 투입자본 1단위당 일수익 (현금 희석 제거)
    x = r_m[inv]
    # HAC lag: 중첩보유 자기상관 horizon ≈ 보유기간. 작으면 t 폭증.
    maxlags = max(S2_HAC_MIN_LAG, int(math.ceil(1.5 * max_hold_days)))
    maxlags = min(maxlags, max(1, n_inv // 4))  # 표본 대비 안정
    reg = hac_ols(y, x, maxlags)
    if reg is None:
        return out
    a, t_a, b, t_b, _ = reg
    alpha_ann = a * TRADING_DAYS * 100.0
    out.update(alpha_daily=round(a, 6), alpha_ann_pct=round(alpha_ann, 3),
               beta=round(b, 3), t_alpha=round(t_a, 3), t_beta=round(t_b, 3),
               maxlags=maxlags)

    # 미분산 진단: 투입일의 동시보유 종목수 분포 + 최대 |y| 1일 제거 시 알파 변화
    # (단일종목 꼬리 지배·아웃라이어 레버리지 가시화 — 판정 게이트 아님)
    if k is not None:
        ki = np.asarray(k, dtype=float)[inv]
        drop1_ann = None
        if n_inv > 2:
            j = int(np.argmax(np.abs(y)))
            keep = np.ones(n_inv, dtype=bool)
            keep[j] = False
            r2 = hac_ols(y[keep], x[keep], maxlags)
            if r2 is not None:
                drop1_ann = round(r2[0] * TRADING_DAYS * 100.0, 3)
        out["concentration"] = {
            "frac_solo": round(float((ki == 1).mean()), 3),
            "median_k": float(np.median(ki)), "max_k": int(ki.max()),
            "drop1_alpha_ann_pct": drop1_ann,
            "undiversified": bool((ki == 1).mean() > 0.5),
        }

    # 소표본 HAC-t 두꺼운 꼬리를 이동블록 부트스트랩으로 보정 (정규 폴백).
    # 블록 길이는 자기상관 horizon 을 담되 블록 수가 충분하도록(≥8) 캡 — 너무 길면
    # 귀무 꼬리를 과소추정(실측: L=maxlags 면 q99 과소). 짧으면 보수적(안전).
    block_len = max(2, min(maxlags, n_inv // 8))

    def _t_hac(xx, yy, _g):
        r = hac_ols(yy, xx, maxlags)
        return r[1] if r is not None else None
    t_boot = wild_bootstrap_t_crit(list(x), list(y), None, _t_hac, n_trials,
                                   method="mbb", block_len=block_len)
    if t_boot is not None:
        t_crit = t_boot
        out["multiplicity"]["t_crit"] = round(t_crit, 3)
        out["multiplicity"]["t_crit_method"] = "mbb_bootstrap"
        out["multiplicity"]["block_len"] = block_len

    # 투입 하락일 시장중립 초과수익 = y − β·x (현금일 제외 → 위장검사 유효)
    excess = y - b * x
    down = list(excess[x < 0])
    dn_n, dn_m, dn_t = _mean_t(down)
    out["down"] = {"n": dn_n, "mean_excess_pct": round(dn_m * 100, 4) if dn_m is not None else None,
                   "t": round(dn_t, 3) if dn_t is not None else None}

    out["checks"]["alpha_significant"] = bool(t_a >= t_crit)
    out["checks"]["alpha_material"] = bool(alpha_ann >= S2_ALPHA_ANN_MIN)
    out["checks"]["beta_controlled"] = bool(abs(b) <= S2_BETA_MAX)
    if dn_n < S2_MIN_DOWN_DAYS or dn_t is None:
        out["checks"]["enough_down"] = False
        return out
    out["checks"]["enough_down"] = True
    out["checks"]["down_not_broken"] = bool(dn_t > S2_DOWN_EXCESS_T_FLOOR)

    out["gate_passed"] = bool(
        out["checks"]["alpha_significant"] and out["checks"]["alpha_material"]
        and out["checks"]["beta_controlled"] and out["checks"]["down_not_broken"]
    )
    out["status"] = "pass" if out["gate_passed"] else "fail"
    return out


def portfolio_daily(spec: dict, tickers: list[str], loader, kospi, usdkrw, w0: str, w1: str | None):
    """배포 구성(30/5%/90%)으로 묶은 포트폴리오 일수익 r_p, 시장 일수익 r_m,
    투입비중 w, 동시보유 종목수 k 를 (창 내, 정렬·결손제거) 반환. (None×4) 가능.

    production(run_universe_backtest_v2)과 동일하게 _combine 으로 결합. overlay 는
    y=r_p/w 비율에서 상쇄되므로 곱하지 않고(죽은 연산 제거), 완전 risk-off
    (mult==0) 날만 투입비중 0 으로 회귀에서 제외한다.
    """
    from tradingagents.hermes import backtest_engine as bt
    from tradingagents.hermes.backtest_engine_v2 import _market_overlay_multiplier, _simulate_v2

    ret_frames: list[pd.Series] = []
    act_frames: list[pd.Series] = []
    for tk in tickers:
        df, _ = loader(tk)
        if df is None or df.empty:
            continue
        try:
            _trades, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
        except Exception:
            continue
        idx = cdf["date"].astype(str).to_numpy()
        ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
        act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
    if not ret_frames:
        return None, None, None, None

    port = bt._combine_korea_stock_portfolio(ret_frames, act_frames).sort_index()
    invested = bt._invested_weight_korea(act_frames).reindex(port.index).fillna(0.0)
    kcount = (pd.concat(act_frames, axis=1).sort_index() == True).sum(axis=1)  # noqa: E712
    kcount = kcount.reindex(port.index).fillna(0)
    if spec.get("market_overlay"):
        # overlay 는 사이징 — 비율 채점에서 상쇄. 완전 risk-off(mult==0) 날만 제외.
        mult = _market_overlay_multiplier(port.index, kospi, spec.get("market_overlay"), usdkrw=usdkrw)
        invested = invested.where(mult > 0, 0.0)
    market = bt._market_daily_returns(kospi, port.index)

    df = pd.DataFrame({"p": port, "m": market, "w": invested, "k": kcount})
    df = df[df.index >= w0]
    if w1 is not None:
        df = df[df.index <= w1]
    df = df.dropna()
    if df.empty:
        return None, None, None, None
    return df["p"].to_numpy(), df["m"].to_numpy(), df["w"].to_numpy(), df["k"].to_numpy()


def score(spec: dict, tickers: list[str], loader, kospi, usdkrw,
          w0: str = DEFAULT_WINDOW_START, w1: str | None = DEFAULT_WINDOW_END, n_trials: int = 1) -> dict:
    if not w1 or w1 > SCORE_DATE_HI:          # 급등 컷오프 강제 상한 — 금지 구간 채점 방지
        w1 = SCORE_DATE_HI
    r_p, r_m, w, k = portfolio_daily(spec, tickers, loader, kospi, usdkrw, w0, w1)
    if r_p is None:
        return {"stage": GATE_VERSION, "status": "insufficient", "n_days": 0,
                "window": [w0, w1], "checks": {"enough_days": False}}
    out = score_portfolio(r_p, r_m, w, n_trials, max_hold_days=_max_hold_days(spec), k=k)
    out["window"] = [w0, w1]
    return out


def _load_spec_from_db(sid: int) -> dict:
    con = sqlite3.connect(f"file:{STRAT_DB}?mode=ro", uri=True)
    row = con.execute("SELECT spec_json FROM strategies WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise SystemExit(f"전략 id {sid} 없음: {STRAT_DB}")
    return json.loads(row[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage2 게이트 — 포트폴리오 시장중립 평가")
    ap.add_argument("--family", required=True,
                    help="가설 계열 키 — 원장엔 '<family>-s2' 로 기록(Stage1 과 예산 공유)")
    ap.add_argument("--hypothesis", default=None, help="가설 한 줄 (s2 family 최초 등록 시)")
    ap.add_argument("--ids", default=None, help="strategies_v2.db 전략 id 콤마 목록")
    ap.add_argument("--spec-file", default=None, help="spec JSON 파일(단일 또는 배열)")
    ap.add_argument("--window", default=f"{DEFAULT_WINDOW_START}:{DEFAULT_WINDOW_END}",
                    help="날짜 창 'YYYY-MM-DD:YYYY-MM-DD' (끝 생략 시 2025-06-30 컷오프)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()
    if not args.ids and not args.spec_file:
        ap.error("--ids 또는 --spec-file 필요")

    # Stage2 도 같은 홀드아웃 시험지를 소모하므로 예산을 공유한다(#4). 원장 키는
    # '<family>-s2' — Stage1 통과로 잠긴 family(status='passed')와 충돌하지 않게 분리하고,
    # 그 자체로 재응시 한도(MAX_SUBMISSIONS)를 독립으로 갖는다. 사전등록·예산 확인은
    # 채점보다 먼저(시험지 마모 방지).
    s2_family = f"{args.family}-s2"
    ledger.register_family(s2_family, args.hypothesis or f"[S2] {args.family} 포트폴리오 시장중립 검증")
    ok, why = ledger.can_submit(s2_family)
    if not ok:
        print(f"제출 거부 [{s2_family}]: {why}")
        return 2

    specs: list[tuple[str, dict]] = []
    if args.ids:
        for s in args.ids.split(","):
            sid = int(s)
            specs.append((f"db:{sid}", _load_spec_from_db(sid)))
    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as f:
            loaded = json.load(f)
        # 깨진 spec 이 0거래→insufficient 로 제출 슬롯을 낭비하지 않도록 전수 검증.
        from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
        for i, sp in enumerate(loaded if isinstance(loaded, list) else [loaded]):
            try:
                validate_spec_v2(sp)
            except Exception as e:
                print(f"spec 무효(제출 미기록): #{i} {sp.get('name', '?')} — {e}")
                return 2
            specs.append((f"file:{sp.get('name', i)}", sp))

    # 다중검정 N = 홀드아웃 누적 채점(Stage1+Stage2 공유) + 이번 배치. 예산 초과는 채점 전 차단.
    cum_before = ledger.cumulative_trials()
    n_eff = cum_before + len(specs)
    budget = ledger.HOLDOUT_TRIAL_BUDGET
    if n_eff > budget:
        print(f"제출 거부 [{s2_family}]: 홀드아웃 예산 초과 "
              f"(누적 {cum_before}+{len(specs)} > {budget}) — 새 홀드아웃 수집 필요")
        return 2

    w0, _, w1 = args.window.partition(":")
    w1 = w1 or None

    from tradingagents.hermes import backtest_engine as bt
    from tradingagents.hermes.backtest_engine_v2 import fetch_usdkrw
    from tradingagents.newloop.holdout import holdout_universe
    tickers, loader, _basket = holdout_universe()
    # production 과 동일하게 KOSPI 를 전구간으로 — 채점창부터 받으면 시뮬의
    # market_filter 가 창 이전 봉에서 NaN→False 로 평가돼 거래집합이 달라진다.
    kospi = bt.fetch_kospi("2018-01-01", w1 or "2027-01-01")
    needs_fx = any(sp.get("market_overlay", {}).get("type") == "usdkrw_trailing_return_ma_scale"
                   for _r, sp in specs)
    usdkrw = fetch_usdkrw("2018-01-01", w1 or "2027-01-01") if needs_fx else None

    print(f"Stage2 {GATE_VERSION} | family={s2_family} | 홀드아웃 {len(tickers)}종목 | "
          f"창 {w0}~{w1 or '끝'}")
    print(f"  다중검정 N={n_eff} (예산 {budget}, 잔여 {budget - n_eff})")
    print(f"{'ref':>10} {'투입일':>5} {'배치%':>6} {'α연율%':>8} {'β':>6} {'t(α)':>7} "
          f"{'t임계':>6} {'단독%':>5} {'하락t':>6}  판정")
    results = []
    for ref, sp in specs:
        r = score(sp, tickers, loader, kospi, usdkrw, w0, w1, n_trials=n_eff)
        results.append({"strategy_ref": ref, "gate_version": GATE_VERSION,
                        "status": r["status"], "result": r})
        dn = r.get("down", {})
        dep = r.get("deployment")
        conc = r.get("concentration") or {}
        tc = r.get("multiplicity", {}).get("t_crit")
        print(f"{ref:>10} {r.get('n_invested_days', 0):>5} "
              f"{round(dep * 100, 1) if dep is not None else '-':>6} "
              f"{r.get('alpha_ann_pct') if r.get('alpha_ann_pct') is not None else '-':>8} "
              f"{r.get('beta') if r.get('beta') is not None else '-':>6} "
              f"{r.get('t_alpha') if r.get('t_alpha') is not None else '-':>7} "
              f"{tc if tc is not None else '-':>6} "
              f"{round(conc.get('frac_solo') * 100) if conc.get('frac_solo') is not None else '-':>5} "
              f"{dn.get('t') if dn.get('t') is not None else '-':>6}  {r['status']}")

    sub = ledger.record_submission(s2_family, results)
    st = ledger.family_state(s2_family)
    print(f"\n원장 기록: 제출 #{sub}/{st['max_submissions']} → family 상태: {st['status']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "stage": GATE_VERSION, "thresholds_provisional": False,
                "thresholds": {"min_invested_days": S2_MIN_INVESTED_DAYS,
                               "min_down_days": S2_MIN_DOWN_DAYS, "alpha_ann_min": S2_ALPHA_ANN_MIN,
                               "beta_max": S2_BETA_MAX, "down_excess_t_floor": S2_DOWN_EXCESS_T_FLOOR},
                "multiplicity": {"n_trials": n_eff, "cumulative_before": cum_before,
                                 "budget": budget, "budget_remaining": budget - n_eff},
                "family": s2_family, "submission": sub,
                "window": [w0, w1], "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
