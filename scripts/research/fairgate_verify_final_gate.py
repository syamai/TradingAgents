"""최종 게이트(fair gate v3) 통계 유효성 검증 — (a) 모수 null, (b) DSR, (c) PBO.

대상 게이트(2026-06-11 최종):
  wf(walk-forward, 생존풀 벤치마크): 모든 OOS 창 초과 IR>0 AND 중앙값>GATE_POOL_MIN_IR(0.3),
  n_windows=5(유효 창 5/5 필수) — DB ``wf_gate_passed``.
  최종 ``gate_passed`` = wf AND xsec 품질(in/out sharpe>1.0, MDD>=-20%, 거래>=50, 승률 제외).

구 검증 스크립트(fairgate_verify_a_multitest / _c_permutation_null)는 구 게이트
(KOSPI 벤치·중앙>0.5·n=6, 1005행 스냅샷) 대상이라 무효 — 본 스크립트가 대체한다.

설계:
(a) 모수 null + 창간 상관 시나리오
    통과 전략의 wf_result_json 이 compact 요약(median/min)뿐이라 창별 IR 원본이 없다.
    재채점 순열(전략당 풀 wf ~12s × 수천)은 30분 예산 초과 → DB 가용 데이터 기반 모수 null:
    - 창내 잡음 σ_w: 전 전략 (median-min) gap 분포로 추정. gap 은 전략 평균(skill)에
      위치 불변이라 skill 유무와 무관하게 σ_w 만 식별한다. 보정상수는 MC 로 산출.
    - 창간 상관 ρ(전략 지속효과 비중): 창별 원본이 보존된 trendmixA 계열(동일 pool
      벤치·gate_min_ir=0.3 설정)에서 between/within 분해로 경험치 추정.
    - 시나리오 {ρ=0(독립), ρ=경험치, ρ=0.5(보수)} 별 p0 = P(5창 min>0 AND median>0.3)
      MC → 기대 위양성 N·p0, P(통과수>=관측) (포아송 + family·창효과 패널 시뮬).
(b) Deflated Sharpe Ratio (Bailey & López de Prado)
    통과 전략 out_sharpe(xsec OOS, 연율) 에 대해 trials N=3106 전수 / N=614(family
    시나리오, 과제 지정) 병기. 표본 길이는 out_n_trades 로 보정(per-trade 환산,
    iid 근사) — 참고로 일별 단위(T=거래일수) 변형 병기. 정규 가정(skew=0, kurt=3).
(c) PBO(CSCV, Bailey et al. 2017)
    전략별 일별 수익률 시계열은 DB/아티팩트에 없음(상위 5개 wealth CSV 만 존재).
    풀 유니버스 재시뮬이 전략당 ~2.4s 로 싸서 wf 통과 16개 + 상위 후보로 일별
    초과수익(생존풀 벤치) 행렬을 직접 생성 후 S=16 CSCV 를 수행한다.

주의: DB 는 읽기 전용(mode=ro)으로만 연다. 라이브 루프가 DB 를 갱신 중일 수 있어
시작 시점 1회 스냅샷 읽기로 고정한다.
실행: uv run python scripts/research/fairgate_verify_final_gate.py
산출: artifacts/research/fairgate_final_gate_verify_{YYYYMMDD}.json + stdout 보고.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import (
    _market_overlay_multiplier,
    _simulate_v2,
)
from tradingagents.hermes.strategy_validation import (
    GATE_POOL_MIN_IR,
    SCORE_DATE_HI,
    _survivor_pool_index,
    engine_version,
    expected_max_sharpe,
    max_justifiable_trials,
)

DB = os.path.expanduser("~/.tradingagents/hermes/strategies_v2.db")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "research")
TODAY = dt.date.today().strftime("%Y%m%d")
OUT_JSON = os.path.abspath(os.path.join(OUT_DIR, f"fairgate_final_gate_verify_{TODAY}.json"))

RNG = np.random.default_rng(20260611)
M_NULL = 2_000_000        # (a) 시나리오별 단일전략 null MC 표본
R_PANEL = 4_000           # (a) 패널(전수 동시) 시뮬 반복수
DSR_THRESHOLD = 0.95      # (b) 통상 임계
N_FAMILY_SCENARIO = 614   # (b) 과제 지정 family 시행수 시나리오
PBO_TOP_EXTRA = 84        # (c) wf 통과 16 + 상위 후보 84 = 100
PBO_S_BLOCKS = 16         # (c) CSCV 블록수 → C(16,8)=12870 조합

report: dict = {"meta": {
    "db": DB, "date": TODAY, "engine_version": engine_version(),
    "gate": "wf pool excess IR(min>0 & median>0.3, n=5) AND xsec(sharpe>1, mdd>=-20, trades>=50)",
}}


def _j(o):
    """numpy → JSON 직렬화."""
    if isinstance(o, dict):
        return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_j(x) for x in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ============ 0. 스냅샷 로드 ============
t0 = time.time()
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT id, name, spec_json, gate_passed, wf_gate_passed, wf_n_windows, "
    "wf_excess_ir_median, wf_excess_ir_min, wf_result_json, "
    "in_sharpe, out_sharpe, in_mdd, out_mdd, in_n_trades, out_n_trades, "
    "out_avg_hold_days FROM strategies"
).fetchall()
con.close()

N_TOTAL = len(rows)
wf_pass = [r for r in rows if r["wf_gate_passed"] == 1]
final_pass = [r for r in rows if r["gate_passed"] == 1]
valid = [r for r in rows if r["wf_excess_ir_median"] is not None
         and r["wf_excess_ir_min"] is not None]
N_VALID = len(valid)
N_OBS_PASS = len(wf_pass)

print("=== 0. 기본 (DB 스냅샷) ===")
print(f"N total={N_TOTAL}, wf 게이트 통과={N_OBS_PASS}, "
      f"최종 gate_passed(wf AND xsec)={len(final_pass)} "
      f"({sorted(r['id'] for r in final_pass)})")
print(f"wf median/min 유효 행 = {N_VALID} (위양성 모수의 분모로 사용)")
meds = np.array([r["wf_excess_ir_median"] for r in valid], float)
mins = np.array([r["wf_excess_ir_min"] for r in valid], float)
print(f"wf median 분포: mean={meds.mean():.3f} sd={meds.std():.3f} "
      f"p50={np.percentile(meds,50):.3f} p90={np.percentile(meds,90):.3f} "
      f"p99={np.percentile(meds,99):.3f} max={meds.max():.3f}")
print("(median 평균<0 이면 평균0 null 은 보수적 — 모집단 평균이 비용만큼 음수)")
report["base"] = {
    "n_total": N_TOTAL, "n_wf_pass": N_OBS_PASS, "n_final_pass": len(final_pass),
    "final_pass_ids": sorted(r["id"] for r in final_pass),
    "wf_pass_ids": sorted(r["id"] for r in wf_pass),
    "n_valid": N_VALID,
    "median_dist": {"mean": meds.mean(), "sd": meds.std(),
                    "p50": np.percentile(meds, 50), "p90": np.percentile(meds, 90),
                    "p99": np.percentile(meds, 99), "max": meds.max()},
}

# ============ 1. (a) 모수 null + 창간 상관 시나리오 ============
print("\n=== 1. (a) 모수 null — 창내 σ_w / 창간 ρ 캘리브레이션 ===")

# --- σ_w: (median - min) gap. 위치(=전략 skill)에 불변 → σ_w 만 식별.
gaps = meds - mins
gaps_pos = gaps[gaps > 1e-9]   # gap==0 은 유효 창<=2 의 퇴화 케이스 → 제외
# 보정상수: 5 iid N(0,1) 의 gap 분포 (MC 1회)
_cal = RNG.standard_normal((1_000_000, 5))
_cal_gap = np.median(_cal, axis=1) - _cal.min(axis=1)
c_mean, c_med = _cal_gap.mean(), np.median(_cal_gap)
sw_mean = gaps_pos.mean() / c_mean
sw_med = np.median(gaps_pos) / c_med
SIGMA_W = sw_med   # 주 추정치 = median 기반(꼬리 강건)
print(f"gap 유효 표본 = {len(gaps_pos)}/{N_VALID} (gap=0 퇴화 {N_VALID-len(gaps_pos)} 제외)")
print(f"보정상수 c_mean={c_mean:.4f} c_med={c_med:.4f} (MC 1e6)")
print(f"σ_w 추정: mean기반={sw_mean:.3f}, median기반={sw_med:.3f} → 채택 {SIGMA_W:.3f}")

# --- ρ·창효과: 창별 원본이 남은 trendmixA 계열(동일 pool 벤치)에서 분해
full = []
for r in rows:
    if r["wf_result_json"] and "out_sample" in r["wf_result_json"]:
        d = json.loads(r["wf_result_json"])
        if d.get("benchmark") != "pool":
            continue
        v = [w["out_sample"].get("excess_sharpe") for w in d.get("windows", [])]
        if len(v) == 5 and all(x is not None for x in v):
            full.append(v)
F = np.array(full, float)
if len(F) >= 20:
    win_eff = F.mean(axis=0)                      # 창(시점) 공통 효과
    Fc = F - win_eff                              # 창효과 제거
    sw_trendmix = float(Fc.std(axis=1, ddof=1).mean())
    sb_trendmix = float(Fc.mean(axis=1).std(ddof=1))
    rho_emp = sb_trendmix**2 / (sb_trendmix**2 + sw_trendmix**2)
    sigma_delta = float(win_eff.std(ddof=1))      # 공유 창효과 크기
    print(f"trendmix 창별 원본 {len(F)}개: within σ={sw_trendmix:.3f} "
          f"(gap 추정 {SIGMA_W:.3f} 교차검증), between σ_b={sb_trendmix:.3f}, "
          f"ρ_경험={rho_emp:.3f}, 창효과 σ_δ={sigma_delta:.3f}")
else:
    rho_emp, sigma_delta = 0.3, 0.0
    print("창별 원본 부족 → ρ_경험=0.3 기본값")
sigma_delta = min(sigma_delta, 0.9 * SIGMA_W)     # σ_ε 실수 보장

# --- family(entry 골격) 군집: 패널 시뮬의 상관 블록
def _entry_skeleton(spec_json: str) -> str:
    """entry 구조(신호 타입·subject 등 비수치 키)만 남긴 family 키."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in sorted(o.items())
                    if not isinstance(v, (int, float))}
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o
    return json.dumps(strip(json.loads(spec_json).get("entry")), sort_keys=True)

fam_keys = [_entry_skeleton(r["spec_json"]) for r in valid]
fam_ids = {k: i for i, k in enumerate(sorted(set(fam_keys)))}
fam_idx = np.array([fam_ids[k] for k in fam_keys])
N_FAM = len(fam_ids)
print(f"family(entry 골격) 수 = {N_FAM} (유효 {N_VALID}행 기준; 과제 지정 시나리오는 614)")

scenarios = [("rho=0 (창간독립)", 0.0), (f"rho={rho_emp:.2f} (경험)", float(rho_emp)),
             ("rho=0.50 (보수)", 0.5)]
null_results = []
print(f"\n--- 시나리오별 null (단일전략 MC M={M_NULL:,} / 패널 R={R_PANEL:,}) ---")
for label, rho in scenarios:
    sigma_b = SIGMA_W * np.sqrt(rho / (1.0 - rho)) if rho > 0 else 0.0
    # (1) 단일전략 p0: w_j = mu + eps_j, mu~N(0,σ_b), eps~N(0,σ_w)
    mu = RNG.normal(0.0, sigma_b, M_NULL) if sigma_b > 0 else np.zeros(M_NULL)
    w = mu[:, None] + RNG.normal(0.0, SIGMA_W, (M_NULL, 5))
    hit = (w.min(axis=1) > 0) & (np.median(w, axis=1) > GATE_POOL_MIN_IR)
    p0 = float(hit.mean())
    del w, mu
    e_fp_valid = p0 * N_VALID
    e_fp_total = p0 * N_TOTAL
    p_poisson = float(poisson.sf(N_OBS_PASS - 1, e_fp_valid))  # P(X>=16), 독립 근사

    # (2) 패널 시뮬: family 군집(κ=0.5) + 공유 창효과 σ_δ → P(통과수>=16)
    sd_eps = float(np.sqrt(max(SIGMA_W**2 - (sigma_delta**2 if rho > 0 else 0.0), 1e-12)))
    counts = np.empty(R_PANEL, dtype=np.int64)
    chunk = 250
    for c0 in range(0, R_PANEL, chunk):
        c1 = min(c0 + chunk, R_PANEL)
        R = c1 - c0
        if sigma_b > 0:
            eta = RNG.standard_normal((R, N_FAM))[:, fam_idx]      # family 공유
            nu = RNG.standard_normal((R, N_VALID))
            mu_i = sigma_b * (np.sqrt(0.5) * eta + np.sqrt(0.5) * nu)
            delta = RNG.normal(0.0, sigma_delta, (R, 1, 5))        # 공유 창효과
        else:
            mu_i = np.zeros((R, N_VALID))
            delta = np.zeros((R, 1, 5))
        eps = RNG.normal(0.0, sd_eps if rho > 0 else SIGMA_W, (R, N_VALID, 5))
        wp = mu_i[:, :, None] + delta + eps
        ok = (wp.min(axis=2) > 0) & (np.median(wp, axis=2) > GATE_POOL_MIN_IR)
        counts[c0:c1] = ok.sum(axis=1)
        del mu_i, delta, eps, wp, ok
    p_panel = float((counts >= N_OBS_PASS).mean())
    q = np.percentile(counts, [50, 90, 99])
    print(f"[{label}] σ_b={sigma_b:.3f} p0={p0:.3e} | 기대 위양성 = {e_fp_valid:.2f}/{N_VALID} "
          f"({e_fp_total:.2f}/{N_TOTAL})")
    print(f"    P(통과>= {N_OBS_PASS}) 포아송={p_poisson:.3e} | "
          f"패널(family+창효과)={p_panel:.4f} (null 통과수 p50={q[0]:.0f} p90={q[1]:.0f} "
          f"p99={q[2]:.0f} max={counts.max()})")
    null_results.append({
        "label": label, "rho": rho, "sigma_b": sigma_b, "p0": p0,
        "expected_fp_valid": e_fp_valid, "expected_fp_total": e_fp_total,
        "p_ge_obs_poisson": p_poisson, "p_ge_obs_panel": p_panel,
        "panel_count_p50": q[0], "panel_count_p90": q[1], "panel_count_p99": q[2],
        "panel_count_max": int(counts.max()),
    })

print("주의: 본 null 은 wf 게이트만 모형화 — 최종 게이트는 xsec 품질 AND 이라 실제")
print("위양성은 이보다 적다(보수적 방향). 유효 창<5 행도 분모에 포함(역시 보수적).")
report["null"] = {
    "method": "parametric (gap-based sigma_w + trendmix rho decomposition)",
    "m_mc": M_NULL, "r_panel": R_PANEL,
    "sigma_w_mean_based": sw_mean, "sigma_w_median_based": sw_med,
    "sigma_w_used": SIGMA_W, "sigma_w_trendmix_crosscheck": sw_trendmix if len(F) >= 20 else None,
    "rho_empirical": float(rho_emp), "sigma_delta": sigma_delta,
    "n_families_entry_skeleton": N_FAM, "scenarios": null_results,
}

# ============ 2. (b) Deflated Sharpe Ratio ============
print("\n=== 2. (b) Deflated Sharpe Ratio (out_sharpe, xsec OOS) ===")
# per-trade 환산: out_sharpe 는 일별 수익 기반 연율 → SR_trade ≈ SR_ann·sqrt(hold/252)
# (iid 근사). 표본 길이 T = out_n_trades (과제 지정 보정).
cand = [r for r in rows if r["out_sharpe"] is not None
        and r["out_avg_hold_days"] and r["out_n_trades"] and r["out_n_trades"] >= 2]
srt_all = np.array([r["out_sharpe"] * np.sqrt(r["out_avg_hold_days"] / 252.0)
                    for r in cand])
sr_std_trade = float(srt_all.std(ddof=0))
sr_std_ann = float(np.array([r["out_sharpe"] for r in cand]).std(ddof=0))
print(f"후보 전수(유효 out_sharpe) = {len(cand)}, per-trade SR 횡단 sd = {sr_std_trade:.4f} "
      f"(연율 sd {sr_std_ann:.3f})")

T_DAYS = 2227   # xsec 채점 구간(2016-06-02~2025-06-30) 유니버스 거래일수 — §3에서 실측 갱신
DATA_YEARS = T_DAYS / 252.0

def _dsr(sr, sr0, T):
    """Bailey-LdP DSR. sr/sr0 동일 단위(per-period), T 관측수. 정규 가정."""
    if T < 2:
        return 0.0
    denom = np.sqrt(1.0 + 0.5 * sr * sr)
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1.0) / denom))

trial_scenarios = [("N=3106 (전수)", N_TOTAL), (f"N={N_FAMILY_SCENARIO} (family)", N_FAMILY_SCENARIO)]
sr0_by = {N: expected_max_sharpe(N, sr_std=sr_std_trade) for _, N in trial_scenarios}
sr0_ann_by = {N: expected_max_sharpe(N, sr_std=sr_std_ann) for _, N in trial_scenarios}
for lbl, N in trial_scenarios:
    print(f"  {lbl}: E[max SR_trade|null]={sr0_by[N]:.4f} (연율 환산 기대최대 {sr0_ann_by[N]:.3f})")

dsr_rows = []
print(f"\n{'id':>5} {'name':38} {'outSR':>7} {'SRtrd':>7} {'T':>5} "
      f"{'DSR@3106':>9} {'DSR@614':>8} {'DSRday@3106':>11}")
for r in sorted(wf_pass, key=lambda x: -(x["wf_excess_ir_median"] or 0)):
    hold = r["out_avg_hold_days"] or 0.0
    T_tr = int(r["out_n_trades"] or 0)
    srt = r["out_sharpe"] * np.sqrt(hold / 252.0)
    d3106 = _dsr(srt, sr0_by[N_TOTAL], T_tr)
    d614 = _dsr(srt, sr0_by[N_FAMILY_SCENARIO], T_tr)
    # 참고: 일별 단위 변형 (T=거래일수, sr_std 도 일별 단위)
    srd = r["out_sharpe"] / np.sqrt(252.0)
    sr0_d = expected_max_sharpe(N_TOTAL, sr_std=sr_std_ann / np.sqrt(252.0))
    dday = _dsr(srd, sr0_d, T_DAYS)
    print(f"{r['id']:>5} {r['name'][:38]:38} {r['out_sharpe']:>7.3f} {srt:>7.4f} "
          f"{T_tr:>5} {d3106:>9.4f} {d614:>8.4f} {dday:>11.4f}")
    dsr_rows.append({
        "id": r["id"], "name": r["name"], "out_sharpe": r["out_sharpe"],
        "sr_trade": srt, "T_trades": T_tr,
        "dsr_n3106": d3106, "dsr_n614": d614, "dsr_daily_n3106": dday,
        "wf_excess_ir_median": r["wf_excess_ir_median"],
        "final_gate_passed": int(r["gate_passed"]),
    })

n_pass_3106 = sum(1 for d in dsr_rows if d["dsr_n3106"] > DSR_THRESHOLD)
n_pass_614 = sum(1 for d in dsr_rows if d["dsr_n614"] > DSR_THRESHOLD)
print(f"\nDSR>{DSR_THRESHOLD}: N=3106 기준 {n_pass_3106}/{N_OBS_PASS}, "
      f"N=614 기준 {n_pass_614}/{N_OBS_PASS}")
mjt = max_justifiable_trials(DATA_YEARS, target_sharpe=1.0)
print(f"max_justifiable_trials(data {DATA_YEARS:.1f}y, SR=1.0) = {mjt} "
      f"→ 실제 시행 {N_TOTAL} 은 정당화 가능 시행수의 {N_TOTAL/max(mjt,1):.0f}배")
report["dsr"] = {
    "convention": "per-trade SR = out_sharpe*sqrt(hold/252), T=out_n_trades, 정규 가정",
    "n_candidates": len(cand), "sr_std_trade": sr_std_trade, "sr_std_annual": sr_std_ann,
    "sr0_trade": {str(k): v for k, v in sr0_by.items()},
    "threshold": DSR_THRESHOLD, "n_pass_n3106": n_pass_3106, "n_pass_n614": n_pass_614,
    "max_justifiable_trials_sr1": mjt, "rows": dsr_rows,
}

# ============ 3. (c) PBO — CSCV (일별 초과수익 재시뮬 생성) ============
print("\n=== 3. (c) PBO/CSCV — 일별 초과수익 행렬 생성 후 평가 ===")
print("저장된 전략별 일별 시계열 없음(wealth CSV 는 5개뿐) → 재시뮬로 생성한다.")
pbo_report: dict = {"available": False}
try:
    from tradingagents.dataflows.kis_history_store import KisHistoryStore
    from tradingagents.hermes.strategy_research_v2 import _preload

    t_pre = time.time()
    raw = KisHistoryStore().list_tickers()
    tickers, loader, kospi_fetcher, usdkrw_fetcher = _preload(raw)
    holdings: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        df, _ = loader(tk)
        if df is None or df.empty:
            continue
        df = df[df["date"].astype(str) <= SCORE_DATE_HI]   # 2025-06 채점 상한
        if not df.empty:
            holdings[tk] = df
    kospi = kospi_fetcher("", "")
    usdkrw = usdkrw_fetcher("", "")
    bench = _survivor_pool_index(holdings)
    uni_dates = sorted({d for df in holdings.values() for d in df["date"].astype(str)})
    T_DAYS = len(uni_dates)
    DATA_YEARS = T_DAYS / 252.0
    print(f"preload {time.time()-t_pre:.0f}s: universe={len(holdings)}, "
          f"거래일수 T={T_DAYS} ({DATA_YEARS:.2f}y)")

    # 대상: wf 통과 16 + 상위 후보(median 내림차순, 퇴화·저거래 제외)
    pass_ids = {r["id"] for r in wf_pass}
    extras = sorted(
        (r for r in valid
         if r["wf_gate_passed"] != 1 and (r["out_n_trades"] or 0) >= 50
         and abs(r["wf_excess_ir_median"] - r["wf_excess_ir_min"]) > 1e-9),
        key=lambda r: -r["wf_excess_ir_median"])[:PBO_TOP_EXTRA]
    targets = list(wf_pass) + extras
    print(f"대상 {len(targets)}개 = wf 통과 {len(wf_pass)} + 상위 후보 {len(extras)} "
          f"(후보 컷: median {extras[-1]['wf_excess_ir_median']:.3f})")

    def _daily_excess(spec: dict) -> pd.Series:
        """전 유니버스 일별 초과수익(생존풀 벤치) — _window_metrics 와 동일 로직, 시계열 반환."""
        ret_frames, act_frames = [], []
        for tk in sorted(holdings):
            sub = holdings[tk]
            if len(sub) < 3:
                continue
            sub = sub.reset_index(drop=True)
            _, daily, active, cdf = _simulate_v2(spec, sub, kospi=kospi)
            idx = cdf["date"].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        port = bt._combine_korea_stock_portfolio(ret_frames, act_frames)
        overlay_mult = _market_overlay_multiplier(
            port.index, kospi, spec.get("market_overlay"), usdkrw=usdkrw)
        if spec.get("market_overlay"):
            port = port * overlay_mult
            market = bt._market_daily_returns(bench, port.index)
            invested = (bt._invested_weight_korea(act_frames)
                        .reindex(port.index).fillna(0.0) * overlay_mult)
            return port - invested * market
        return bt._excess_daily_korea(port, act_frames, bench)

    t_sim = time.time()
    series = {}
    for i, r in enumerate(targets, 1):
        series[r["id"]] = _daily_excess(json.loads(r["spec_json"]))
        if i % 20 == 0:
            print(f"  sim {i}/{len(targets)} ({time.time()-t_sim:.0f}s)", flush=True)
    X = pd.DataFrame(series).reindex(uni_dates).fillna(0.0)
    cols = list(X.columns)
    Xv = X.to_numpy()
    T, Nst = Xv.shape
    print(f"행렬 {T}x{Nst} 생성 완료 ({time.time()-t_sim:.0f}s)")

    # CSCV: S 블록 → C(S, S/2) 조합. 블록 합/제곱합 집계로 벡터화.
    blocks = np.array_split(np.arange(T), PBO_S_BLOCKS)
    s_b = np.array([Xv[b].sum(axis=0) for b in blocks])            # S x N
    q_b = np.array([(Xv[b] ** 2).sum(axis=0) for b in blocks])     # S x N
    n_b = np.array([len(b) for b in blocks], float)                # S
    combos = list(itertools.combinations(range(PBO_S_BLOCKS), PBO_S_BLOCKS // 2))
    ind = np.zeros((len(combos), PBO_S_BLOCKS))
    for ci, c in enumerate(combos):
        ind[ci, list(c)] = 1.0

    def _ir(ind_m):
        n = ind_m @ n_b                                            # C
        mean = (ind_m @ s_b) / n[:, None]
        var = (ind_m @ q_b) / n[:, None] - mean**2
        sd = np.sqrt(np.clip(var, 1e-18, None))
        return mean / sd * np.sqrt(252.0)

    ir_tr = _ir(ind)
    ir_te = _ir(1.0 - ind)
    best = ir_tr.argmax(axis=1)                                    # IS 최강 전략
    te_best = ir_te[np.arange(len(combos)), best]
    rank_bot = (ir_te < te_best[:, None]).sum(axis=1) + 0.5 * (
        (ir_te == te_best[:, None]).sum(axis=1) - 1) + 1.0
    omega = rank_bot / (Nst + 1.0)
    lam = np.log(omega / (1.0 - omega))
    pbo = float((lam < 0).mean())

    # 16개 통과 전략의 OOS 백분위(조합 전체 중앙값)
    order = ir_te.argsort(axis=1).argsort(axis=1)                  # 0=최하
    pct_te = (order + 1.0) / Nst
    pass_pos = [cols.index(pid) for pid in cols if pid in pass_ids]
    passer_pct = {int(cols[p]): float(np.median(pct_te[:, p])) for p in pass_pos}
    best_from_pass = float(np.isin(best, pass_pos).mean())
    bc = np.bincount(best, minlength=Nst)
    top_best = sorted(((int(cols[i]), int(c)) for i, c in enumerate(bc) if c),
                      key=lambda x: -x[1])[:5]

    print(f"\nCSCV S={PBO_S_BLOCKS}, 조합 {len(combos)}개")
    print(f"PBO = {pbo:.4f} (IS 최강이 OOS 중앙값 미달일 확률)")
    print(f"λ 분포: p10={np.percentile(lam,10):.3f} p50={np.percentile(lam,50):.3f} "
          f"p90={np.percentile(lam,90):.3f}")
    print(f"IS 최강이 wf 통과 16개 중 하나일 비율 = {best_from_pass:.3f}; "
          f"IS 최강 빈도 top5 = {top_best}")
    print("wf 통과 16개의 OOS IR 백분위 중앙값(조합 전체, 1.0=최상):")
    for pid, pc in sorted(passer_pct.items(), key=lambda x: -x[1]):
        print(f"  #{pid}: {pc:.3f}")
    print("주의: 본 PBO 는 16+상위84 '엘리트 집합' 내 상대평가 — 전수 3106 PBO 가")
    print("아니라 '엘리트 안에서의 선택 과적합' 측정이다(전수는 시뮬 ~2h 로 별도).")
    pbo_report = {
        "available": True, "n_strategies": Nst, "n_days": T, "s_blocks": PBO_S_BLOCKS,
        "n_combos": len(combos), "pbo": pbo,
        "lambda_p10": np.percentile(lam, 10), "lambda_p50": np.percentile(lam, 50),
        "lambda_p90": np.percentile(lam, 90),
        "best_from_passers_ratio": best_from_pass, "top_is_best": top_best,
        "passer_oos_pct_median": passer_pct,
        "note": "엘리트(통과16+상위84) 집합 내 CSCV — 전수 3106 아님",
    }
except Exception as e:                                             # noqa: BLE001
    print(f"PBO 실패 — 스킵: {e!r}")
    print("필요 데이터: 전략별 일별 (초과)수익 시계열. 수집 방법: 연구 루프에서")
    print("run_walk_forward_validation 채점 시 excess_daily 를 parquet 로 저장하거나,")
    print("본 스크립트의 _daily_excess() 재시뮬(전략당 ~2.4s)을 전수로 확장.")
    pbo_report = {"available": False, "error": repr(e),
                  "required": "per-strategy daily excess return series",
                  "how": "research 루프에서 excess_daily 저장 또는 재시뮬 확장"}
report["pbo"] = pbo_report

# ============ 4. 결론 ============
print("\n" + "=" * 70)
print("=== 4. 결론 요약 ===")
worst = max(null_results, key=lambda s: s["expected_fp_valid"])
base_s = null_results[0]
print(f"① wf 통과 {N_OBS_PASS}/{N_TOTAL} vs null:")
for s in null_results:
    print(f"   [{s['label']}] 기대 위양성 {s['expected_fp_valid']:.1f}개, "
          f"P(>= {N_OBS_PASS}) 포아송 {s['p_ge_obs_poisson']:.2e} / "
          f"패널 {s['p_ge_obs_panel']:.4f}")
sig = worst["p_ge_obs_panel"] < 0.05
print(f"   → 최보수 시나리오 패널 p={worst['p_ge_obs_panel']:.4f} → "
      f"{'유의' if sig else '비유의(운으로 설명 가능)'}")
dsr_pass_ids = [d["id"] for d in dsr_rows if d["dsr_n3106"] > DSR_THRESHOLD]
print(f"② DSR>{DSR_THRESHOLD} (N=3106): {n_pass_3106}/{N_OBS_PASS} → {dsr_pass_ids}")
print(f"   (family N=614 시나리오: {n_pass_614}/{N_OBS_PASS})")
print(f"③ 다중검정 권고: 데이터 {DATA_YEARS:.1f}y 에서 SR 1.0 을 정당화할 수 있는")
print(f"   최대 시행수 ≈ {max_justifiable_trials(DATA_YEARS, 1.0)} — 현재 {N_TOTAL}회 시행은 이를")
print(f"   초과. 신규 탐색은 family 단위 사전등록·시행 카운트 동결 후 forward 검증 권장.")
report["conclusion"] = {
    "observed_pass": N_OBS_PASS, "worst_scenario_p_panel": worst["p_ge_obs_panel"],
    "significant_5pct": bool(sig), "dsr_pass_ids_n3106": dsr_pass_ids,
    "max_justifiable_trials_sr1": max_justifiable_trials(DATA_YEARS, 1.0),
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(_j(report), f, ensure_ascii=False, indent=1)
print(f"\n산출물: {OUT_JSON}  (총 {time.time()-t0:.0f}s)")
