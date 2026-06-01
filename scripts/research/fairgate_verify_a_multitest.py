import sqlite3, json, math
import numpy as np
from tradingagents.hermes.strategy_validation import expected_max_sharpe

DB = "/Users/selab/.tradingagents/hermes_fairgate/strategies_v2.db"
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT name, gate_passed, wf_n_windows, wf_excess_ir_median, wf_excess_ir_min, wf_result_json "
    "FROM strategies"
).fetchall()
con.close()

def win_vec(rj):
    d = json.loads(rj)
    return [w["out_sample"].get("excess_sharpe") for w in d["windows"]]

N = len(rows)
strict = [r for r in rows if r["gate_passed"] == 1]
conc   = [r for r in rows if r["name"].startswith("conc")]
conc_strict = [r for r in conc if r["gate_passed"] == 1]

print(f"=== 0. 기본 ===")
print(f"N total={N}, strict={len(strict)}, conc total={len(conc)}, conc strict={len(conc_strict)}")
nwins = sorted(set(r["wf_n_windows"] for r in rows))
print(f"n_windows distinct={nwins}")

# ============ 1) 독립 null 기대 위양성 ============
print("\n=== 1. 독립 null 기대 위양성 ===")
# strict gate = 모든 OOS 창 초과IR>0  (확률 0.5^n)  AND  median 초과IR>0.5
# (a) 단순 'all windows positive' 기대 = sum over strategies of 0.5^n_i
E_all_pos = sum(0.5 ** r["wf_n_windows"] for r in rows)
# (b) 'median>0' 추가 1 비트 가중 근사: 0.5^(n+1)
E_median_weighted = sum(0.5 ** (r["wf_n_windows"] + 1) for r in rows)
print(f"E[all windows >0]  = sum 0.5^n   = {E_all_pos:.3f}")
print(f"E[+median bit]     = sum 0.5^(n+1)= {E_median_weighted:.3f}")
print(f"관측 strict        = {len(strict)}")

# 보다 정확한 strict null: median 초과IR>0.5 는 단순 부호보다 훨씬 빡빡.
# n=6 독립 N(0,1) OOS IR 가정 하 P(all 6 >0)=0.5^6, 그리고 median(6개)>0.5.
# 시뮬레이션으로 진짜 strict null 위양성률 추정.
rng = np.random.default_rng(0)
M = 2_000_000
n = 6
# 창별 IR 의 노이즈 스케일: 관측 전체 분포에서 표준편차 추정
all_win = np.array([s for r in rows for s in win_vec(r["wf_result_json"]) if s is not None], float)
sd_emp = all_win.std()
mu_emp = all_win.mean()
print(f"\n전체 창별 OOS 초과IR: n={all_win.size}, mean={mu_emp:.3f}, sd={sd_emp:.3f}")

# null = 평균0, 표준편차=경험 sd 의 i.i.d. (skill 없음 가정, 분산은 데이터의 finite-sample noise)
sim = rng.normal(0.0, sd_emp, size=(M, n))
all_pos = (sim > 0).all(axis=1)
med = np.median(sim, axis=1)
p_strict_iid = float((all_pos & (med > 0.5)).mean())
print(f"i.i.d. null(sd={sd_emp:.2f}) strict 위양성률 p={p_strict_iid:.4e}  -> 기대 위양성 = {p_strict_iid*N:.2f} / {N}")
p_allpos = float(all_pos.mean())
print(f"  (참고) all-pos only p={p_allpos:.4e} -> {p_allpos*N:.2f}; 이론치 0.5^6*N={0.5**6*N:.2f}")

# ============ 2) 유효 시행수 (effective N) ============
print("\n=== 2. 유효 시행수 (effective N) ===")
# 창별 OOS 초과IR 벡터를 6차원 점으로 보고 strict 18개 / conc 11개 상호상관.
def mat(strats):
    return np.array([win_vec(r["wf_result_json"]) for r in strats], float)

def part_ratio(M_):
    # 전략들을 변수로 보는 상관행렬의 participation ratio of eigenvalues
    # 각 전략 = 6창 시계열. 전략간 상관 -> k x k corr matrix.
    X = M_  # k x 6
    # 표준화 후 전략간 상관 (창 축으로)
    C = np.corrcoef(X)  # k x k
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-10]
    pr = (ev.sum() ** 2) / (ev ** 2).sum()
    return C, ev, pr

for label, strats in [("strict 18", strict), ("conc strict 11", conc_strict), ("conc all 16", conc)]:
    Mk = mat(strats)
    C, ev, pr = part_ratio(Mk)
    k = Mk.shape[0]
    offdiag = C[np.triu_indices(k, 1)]
    print(f"\n[{label}] k={k}")
    print(f"  창간 상관 off-diag: mean={offdiag.mean():.3f} median={np.median(offdiag):.3f} min={offdiag.min():.3f} max={offdiag.max():.3f}")
    print(f"  participation ratio (독립 베팅 수 추정) = {pr:.2f}")
    print(f"  최대 고유값 점유율 = {ev.max()/ev.sum():.3f}")

# near-duplicate 군집: 상관 > 0.95 를 동일 베팅으로 묶음
def cluster(strats, thr=0.95):
    Mk = mat(strats)
    k = Mk.shape[0]
    C = np.corrcoef(Mk)
    parent = list(range(k))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        parent[find(a)]=find(b)
    for i in range(k):
        for j in range(i+1,k):
            if C[i,j] >= thr:
                union(i,j)
    groups={}
    for i in range(k):
        groups.setdefault(find(i),[]).append(strats[i]["name"])
    return list(groups.values())

print("\n--- near-duplicate 군집 (corr>=0.95) ---")
for label, strats in [("strict 18", strict), ("conc strict 11", conc_strict)]:
    cl = cluster(strats)
    print(f"[{label}] {len(cl)} 군집:")
    for g in cl:
        print(f"   ({len(g)}) {g}")

# ============ 3) deflated 천장 ============
print("\n=== 3. deflated 천장 (expected_max_sharpe; IR 동형) ===")
# OOS 초과IR median 들의 표준편차로 sr_std 추정 (전체 1005개 분포)
med_all = np.array([r["wf_excess_ir_median"] for r in rows if r["wf_excess_ir_median"] is not None], float)
sr_std_med = med_all.std()
obs_max_med = med_all.max()
print(f"전체 1005 median-IR 분포: p50={np.percentile(med_all,50):.3f} p90={np.percentile(med_all,90):.3f} max={obs_max_med:.3f}, sd={sr_std_med:.3f}")

for Ntr, lbl in [(N, "N=1005 (raw)"),
                 (None, "effective-N (strict PR)"),
                 (None, "effective-N (conc PR)")]:
    pass

# 천장: expected_max at various trial counts, scaled by sr_std of median-IR
Mk = mat(strict); _,_,pr_strict = part_ratio(Mk)
Mk = mat(conc_strict); _,_,pr_conc = part_ratio(Mk)
print(f"\nsr_std(median-IR over 1005) = {sr_std_med:.3f}")
for Ntr, lbl in [(1005, "N=1005 raw"),
                 (max(2,round(pr_strict)), f"eff-N=strict PR≈{pr_strict:.1f}"),
                 (max(2,round(pr_conc)), f"eff-N=conc PR≈{pr_conc:.1f}"),
                 (18, "N=18 (strict 후보군)")]:
    em = expected_max_sharpe(Ntr, sr_std=sr_std_med)
    print(f"  {lbl:28s} expected_max median-IR = {em:.3f}")
print(f"  관측 최고 median-IR(strict) = {obs_max_med:.3f}")
print(f"  관측 최고 단일창 초과IR     = {all_win.max():.3f}")

# conc 가족 median-IR 들
conc_meds = sorted([r["wf_excess_ir_median"] for r in conc_strict], reverse=True)
print(f"\n  conc strict median-IR 들: {[round(x,3) for x in conc_meds]}")
