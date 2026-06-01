import sqlite3, json
import numpy as np
from tradingagents.hermes.strategy_validation import expected_max_sharpe

DB = "/Users/selab/.tradingagents/hermes_fairgate/strategies_v2.db"
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
rows = con.execute("SELECT name, gate_passed, wf_n_windows, wf_excess_ir_median, wf_result_json FROM strategies").fetchall()
con.close()

def win_vec(rj):
    return [w["out_sample"].get("excess_sharpe") for w in json.loads(rj)["windows"]]

# 전체 1005 전략의 유효 시행수: 6창 OOS 초과IR 벡터를 변수로 본 상관행렬 PR
allM = np.array([win_vec(r["wf_result_json"]) for r in rows], float)  # 1005 x 6
allM = np.nan_to_num(allM, nan=0.0)
# 분산 0 (모든 창 동일, 보통 거래 없음) 전략은 PR 계산에서 제외 — 상관 정의 안 됨
sd_row = allM.std(axis=1)
keep = sd_row > 1e-9
print(f"전략 {allM.shape[0]} 중 분산>0 = {keep.sum()} (분산0={(~keep).sum()}는 거래없음·상수)")
Mv = allM[keep]
C = np.corrcoef(Mv)
ev = np.linalg.eigvalsh(C); ev = ev[ev > 1e-9]
PR_full = (ev.sum()**2)/(ev**2).sum()
print(f"유효 시행수 (participation ratio over {keep.sum()} non-degenerate candidates) = {PR_full:.1f}")

med_all = np.array([r["wf_excess_ir_median"] for r in rows if r["wf_excess_ir_median"] is not None], float)
sr_std = med_all.std()
obs_max = med_all.max()
conc_max = max(r["wf_excess_ir_median"] for r in rows if r["name"].startswith("conc") and r["gate_passed"]==1)

print(f"\nsr_std(median-IR)={sr_std:.3f}  관측 max(median-IR)={obs_max:.3f}  conc 최고={conc_max:.3f}")
print("\nexpected_max(median-IR) 천장 vs 관측:")
for Ntr, lbl in [(1005,"raw N=1005"), (round(PR_full),f"eff-N(전체 PR)={round(PR_full)}"),
                 (200,"보수: eff-N=200"), (50,"보수: eff-N=50"), (20,"보수: eff-N=20")]:
    em = expected_max_sharpe(Ntr, sr_std=sr_std)
    flag = "관측<천장(운)" if obs_max < em else "관측>천장"
    print(f"  {lbl:24s} N={Ntr:5d}  ceil={em:.3f}   [{flag}]  conc최고 vs ceil: {'<' if conc_max<em else '>'}")
