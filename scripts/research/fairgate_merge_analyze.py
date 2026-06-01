"""8개 샤드 DB 합치고 공정 게이트 최종 집계 + 다중검정/생존편향 점검."""
import json
import math
import sqlite3
from pathlib import Path

SHARDS = [Path.home() / f".tradingagents/hermes_fairgate_sh{i}/strategies_v2.db"
          for i in range(8)]
FINAL = Path.home() / ".tradingagents/hermes_fairgate/strategies_v2.db"

rows = []
errs = 0
for sh in SHARDS:
    if not sh.exists():
        print(f"  WARN missing {sh}"); continue
    con = sqlite3.connect(str(sh)); con.row_factory = sqlite3.Row
    rows.extend(dict(r) for r in con.execute("SELECT * FROM strategies"))
    con.close()

# spec_hash 중복 제거(샤드는 disjoint 라 없어야 정상)
seen = {}
for r in rows:
    seen[r["spec_hash"]] = r
rows = list(seen.values())
N = len(rows)
print(f"=== 합산: {N} 전략 (8 샤드) ===")

strict = [r for r in rows if r["gate_passed"] == 1]
lenient = [r for r in rows if (r["wf_excess_ir_median"] or -9) > 0.5]
has_wf = [r for r in rows if r["wf_excess_ir_median"] is not None]
print(f"엄격 합격(모든 구간 시장초과 AND 중앙IR>0.5): {len(strict)} ({len(strict)/N*100:.1f}%)")
print(f"보통 합격(중앙IR>0.5만):                    {len(lenient)} ({len(lenient)/N*100:.1f}%)")
print(f"wf 평가된 전략: {len(has_wf)}/{N}")

# --- 다중검정: 독립 가정 null 에서 '모든 구간 시장초과' 기대 위양성 ---
# 구간별 시장초과 확률 0.5(동전), n_windows 독립 가정.
exp_fp_allpos = sum(0.5 ** (r["wf_n_windows"] or 6) for r in has_wf)
exp_fp_strict = sum(0.5 ** ((r["wf_n_windows"] or 6) + 1) for r in has_wf)  # +median>0 ~0.5
print(f"\n=== 다중검정(독립 null) ===")
print(f"'모든 구간 시장초과' 기대 위양성: ~{exp_fp_allpos:.1f}개 (관측 strict {len(strict)})")
print(f"'+ 중앙IR>0' 가중 시 기대 위양성: ~{exp_fp_strict:.1f}개")
print("주: 생존편향으로 구간들이 양(+)으로 상관 → 실제 위양성은 이보다 더 많을 수 있음(독립가정은 하한).")

# --- 엄격 합격자 분포: 이름 prefix(아키타입/신호족) ---
from collections import Counter
def prefix(name):
    for p in ("dipsup","diptr","conc","diponly","rgx","frev","fscale","orot",
              "mrp","mconc","mfscale","defrot","v2_","batch","rcfu"):
        if name.startswith(p) or p in name[:6]:
            return p
    return name.split("-")[0][:8]
pc = Counter(prefix(r["name"]) for r in strict)
print(f"\n=== 엄격 합격 {len(strict)}개 신호족 분포 ===")
for k, v in pc.most_common():
    print(f"  {k:10} {v}")

# --- 엄격 합격 상위 (min IR 큰 순 = 최악 구간도 강한) ---
strict_sorted = sorted(strict, key=lambda r: r["wf_excess_ir_min"] or -9, reverse=True)
print(f"\n=== 엄격 합격 상위 10 (최악구간 초과IR 큰 순) ===")
print(f"{'name':36} {'med IR':>7} {'min IR':>7} {'nwin':>4} {'raw inSh':>8}")
for r in strict_sorted[:10]:
    print(f"{r['name'][:36]:36} {r['wf_excess_ir_median']:7.2f} "
          f"{r['wf_excess_ir_min']:7.2f} {r['wf_n_windows']:>4} "
          f"{(r['in_sharpe'] or 0):8.2f}")

# --- 초과IR 중앙값 분포 ---
meds = sorted(r["wf_excess_ir_median"] for r in has_wf if r["wf_excess_ir_median"] is not None)
def pct(p):
    i = int(len(meds)*p); return meds[min(i, len(meds)-1)]
print(f"\n=== 전체 초과IR(중앙) 분포 ===")
print(f"  p10={pct(.1):.2f} p25={pct(.25):.2f} p50={pct(.5):.2f} "
      f"p75={pct(.75):.2f} p90={pct(.9):.2f} max={meds[-1]:.2f}")

# --- 최종 DB 로 합치기(id 재발급) ---
FINAL.parent.mkdir(parents=True, exist_ok=True)
if FINAL.exists():
    FINAL.unlink()
# 첫 샤드를 스키마 베이스로 복사 후 나머지 INSERT
import shutil
shutil.copy(str(SHARDS[0]), str(FINAL))
fcon = sqlite3.connect(str(FINAL))
cols = [r[1] for r in fcon.execute("PRAGMA table_info(strategies)") if r[1] != "id"]
collist = ",".join(cols)
ph = ",".join("?" * len(cols))
inserted = fcon.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
for sh in SHARDS[1:]:
    if not sh.exists(): continue
    scon = sqlite3.connect(str(sh)); scon.row_factory = sqlite3.Row
    for r in scon.execute("SELECT * FROM strategies"):
        fcon.execute(f"INSERT INTO strategies ({collist}) VALUES ({ph})",
                     tuple(r[c] for c in cols))
        inserted += 1
    scon.close()
fcon.commit()
final_n = fcon.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
final_pass = fcon.execute("SELECT SUM(gate_passed) FROM strategies").fetchone()[0]
fcon.close()
print(f"\n=== 최종 DB {FINAL} ===")
print(f"  행수={final_n} 엄격합격={final_pass}")
