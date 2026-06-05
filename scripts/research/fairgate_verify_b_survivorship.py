"""테스트 B — 생존편향 discriminator.

conc 가족(gate_passed=1, name LIKE 'conc%')의 초과수익 벤치마크를
KOSPI -> universe-EW(199종목 동일가중 NAV)로 바꿔 wf 재채점.
"""
import json
import sqlite3
import statistics as st

import numpy as np
import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_validation import run_walk_forward_validation, engine_version

DB = "/Users/selab/.tradingagents/hermes_fairgate/strategies_v2.db"

# 1) preload (199종목 holdings + KOSPI 1회)
raw = KisHistoryStore().list_tickers()
print(f"[preload] raw_tickers={len(raw)}", flush=True)
tickers, loader, kospi_fetcher, usdkrw_fetcher = _preload(raw)
print(f"[preload] loaded universe={len(tickers)}", flush=True)

# 2) universe-EW NAV 구성: 전 종목 일별수익(close pct_change, 거래일만)의 동일가중 평균
#    각 날짜에 그날 가격이 있는 종목들만 평균(생존편향 유니버스 그대로).
ret_series = []
for tk in tickers:
    df, _ = loader(tk)
    d = df[["date", "close"]].copy()
    d["date"] = d["date"].astype(str)
    d["close"] = pd.to_numpy if False else pd.to_numeric(d["close"], errors="coerce")
    d = d[d["close"] > 0].dropna(subset=["close"])
    d = d.sort_values("date").drop_duplicates("date", keep="last")
    r = pd.Series(d["close"].to_numpy(dtype=float), index=d["date"].to_numpy())
    r = r.pct_change().replace([np.inf, -np.inf], np.nan)
    ret_series.append(r.rename(tk))

R = pd.concat(ret_series, axis=1).sort_index()       # date x ticker 일별수익
ew_ret = R.mean(axis=1, skipna=True)                 # 동일가중 평균(그날 존재 종목만)
ew_ret = ew_ret.fillna(0.0)
ew_nav = (1.0 + ew_ret).cumprod()
ew_nav_df = pd.DataFrame({"date": ew_nav.index.astype(str), "close": ew_nav.to_numpy(dtype=float)})
print(f"[ew] nav span={ew_nav_df['date'].iloc[0]}..{ew_nav_df['date'].iloc[-1]} "
      f"n_days={len(ew_nav_df)} final_nav={ew_nav_df['close'].iloc[-1]:.4f}", flush=True)

def ew_fetcher(_s, _e):
    return ew_nav_df

# 참고: (universe EW - KOSPI) 자체의 연율 IR (생존/구성 프리미엄)
kospi_df = kospi_fetcher(None, None)
def _to_ret(df, idx):
    k = df[["date", "close"]].copy()
    k["date"] = k["date"].astype(str)
    k = k.sort_values("date").drop_duplicates("date", keep="last")
    s = pd.Series(pd.to_numeric(k["close"], errors="coerce").to_numpy(dtype=float), index=k["date"].to_numpy())
    return s.reindex(idx).ffill().pct_change().replace([np.inf,-np.inf],np.nan).fillna(0.0)
grid = ew_nav_df["date"].to_numpy()
ew_grid_ret = _to_ret(ew_nav_df, grid)
kospi_grid_ret = _to_ret(kospi_df, grid) if kospi_df is not None else pd.Series(0.0, index=grid)
prem = ew_grid_ret - kospi_grid_ret
prem_ir = (prem.mean() / prem.std() * np.sqrt(252)) if prem.std() > 0 else float("nan")
print(f"[premium] (EW-KOSPI) daily mean={prem.mean()*1e4:.2f}bp std={prem.std()*1e4:.2f}bp "
      f"annualized IR={prem_ir:.3f}", flush=True)

# 3) conc 가족 재채점
con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT name, spec_json, wf_excess_ir_median, wf_excess_ir_min, wf_n_windows "
    "FROM strategies WHERE gate_passed=1 AND name LIKE 'conc%' ORDER BY name"
).fetchall()
print(f"\n[conc] re-scoring {len(rows)} strategies with universe-EW benchmark\n", flush=True)
print(f"{'name':38} | {'KOSPI-rel med/min':>20} | {'UNIV-rel med/min':>20} | {'nwin':>4} | gate(K->U)")
print("-"*120)

ev = engine_version()
results = []
for name, spec_json, k_med, k_min, k_nwin in rows:
    spec = json.loads(spec_json)
    wf = run_walk_forward_validation(spec, tickers, loader=loader, kospi_fetcher=ew_fetcher)
    u_med = wf["oos_excess_ir_median"]
    u_min = wf["oos_excess_ir_min"]
    u_gate = wf["gate_passed"]
    u_nwin = wf["n_windows"]
    # 창별 OOS universe-relative IR
    win_ex = [round(w["out_sample"].get("excess_sharpe"),3) if w["out_sample"].get("excess_sharpe") is not None else None
              for w in wf["windows"]]
    results.append((name, k_med, k_min, u_med, u_min, u_gate, u_nwin, win_ex))
    print(f"{name:38} | {k_med:9.4f}/{k_min:8.4f} | {(u_med if u_med is not None else float('nan')):9.4f}/"
          f"{(u_min if u_min is not None else float('nan')):8.4f} | {u_nwin:>4} | "
          f"{'PASS->'+ ('PASS' if u_gate else 'FAIL')}", flush=True)
    print(f"{'':38}   per-window UNIV OOS exIR: {win_ex}", flush=True)

print("\n=== SUMMARY ===")
k_meds = [r[1] for r in results]
u_meds = [r[3] for r in results if r[3] is not None]
u_mins = [r[4] for r in results if r[4] is not None]
n_u_pass = sum(1 for r in results if r[5])
print(f"engine_version={ev}")
print(f"conc n={len(results)}")
print(f"KOSPI-rel exIR median: median-of-strategies={st.median(k_meds):.4f} "
      f"min={min(k_meds):.4f} max={max(k_meds):.4f}")
print(f"UNIV-rel  exIR median: median-of-strategies={st.median(u_meds):.4f} "
      f"min={min(u_meds):.4f} max={max(u_meds):.4f}")
print(f"UNIV-rel  exIR min(across windows): median={st.median(u_mins):.4f} "
      f"min={min(u_mins):.4f} max={max(u_mins):.4f}")
print(f"strict gate pass under UNIV benchmark: {n_u_pass}/{len(results)}")
print(f"premium (EW-KOSPI) annualized IR = {prem_ir:.3f}")
