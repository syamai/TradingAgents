
import csv, math, statistics, json
from pathlib import Path
from datetime import datetime, timezone
IN=Path('/Users/selab/Source/trading-ai/artifacts/recent_passed_100m_wealth_curves_20160602_20250630.csv')
OUTDIR=Path('/Users/selab/Source/trading-ai/artifacts/market_overlay_experiments'); OUTDIR.mkdir(parents=True, exist_ok=True)
rows=[]
for r in csv.DictReader(IN.open()):
    if r['strategy_id']=='1446': rows.append({'date':r['date'],'strat_ret':float(r['strategy_daily_return']),'kospi_ret':float(r['kospi_daily_return'])})
# wealth arrays
w=1.0
for r in rows:
    w*=1+r['kospi_ret']; r['kospi']=w
w=1.0
for r in rows:
    w*=1+r['strat_ret']; r['orig']=w
kospi=[r['kospi'] for r in rows]

def tr(i,w):
    if i<w: return None
    return kospi[i]/kospi[i-w]-1

def mdd(vals):
    p=vals[0]; pi=0; worst=0; wi=0; wpi=0
    for i,v in enumerate(vals):
        if v>p: p=v; pi=i
        dd=v/p-1
        if dd<worst: worst=dd; wi=i; wpi=pi
    return worst,wpi,wi

def eval_variant(name, fn):
    wealth=[]; w=1.0; mult=[]
    for i,r in enumerate(rows):
        m=fn(i); m=max(0,min(1,m)); mult.append(m); w*=1+r['strat_ret']*m; wealth.append(w)
    n=len(rows); years=n/252; dd,pi,ti=mdd(wealth)
    rets=[rows[i]['strat_ret']*mult[i] for i in range(n)]
    sd=statistics.stdev(rets); mean=sum(rets)/n
    return {'name':name,'final_wealth_krw':w*1e8,'cum_return_pct':(w-1)*100,'annual_return_pct':(w**(1/years)-1)*100,'mdd_pct':dd*100,'mdd_peak_date':rows[pi]['date'],'mdd_trough_date':rows[ti]['date'],'sharpe_daily':(mean/sd*math.sqrt(252) if sd else 0),'avg_multiplier_pct':sum(mult)/n*100,'riskoff_days_pct':sum(x<.999 for x in mult)/n*100,'min_multiplier_pct':min(mult)*100,'final_vs_kospi_krw':(w-rows[-1]['kospi'])*1e8,'final_vs_orig_krw':(w-rows[-1]['orig'])*1e8}

variants=[eval_variant('baseline', lambda i:1.0)]
# grid: if previous KOSPI window return <= threshold, apply multiplier. Optional 20d shock cap.
for w60 in [40,60,80,120]:
  for th60 in [-0.03,-0.05,-0.07,-0.10,0.0]:
    for risk_mult in [0.0, 20/90, 30/90, 45/90, 60/90, 70/90]:
      def make(w60=w60,th60=th60,risk_mult=risk_mult):
        def f(i):
          j=i-1
          if j<0: return 1.0
          r60=tr(j,w60)
          if r60 is not None and r60<=th60: return risk_mult
          return 1.0
        return f
      variants.append(eval_variant(f'ret{w60}_le_{th60:+.0%}_mult_{risk_mult:.3f}', make()))
# combine 60d/20d shock cap
for th60 in [-0.03,-0.05,-0.07,0.0]:
  for base_mult in [0.0,30/90,45/90,60/90,70/90]:
    for th20 in [-0.06,-0.08,-0.10]:
      for cap_mult in [20/90,30/90,45/90]:
        def make(th60=th60,base_mult=base_mult,th20=th20,cap_mult=cap_mult):
          def f(i):
            j=i-1
            if j<0: return 1.0
            m=1.0
            r60=tr(j,60); r20=tr(j,20)
            if r60 is not None and r60<=th60: m=base_mult
            if r20 is not None and r20<=th20: m=min(m,cap_mult)
            return m
          return f
        variants.append(eval_variant(f'ret60_le_{th60:+.0%}_m{base_mult:.3f}_cap20_{th20:+.0%}_m{cap_mult:.3f}', make()))
# Score: prefer ann>=8 and lower mdd; otherwise include all. Pareto-ish sort by mdd then ann.
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
all_path=OUTDIR/f'overlay_1446_grid_{stamp}.csv'
cols=list(variants[0].keys())
with all_path.open('w', newline='') as f:
    wr=csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(variants)
# candidates with ann>=8, final>KOSPI, mdd improvement
cands=[v for v in variants if v['annual_return_pct']>=8 and v['final_vs_kospi_krw']>0 and v['mdd_pct']>-43.9468]
by_mdd=sorted(cands, key=lambda v:(-v['mdd_pct'],-v['annual_return_pct']))[:15]
by_score=sorted(cands, key=lambda v: (v['annual_return_pct'] / abs(v['mdd_pct'])), reverse=True)[:15]
print('WROTE', all_path, 'n=', len(variants))
print('BEST_MDD_ANN>=8')
for v in by_mdd[:10]: print(v['name'], 'ann',round(v['annual_return_pct'],2),'mdd',round(v['mdd_pct'],2),'final억',round(v['final_wealth_krw']/1e8,3),'avgM',round(v['avg_multiplier_pct'],1),'riskoff',round(v['riskoff_days_pct'],1))
print('BEST_ANN_OVER_MDD')
for v in by_score[:10]: print(v['name'], 'ann',round(v['annual_return_pct'],2),'mdd',round(v['mdd_pct'],2),'final억',round(v['final_wealth_krw']/1e8,3),'avgM',round(v['avg_multiplier_pct'],1),'riskoff',round(v['riskoff_days_pct'],1))
