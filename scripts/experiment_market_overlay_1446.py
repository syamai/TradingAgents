
import csv, math, json, statistics
from pathlib import Path
from datetime import datetime

IN = Path('/Users/selab/Source/trading-ai/artifacts/recent_passed_100m_wealth_curves_20160602_20250630.csv')
OUTDIR = Path('/Users/selab/Source/trading-ai/artifacts/market_overlay_experiments')
OUTDIR.mkdir(parents=True, exist_ok=True)
rows=[]
with IN.open() as f:
    for r in csv.DictReader(f):
        if r['strategy_id']=='1446':
            rows.append({
                'date': r['date'],
                'strat_ret': float(r['strategy_daily_return']),
                'kospi_ret': float(r['kospi_daily_return']),
                'active_w': float(r['avg_active_weight_pct'])/100.0,
            })
if not rows:
    raise SystemExit('no rows')
# reconstruct KOSPI and original strategy wealth from daily returns to avoid CSV rounded wealth issues
for key in ['kospi','orig']:
    w=1.0
    for i,r in enumerate(rows):
        ret = r['kospi_ret'] if key=='kospi' else r['strat_ret']
        w *= (1+ret)
        r[key] = w

def trailing_ret(i, arr, window):
    if i < window: return None
    base = arr[i-window]
    return arr[i]/base - 1 if base else None

def ma(i, arr, window):
    if i+1 < window: return None
    return sum(arr[i-window+1:i+1])/window

def vol_ann(i, rets, window):
    if i+1 < window: return None
    xs=rets[i-window+1:i+1]
    if len(xs)<2: return None
    m=sum(xs)/len(xs)
    var=sum((x-m)**2 for x in xs)/(len(xs)-1)
    return math.sqrt(var)*math.sqrt(252)

def mdd(vals):
    peak=vals[0]; peak_i=0; worst=0; wi=0; wpi=0
    for i,v in enumerate(vals):
        if v>peak:
            peak=v; peak_i=i
        dd=v/peak-1
        if dd<worst:
            worst=dd; wi=i; wpi=peak_i
    return worst, wpi, wi

def metrics(name, mult_fn):
    wealth=[]; w=1.0; mults=[]
    for i,r in enumerate(rows):
        mult=mult_fn(i,r)
        mult=max(0.0, min(1.0, mult))
        mults.append(mult)
        # overlay scales the existing 90/10 portfolio's stock-risk return.
        w *= (1 + r['strat_ret']*mult)
        wealth.append(w)
    n=len(rows); years=n/252
    cum=w-1
    ann=w**(1/years)-1
    dd, pi, ti=mdd(wealth)
    # daily sharpe on scaled returns
    scaled=[rows[i]['strat_ret']*mults[i] for i in range(n)]
    mean=sum(scaled)/n
    sd=statistics.stdev(scaled) if n>1 else 0
    sharpe=mean/sd*math.sqrt(252) if sd else 0
    return {
        'name': name,
        'final_wealth_krw': w*100_000_000,
        'cum_return_pct': cum*100,
        'annual_return_pct': ann*100,
        'mdd_pct': dd*100,
        'mdd_peak_date': rows[pi]['date'],
        'mdd_trough_date': rows[ti]['date'],
        'sharpe_daily': sharpe,
        'avg_multiplier_pct': sum(mults)/n*100,
        'riskoff_days_pct': sum(1 for x in mults if x < 0.999)/n*100,
        'min_multiplier_pct': min(mults)*100,
        'final_vs_orig_krw': (w - rows[-1]['orig'])*100_000_000,
        'final_vs_kospi_krw': (w - rows[-1]['kospi'])*100_000_000,
    }

kospi=[r['kospi'] for r in rows]
krets=[r['kospi_ret'] for r in rows]

variants=[]
# baseline
variants.append(metrics('baseline_no_overlay', lambda i,r: 1.0))
# Hard filters / scaling: signal uses same-day close info but scales next day? To avoid lookahead, use previous row signal.
def prev_signal(i, fn, default=True):
    if i==0: return default
    return fn(i-1)

variants.append(metrics('hard_kospi_close_gt_200ma', lambda i,r: 1.0 if prev_signal(i, lambda j: ma(j,kospi,200) is not None and kospi[j] > ma(j,kospi,200), True) else 0.0))
variants.append(metrics('scale_200ma_90_to_45', lambda i,r: 1.0 if prev_signal(i, lambda j: ma(j,kospi,200) is None or kospi[j] > ma(j,kospi,200), True) else 0.5))
variants.append(metrics('hard_kospi_60d_ret_gt_minus5', lambda i,r: 1.0 if prev_signal(i, lambda j: trailing_ret(j,kospi,60) is None or trailing_ret(j,kospi,60) > -0.05, True) else 0.0))
variants.append(metrics('scale_60d_minus5_90_to_45', lambda i,r: 1.0 if prev_signal(i, lambda j: trailing_ret(j,kospi,60) is None or trailing_ret(j,kospi,60) > -0.05, True) else 0.5))
variants.append(metrics('hard_20d_ret_gt_minus8', lambda i,r: 1.0 if prev_signal(i, lambda j: trailing_ret(j,kospi,20) is None or trailing_ret(j,kospi,20) > -0.08, True) else 0.0))
variants.append(metrics('scale_20d_minus8_90_to_30', lambda i,r: 1.0 if prev_signal(i, lambda j: trailing_ret(j,kospi,20) is None or trailing_ret(j,kospi,20) > -0.08, True) else (30/90)))
variants.append(metrics('hard_60d_gt_m5_and_20d_gt_m8', lambda i,r: 1.0 if prev_signal(i, lambda j: (trailing_ret(j,kospi,60) is None or trailing_ret(j,kospi,60) > -0.05) and (trailing_ret(j,kospi,20) is None or trailing_ret(j,kospi,20) > -0.08), True) else 0.0))
variants.append(metrics('scale_60d_m5_and_20d_m8_to_30', lambda i,r: 1.0 if prev_signal(i, lambda j: (trailing_ret(j,kospi,60) is None or trailing_ret(j,kospi,60) > -0.05) and (trailing_ret(j,kospi,20) is None or trailing_ret(j,kospi,20) > -0.08), True) else (30/90)))
variants.append(metrics('vol20_gt25_and_60d_lt0_to45', lambda i,r: 0.5 if prev_signal(i, lambda j: (vol_ann(j,krets,20) is not None and vol_ann(j,krets,20) > 0.25 and trailing_ret(j,kospi,60) is not None and trailing_ret(j,kospi,60) < 0), False) else 1.0))
# step scaling based on 60d ret and 20d shock cap
def step_scaling(i,r):
    def sig(j):
        r60=trailing_ret(j,kospi,60)
        r20=trailing_ret(j,kospi,20)
        mult=1.0
        if r60 is not None:
            if r60 <= -0.10: mult=20/90
            elif r60 <= -0.05: mult=45/90
            elif r60 <= 0: mult=70/90
            else: mult=1.0
        if r20 is not None and r20 <= -0.08:
            mult=min(mult, 30/90)
        return mult
    return sig(i-1) if i>0 else 1.0
variants.append(metrics('step_60d_90_70_45_20_plus_20d_cap30', step_scaling))
# alternative: less aggressive step no 0.2 floor but 0.5 min
variants.append(metrics('step_60d_90_70_45_min45_plus_20d_cap45', lambda i,r: (lambda j: 1.0 if j<0 else (min((1.0 if (trailing_ret(j,kospi,60) is None or trailing_ret(j,kospi,60)>0) else (70/90 if trailing_ret(j,kospi,60)>-0.05 else 0.5)), 0.5 if (trailing_ret(j,kospi,20) is not None and trailing_ret(j,kospi,20)<=-0.08) else 1.0)))(i-1)))

variants=sorted(variants, key=lambda x: (x['mdd_pct'], -x['annual_return_pct']), reverse=True)  # less negative mdd first
stamp=datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
json_path=OUTDIR/f'overlay_1446_{stamp}.json'
csv_path=OUTDIR/f'overlay_1446_{stamp}.csv'
with json_path.open('w') as f: json.dump({'generated_at':stamp,'source':str(IN),'assumption':'daily multiplier applied to existing portfolio daily return; signals lagged 1 day to avoid lookahead','results':variants}, f, indent=2, ensure_ascii=False)
with csv_path.open('w', newline='') as f:
    cols=list(variants[0].keys())
    w=csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(variants)
print('WROTE', json_path)
print('WROTE', csv_path)
print('TOP_BY_MDD')
for v in variants[:8]:
    print(v['name'], 'ann', round(v['annual_return_pct'],2), 'mdd', round(v['mdd_pct'],2), 'final억', round(v['final_wealth_krw']/1e8,3), 'avgMult', round(v['avg_multiplier_pct'],1), 'riskoff%', round(v['riskoff_days_pct'],1), 'vsKospi억', round(v['final_vs_kospi_krw']/1e8,3))
print('BASELINE')
base=[v for v in variants if v['name']=='baseline_no_overlay'][0]
print(base)
