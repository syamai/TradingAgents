#!/usr/bin/env python3
"""Build interactive HTML dashboard with 100M KRW wealth curves for recent gate-passed strategies."""
from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import pandas as pd
import plotly.graph_objects as go

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _simulate_v2
from tradingagents.hermes.strategy_research_v2 import _preload

ART = Path('/Users/selab/Source/trading-ai/artifacts')
YEAR_CSV = ART / 'recent_passed_calendar_year_returns_20160602_20250630.csv'
SEG_CSV = ART / 'recent_passed_yearly_contiguous_regime_segments_20160602_20250630.csv'
OUT = ART / 'recent_passed_strategy_regime_dashboard.html'
DAILY_CSV = ART / 'recent_passed_100m_wealth_curves_20160602_20250630.csv'
DB = Path.home() / '.tradingagents' / 'hermes' / 'strategies_v2.db'
IDS = [1473, 1446, 1401, 1346, 1345]
CUTOFF = '2025-06-30'
INITIAL_KRW = 100_000_000
REGIME_COLORS = {'상승': '#ef4444', '보합': '#64748b', '하락': '#2563eb'}


def template(fig):
    fig.update_layout(
        template='plotly_white',
        font=dict(family='-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', size=12),
        hovermode='closest',
        margin=dict(l=50, r=30, t=70, b=45),
    )
    return fig


def money_tick(v):
    return f'{v/100_000_000:.2f}억'


def compute_daily_wealth() -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    placeholders = ','.join('?' for _ in IDS)
    rows = conn.execute(
        f'SELECT id,name,spec_json FROM strategies WHERE id IN ({placeholders})', IDS
    ).fetchall()
    by_id = {int(r['id']): r for r in rows}

    raw_tickers = KisHistoryStore().list_tickers()
    tickers, loader, _, _uf = _preload(raw_tickers)
    holdings = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader(tk)
        if df is None or df.empty:
            continue
        d = df['date'].astype(str)
        df = df[d <= CUTOFF].reset_index(drop=True)
        if len(df) < 3:
            continue
        holdings[tk] = df
        d0, d1 = str(df['date'].iloc[0]), str(df['date'].iloc[-1])
        min_d = d0 if min_d is None or d0 < min_d else min_d
        max_d = d1 if max_d is None or d1 > max_d else max_d
    if min_d is None or max_d is None:
        raise RuntimeError('no holdings')

    kospi = fetch_kospi(min_d, max_d)
    all_rows = []
    kospi_ret_ref = None
    for sid in IDS:
        row = by_id[sid]
        spec = json.loads(row['spec_json'])
        ret_frames, act_frames = [], []
        for tk, df in holdings.items():
            trades, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
            idx = cdf['date'].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        port = bt._combine_korea_stock_portfolio(ret_frames, act_frames).sort_index()
        kret = bt._market_daily_returns(kospi, port.index).sort_index()
        if kospi_ret_ref is None:
            kospi_ret_ref = kret
        strategy_wealth = INITIAL_KRW * (1.0 + port).cumprod()
        kospi_wealth = INITIAL_KRW * (1.0 + kret).cumprod()
        active_weight = bt._invested_weight_korea(act_frames).reindex(port.index).fillna(0.0)
        for date in port.index:
            all_rows.append({
                'strategy_id': sid,
                'name': row['name'],
                'date': date,
                'strategy_daily_return': float(port.loc[date]),
                'kospi_daily_return': float(kret.loc[date]),
                'strategy_wealth_krw': float(strategy_wealth.loc[date]),
                'kospi_wealth_krw': float(kospi_wealth.loc[date]),
                'strategy_wealth_억': float(strategy_wealth.loc[date] / 100_000_000),
                'kospi_wealth_억': float(kospi_wealth.loc[date] / 100_000_000),
                'excess_wealth_krw': float(strategy_wealth.loc[date] - kospi_wealth.loc[date]),
                'avg_active_weight_pct': float(active_weight.loc[date] * 100),
            })
    df = pd.DataFrame(all_rows)
    df.to_csv(DAILY_CSV, index=False, encoding='utf-8')
    return df


def main():
    y = pd.read_csv(YEAR_CSV)
    s = pd.read_csv(SEG_CSV)
    d = compute_daily_wealth()
    y['strategy_label'] = '#' + y.strategy_id.astype(str) + ' ' + y.name
    s['strategy_label'] = '#' + s.strategy_id.astype(str) + ' ' + s.name
    d['strategy_label'] = '#' + d.strategy_id.astype(str) + ' ' + d.name
    s['start_dt'] = pd.to_datetime(s['start'])
    s['end_dt'] = pd.to_datetime(s['end'])
    s['mid_dt'] = s['start_dt'] + (s['end_dt'] - s['start_dt']) / 2
    d['date_dt'] = pd.to_datetime(d['date'])

    # 0) 100M wealth curve per strategy dropdown.
    fig_wealth = go.Figure()
    labels = []
    for i, (sid, grp) in enumerate(d.groupby('strategy_id')):
        label = grp['strategy_label'].iloc[0]
        labels.append(label)
        visible = i == 0
        final_s = grp['strategy_wealth_krw'].iloc[-1]
        final_k = grp['kospi_wealth_krw'].iloc[-1]
        fig_wealth.add_trace(go.Scatter(
            x=grp['date_dt'], y=grp['strategy_wealth_krw'], mode='lines', name='전략 1억원', visible=visible,
            line=dict(color='#10b981', width=2.4),
            customdata=grp[['strategy_wealth_억','strategy_daily_return','avg_active_weight_pct']].to_numpy(),
            hovertemplate='<b>'+label+'</b><br>%{x|%Y-%m-%d}<br>전략 자산=%{customdata[0]:.3f}억 원<br>일수익률=%{customdata[1]:.3%}<br>노출=%{customdata[2]:.1f}%<extra></extra>'
        ))
        fig_wealth.add_trace(go.Scatter(
            x=grp['date_dt'], y=grp['kospi_wealth_krw'], mode='lines', name='KOSPI 지수투자 1억원', visible=visible,
            line=dict(color='#111827', width=2.4, dash='dash'),
            customdata=grp[['kospi_wealth_억','kospi_daily_return']].to_numpy(),
            hovertemplate='<b>KOSPI 지수투자</b><br>%{x|%Y-%m-%d}<br>자산=%{customdata[0]:.3f}억 원<br>일수익률=%{customdata[1]:.3%}<extra></extra>'
        ))
        fig_wealth.add_trace(go.Scatter(
            x=grp['date_dt'], y=grp['excess_wealth_krw'], mode='lines', name='전략-KOSPI 자산차이', visible=False,
            line=dict(color='#f59e0b', width=1.8),
            hovertemplate='<b>자산차이</b><br>%{x|%Y-%m-%d}<br>%{y:,.0f}원<extra></extra>'
        ))
    buttons = []
    for i, label in enumerate(labels):
        v = [False] * len(fig_wealth.data)
        v[3*i] = True
        v[3*i+1] = True
        # third trace hidden by default; user can click legend if desired after selecting strategy
        v[3*i+2] = False
        buttons.append(dict(label=label, method='update', args=[{'visible': v}, {'title': f'1억원 투자 시 자산 변화: {label} vs KOSPI 지수투자'}]))
    fig_wealth.update_layout(
        title=f'1억원 투자 시 자산 변화: {labels[0]} vs KOSPI 지수투자',
        xaxis_title='날짜', yaxis_title='자산가치 (원)',
        updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.18, xanchor='left', yanchor='top')],
    )
    fig_wealth.update_yaxes(tickformat=',.0f')
    template(fig_wealth)

    # 1) Annual calendar returns: grouped bars, all strategies + KOSPI reference.
    fig_year = go.Figure()
    for sid, grp in y.groupby('strategy_id'):
        label = grp['strategy_label'].iloc[0]
        fig_year.add_trace(go.Bar(
            x=grp['year'], y=grp['strategy_calendar_return_pct'], name=label,
            customdata=grp[['strategy_annualized_return_pct','avg_active_weight_pct']].to_numpy(),
            hovertemplate='<b>%{fullData.name}</b><br>연도=%{x}<br>연간 누적=%{y:.2f}%<br>연율화=%{customdata[0]:.2f}%<br>평균 노출=%{customdata[1]:.2f}%<extra></extra>'
        ))
    kg = y.drop_duplicates('year').sort_values('year')
    fig_year.add_trace(go.Scatter(
        x=kg['year'], y=kg['kospi_calendar_return_pct'], mode='lines+markers', name='KOSPI 연간 수익률',
        line=dict(color='black', width=3),
        hovertemplate='<b>KOSPI</b><br>연도=%{x}<br>연간 누적=%{y:.2f}%<extra></extra>'
    ))
    fig_year.update_layout(title='연도별 실제 연간 수익률: 전략 vs KOSPI', barmode='group', yaxis_title='연간 누적수익률 (%)', xaxis_title='연도')
    template(fig_year)

    # 2) Excess returns heatmap.
    yh = y.copy()
    yh['excess_pct'] = yh['strategy_calendar_return_pct'] - yh['kospi_calendar_return_pct']
    mat = yh.pivot(index='strategy_label', columns='year', values='excess_pct').sort_index()
    fig_heat = go.Figure(go.Heatmap(
        z=mat.values, x=mat.columns.astype(str), y=mat.index,
        colorscale='RdBu', reversescale=True, zmid=0,
        colorbar=dict(title='초과수익 %p'),
        hovertemplate='<b>%{y}</b><br>연도=%{x}<br>KOSPI 대비 초과=%{z:.2f}%p<extra></extra>'
    ))
    fig_heat.update_layout(title='연도별 KOSPI 대비 초과수익 히트맵', xaxis_title='연도', yaxis_title='전략', height=430)
    template(fig_heat)

    # 3) Per-strategy annual bar dropdown.
    fig_one = go.Figure()
    labels_one=[]
    for i, (sid, grp) in enumerate(y.groupby('strategy_id')):
        label = grp['strategy_label'].iloc[0]
        labels_one.append(label)
        visible = i == 0
        fig_one.add_trace(go.Bar(x=grp['year'], y=grp['strategy_calendar_return_pct'], name='전략', visible=visible,
            marker_color='#10b981', customdata=grp[['strategy_annualized_return_pct','avg_active_weight_pct']].to_numpy(),
            hovertemplate='<b>'+label+'</b><br>연도=%{x}<br>전략 연간 누적=%{y:.2f}%<br>연율화=%{customdata[0]:.2f}%<br>평균 노출=%{customdata[1]:.2f}%<extra></extra>'))
        fig_one.add_trace(go.Bar(x=grp['year'], y=grp['kospi_calendar_return_pct'], name='KOSPI', visible=visible,
            marker_color='#111827', customdata=grp[['kospi_annualized_return_pct']].to_numpy(),
            hovertemplate='<b>KOSPI</b><br>연도=%{x}<br>KOSPI 연간 누적=%{y:.2f}%<br>연율화=%{customdata[0]:.2f}%<extra></extra>'))
    buttons=[]
    for i, label in enumerate(labels_one):
        v=[False]*len(fig_one.data); v[2*i]=True; v[2*i+1]=True
        buttons.append(dict(label=label, method='update', args=[{'visible':v}, {'title':f'전략별 연간 수익률: {label}'}]))
    fig_one.update_layout(title=f'전략별 연간 수익률: {labels_one[0]}', barmode='group', yaxis_title='연간 누적수익률 (%)', xaxis_title='연도',
                          updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.18, xanchor='left', yanchor='top')])
    template(fig_one)

    # 4) Contiguous regime segment return chart per strategy.
    fig_seg = go.Figure()
    labels_seg=[]
    traces_per_strategy=3
    for i, (sid, grp_all) in enumerate(s.groupby('strategy_id')):
        label=grp_all['strategy_label'].iloc[0]; labels_seg.append(label)
        for regime in ['상승','보합','하락']:
            grp=grp_all[grp_all.regime==regime].sort_values('start_dt')
            visible=i==0
            fig_seg.add_trace(go.Bar(
                x=grp['mid_dt'], y=grp['strategy_segment_cum_return_pct'], name=regime,
                visible=visible, marker_color=REGIME_COLORS[regime],
                width=(grp['trading_days'].clip(lower=1)*24*60*60*1000).astype(float),
                customdata=grp[['start','end','trading_days','kospi_segment_cum_return_pct','avg_active_weight_pct']].to_numpy(),
                hovertemplate='<b>'+label+'</b><br>regime='+regime+'<br>%{customdata[0]}~%{customdata[1]} (%{customdata[2]}거래일)<br>전략 구간 누적=%{y:.2f}%<br>KOSPI 구간 누적=%{customdata[3]:.2f}%<br>평균 노출=%{customdata[4]:.2f}%<extra></extra>'
            ))
    buttons=[]
    for i,label in enumerate(labels_seg):
        v=[False]*len(fig_seg.data)
        for j in range(traces_per_strategy): v[i*traces_per_strategy+j]=True
        buttons.append(dict(label=label, method='update', args=[{'visible':v}, {'title':f'연속 regime 구간별 누적수익률: {label}'}]))
    fig_seg.update_layout(title=f'연속 regime 구간별 누적수익률: {labels_seg[0]}', xaxis_title='구간 중간 날짜', yaxis_title='구간 누적수익률 (%)', barmode='overlay',
                          updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.18, xanchor='left', yanchor='top')])
    template(fig_seg)

    # 5) Segment timeline.
    fig_tl = go.Figure()
    ymap = {lab:i for i,lab in enumerate(sorted(s.strategy_label.unique()))}
    for regime in ['상승','보합','하락']:
        xs=[]; ys=[]; custom=[]
        for _,r in s[s.regime==regime].iterrows():
            yy=ymap[r.strategy_label]
            xs += [r.start_dt, r.end_dt, None]
            ys += [yy, yy, None]
            custom += [[r.strategy_label,r.start,r.end,r.trading_days,r.strategy_segment_cum_return_pct,r.kospi_segment_cum_return_pct],
                       [r.strategy_label,r.start,r.end,r.trading_days,r.strategy_segment_cum_return_pct,r.kospi_segment_cum_return_pct],
                       [None,None,None,None,None,None]]
        fig_tl.add_trace(go.Scatter(x=xs,y=ys,mode='lines',name=regime,line=dict(color=REGIME_COLORS[regime],width=8),
            customdata=custom, hovertemplate='<b>%{customdata[0]}</b><br>regime='+regime+'<br>%{customdata[1]}~%{customdata[2]} (%{customdata[3]}거래일)<br>전략 구간 누적=%{customdata[4]:.2f}%<br>KOSPI 구간 누적=%{customdata[5]:.2f}%<extra></extra>'))
    fig_tl.update_yaxes(tickmode='array', tickvals=list(ymap.values()), ticktext=list(ymap.keys()))
    fig_tl.update_layout(title='전략별 연속 regime 타임라인', xaxis_title='날짜', yaxis_title='전략', height=430)
    template(fig_tl)

    figs = [fig_wealth, fig_year, fig_heat, fig_one, fig_seg, fig_tl]
    html_parts=[]
    for idx,fig in enumerate(figs):
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn' if idx==0 else False, div_id=f'fig{idx}'))

    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
      header { padding: 28px 36px; background: linear-gradient(135deg,#0f172a,#334155); color: white; }
      header h1 { margin: 0 0 8px 0; font-size: 26px; }
      header p { margin: 4px 0; opacity: 0.88; }
      section { margin: 22px 28px; padding: 18px; background: white; border-radius: 14px; box-shadow: 0 1px 6px rgba(15,23,42,.08); }
      .note { font-size: 13px; color: #475569; line-height: 1.55; }
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; color:white; font-size:12px; margin-right:5px; }
      .up { background:#ef4444; } .flat { background:#64748b; } .down { background:#2563eb; }
      code { background:#f1f5f9; padding:2px 4px; border-radius:4px; }
    </style>
    """
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>Recent Passed Strategy Regime Dashboard</title>{style}</head><body>
<header>
  <h1>새 최종 통과 후보 — 1억원 자산 변화 / 연도 / Regime 결과</h1>
  <p>기간: 2016-06-02 ~ 2025-06-30 · 2025-06-30 이후 제외 · 전략 포트폴리오: 주식 90%, 현금 10%, 30종목, 종목당 최대 5%</p>
  <p>지수투자 비교: 같은 시작일에 KOSPI에 1억원 100% 투자한 buy-and-hold 기준</p>
  <p>Regime 기준: KOSPI 60거래일 trailing 수익률 · <span class='pill up'>상승 ≥ +5%</span><span class='pill flat'>보합 -5%~+5%</span><span class='pill down'>하락 ≤ -5%</span></p>
</header>
<section class='note'>
  <b>읽는 법</b>: “1억원 자산 변화”는 전략 일별 포트폴리오 수익률을 시간 순서대로 복리 누적한 자산가치입니다. KOSPI 지수투자는 동일 기간 KOSPI 일별 수익률을 100% 노출로 복리 누적한 기준입니다. “연간 수익률”은 해당 연도 전체 일별수익률을 시간 순서대로 복리 누적한 실제 calendar-year return입니다.
</section>
<section>{html_parts[0]}</section>
<section>{html_parts[1]}</section>
<section>{html_parts[2]}</section>
<section>{html_parts[3]}</section>
<section>{html_parts[4]}</section>
<section>{html_parts[5]}</section>
<section class='note'>
  자산곡선 CSV: <code>{DAILY_CSV}</code><br>
  연간 CSV: <code>{YEAR_CSV}</code><br>
  구간 CSV: <code>{SEG_CSV}</code>
</section>
</body></html>"""
    OUT.write_text(body, encoding='utf-8')
    print(json.dumps({'html': str(OUT), 'daily_csv': str(DAILY_CSV), 'size_bytes': OUT.stat().st_size, 'daily_rows': len(d)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
