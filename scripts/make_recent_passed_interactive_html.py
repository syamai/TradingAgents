#!/usr/bin/env python3
"""Build an interactive HTML dashboard for recent gate-passed strategy regime results."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ART = Path('/Users/selab/Source/trading-ai/artifacts')
YEAR_CSV = ART / 'recent_passed_calendar_year_returns_20160602_20250630.csv'
SEG_CSV = ART / 'recent_passed_yearly_contiguous_regime_segments_20160602_20250630.csv'
OUT = ART / 'recent_passed_strategy_regime_dashboard.html'

REGIME_COLORS = {'상승': '#ef4444', '보합': '#64748b', '하락': '#2563eb'}


def template(fig):
    fig.update_layout(
        template='plotly_white',
        font=dict(family='-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', size=12),
        hovermode='closest',
        margin=dict(l=50, r=30, t=70, b=45),
    )
    return fig


def main():
    y = pd.read_csv(YEAR_CSV)
    s = pd.read_csv(SEG_CSV)
    y['strategy_label'] = '#' + y.strategy_id.astype(str) + ' ' + y.name
    s['strategy_label'] = '#' + s.strategy_id.astype(str) + ' ' + s.name
    s['start_dt'] = pd.to_datetime(s['start'])
    s['end_dt'] = pd.to_datetime(s['end'])
    s['mid_dt'] = s['start_dt'] + (s['end_dt'] - s['start_dt']) / 2
    strategies = y[['strategy_id','strategy_label']].drop_duplicates().sort_values('strategy_id').to_dict('records')

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

    # 2) Excess returns heatmap (strategy - KOSPI), calendar year.
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

    # 3) Per-strategy annual bar (dropdown).
    fig_one = go.Figure()
    vis = []
    for i, (sid, grp) in enumerate(y.groupby('strategy_id')):
        label = grp['strategy_label'].iloc[0]
        visible = i == 0
        fig_one.add_trace(go.Bar(x=grp['year'], y=grp['strategy_calendar_return_pct'], name='전략', visible=visible,
            marker_color='#10b981', customdata=grp[['strategy_annualized_return_pct','avg_active_weight_pct']].to_numpy(),
            hovertemplate='<b>'+label+'</b><br>연도=%{x}<br>전략 연간 누적=%{y:.2f}%<br>연율화=%{customdata[0]:.2f}%<br>평균 노출=%{customdata[1]:.2f}%<extra></extra>'))
        fig_one.add_trace(go.Bar(x=grp['year'], y=grp['kospi_calendar_return_pct'], name='KOSPI', visible=visible,
            marker_color='#111827', customdata=grp[['kospi_annualized_return_pct']].to_numpy(),
            hovertemplate='<b>KOSPI</b><br>연도=%{x}<br>KOSPI 연간 누적=%{y:.2f}%<br>연율화=%{customdata[0]:.2f}%<extra></extra>'))
        vis.append(label)
    buttons=[]
    for i, label in enumerate(vis):
        v=[False]*len(fig_one.data); v[2*i]=True; v[2*i+1]=True
        buttons.append(dict(label=label, method='update', args=[{'visible':v}, {'title':f'전략별 연간 수익률: {label}'}]))
    fig_one.update_layout(title=f'전략별 연간 수익률: {vis[0]}', barmode='group', yaxis_title='연간 누적수익률 (%)', xaxis_title='연도',
                          updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.18, xanchor='left', yanchor='top')])
    template(fig_one)

    # 4) Contiguous regime segment return chart per strategy (dropdown) + regime colored bars.
    fig_seg = go.Figure()
    labels=[]
    traces_per_strategy=3  # one per regime for color grouping
    for i, (sid, grp_all) in enumerate(s.groupby('strategy_id')):
        label=grp_all['strategy_label'].iloc[0]; labels.append(label)
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
    for i,label in enumerate(labels):
        v=[False]*len(fig_seg.data)
        for j in range(traces_per_strategy): v[i*traces_per_strategy+j]=True
        buttons.append(dict(label=label, method='update', args=[{'visible':v}, {'title':f'연속 regime 구간별 누적수익률: {label}'}]))
    fig_seg.update_layout(title=f'연속 regime 구간별 누적수익률: {labels[0]}', xaxis_title='구간 중간 날짜', yaxis_title='구간 누적수익률 (%)', barmode='overlay',
                          updatemenus=[dict(active=0, buttons=buttons, x=0, y=1.18, xanchor='left', yanchor='top')])
    template(fig_seg)

    # 5) Segment timeline: all strategies as rows, color by KOSPI regime. Use line segments.
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

    figs = [fig_year, fig_heat, fig_one, fig_seg, fig_tl]
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
      table { border-collapse: collapse; font-size: 13px; }
      td, th { border: 1px solid #e2e8f0; padding: 6px 8px; }
    </style>
    """
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>Recent Passed Strategy Regime Dashboard</title>{style}</head><body>
<header>
  <h1>새 최종 통과 후보 — 연도/Regime별 인터랙티브 결과</h1>
  <p>기간: 2016-06-02 ~ 2025-06-30 · 2025-06-30 이후 제외 · 포트폴리오: 주식 90%, 현금 10%, 30종목, 종목당 최대 5%</p>
  <p>Regime 기준: KOSPI 60거래일 trailing 수익률 · <span class='pill up'>상승 ≥ +5%</span><span class='pill flat'>보합 -5%~+5%</span><span class='pill down'>하락 ≤ -5%</span></p>
</header>
<section class='note'>
  <b>읽는 법</b>: “연간 수익률”은 해당 연도 전체 일별수익률을 시간 순서대로 복리 누적한 실제 calendar-year return입니다. “구간별 수익률”은 연도 안에서 KOSPI regime이 바뀔 때마다 자른 연속 구간의 누적수익률입니다. 1~2거래일짜리 구간의 연율화는 과장되므로 구간 그래프는 누적수익률 중심으로 보세요.
</section>
<section>{html_parts[0]}</section>
<section>{html_parts[1]}</section>
<section>{html_parts[2]}</section>
<section>{html_parts[3]}</section>
<section>{html_parts[4]}</section>
<section class='note'>
  원본 CSV: <code>{YEAR_CSV}</code><br>
  구간 CSV: <code>{SEG_CSV}</code>
</section>
</body></html>"""
    OUT.write_text(body, encoding='utf-8')
    print(json.dumps({'html': str(OUT), 'size_bytes': OUT.stat().st_size, 'strategies': len(strategies), 'year_rows': len(y), 'segment_rows': len(s)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
