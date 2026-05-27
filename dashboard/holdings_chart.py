"""KIS holdings 데이터 → Plotly 3-패널 인터랙티브 차트.

DRY 원칙: ``make_figure`` 를 CLI(``build_holdings_chart.py``), Jupyter
노트북, Streamlit 대시보드(``app.py``) 모두에서 import.

3 패널 구성:
  Row 1 — 종가 line + 가격 변동률 bar (secondary y-axis)
  Row 2 — 11 주체 누적 보유량(cum_qty) line, legendgroup 으로 카테고리 토글
  Row 3 — 10 sub 보유 비율(pct) 100% 누적 영역 (외국인 통합은 제외 → sub 합 = 100%)

색상 정책 — 카테고리별 색상 그룹:
  외국인(red 계열): 통합/등록/비등록
  기관(blue/purple 계열): pension / private_equity / investment_trust /
                          securities / bank / insurance
  개인: green
  기타법인: yellow/khaki
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.kis_holdings import (
    ALL_SUBJECTS, SUBS_10,
)

# --- 표시 순서 (11 주체) -----------------------------------------------------
# 외국인 통합 → 등록 → 비등록 → 기관 6 sub → 개인 → 기타법인
SUBJECTS_ORDER: tuple[str, ...] = (
    "foreign", "foreign_registered", "foreign_unregistered",
    "pension", "private_equity", "investment_trust", "securities",
    "bank", "insurance",
    "retail", "other_corp",
)

# Sanity: SUBJECTS_ORDER 는 ALL_SUBJECTS 의 순열이어야 한다.
assert set(SUBJECTS_ORDER) == set(ALL_SUBJECTS), (
    f"SUBJECTS_ORDER mismatch: {set(SUBJECTS_ORDER) ^ set(ALL_SUBJECTS)}"
)

SUBJECT_LABELS: dict[str, str] = {
    "foreign": "외국인(통합)",
    "foreign_registered": "외국인(등록)",
    "foreign_unregistered": "외국인(비등록)",
    "pension": "연기금",
    "private_equity": "사모펀드",
    "investment_trust": "투신",
    "securities": "금융투자",
    "bank": "은행",
    "insurance": "보험",
    "retail": "개인",
    "other_corp": "기타법인",
}

SUBJECT_COLORS: dict[str, str] = {
    # 외국인 — red 계열 (통합은 진한 빨강, sub 는 옅게)
    "foreign": "#d62728",
    "foreign_registered": "#ff7f0e",
    "foreign_unregistered": "#ffbb78",
    # 기관 — blue/purple 계열
    "pension": "#1f3b73",
    "private_equity": "#8c6bb1",
    "investment_trust": "#17becf",
    "securities": "#1f77b4",
    "bank": "#7f8fa6",
    "insurance": "#2c3e50",
    # 개인 — green
    "retail": "#2ca02c",
    # 기타법인 — yellow/khaki
    "other_corp": "#bcbd22",
}

SUBJECT_LEGENDGROUP: dict[str, str] = {
    "foreign": "외국인", "foreign_registered": "외국인", "foreign_unregistered": "외국인",
    "pension": "기관", "private_equity": "기관", "investment_trust": "기관",
    "securities": "기관", "bank": "기관", "insurance": "기관",
    "retail": "개인", "other_corp": "기타법인",
}

# Sanity: 라벨·색상·legendgroup 11개 모두 정의되어 있어야 한다.
assert set(SUBJECT_LABELS) == set(SUBJECTS_ORDER)
assert set(SUBJECT_COLORS) == set(SUBJECTS_ORDER)
assert set(SUBJECT_LEGENDGROUP) == set(SUBJECTS_ORDER)


# --- 데이터 로드 -------------------------------------------------------------

def load_holdings(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    store: Optional[KisHistoryStore] = None,
) -> tuple[pd.DataFrame, dict]:
    """holdings derived 테이블 + ticker 메타 dict 반환.

    ``store`` 미지정 시 디폴트 경로(``~/.tradingagents/kis_history``).
    메타는 ``{"ticker", "company_name", "market", "updated_at"}`` 또는 빈 dict.
    """
    store = store or KisHistoryStore()
    df = store.read(ticker, "holdings", start_date=start, end_date=end)
    meta = store.get_ticker_metadata(ticker) or {}
    return df, meta


# --- Figure 생성 -------------------------------------------------------------

def _net_col(s: str) -> str:
    return f"{s}_net_qty"


def _cum_col(s: str) -> str:
    return f"{s}_cum_qty"


def _pct_col(s: str) -> str:
    return f"{s}_pct"


PCT_MODES = ("cumulative", "daily")


def make_figure(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    visible_subjects: Optional[Iterable[str]] = None,
    pct_mode: str = "cumulative",
) -> go.Figure:
    """3-패널 figure 생성. 빈 DataFrame 도 안전(빈 figure 반환).

    ``visible_subjects`` 미지정 시 모두 표시. 일부만 지정하면 나머지는
    ``legendonly`` 로 숨김(범례 클릭으로 재표시 가능).

    ``pct_mode``:
      - ``"cumulative"`` (디폴트): row 3 = ``|cum_qty|`` / ``sum(|10 sub cum_qty|)``
        × 100. 첫 거래일부터 누적된 매매 영향력 비중.
      - ``"daily"``: row 3 = ``|net_qty|`` / ``sum(|10 sub net_qty|)`` × 100.
        그날 일별 순매수 절댓값 비중. 0 거래일은 분모 0 → 0%.
    """
    if pct_mode not in PCT_MODES:
        raise ValueError(f"pct_mode must be one of {PCT_MODES}, got {pct_mode!r}")

    row3_title = {
        "cumulative": "주체별 보유 비율 — 누적 (10 sub, 합=100%)",
        "daily": "주체별 보유 비율 — 일별 (10 sub, 합=100%)",
    }[pct_mode]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.40, 0.35, 0.25],
        specs=[
            [{"secondary_y": True}],   # row 1: 종가 + 가격변동률
            [{"secondary_y": False}],  # row 2: cum_qty
            [{"secondary_y": False}],  # row 3: pct 100% stack
        ],
        subplot_titles=("종가 / 가격변동률", "주체별 누적 보유량", row3_title),
    )

    if df.empty:
        fig.update_layout(
            title=title or "(데이터 없음)",
            height=900,
            annotations=[dict(
                text="holdings 데이터가 없습니다.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16),
            )],
        )
        return fig

    visible_set = set(visible_subjects) if visible_subjects is not None else None

    def _vis(s: str):
        if visible_set is None:
            return True
        return True if s in visible_set else "legendonly"

    x = df["date"]

    # === Row 1 — 종가 (left y) + 가격변동률 bar (right y) ===
    # 변동률 bar 를 먼저(아래) → 종가 line 을 나중(위). bar 는 반투명(0.30)
    # 으로 깔고 line 은 진한 인디고 + 두께 2.4 로 메인 시그널이 묻히지 않게.
    pct = df["price_change_pct"]
    bar_colors = ["#e74c3c" if v >= 0 else "#3498db" for v in pct]
    fig.add_trace(
        go.Bar(
            x=x, y=pct,
            name="가격 변동률 (%)",
            marker_color=bar_colors,
            opacity=0.30,
            hovertemplate="%{x|%Y-%m-%d}<br>변동: %{y:+.2f}%<extra></extra>",
            legendgroup="가격", legendgrouptitle_text="가격",
        ),
        row=1, col=1, secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["close"],
            name="종가", mode="lines",
            line=dict(color="#1a237e", width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>종가: %{y:,.0f}원<extra></extra>",
            legendgroup="가격",
        ),
        row=1, col=1, secondary_y=False,
    )

    # === Row 2 — 11 주체 cum_qty 라인 ===
    for s in SUBJECTS_ORDER:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[_cum_col(s)],
                name=SUBJECT_LABELS[s],
                mode="lines",
                line=dict(color=SUBJECT_COLORS[s], width=1.6),
                legendgroup=SUBJECT_LEGENDGROUP[s],
                legendgrouptitle_text=SUBJECT_LEGENDGROUP[s],
                visible=_vis(s),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    + SUBJECT_LABELS[s]
                    + "<br>누적: %{y:,.0f}주<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    # === Row 3 — 10 sub pct 100% 누적영역 (외국인 통합 제외) ===
    # cumulative: 저장된 _pct 컬럼 그대로. daily: net_qty 절댓값으로 즉석 계산.
    if pct_mode == "cumulative":
        pct_series = {s: df[_pct_col(s)] for s in SUBS_10}
    else:  # daily
        abs_net = {s: df[_net_col(s)].abs() for s in SUBS_10}
        denom = sum(abs_net.values())
        safe_denom = denom.where(denom != 0, other=1)
        pct_series = {
            s: (abs_net[s] / safe_denom * 100).round(4) for s in SUBS_10
        }

    for s in SUBS_10:
        fig.add_trace(
            go.Scatter(
                x=x, y=pct_series[s],
                name=SUBJECT_LABELS[s] + " (비중)",
                mode="lines",
                line=dict(width=0.5, color=SUBJECT_COLORS[s]),
                stackgroup="pct",
                groupnorm="percent",
                fillcolor=SUBJECT_COLORS[s],
                legendgroup=SUBJECT_LEGENDGROUP[s],
                showlegend=False,  # Row 2 범례와 토글 공유
                visible=_vis(s),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    + SUBJECT_LABELS[s]
                    + ": %{y:.2f}%<extra></extra>"
                ),
            ),
            row=3, col=1,
        )

    # === 축 설정 ===
    # tickformat=",.0f" — 천 단위 콤마. 0d3s 같은 SI 약어 대신 명시적 풀 숫자.
    fig.update_yaxes(title_text="종가 (원)", row=1, col=1, secondary_y=False,
                     tickformat=",.0f")
    fig.update_yaxes(
        title_text="변동률 (%)", row=1, col=1, secondary_y=True,
        zeroline=True, zerolinecolor="#888", zerolinewidth=1,
        tickformat=",.2f",
    )
    fig.update_yaxes(title_text="누적 보유량 (주)", row=2, col=1,
                     zeroline=True, zerolinecolor="#888", zerolinewidth=1,
                     tickformat=",.0f")
    fig.update_yaxes(title_text="비중 (%)", row=3, col=1, range=[0, 100],
                     tickformat=",.0f")

    # bottom range slider + 1Y/3Y/5Y/전체 버튼 — row 3 xaxis 에만
    fig.update_xaxes(
        row=3, col=1,
        rangeslider=dict(visible=True, thickness=0.04),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="전체"),
            ],
            x=0, y=-0.15,
        ),
    )

    fig.update_layout(
        title=dict(text=title or "수급 보유 변화", x=0.01, xanchor="left"),
        height=900,
        hovermode="x unified",
        barmode="relative",
        font=dict(family='"Apple SD Gothic Neo", "Noto Sans KR", sans-serif',
                  size=12),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            groupclick="togglegroup",
        ),
        margin=dict(l=70, r=180, t=70, b=80),
    )
    return fig


# --- Plotly 공용 config ------------------------------------------------------

PLOTLY_CONFIG: dict = {
    # 마우스 스크롤로 호버 중인 subplot 확대/축소.
    # 박스 드래그 줌(디폴트), 더블클릭 리셋, modebar 줌버튼은 plotly 내장.
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
}


# --- HTML 저장 ---------------------------------------------------------------

def save_html(fig: go.Figure, path: str | Path) -> Path:
    """plotly figure → 단일 HTML. plotly.js 는 CDN 로딩으로 파일 크기 ↓."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path), include_plotlyjs="cdn", full_html=True,
        config=PLOTLY_CONFIG,
    )
    return path
