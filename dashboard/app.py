"""Streamlit 다종목 수급 보유 대시보드.

실행:
    uv run streamlit run dashboard/app.py

기능:
- 종목 셀렉트박스 (회사명·시장 함께 표시)
- 기간 라디오: 1년 / 3년 / 5년 / 전체 / 사용자 지정
- 표시 주체 multiselect (한국어 라벨)
- 종목 비교 모드 (옵션) — 2종목 좌우 배치

캐시: ``@st.cache_data(ttl=3600)`` 으로 종목 전환 시 빠른 응답.
재수집 후엔 1시간 내 자동 무효화. 즉시 무효화는 UI "Clear cache".
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

import tradingagents  # noqa: F401 — dotenv
from dashboard.holdings_chart import (
    SUBJECTS_ORDER, SUBJECT_LABELS, load_holdings, make_figure,
)
from tradingagents.dataflows.kis_history_store import KisHistoryStore


st.set_page_config(
    page_title="수급 보유 변화 대시보드",
    page_icon="📊",
    layout="wide",
)


# --- 데이터 로드 (캐시) ------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_tickers() -> list[tuple[str, str, str]]:
    """(ticker, company_name, market) 리스트. 회사명 가나다순."""
    store = KisHistoryStore()
    tickers = store.list_tickers()
    out: list[tuple[str, str, str]] = []
    for t in tickers:
        meta = store.get_ticker_metadata(t) or {}
        out.append((t, meta.get("company_name") or "-", meta.get("market") or "-"))
    return sorted(out, key=lambda r: r[1])


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_holdings(
    ticker: str, start: Optional[str], end: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    return load_holdings(ticker, start=start, end=end)


# --- 기간 계산 ---------------------------------------------------------------

def _period_to_start(period: str, last_date: Optional[date]) -> Optional[str]:
    """라디오 선택 → start_date (YYYY-MM-DD) 또는 None(전체)."""
    if period == "전체" or last_date is None:
        return None
    years = {"1년": 1, "3년": 3, "5년": 5}.get(period)
    if years is None:
        return None
    start = last_date - timedelta(days=int(years * 365.25))
    return start.isoformat()


# --- 단일 종목 패널 ----------------------------------------------------------

def _render_panel(
    ticker: str, company_name: str, market: str,
    start: Optional[str], end: Optional[str],
    visible_subjects: list[str],
) -> None:
    df, meta = _cached_holdings(ticker, start, end)

    name = meta.get("company_name") or company_name
    mkt = meta.get("market") or market

    st.subheader(f"{name} ({ticker})")
    if df.empty:
        st.warning("선택 기간에 holdings 데이터가 없습니다.")
        return

    st.caption(
        f"시장: {mkt} · 기간: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} "
        f"· 총 {len(df)}일"
    )
    fig = make_figure(df, title=None, visible_subjects=visible_subjects)
    # height 는 비교 모드에서 column 폭이 절반이라 살짝 줄임
    st.plotly_chart(fig, use_container_width=True)


# --- 메인 --------------------------------------------------------------------

def main() -> None:
    st.title("📊 수급 보유 변화 대시보드")
    st.caption(
        "KIS 일별 투자자 매매동향 → 누적 보유량(cum_qty) + 비중(pct, 10 sub 합 = 100%)"
    )

    tickers = _cached_tickers()
    if not tickers:
        st.error(
            "저장된 종목이 없습니다. 먼저 `collect_kis_history.py` 로 수집하세요."
        )
        return

    # === Sidebar ===
    with st.sidebar:
        st.header("종목·기간")

        options = [f"{t} — {n} ({m})" for t, n, m in tickers]
        selected_idx = st.selectbox(
            "종목 선택", range(len(options)),
            format_func=lambda i: options[i],
            key="primary",
        )

        period = st.radio(
            "기간",
            ["1년", "3년", "5년", "전체", "사용자 지정"],
            index=3,
        )

        # primary 종목의 last_date 로 라디오 → start 변환
        primary_ticker = tickers[selected_idx][0]
        primary_df, _ = _cached_holdings(primary_ticker, None, None)
        last_dt = (
            date.fromisoformat(primary_df["date"].iloc[-1])
            if not primary_df.empty else None
        )

        if period == "사용자 지정":
            cols = st.columns(2)
            default_start = last_dt - timedelta(days=365) if last_dt else date.today()
            with cols[0]:
                start_dt = st.date_input("시작일", value=default_start)
            with cols[1]:
                end_dt = st.date_input("종료일", value=last_dt or date.today())
            start_str: Optional[str] = start_dt.isoformat() if start_dt else None
            end_str: Optional[str] = end_dt.isoformat() if end_dt else None
        else:
            start_str = _period_to_start(period, last_dt)
            end_str = None

        st.header("표시 주체")
        visible = st.multiselect(
            "범례 토글로도 가능 — 여기서 미리 필터",
            options=list(SUBJECTS_ORDER),
            default=list(SUBJECTS_ORDER),
            format_func=lambda s: SUBJECT_LABELS[s],
        )

        st.header("비교 모드")
        compare = st.checkbox("두 종목 비교", value=False)
        compare_idx: Optional[int] = None
        if compare:
            other_options = [
                opt for i, opt in enumerate(options) if i != selected_idx
            ]
            other_map = [i for i in range(len(options)) if i != selected_idx]
            j = st.selectbox(
                "비교 종목", range(len(other_options)),
                format_func=lambda i: other_options[i],
                key="secondary",
            )
            compare_idx = other_map[j]

    # === Main ===
    primary = tickers[selected_idx]

    if compare and compare_idx is not None:
        col1, col2 = st.columns(2)
        with col1:
            _render_panel(primary[0], primary[1], primary[2],
                          start_str, end_str, visible)
        with col2:
            other = tickers[compare_idx]
            _render_panel(other[0], other[1], other[2],
                          start_str, end_str, visible)
    else:
        _render_panel(primary[0], primary[1], primary[2],
                      start_str, end_str, visible)

    with st.expander("ℹ️ 차트 해석 가이드"):
        st.markdown(
            """
- **Row 1** — 종가(line, 좌측 y) + 가격 변동률(bar, 우측 y, 양수 빨강 / 음수 파랑)
- **Row 2** — 11 주체 *누적* 보유량. 라인 기울기 = 매수·매도 강도.
  카테고리(외국인 / 기관 / 개인 / 기타법인)별 범례 그룹 클릭으로 한 번에 토글.
- **Row 3** — 10 sub 의 *비중* 100% 누적영역. 외국인은 통합이 아니라 등록 + 비등록으로 분해되어 합 = 100%.
- 첫 거래일 근처는 누적이 작아 row 3 비중이 흔들리는 게 정상.
            """.strip()
        )


if __name__ == "__main__":
    main()
