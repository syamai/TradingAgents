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

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox

import tradingagents  # noqa: F401 — dotenv
import dashboard.interpretation as itp
from dashboard.advanced_analysis import compute_advanced_report, pnl_attribution
from dashboard.correlation_analysis import compute_correlation_report
from dashboard.holdings_chart import (
    PLOTLY_CONFIG, SUBJECTS_ORDER, SUBJECT_LABELS, load_holdings, make_figure,
)
from dashboard.llm_synthesis import (
    DEFAULT_MODEL, DEFAULT_PROVIDER, synthesize_conclusion,
)
from dashboard.trend_analysis import compute_trend_report, render_trend_markdown
from dashboard.event_retro import (
    compute_event_retro, render_event_retro_markdown,
)
from dashboard.event_retro_synthesis import (
    DEFAULT_PROVIDER as RETRO_SYN_PROVIDER,
    DEFAULT_MODEL as RETRO_SYN_MODEL,
    synthesize_event_retro, render_synthesis_markdown,
)
from tradingagents.dataflows.kis_history_store import KisHistoryStore


st.set_page_config(
    page_title="수급 보유 변화 대시보드",
    page_icon="📊",
    layout="wide",
)


# --- 데이터 로드 (캐시) ------------------------------------------------------

def _tickers_db_mtime() -> float:
    """``kis.db`` mtime — `_cached_tickers` 의 캐시 무효화 키.

    데이터 수집/메타 변경 시 DB가 갱신되면 mtime이 바뀌어 캐시 자동 무효화.
    """
    db = KisHistoryStore()._sqlite_path()
    return db.stat().st_mtime if db.exists() else 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_tickers(db_mtime: float) -> list[tuple[str, str, str]]:
    """(ticker, company_name, market) 리스트. 회사명 가나다순.

    ``tickers`` 메타에 등록된 종목만 노출 — investor 데이터 수집이 완료된
    표식. parquet 디렉토리만 있고 메타 미등록인 종목(부분 수집)은 제외해
    "회사명 -" + "holdings 데이터 없음" 동시 발생을 방지한다.

    ``db_mtime`` 인자는 함수 본문에서 사용하지 않지만 cache key 의 일부 —
    DB 가 갱신되면 캐시가 자동 무효화된다.
    """
    del db_mtime  # cache key only
    store = KisHistoryStore()
    out: list[tuple[str, str, str]] = []
    for t in store.list_tickers():
        meta = store.get_ticker_metadata(t)
        if meta is None:
            continue
        out.append((t, meta["company_name"], meta["market"]))
    return sorted(out, key=lambda r: r[1])


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_holdings(
    ticker: str, start: Optional[str], end: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    return load_holdings(ticker, start=start, end=end)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_correlation(
    ticker: str, start: Optional[str], end: Optional[str],
) -> dict:
    df, meta = load_holdings(ticker, start=start, end=end)
    return compute_correlation_report(
        df, ticker=ticker,
        company_name=meta.get("company_name"),
        market=meta.get("market"),
    )


# 핵심 주체 → raw investor 거래대금(amount) 컬럼. foreign 은 holdings net_qty
# (등록+비등록 합) 와 단가 기준을 맞추려 등록+비등록 amount 를 합산한다.
_PNL_AMOUNT_SOURCES: dict[str, tuple[str, ...]] = {
    "foreign": ("foreign_registered_amount", "foreign_unregistered_amount"),
    "pension": ("pension_amount",),
    "securities": ("securities_amount",),
    "private_equity": ("private_equity_amount",),
    "retail": ("retail_amount",),
}

# KIS ``*_ntby_tr_pbmn`` 필드는 백만원(百萬) 단위 → close(원) 와 맞추려 ×1e6.
_PBMN_TO_KRW = 1_000_000


def _load_holdings_with_amounts(
    ticker: str, start: Optional[str], end: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    """holdings df + 핵심 주체별 거래대금(amount, 원 환산) 병합 — 손익 근사용.

    amount 는 raw ``investor`` 테이블에만 있어 holdings 에 별도 병합한다
    (``compute_holdings`` 는 qty 만 derive). raw 는 백만원 단위라 ``close``(원)
    와 단위를 맞추려 ×1e6 환산해 넘긴다. 병합 실패해도 holdings 만 반환해
    기존 분석은 정상 동작.
    """
    df, meta = load_holdings(ticker, start=start, end=end)
    if df.empty:
        return df, meta
    inv = KisHistoryStore().read(ticker, "investor", start_date=start, end_date=end)
    if inv.empty or "date" not in inv.columns:
        return df, meta
    amt = pd.DataFrame({"date": inv["date"]})
    for subj, cols in _PNL_AMOUNT_SOURCES.items():
        present = [c for c in cols if c in inv.columns]
        if present:
            amt[f"{subj}_amount"] = _PBMN_TO_KRW * sum(
                pd.to_numeric(inv[c], errors="coerce").fillna(0) for c in present
            )
    return df.merge(amt, on="date", how="left"), meta


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_advanced(
    ticker: str, start: Optional[str], end: Optional[str],
) -> dict:
    df, _ = _load_holdings_with_amounts(ticker, start, end)
    return compute_advanced_report(df)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_pnl(
    ticker: str, start: Optional[str], end: Optional[str],
) -> list[dict]:
    """손익 근사만 — 정교한 분석(50행 컷오프·statsmodels)과 분리해 짧은 구간도 계산."""
    df, _ = _load_holdings_with_amounts(ticker, start, end)
    return pnl_attribution(df)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_trend(
    ticker: str, start: Optional[str], end: Optional[str],
) -> dict:
    df, _ = load_holdings(ticker, start=start, end=end)
    return compute_trend_report(df)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_event_retro(
    ticker: str, start: str, end: str,
    enable_search: bool, enable_macro: bool, enable_dart: bool,
    enable_peers: bool,
    industry_keyword: str,
    peers_override_csv: str,
) -> dict:
    """이벤트 리뷰 dict 캐시 — 3주차: peers + LLM 분류는 후처리."""
    peers_override = (
        [p.strip() for p in peers_override_csv.split(",") if p.strip()]
        if peers_override_csv else None
    )
    return compute_event_retro(
        ticker, start, end,
        enable_search=enable_search,
        enable_macro=enable_macro,
        enable_dart=enable_dart,
        enable_peers=enable_peers,
        industry_keyword=industry_keyword or None,
        peers_override=peers_override,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_event_retro_synthesis(
    ticker: str, start: str, end: str,
    provider: str, model: str,
    # report 캐시 키 — JSON 직렬화 해시. dict 는 hashable 아니라서 timeline 길이로 약한 키
    timeline_len: int,
    report_hash: str,
) -> dict:
    """LLM 합성 결과 캐시 — provider·model·timeline 길이 키. 부분 캐싱 한계."""
    # 캐시 키 한계로 호출자가 report 를 인자로 못 넘김 — 별도 함수 _synthesize_now 사용.
    raise RuntimeError("Use _synthesize_now instead — report dict is not hashable")


def _synthesize_now(report: dict, provider: str, model: str) -> dict:
    """LLM 합성 — 캐시 안 함 (report dict 가 hashable 아님)."""
    return synthesize_event_retro(report, provider=provider, model=model)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_llm_synthesis(
    ticker: str, start: Optional[str], end: Optional[str],
    provider: str, model: str,
) -> str:
    """LLM 호출 결과 캐시 — provider·model 도 캐시 키."""
    df, meta = _load_holdings_with_amounts(ticker, start, end)
    report = compute_correlation_report(
        df, ticker=ticker,
        company_name=meta.get("company_name"),
        market=meta.get("market"),
    )
    advanced = compute_advanced_report(df)
    return synthesize_conclusion(
        report, advanced, provider=provider, model=model,
    )


def _df_height(n_rows: int) -> int:
    """streamlit dataframe height — 헤더 + n_rows 행이 스크롤 없이 보이게.

    행 높이 ~35px + 헤더 38px + 패딩 6px. 최대 800px 까지(매우 긴 표는 스크롤).
    """
    return min(35 * n_rows + 38 + 6, 800)


# --- 검색 필터 ---------------------------------------------------------------

_SEARCHBOX_MAX_RESULTS = 30


def _filter_tickers(
    tickers: list[tuple[str, str, str]], query: str,
) -> list[tuple[str, str, str]]:
    """검색어로 (ticker, company_name) 부분일치 필터. 공백/빈 문자열은 전체 반환.

    대소문자 무시. 시장(market)은 검색 대상 아님 — "KOSPI"가 너무 광범위.
    """
    q = (query or "").strip().lower()
    if not q:
        return tickers
    return [
        (t, n, m) for t, n, m in tickers
        if q in t.lower() or q in n.lower()
    ]


def _make_searchbox_options(
    tickers: list[tuple[str, str, str]], query: str,
) -> list[tuple[str, str]]:
    """``st_searchbox`` 용 ``(label, value)`` 리스트. value 는 ticker 코드.

    드롭다운 길이 제한 — 빈 query 면 가나다순 앞쪽 30종목만.
    """
    matches = _filter_tickers(tickers, query)
    return [
        (f"{t} — {n} ({m})", t)
        for t, n, m in matches[:_SEARCHBOX_MAX_RESULTS]
    ]


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
    pct_mode: str,
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
    fig = make_figure(
        df, title=None,
        visible_subjects=visible_subjects, pct_mode=pct_mode,
    )
    # height 는 비교 모드에서 column 폭이 절반이라 살짝 줄임
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# --- 메인 --------------------------------------------------------------------

def main() -> None:
    st.title("📊 수급 보유 변화 대시보드")
    st.caption(
        "KIS 일별 투자자 매매동향 → 누적 보유량(cum_qty) + 비중(pct, 10 sub 합 = 100%)"
    )

    tickers = _cached_tickers(_tickers_db_mtime())
    if not tickers:
        st.error(
            "저장된 종목이 없습니다. 먼저 `collect_kis_history.py` 로 수집하세요."
        )
        return

    # === Sidebar ===
    ticker_lookup = {t: (t, n, m) for t, n, m in tickers}
    default_primary = tickers[0]  # 가나다순 첫 종목

    with st.sidebar:
        st.header("종목·기간")

        selected_primary = st_searchbox(
            search_function=lambda q: _make_searchbox_options(tickers, q),
            placeholder="이름 또는 코드",
            label=f"종목 ({len(tickers)})",
            default=default_primary[0],
            default_options=_make_searchbox_options(tickers, ""),
            key="primary_search",
        )
        primary = ticker_lookup.get(selected_primary, default_primary)

        period = st.radio(
            "기간",
            ["1년", "3년", "5년", "전체", "사용자 지정"],
            index=3,
        )

        # primary 종목의 last_date 로 라디오 → start 변환
        primary_df, _ = _cached_holdings(primary[0], None, None)
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

        st.header("Row 3 비중")
        pct_mode_label = st.radio(
            "분모 기준",
            ["누적 (첫 거래일~오늘)", "일별 (그날 net_qty)"],
            index=0,
            help="누적: |cum_qty| / Σ|cum_qty|. "
                 "일별: |net_qty| / Σ|net_qty| (그날 분).",
        )
        pct_mode = "cumulative" if pct_mode_label.startswith("누적") else "daily"

        st.header("비교 모드")
        compare = st.checkbox("두 종목 비교", value=False)
        other: Optional[tuple[str, str, str]] = None
        if compare:
            others = [r for r in tickers if r[0] != primary[0]]
            selected_other = st_searchbox(
                search_function=lambda q: _make_searchbox_options(others, q),
                placeholder="이름 또는 코드",
                label="비교 종목",
                default_options=_make_searchbox_options(others, ""),
                key="secondary_search",
            )
            if selected_other:
                other = ticker_lookup.get(selected_other)

    # === Main ===
    if compare and other is not None:
        col1, col2 = st.columns(2)
        with col1:
            _render_panel(primary[0], primary[1], primary[2],
                          start_str, end_str, visible, pct_mode)
        with col2:
            _render_panel(other[0], other[1], other[2],
                          start_str, end_str, visible, pct_mode)
    else:
        _render_panel(primary[0], primary[1], primary[2],
                      start_str, end_str, visible, pct_mode)

    # === 탭 구조 — 차트 아래 ===
    st.divider()
    if compare:
        # 비교 모드: 손익 근사는 기준 종목 기준. 방법 가이드와 2 탭.
        tab1, tab2 = st.tabs(["💰 주체별 손익 근사", "🧪 분석 방법"])
        with tab1:
            _render_pnl_section(primary[0], primary[1], start_str, end_str)
        with tab2:
            _render_methodology_guide()
    else:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "💰 주체별 손익 근사",
            f"📊 상관 분석",
            f"🧬 정교한 분석",
            f"📈 추세 분석",
            f"🧠 최종 종합 의견",
            "🔄 이벤트 리뷰",
            "🧪 분석 방법",
        ])
        with tab1:
            _render_pnl_section(primary[0], primary[1], start_str, end_str)
        with tab2:
            _render_correlation_section(primary[0], primary[1], start_str, end_str)
        with tab3:
            _render_advanced_section(primary[0], primary[1], start_str, end_str)
        with tab4:
            _render_trend_section(primary[0], primary[1], start_str, end_str)
        with tab5:
            _render_llm_synthesis_section(primary[0], primary[1], start_str, end_str)
        with tab6:
            _render_event_retro_section(primary[0], primary[1])
        with tab7:
            _render_methodology_guide()


def _render_pnl_section(
    ticker: str, company_name: str,
    start: Optional[str], end: Optional[str],
) -> None:
    """탭 안에서 호출 — 주체별 손익 근사(구간 mark-to-market)."""
    st.markdown(f"### 💰 주체별 손익 근사 — 구간 mark-to-market ({company_name})")
    with st.spinner("손익 근사 계산 중..."):
        pnl_rows = _cached_pnl(ticker, start, end)
    if not pnl_rows:
        st.info("거래대금(amount) 데이터 없음 — 손익 추정 생략.")
        return
    lc = pnl_rows[0]["last_close"]
    st.caption(
        f"구간 net 흐름을 **순매수일/순매도일로 분리**한 거래량가중 평단 "
        f"(현재가 {lc:,.0f}원).  \n"
        "**실현손익** = 왕복 매칭분 `min(순매수,순매도)×(매도평단−매수평단)`.  \n"
        "**미실현 평가손익** = 순매수 잔여(실제 보유)의 현재가 평가. "
        "**미실현 기회손익** = 순매도 초과분의 매도 타이밍 평가(보유 아님, 현재가 "
        "대비 잘 팔았나). net·구간 상대 **근사치**."
    )
    pnl_df = pd.DataFrame([
        {"주체": r["label"],
         "순매수량": r["buy_qty"], "순매수평단": r["buy_avg"],
         "순매도량": r["sell_qty"], "순매도평단": r["sell_avg"],
         "실현손익(억)": r["realized_pnl"] / 1e8,
         "미실현 평가손익(억)":
             r["unrealized_pnl"] / 1e8 if r["unrealized_kind"] == "평가" else None,
         "미실현 기회손익(억)":
             r["unrealized_pnl"] / 1e8 if r["unrealized_kind"] == "기회" else None}
        for r in pnl_rows
    ])
    st.dataframe(
        pnl_df, use_container_width=True, hide_index=True,
        height=_df_height(len(pnl_df)),
        column_config={
            "순매수량": st.column_config.NumberColumn(format="%,d"),
            "순매수평단": st.column_config.NumberColumn(format="%,.0f"),
            "순매도량": st.column_config.NumberColumn(format="%,d"),
            "순매도평단": st.column_config.NumberColumn(format="%,.0f"),
            "실현손익(억)": st.column_config.NumberColumn(format="%+.1f"),
            "미실현 평가손익(억)": st.column_config.NumberColumn(format="%+.1f"),
            "미실현 기회손익(억)": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
    top, bot = pnl_rows[0], pnl_rows[-1]
    st.info(
        f"구간 추정 최대 이익(실현+미실현): **{top['label']}** "
        f"{top['pnl'] / 1e8:+,.1f}억 · 최대 손실: **{bot['label']}** "
        f"{bot['pnl'] / 1e8:+,.1f}억."
    )

    with st.expander("ℹ️ 미실현 평가손익 / 기회손익 부호 해석"):
        st.markdown(
            "**A. 순매수 잔여 (net_qty > 0) — 실제로 들고 있는 물량**  \n"
            "기준단가 = 매수평단. 정석적인 \"보유 평가손익\".\n\n"
            "| 부호 | 조건 | 의미 |\n"
            "|:---:|---|---|\n"
            "| **+** | 현재가 > 매수평단 | 산 값보다 올라서 **평가이익** (지금 팔면 이익) |\n"
            "| **−** | 현재가 < 매수평단 | 산 값보다 내려서 **평가손실** (물려있음) |\n\n"
            "**B. 순매도 초과 (net_qty < 0) — 비워버린 포지션 (기회손익)**  \n"
            "기준단가 = 매도평단. 실제 보유가 아니라(구간 시작 이전 물량까지 순매도한 "
            "notional 상태), \"그때 판 게 지금 기준으로 잘한 거냐\"는 기회손익.\n\n"
            "| 부호 | 조건 | 의미 |\n"
            "|:---:|---|---|\n"
            "| **+** | 현재가 < 매도평단 | 판 값보다 지금이 쌈 → **잘 팔았다** (기회이익) |\n"
            "| **−** | 현재가 > 매도평단 | 판 값보다 지금이 비쌈 → **싸게 팔아버렸다** (기회손실) |\n"
        )


# === 상관 분석 패널 ----------------------------------------------------------

def _render_correlation_section(
    ticker: str, company_name: str,
    start: Optional[str], end: Optional[str],
) -> None:
    """탭 안에서 호출 — expander/divider 없이 내용만."""
    st.markdown(f"### 📊 상관 분석 — {company_name} ({ticker})")
    with st.spinner("상관관계 계산 중..."):
        report = _cached_correlation(ticker, start, end)
    if report["n_days"] == 0:
        st.info("데이터 없음.")
        return

    pr = report["price"]
    w = report["window"]
    st.markdown(
        f"**기간** {w['start']} ~ {w['end']} · {report['n_days']}일 · "
        f"**종가** {pr['start']:,.0f} → {pr['end']:,.0f} "
        f"({pr['total_return_pct']:+.1f}%) · "
        f"**일별 σ** {pr['daily_std_pct']:.2f}%"
    )

    c1, c2 = st.columns(2)

    # [1] 5년 누적 — 정수, 천 단위 콤마
    with c1:
        st.markdown("**1. 누적 매수·매도 (cum_qty 최종)**")
        cum_df = pd.DataFrame([
            {
                "주체": ("ⓘ " if r["is_info_total"] else "") + r["label"],
                "누적": r["cum_qty"],
                "비중%": None if r["is_info_total"] else r["pct"],
            }
            for r in report["cumulative"]
        ])
        st.dataframe(
            cum_df, use_container_width=True, hide_index=True,
            height=_df_height(len(cum_df)),
            column_config={
                "누적": st.column_config.NumberColumn(format="%,d"),
                "비중%": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.info(itp.interpret_cumulative(report))

    # [2] 동시 상관 — r 4자리, p 과학표기
    with c2:
        st.markdown("**2. 동시 상관 — `ret(t) ↔ net_qty(t)`**")
        conc_df = pd.DataFrame([
            {
                "주체": r["label"],
                "r": r["r"],
                "p": f"{r['p_value']:.1e}" if r["p_value"] > 0 else "≈0",
                "해석": r["interpretation"] + " " + r["significance"],
            }
            for r in report["concurrent"]
        ])
        st.dataframe(
            conc_df, use_container_width=True, hide_index=True,
            height=_df_height(len(conc_df)),
            column_config={
                "r": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.info(itp.interpret_concurrent(report))

    # [3] 상승/하락일 — 정수, 천 단위 콤마
    meta = report["up_down_meta"]
    st.markdown(
        f"**3. 상승일 vs 하락일 평균 순매수** "
        f"(상승 {meta['n_up']:,} · 하락 {meta['n_down']:,} · 보합 {meta['n_flat']:,})"
    )
    pat_label = {"accumulating_up": "추세 추종",
                 "counter_trend": "역행 매매", "mixed": "혼합"}
    ud_df = pd.DataFrame([
        {"주체": r["label"], "상승일 평균": r["up_mean"],
         "하락일 평균": r["down_mean"], "차이": r["diff"],
         "패턴": pat_label[r["pattern"]]}
        for r in report["up_down"]
    ])
    st.dataframe(
        ud_df, use_container_width=True, hide_index=True,
        height=_df_height(len(ud_df)),
        column_config={
            "상승일 평균": st.column_config.NumberColumn(format="%,.0f"),
            "하락일 평균": st.column_config.NumberColumn(format="%,.0f"),
            "차이": st.column_config.NumberColumn(format="%,.0f"),
        },
    )
    st.info(itp.interpret_up_down(report))

    # [4] Lag — r 3자리
    st.markdown("**4. Lag (CCF) — 수급이 가격을 선도? 후행?**")
    if report["lag"]:
        lag_keys = list(report["lag"][0]["lags"].keys())
        lag_df = pd.DataFrame([
            {"주체": r["label"], **r["lags"]} for r in report["lag"]
        ])
        st.dataframe(
            lag_df, use_container_width=True, hide_index=True,
            height=_df_height(len(lag_df)),
            column_config={
                k: st.column_config.NumberColumn(format="%+.3f")
                for k in lag_keys
            },
        )
        st.caption(
            "`t=0` 동시 · `t+k` 수급이 k일 *선도* · `t-k` 수급이 k일 *후행*."
        )
        st.info(itp.interpret_lag(report))

    # [5] Regime — r 3자리, None 은 자동 빈칸
    st.markdown("**5. Regime — 기간별 동시 상관**")
    if report["regime"]:
        win_keys = list(report["regime"][0]["windows"].keys())
        rg_df = pd.DataFrame([
            {"주체": r["label"],
             **{k: r["windows"][k] for k in win_keys}}
            for r in report["regime"]
        ])
        st.dataframe(
            rg_df, use_container_width=True, hide_index=True,
            height=_df_height(len(rg_df)),
            column_config={
                k: st.column_config.NumberColumn(format="%+.3f")
                for k in win_keys
            },
        )
        st.info(itp.interpret_regime(report))

    # [6] Level — r 3자리
    st.markdown("**6. 누적 수준 상관 — `r(cum_qty, close)`** *(트렌드 영향 큼 — 참고용)*")
    lv_df = pd.DataFrame([
        {"주체": r["label"], "r": r["r"]} for r in report["level"]
    ])
    st.dataframe(
        lv_df, use_container_width=True, hide_index=True,
        height=_df_height(len(lv_df)),
        column_config={
            "r": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    st.info(itp.interpret_level(report))


# === 정교한 분석 패널 (동적) -------------------------------------------------

def _render_advanced_section(
    ticker: str, company_name: str,
    start: Optional[str], end: Optional[str],
) -> None:
    """탭 안에서 호출 — expander 없이 내용만."""
    st.markdown(f"### 🧬 정교한 분석 — Granger·VAR·MI 외 ({company_name})")
    st.caption(
        "⚠️ Granger·VAR fit·MI 등 statsmodels/sklearn 계산이 수 초 소요됨. "
        "탭 진입 시 1회 계산 후 1시간 캐시."
    )
    with st.spinner("정교한 분석 계산 중..."):
        adv = _cached_advanced(ticker, start, end)
    if not adv["adf"]:
        st.info("데이터 부족 — 분석 미수행.")
        return

    cfg = adv["config"]
    key_labels = ", ".join(SUBJECT_LABELS[s] for s in cfg["key_subjects"])
    st.caption(
        f"핵심 주체: {key_labels} · "
        f"Granger max lag={cfg['granger_max_lag']} · "
        f"VAR horizon={cfg['var_horizon']}일 · "
        f"Rolling window={cfg['rolling_window']}일"
    )

    # 7-1. ADF
    st.markdown("**7-1. ADF 정상성 검정**")
    adf_df = pd.DataFrame([
        {"시리즈": r["label"], "ADF": r["adf_stat"],
         "p": r["p_value"], "정상성": "✓" if r["is_stationary"] else "✗ 비정상",
         "n": r["n_obs"]}
        for r in adv["adf"]
    ])
    st.dataframe(
        adf_df, use_container_width=True, hide_index=True,
        height=_df_height(len(adf_df)),
        column_config={
            "ADF": st.column_config.NumberColumn(format="%+.3f"),
            "p": st.column_config.NumberColumn(format="%.4f"),
            "n": st.column_config.NumberColumn(format="%,d"),
        },
    )
    st.caption("종가·cum_qty 같은 누적 시계열이 비정상으로 나오는 게 보통 — "
               "level r 은 spurious 위험. Cointegration 결과로 보강.")
    st.info(itp.interpret_adf(adv))

    # 7-2. Granger
    st.markdown("**7-2. Granger Causality**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("`net_qty → return` — 매매가 미래 가격 *예측*")
        granger_a = _granger_df(adv["granger"], "net_causes_return")
        st.dataframe(
            granger_a, use_container_width=True, hide_index=True,
            height=_df_height(len(granger_a)),
            column_config={
                f"lag {k}": st.column_config.NumberColumn(format="%.4f")
                for k in range(1, cfg["granger_max_lag"] + 1)
            },
        )
    with c2:
        st.markdown("`return → net_qty` — 가격이 매매 유발 (chasing)")
        granger_b = _granger_df(adv["granger"], "return_causes_net")
        st.dataframe(
            granger_b, use_container_width=True, hide_index=True,
            height=_df_height(len(granger_b)),
            column_config={
                f"lag {k}": st.column_config.NumberColumn(format="%.4f")
                for k in range(1, cfg["granger_max_lag"] + 1)
            },
        )
    st.caption("값은 p-value. p<0.05 면 Granger 인과 — 셀 옆 ✓ 의미는 "
               "데이터프레임 값이 0.05 미만인 셀.")
    st.info(itp.interpret_granger(adv))

    # 7-3. VAR + IRF
    st.markdown("**7-3. VAR + 누적 충격반응 (IRF)**")
    irf = adv["var_irf"] or {}
    if not irf or irf.get("order") is None:
        st.info("VAR fit 실패 — 데이터 부족 또는 공선성.")
    else:
        st.caption(
            f"VAR 차수(AIC) = **{irf['order']}**, horizon = {irf['horizon']}일. "
            "1-σ 매매 충격이 누적 수익률(%)에 미치는 효과."
        )
        irf_rows = []
        for s in cfg["key_subjects"]:
            key = f"{s}_net->return"
            vals = irf["irf_cum"].get(key, [])
            if not vals:
                continue
            row = {"충격원 → return": SUBJECT_LABELS[s]}
            for h in [0, 1, 3, 5, 10]:
                if h < len(vals):
                    row[f"t={h}"] = vals[h]
            irf_rows.append(row)
        if irf_rows:
            irf_df = pd.DataFrame(irf_rows)
            st.dataframe(
                irf_df, use_container_width=True, hide_index=True,
                height=_df_height(len(irf_df)),
                column_config={
                    c: st.column_config.NumberColumn(format="%+.5f")
                    for c in irf_df.columns if c.startswith("t=")
                },
            )
            st.info(itp.interpret_var_irf(adv))

    # 7-4. Cointegration
    st.markdown("**7-4. Cointegration — `cum_qty ↔ close`**")
    coint_df = pd.DataFrame([
        {"주체": r["label"], "score": r["score"], "p": r["p_value"],
         "공적분": "✓ 장기 균형" if r["is_cointegrated"] else "✗"}
        for r in adv["cointegration"]
    ])
    st.dataframe(
        coint_df, use_container_width=True, hide_index=True,
        height=_df_height(len(coint_df)),
        column_config={
            "score": st.column_config.NumberColumn(format="%+.3f"),
            "p": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption("✓면 두 시계열이 장기적으로 함께 움직임 → level r 신뢰. "
               "✗면 추세 동조성으로 spurious 가능.")
    st.info(itp.interpret_cointegration(adv))

    # 7-5. MI
    st.markdown("**7-5. Mutual Information — 비선형 의존성**")
    mi_df = pd.DataFrame([
        {"주체": r["label"], "MI": r["mi"]} for r in adv["mutual_info"]
    ])
    st.dataframe(
        mi_df, use_container_width=True, hide_index=True,
        height=_df_height(len(mi_df)),
        column_config={
            "MI": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption("Pearson r 이 작은데 MI 가 크면 비선형 의존성 신호.")
    st.info(itp.interpret_mi(adv))

    # 7-6. Rolling
    rw = cfg["rolling_window"]
    st.markdown(f"**7-6. Rolling correlation ({rw}일) — DCC 단순판**")
    roll_df = pd.DataFrame([
        {"주체": r["label"], "mean": r["mean"], "std": r["std"],
         "min": r["min"], "p10": r["p10"], "p90": r["p90"], "max": r["max"]}
        for r in adv["rolling_r"] if r["mean"] is not None
    ])
    st.dataframe(
        roll_df, use_container_width=True, hide_index=True,
        height=_df_height(len(roll_df)),
        column_config={
            c: st.column_config.NumberColumn(format="%+.3f")
            for c in ["mean", "min", "p10", "p90", "max"]
        } | {"std": st.column_config.NumberColumn(format="%.3f")},
    )
    st.caption("std 가 크면 상관 강도가 시기마다 크게 변동(regime 변화 시사).")
    st.info(itp.interpret_rolling(adv))


def _granger_df(rows: list[dict], direction_key: str) -> pd.DataFrame:
    """granger 양방향 결과 → pivot DataFrame (행: 주체, 열: lag 1..N)."""
    out_rows = []
    for r in rows:
        row = {"주체": r["label"]}
        for x in r[direction_key]:
            row[f"lag {x['lag']}"] = x["p_value"]
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# === 추세 분석 패널 ---------------------------------------------------------

def _render_trend_section(
    ticker: str, company_name: str,
    start: Optional[str], end: Optional[str],
) -> None:
    """누적 보유 추세 phase 분할 + 동행성 — 일별 Granger 가 못 잡는 중기 신호."""
    with st.spinner("추세 phase 분할 중..."):
        trend = _cached_trend(ticker, start, end)
    if not trend.get("subjects"):
        st.info("데이터 부족 — 분석 미수행.")
        return
    st.markdown(render_trend_markdown(trend))


# === 이벤트 리뷰 — 가격·이벤트·뉴스·재무·거시·동종 통합 분석 -----------------

_OBSIDIAN_REL = Path("Projects/trading-ai/reports")


def _find_obsidian_vault() -> Optional[Path]:
    """``~/Documents/*/.obsidian`` 첫 매치 vault root. 없으면 None."""
    docs = Path.home() / "Documents"
    if not docs.exists():
        return None
    for entry in docs.iterdir():
        if (entry / ".obsidian").is_dir():
            return entry
    return None


def _event_retro_frontmatter(report: dict) -> str:
    meta = report.get("meta", {})
    return "\n".join([
        "---",
        f"ticker: {meta.get('ticker', '')}",
        f"company_name: {meta.get('company_name', '')}",
        f"market: {meta.get('market', '')}",
        f"report_kind: event_retro",
        f"period_start: {meta.get('start', '')}",
        f"period_end: {meta.get('end', '')}",
        f"generated_at: {meta.get('generated_at', '')}",
        f"tags: [analysis, event-retro]",
        "---",
        "",
    ])


def _event_retro_obsidian_path(
    vault: Path, ticker: str, company_name: str, start: str, end: str,
) -> Path:
    safe = (company_name or "unknown").replace(" ", "_").replace("/", "_")
    return vault / _OBSIDIAN_REL / f"event_{ticker}_{safe}_{start}_to_{end}.md"


def _render_event_retro_section(
    ticker: str, company_name: str,
) -> None:
    """이벤트 리뷰 탭 — 사이드바와 별도 기간.

    내부 탭 2개:
      📊 회고 생성 — 입력 + 기본 결과 (§0~§7 + 저장)
      🧠 LLM 합성 — 모델 선택 + 합성 결과 (§8 자연어 단락)
    """
    st.markdown(f"### 🔄 이벤트 리뷰 — {company_name} ({ticker})")
    st.caption(
        "사이드바 기간과 *별도*로 회고할 사건 기간을 지정하세요. "
        "가격·거래량 이상치 + 뉴스 검색 + 거시 + DART 공시 + 동종 비교 + LLM 합성까지."
    )

    inner_tab1, inner_tab2 = st.tabs(["📊 회고 생성", "🧠 LLM 합성"])
    with inner_tab1:
        _render_retro_generation_tab(ticker, company_name)
    with inner_tab2:
        _render_retro_synthesis_tab(ticker, company_name)


def _render_retro_generation_tab(ticker: str, company_name: str) -> None:
    """탭 1: 회고 생성 — 기간 입력 + 고급 옵션 + 본문 + 저장."""
    col1, col2 = st.columns(2)
    with col1:
        retro_start = st.date_input(
            "회고 시작일", value=date(2025, 2, 14),
            key=f"retro_start_{ticker}",
        )
    with col2:
        retro_end = st.date_input(
            "회고 종료일", value=date(2025, 4, 14),
            key=f"retro_end_{ticker}",
        )

    with st.expander("고급 옵션", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            enable_search = st.checkbox(
                "뉴스 검색", value=True, key=f"retro_search_{ticker}",
            )
        with c2:
            enable_macro = st.checkbox(
                "거시 지표", value=True, key=f"retro_macro_{ticker}",
            )
        with c3:
            enable_dart = st.checkbox(
                "DART 공시", value=True, key=f"retro_dart_{ticker}",
            )
        with c4:
            enable_peers = st.checkbox(
                "동종 비교", value=True, key=f"retro_peers_{ticker}",
            )
        industry_keyword = st.text_input(
            "산업 키워드 (검색 보강)",
            value="", placeholder="예: 손해보험, 반도체, 자동차",
            key=f"retro_industry_{ticker}",
        )
        peers_override = st.text_input(
            "동종 수동 (콤마 구분, 6자리 코드)",
            value="", placeholder="예: 000810,001450,000060,000540",
            key=f"retro_peers_override_{ticker}",
            help="HARD_CODED_PEERS 자동 매칭 대신 직접 지정.",
        )

    run_btn = st.button(
        "회고 생성", type="primary", key=f"retro_run_{ticker}",
    )

    state_key = f"retro_report_{ticker}"
    start_str = retro_start.strftime("%Y-%m-%d")
    end_str = retro_end.strftime("%Y-%m-%d")

    if run_btn:
        with st.status("회고 생성 중...", expanded=True) as status:
            status.update(label="1/4 가격 데이터 로드 + 이벤트 탐지")
            report = _cached_event_retro(
                ticker, start_str, end_str,
                enable_search, enable_macro, enable_dart, enable_peers,
                industry_keyword, peers_override,
            )
            status.update(label="2/4 검색 + 거시 + DART + 동종 병렬 수집")
            status.update(label="3/4 이벤트-뉴스 매칭 + 분해 계산")
            status.update(label="4/4 마크다운 렌더", state="complete")
        st.session_state[state_key] = report

    report = st.session_state.get(state_key)
    if not report:
        st.info("기간 지정 후 **회고 생성** 버튼을 클릭하세요.")
        return

    # 경고
    warnings = report.get("warnings") or []
    if warnings:
        with st.expander(f"⚠️ 경고 {len(warnings)}건", expanded=False):
            for w in warnings:
                st.write(f"- {w}")

    # 본문 — LLM 합성 §8 포함 (있으면)
    md_body = render_event_retro_markdown(report)
    syn_md = render_synthesis_markdown(report)
    if syn_md:
        md_body += "\n\n" + syn_md

    # §2-A 매칭 뉴스 섹션이 <details> 로 감싸여 있어 HTML 허용 필요
    st.markdown(md_body, unsafe_allow_html=True)

    # 저장 행
    st.markdown("---")
    save_col1, save_col2, save_col3 = st.columns(3)
    file_name = f"event_{ticker}_{company_name}_{start_str}_to_{end_str}.md"
    md_full = _event_retro_frontmatter(report) + md_body + "\n"

    with save_col1:
        st.download_button(
            "📥 다운로드", data=md_full,
            file_name=file_name, mime="text/markdown",
            key=f"retro_download_{ticker}",
        )
    with save_col2:
        save_db = st.checkbox(
            "SQLite 저장", value=True, key=f"retro_save_db_{ticker}",
        )
        if st.button("💾 SQLite", key=f"retro_save_db_btn_{ticker}",
                     disabled=not save_db):
            try:
                store = KisHistoryStore()
                store.write_analysis_report(
                    ticker, "event_retro", end_str, report,
                )
                st.success(
                    f"SQLite analysis_reports[{ticker}, event_retro, {end_str}]"
                )
            except Exception as exc:
                st.error(f"SQLite 저장 실패: {exc}")
    with save_col3:
        if st.button("🗒 옵시디언", key=f"retro_save_obs_{ticker}"):
            vault = _find_obsidian_vault()
            if vault is None:
                st.error("옵시디언 vault 못 찾음 (~/Documents/*/.obsidian)")
            else:
                path = _event_retro_obsidian_path(
                    vault, ticker, company_name, start_str, end_str,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    path.write_text(md_full, encoding="utf-8")
                    st.success(f"저장: `{path}`")
                except Exception as exc:
                    st.error(f"옵시디언 저장 실패: {exc}")


def _render_retro_synthesis_tab(ticker: str, company_name: str) -> None:
    """탭 2: LLM 합성 — 모델 선택 + 합성 버튼 + §8 단락 + 분해표 갱신."""
    state_key = f"retro_report_{ticker}"
    report = st.session_state.get(state_key)
    if not report:
        st.info("📊 **회고 생성** 탭에서 먼저 회고를 만든 후 이 탭으로 돌아오세요.")
        return

    st.caption(
        "LLM 1회(또는 chained 6회) 호출로 이벤트 분류 + 클러스터별 자연어 단락 생성. "
        "분해표(§7)도 LLM 분류 기준으로 자동 재계산. ollama 디폴트는 무료(로컬)."
    )

    # === 모델 선택 ===
    CUSTOM = "✏️ 직접 입력..."
    PRESET_MODELS = [
        f"{RETRO_SYN_PROVIDER}:{RETRO_SYN_MODEL}",
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-haiku-4-5",
        "openai:gpt-5",
        "openai:gpt-4.1",
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        "google:gemini-2.5-pro",
        "ollama:gemma4:26b-a4b",
        "ollama:llama3.2:8b",
        CUSTOM,
    ]
    seen = set(); PRESET_MODELS = [
        x for x in PRESET_MODELS if not (x in seen or seen.add(x))
    ]
    selected = st.selectbox(
        "모델 (provider:model)", PRESET_MODELS, index=0,
        key=f"retro_llm_combo_{ticker}",
        help="외부 API 모델은 해당 API_KEY 환경변수가 설정되어야 합니다. "
             "ollama 는 로컬 무료 — chained 6회 호출로 단락 분량 자동 강화.",
    )
    if selected == CUSTOM:
        combo = st.text_input(
            "provider:model 직접 입력",
            value=f"{RETRO_SYN_PROVIDER}:{RETRO_SYN_MODEL}",
            key=f"retro_llm_custom_{ticker}",
            help="예: `ollama:gemma2:27b`, `openai:o4-mini`",
        )
    else:
        combo = selected

    if ":" in combo:
        llm_provider, _, llm_model = combo.partition(":")
    else:
        llm_provider, llm_model = RETRO_SYN_PROVIDER, combo

    # 키 부재 경고
    _RETRO_KEY_ENV = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    key_env = _RETRO_KEY_ENV.get(llm_provider)
    if key_env and not os.environ.get(key_env):
        st.warning(
            f"⚠️ 환경변수 `{key_env}` 미설정 — LLM 호출 실패 가능. "
            f"룰베이스 키워드 fallback 으로 분류됩니다."
        )

    # === 합성 버튼 ===
    syn_btn = st.button(
        "🧠 LLM 합성 추가/재실행", type="primary",
        key=f"retro_synthesize_{ticker}",
        help="이미 합성된 결과를 다른 모델로 재실행할 때도 사용.",
    )

    if syn_btn:
        with st.status("LLM 통합 호출 중...", expanded=True) as status:
            status.update(label=f"{llm_provider}/{llm_model} 호출 (ollama=chained 6회)")
            report = _synthesize_now(report, llm_provider, llm_model)
            status.update(label="JSON 파싱 + 분해 재계산", state="complete")
        st.session_state[state_key] = report

    # === 결과 표시 ===
    syn = report.get("synthesis") or {}
    if not syn:
        st.info("🧠 **LLM 합성 추가/재실행** 버튼을 눌러 합성을 시작하세요.")
        return

    # 메타
    method = syn.get("method", "?")
    st.caption(
        f"**모델**: `{syn.get('provider')}:{syn.get('model')}` · "
        f"**방식**: `{method}`"
    )

    # 갱신된 분해표 (LLM 분류 기준)
    decomp = report.get("decomposition") or {}
    if decomp:
        from dashboard.event_retro import CLUSTER_ENUM, CLUSTER_LABEL_KO
        decomp_lines = ["**§7. 분해 (LLM 분류 기준)**", "",
                        "| 클러스터 | 변동 기여 |", "|---|---:|"]
        for c in CLUSTER_ENUM:
            v = decomp.get(c, 0.0)
            decomp_lines.append(f"| {CLUSTER_LABEL_KO.get(c, c)} | {v:+.2f}% |")
        total = sum(decomp.get(c, 0.0) for c in CLUSTER_ENUM)
        decomp_lines.append(f"| **합계** | **{total:+.2f}%** |")
        st.markdown("\n".join(decomp_lines))
        st.markdown("")

    # §8 자연어 단락
    syn_md = render_synthesis_markdown(report)
    if syn_md:
        st.markdown(syn_md, unsafe_allow_html=True)


# === 최종 종합 의견 — 로컬 LLM 호출 -----------------------------------------

def _render_llm_synthesis_section(
    ticker: str, company_name: str,
    start: Optional[str], end: Optional[str],
) -> None:
    """탭 안에서 호출 — expander 없이 내용만."""
    st.markdown(f"### 🧠 최종 종합 의견 — {company_name}")
    st.caption(
        f"LLM 이 위 12 섹션 자동 해석을 통합해 최종 의견을 작성합니다. "
        f"로컬(Ollama) 모델은 무료지만 외부 API 모델(OpenAI/Anthropic 등)은 "
        f"호출당 비용 발생. 첫 호출 10~30 초, 1 시간 캐시. 매매 추천 미포함."
    )

    CUSTOM = "✏️ 직접 입력..."
    PRESET_MODELS = [
        # 로컬 (Ollama)
        f"{DEFAULT_PROVIDER}:{DEFAULT_MODEL}",
        "ollama:llama3.2:8b",
        "ollama:qwen2.5:14b",
        # OpenAI
        "openai:gpt-5",
        "openai:gpt-4.1",
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        # Anthropic
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-haiku-4-5",
        # Google
        "google:gemini-2.5-pro",
        CUSTOM,
    ]

    col_a, col_b = st.columns([3, 1])
    with col_a:
        selected = st.selectbox(
            "모델 선택", PRESET_MODELS, index=0,
            help="외부 API 모델(openai/anthropic/google) 은 해당 API_KEY "
                 "환경변수가 .env 또는 셸에 설정되어 있어야 합니다.",
        )
        if selected == CUSTOM:
            model = st.text_input(
                "provider:model 직접 입력",
                value=f"{DEFAULT_PROVIDER}:{DEFAULT_MODEL}",
                help="예: `ollama:gemma2:27b`, `openai:o4-mini`",
            )
        else:
            model = selected
    with col_b:
        st.write("")  # vertical spacer
        run = st.button("의견 생성", type="primary")

    # provider:model 파싱
    if ":" in model:
        provider, _, model_name = model.partition(":")
    else:
        provider, model_name = DEFAULT_PROVIDER, model

    # 외부 API provider 선택 시 키 부재 경고
    _KEY_ENV = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    key_env = _KEY_ENV.get(provider)
    if key_env and not os.environ.get(key_env):
        st.warning(
            f"⚠️ 환경변수 `{key_env}` 가 설정되지 않았습니다. "
            f"호출이 실패할 수 있습니다 — `.env` 파일에 키 추가 필요."
        )

    if not run:
        st.info("'의견 생성' 버튼을 누르면 LLM 호출이 시작됩니다.")
        return

    with st.spinner(f"{provider}/{model_name} 호출 중... (10~30초)"):
        text = _cached_llm_synthesis(ticker, start, end, provider, model_name)
    st.markdown(text)


# === 분석 기법 가이드 (정적) -------------------------------------------------

def _render_methodology_guide() -> None:
    """탭 안에서 호출 — expander 없이 markdown 만."""
    st.markdown(
        """
### r — Pearson 상관계수

두 변수의 **선형 동조** 정도. 수식 `r = Cov(X,Y) / (σ_X · σ_Y)`.

| r | 의미 |
|---|---|
| +1.0 | 완벽 양의 선형 |
| +0.5 ~ +0.7 | 강한 양의 동조 |
| +0.1 ~ +0.3 | 약한 동조 |
| 0 | 무관 |
| −0.5 이하 | 강한 역행 |
| −1.0 | 완벽 음의 선형 |

**p-value** — 우연일 확률. p<0.05 면 통계적 유의(`*`). 1222일 표본이면 |r|≈0.06만 넘어도 p<0.05라서 의미는 **r 의 절댓값 크기** 자체로 판단.

⚠️ **상관 ≠ 인과** — 같은 제3 요인(시장 분위기·뉴스)에 둘 다 반응해도 r 이 커진다.

---

### lag — 시간 지연

한 변수를 *k 일* 어긋나게 놓고 상관 계산.

| 표시 | 의미 |
|---|---|
| **t=0** | 같은 날: 오늘 수급 ↔ 오늘 수익률 |
| **t+1** | 오늘 수급 ↔ **내일** 수익률 — 수급이 가격을 1일 선도? |
| **t+3** | 오늘 수급 ↔ 3일 뒤 수익률 |
| **t−1** | 어제 수급 ↔ 오늘 수익률 — 가격이 어제 수급에 영향?(chasing) |

t=0 만 강하고 t±1, ±3 이 0 에 가까우면 → 수급은 가격을 1 일 이상 선도/후행하지 않는다(인트라데이 영향 또는 즉시 chasing).

---

### 본 패널이 사용한 기법

| # | 기법 | 어디서 |
|---|---|---|
| 1 | **Pearson product-moment correlation** | 동시 상관 표 (r, p-value) |
| 2 | **Cross-correlation function (CCF)** | lag −3 ~ +3 표 (시계열 기본) |
| 3 | **Sub-period (regime-split) correlation** | 2021-22 / 2023-24 / 2025- 비교 |
| 4 | **Conditional mean** (event-study 변형) | 상승일 vs 하락일 평균 net_qty |
| 5 | **Compositional analysis** | 5년 누적 매수 주체 비중 |
| 6 | **Level correlation** | cum_qty 수준 vs 종가 수준 (trend-spurious 가능) |

---

### 더 정교한 기법 — 구현 적용 6 종 (위 "🧬 정교한 분석" expander 참조)

| 기법 | 무엇을 잡나 | 구현 위치 |
|---|---|---|
| **ADF (Augmented Dickey-Fuller)** | 시리즈가 정상(stationary)인가 — level r 의 spurious 위험 정량화 | 7-1 |
| **Granger causality** | X 가 Y 의 *미래값* 예측에 기여 (선형, F-test). 양방향: 매매→가격 / 가격→매매(chasing) | 7-2 |
| **VAR + Impulse Response** | 다변량 lag 회귀. 1-σ 매매 충격이 누적 수익률에 미치는 효과를 horizon 별로 추적 | 7-3 |
| **Cointegration (Engle-Granger)** | `cum_qty` 와 `close` 같은 비정상 시계열의 **장기 균형** | 7-4 |
| **Mutual Information** | **비선형** 의존성 (Pearson r 은 선형만). KSG 추정기. | 7-5 |
| **Rolling correlation** | 시간에 따른 상관 변화. DCC-GARCH 의 단순판. | 7-6 |

### 구현하지 않은 4 종 — 사유

| 기법 | 사유 |
|---|---|
| **DCC-GARCH** | `arch` 라이브러리(C 확장) 추가 필요, fit 시간 길고 종목·주체별로 모델링이 무거움. **Rolling correlation (7-6)** 으로 1차 근사. |
| **Transfer Entropy** | 안정적 PyPI 패키지 부재(직접 구현 필요), 표본 크기에 매우 민감해 1,200일로는 추정 분산이 큼. **Granger (7-2) + MI (7-5)** 로 선·비선형 정보 흐름은 부분 커버. |
| **Hawkes process** | `tick` 라이브러리는 macOS arm64 빌드 이슈가 잦고, 이벤트 자기·교차 유발 모델은 일별 데이터(저빈도)보다 분/틱 데이터에서 가치가 높음. |
| **Wavelet coherence** | `pywt` + 자체 구현 부담. 결과 해석에 푸리에/웨이블릿 도메인 친숙도 필요. 본 분석 목적(주체별 상관 구조)에 비해 가독성 낮음. |

### 본 패널에서 보강 가능한 다음 단계

- Pearson 외에 **Spearman 순위 상관** 추가 — 비선형 단조 관계 보존, outlier 강건
- 가중 상관 — 거래대금 가중 r (큰 거래일에 가중)
- **VECM** (Vector Error Correction Model) — cointegration ✓ 인 시리즈의 장단기 분리
        """.strip()
    )


if __name__ == "__main__":
    main()
