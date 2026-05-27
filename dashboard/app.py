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
import dashboard.interpretation as itp
from dashboard.advanced_analysis import compute_advanced_report
from dashboard.correlation_analysis import compute_correlation_report
from dashboard.holdings_chart import (
    PLOTLY_CONFIG, SUBJECTS_ORDER, SUBJECT_LABELS, load_holdings, make_figure,
)
from dashboard.llm_synthesis import (
    DEFAULT_MODEL, DEFAULT_PROVIDER, synthesize_conclusion,
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


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_advanced(
    ticker: str, start: Optional[str], end: Optional[str],
) -> dict:
    df, _ = load_holdings(ticker, start=start, end=end)
    return compute_advanced_report(df)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_llm_synthesis(
    ticker: str, start: Optional[str], end: Optional[str],
    provider: str, model: str,
) -> str:
    """LLM 호출 결과 캐시 — provider·model 도 캐시 키."""
    df, meta = load_holdings(ticker, start=start, end=end)
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
                          start_str, end_str, visible, pct_mode)
        with col2:
            other = tickers[compare_idx]
            _render_panel(other[0], other[1], other[2],
                          start_str, end_str, visible, pct_mode)
    else:
        _render_panel(primary[0], primary[1], primary[2],
                      start_str, end_str, visible, pct_mode)

    # === 차트 해석 가이드 ===
    with st.expander("ℹ️ 차트 해석 가이드"):
        st.markdown(
            """
- **Row 1** — 종가(line, 좌측 y) + 가격 변동률(bar, 우측 y, 양수 빨강 / 음수 파랑)
- **Row 2** — 11 주체 *누적* 보유량. 라인 기울기 = 매수·매도 강도.
  카테고리(외국인 / 기관 / 개인 / 기타법인)별 범례 그룹 클릭으로 한 번에 토글.
- **Row 3** — 10 sub 의 *비중* 100% 누적영역. 외국인은 통합이 아니라 등록 + 비등록으로 분해되어 합 = 100%.
  - **누적 모드** — 첫 거래일부터의 매매 영향력 누적 비중. 시간이 지나면 안정됨.
  - **일별 모드** — 그날 하루의 매매 비중. 누적과 달리 일변동이 크고, 거래가 없는 주체는 즉시 0%.
- 첫 거래일 근처는 누적이 작아 row 3 (누적 모드) 비중이 흔들리는 게 정상.

**조작법**

- **마우스 스크롤** — 호버 중인 패널 확대/축소
- **드래그(박스)** — 선택 영역 확대
- **더블클릭** — 줌 리셋
- **우상단 modebar** — 줌인/아웃 버튼, 카메라(PNG), 팬/줌 모드 전환
            """.strip()
        )

    # === 상관 분석 패널 (동적 — 현재 선택 종목) ===
    if not compare:
        _render_correlation_section(primary[0], primary[1], start_str, end_str)
        _render_advanced_section(primary[0], primary[1], start_str, end_str)
        _render_llm_synthesis_section(primary[0], primary[1], start_str, end_str)

    # === 분석 기법 설명 (정적) ===
    _render_methodology_guide()


# === 상관 분석 패널 ----------------------------------------------------------

def _render_correlation_section(
    ticker: str, company_name: str,
    start: Optional[str], end: Optional[str],
) -> None:
    st.divider()
    with st.expander(f"📊 상관 분석 — {company_name} ({ticker})", expanded=False):
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
    with st.expander(
        f"🧬 정교한 분석 — Granger·VAR·MI 외 ({company_name})", expanded=False,
    ):
        st.caption(
            "⚠️ Granger·VAR fit·MI 등 statsmodels/sklearn 계산이 수 초 소요됨. "
            "expander 열면 1회 계산 후 1시간 캐시."
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


# === 최종 종합 의견 — 로컬 LLM 호출 -----------------------------------------

def _render_llm_synthesis_section(
    ticker: str, company_name: str,
    start: Optional[str], end: Optional[str],
) -> None:
    st.divider()
    with st.expander(
        f"🧠 최종 종합 의견 — {company_name} (로컬 LLM)", expanded=False,
    ):
        st.caption(
            f"로컬 LLM (`{DEFAULT_PROVIDER}` · `{DEFAULT_MODEL}`)이 위 12 섹션 "
            "자동 해석을 통합해 최종 의견을 작성합니다. 첫 호출에 10~30 초 "
            "소요 후 1 시간 캐시. 매매 추천은 포함하지 않습니다."
        )
        col_a, col_b = st.columns([3, 1])
        with col_a:
            model = st.text_input(
                "모델 (provider:model 또는 model 만)",
                value=f"{DEFAULT_PROVIDER}:{DEFAULT_MODEL}",
                help="예: `ollama:gemma4:26b-a4b` 또는 `ollama:llama3.2:8b`",
            )
        with col_b:
            st.write("")  # vertical spacer
            run = st.button("의견 생성", type="primary")

        # provider:model 파싱
        if ":" in model:
            provider, _, model_name = model.partition(":")
        else:
            provider, model_name = DEFAULT_PROVIDER, model

        if not run:
            st.info("'의견 생성' 버튼을 누르면 로컬 LLM 호출이 시작됩니다.")
            return

        with st.spinner(f"{provider}/{model_name} 호출 중... (10~30초)"):
            text = _cached_llm_synthesis(ticker, start, end, provider, model_name)
        st.markdown(text)


# === 분석 기법 가이드 (정적) -------------------------------------------------

def _render_methodology_guide() -> None:
    with st.expander("🧪 분석 방법·기법 설명 (r·lag·사용 기법)", expanded=False):
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
