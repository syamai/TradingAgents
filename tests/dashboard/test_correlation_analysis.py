"""correlation_analysis + analysis_reports store API 단위 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from dashboard.correlation_analysis import (
    LAG_OFFSETS, compute_correlation_report, pearson, render_markdown,
)
from dashboard.holdings_chart import SUBJECTS_ORDER
from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.kis_holdings import SUBS_10, compute_holdings


def _zero_subs() -> dict:
    return {f"{s}_qty": 0 for s in SUBS_10}


def _investor_row(date: str, close: int, **overrides) -> dict:
    base = {"date": date, "close": close, **_zero_subs()}
    base.update(overrides)
    return base


def _sample_holdings_signed(n: int = 60) -> pd.DataFrame:
    """retail = -ret 강한 역행, foreign_registered = +ret 강한 동조."""
    import numpy as np
    rng = np.random.default_rng(42)
    close = 100.0
    rows: list[dict] = []
    for i in range(n):
        d = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
        # 가격은 랜덤워크
        delta = rng.normal(0, 1.5)
        new_close = max(close + delta, 1.0)
        # retail 은 가격 변화의 정확히 -100 배 (강한 역행)
        # foreign_registered 는 가격 변화 +100 배 (강한 동조)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "close": new_close,
            **{f"{s}_qty": 0 for s in SUBS_10},
            "retail_qty": int(round(-100 * delta)),
            "foreign_registered_qty": int(round(+100 * delta)),
        })
        close = new_close
    return compute_holdings(pd.DataFrame(rows))


# === pearson 함수 ===

@pytest.mark.unit
class TestPearson:
    def test_perfect_positive(self):
        r, p = pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r == pytest.approx(1.0, abs=1e-6)
        assert p == pytest.approx(0.0, abs=1e-6)

    def test_perfect_negative(self):
        r, p = pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r == pytest.approx(-1.0, abs=1e-6)

    def test_zero_variance_returns_zero(self):
        r, p = pearson([1, 1, 1, 1], [1, 2, 3, 4])
        assert r == 0.0
        assert p == 1.0

    def test_too_short_returns_zero(self):
        r, p = pearson([1], [2])
        assert r == 0.0

    def test_nan_filtered(self):
        import numpy as np
        r, _ = pearson([1, 2, np.nan, 4, 5], [2, 4, 6, 8, 10])
        # nan 행 제거 후 r=1
        assert r == pytest.approx(1.0, abs=1e-6)


# === compute_correlation_report ===

@pytest.mark.unit
class TestComputeReport:
    def test_empty_input_returns_skeleton(self):
        r = compute_correlation_report(pd.DataFrame(), ticker="005930")
        assert r["n_days"] == 0
        assert r["cumulative"] == []
        assert r["concurrent"] == []

    def test_sample_data_top_correlations(self):
        df = _sample_holdings_signed(60)
        r = compute_correlation_report(df, ticker="005930",
                                        company_name="삼성전자", market="KOSPI")
        assert r["n_days"] == 60
        assert r["company_name"] == "삼성전자"
        # concurrent 정렬: |r| 큰 순. 3 강한 시계열: foreign(통합), foreign_registered,
        # retail. 통합과 등록은 동일 시리즈라 r 도 같음 → 상위 3 가 이 세 주체.
        top_3_subjects = {row["subject"] for row in r["concurrent"][:3]}
        assert top_3_subjects == {"foreign", "foreign_registered", "retail"}
        retail = next(x for x in r["concurrent"] if x["subject"] == "retail")
        fr = next(x for x in r["concurrent"] if x["subject"] == "foreign_registered")
        assert retail["r"] < -0.95
        assert fr["r"] > 0.95

    def test_lag_keys_match_offsets(self):
        df = _sample_holdings_signed(60)
        r = compute_correlation_report(df, ticker="005930")
        assert len(r["lag"]) > 0
        expected = {f"t{k:+d}" if k != 0 else "t0" for k in LAG_OFFSETS}
        assert set(r["lag"][0]["lags"].keys()) == expected

    def test_up_down_pattern_classified(self):
        df = _sample_holdings_signed(60)
        r = compute_correlation_report(df, ticker="005930")
        # retail: 상승일에 음수, 하락일에 양수 → counter_trend
        retail = next(x for x in r["up_down"] if x["subject"] == "retail")
        assert retail["pattern"] == "counter_trend"
        # foreign_registered: 상승일에 양수, 하락일에 음수 → accumulating_up
        fr = next(x for x in r["up_down"] if x["subject"] == "foreign_registered")
        assert fr["pattern"] == "accumulating_up"

    def test_regime_windows_keys(self):
        df = _sample_holdings_signed(60)
        r = compute_correlation_report(df, ticker="005930")
        # 60일 데이터(2026년)는 "2025-" 구간만 충분, 나머지는 n/a 가능
        for row in r["regime"]:
            assert "전체" in row["windows"]


# === render_markdown ===

@pytest.mark.unit
class TestRenderMarkdown:
    def test_empty_data_renders_message(self):
        r = compute_correlation_report(pd.DataFrame(), ticker="005930")
        md = render_markdown(r)
        assert "데이터 없음" in md

    def test_markdown_includes_sections(self):
        df = _sample_holdings_signed(40)
        r = compute_correlation_report(df, ticker="005930", company_name="삼성전자")
        md = render_markdown(r)
        for section in ["1. 5년 누적", "2. 동시 상관", "3. 상승일 vs 하락일",
                        "4. Lag", "5. Regime", "6. 누적 수준 상관"]:
            assert section in md
        assert "삼성전자 (005930)" in md


# === SQLite analysis_reports ===

@pytest.mark.unit
class TestAnalysisReportsStore:
    def test_write_then_read_latest(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        store.write_analysis_report("005930", "correlation", "2026-05-26",
                                     {"r": 0.4, "k": [1, 2, 3]})
        r = store.read_analysis_report("005930", "correlation")
        assert r["ticker"] == "005930"
        assert r["kind"] == "correlation"
        assert r["as_of_date"] == "2026-05-26"
        assert r["payload"]["r"] == 0.4
        assert r["payload"]["k"] == [1, 2, 3]

    def test_upsert_replaces_payload(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        store.write_analysis_report("005930", "correlation", "2026-05-26", {"r": 0.4})
        store.write_analysis_report("005930", "correlation", "2026-05-26", {"r": 0.5})
        r = store.read_analysis_report("005930", "correlation")
        assert r["payload"]["r"] == 0.5

    def test_read_by_as_of_date(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        store.write_analysis_report("005930", "correlation", "2026-05-26", {"r": 0.4})
        store.write_analysis_report("005930", "correlation", "2026-05-27", {"r": 0.5})
        # latest 는 5-27
        assert store.read_analysis_report("005930", "correlation")["payload"]["r"] == 0.5
        # 특정 날짜 지정
        assert store.read_analysis_report("005930", "correlation", "2026-05-26")["payload"]["r"] == 0.4

    def test_missing_returns_none(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        assert store.read_analysis_report("999999", "correlation") is None

    def test_list_filters_by_kind(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        store.write_analysis_report("005930", "correlation", "2026-05-26", {})
        store.write_analysis_report("000660", "correlation", "2026-05-26", {})
        store.write_analysis_report("005930", "other_kind", "2026-05-26", {})
        all_rows = store.list_analysis_reports()
        assert len(all_rows) == 3
        only_corr = store.list_analysis_reports(kind="correlation")
        assert len(only_corr) == 2
        assert {r["ticker"] for r in only_corr} == {"005930", "000660"}
