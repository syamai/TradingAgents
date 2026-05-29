"""statistics_tools wrap 검증.

진짜 통계 함수를 호출하되 ``load_holdings`` (SQLite 의존성) 만 monkeypatch.
dashboard 의 ``_sample_holdings_signed`` fixture 패턴 재사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.holdings_chart import SUBS_10
from tradingagents.dataflows.kis_holdings import compute_holdings
from tradingagents.hermes import statistics_tools


def _sample_holdings(n: int = 60) -> pd.DataFrame:
    """retail 역행 + foreign_registered 동조 가짜 holdings."""
    rng = np.random.default_rng(42)
    close = 100.0
    rows: list[dict] = []
    for i in range(n):
        d = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
        delta = rng.normal(0, 1.5)
        new_close = max(close + delta, 1.0)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "close": new_close,
            **{f"{s}_qty": 0 for s in SUBS_10},
            "retail_qty": int(round(-100 * delta)),
            "foreign_registered_qty": int(round(+100 * delta)),
        })
        close = new_close
    return compute_holdings(pd.DataFrame(rows))


@pytest.mark.unit
class TestComputeCorrelation:
    def test_returns_report_with_ticker_meta(self, monkeypatch):
        df = _sample_holdings(n=60)
        meta = {"company_name": "삼성전자", "market": "KOSPI"}
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (df, meta),
        )

        report = statistics_tools.compute_correlation("005930.KS")

        assert report["ticker"] == "005930.KS"
        assert report["company_name"] == "삼성전자"
        assert report["market"] == "KOSPI"
        assert report["n_days"] > 0
        assert "concurrent" in report
        assert "cumulative" in report

    def test_passes_date_range_to_load_holdings(self, monkeypatch):
        df = _sample_holdings(n=30)
        captured: dict = {}

        def _capture(ticker, start=None, end=None):
            captured["ticker"] = ticker
            captured["start"] = start
            captured["end"] = end
            return (df, {})

        monkeypatch.setattr(statistics_tools, "load_holdings", _capture)

        statistics_tools.compute_correlation(
            "005930.KS", start_date="2026-01-01", end_date="2026-03-01",
        )

        assert captured == {
            "ticker": "005930.KS",
            "start": "2026-01-01",
            "end": "2026-03-01",
        }

    def test_empty_holdings_returns_empty_report(self, monkeypatch):
        empty_df = pd.DataFrame()
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (empty_df, {}),
        )

        report = statistics_tools.compute_correlation("UNKNOWN.KS")

        assert report["ticker"] == "UNKNOWN.KS"
        assert report["n_days"] == 0
        assert report["company_name"] is None
        assert report["market"] is None

    def test_missing_meta_keys_default_to_none(self, monkeypatch):
        df = _sample_holdings(n=30)
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (df, {}),
        )

        report = statistics_tools.compute_correlation("005930.KS")

        assert report["company_name"] is None
        assert report["market"] is None


@pytest.mark.unit
class TestComputeTrend:
    def test_returns_report_with_phases(self, monkeypatch):
        df = _sample_holdings(n=120)
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (df, {}),
        )

        report = statistics_tools.compute_trend("005930.KS")

        assert report["n_days"] == 120
        assert "subjects" in report
        assert "concordance_ranking" in report

    def test_empty_holdings_returns_minimal_report(self, monkeypatch):
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (pd.DataFrame(), {}),
        )

        report = statistics_tools.compute_trend("UNKNOWN.KS")

        assert report["n_days"] == 0
        assert report["subjects"] == {}


@pytest.mark.unit
class TestComputeAdvanced:
    def test_returns_report_when_sufficient_data(self, monkeypatch):
        df = _sample_holdings(n=120)
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (df, {}),
        )

        report = statistics_tools.compute_advanced("005930.KS")

        assert "adf" in report
        assert "granger" in report
        assert "var_irf" in report

    def test_n_days_below_threshold_returns_empty_sections(self, monkeypatch):
        # advanced 는 n<50 이면 빈 sections 반환.
        df = _sample_holdings(n=30)
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (df, {}),
        )

        report = statistics_tools.compute_advanced("005930.KS")

        assert report["adf"] == []
        assert report["granger"] == []


@pytest.mark.unit
class TestGetHoldingsWindow:
    def test_returns_last_n_rows_as_dicts(self, monkeypatch):
        df = _sample_holdings(n=60)
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (df, {}),
        )

        rows = statistics_tools.get_holdings_window("005930.KS", days=5)

        assert isinstance(rows, list)
        assert len(rows) == 5
        # date 가 ISO 문자열로 직렬화됐는지.
        assert isinstance(rows[0]["date"], str)

    def test_empty_holdings_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (pd.DataFrame(), {}),
        )

        assert statistics_tools.get_holdings_window("UNKNOWN.KS") == []

    def test_default_days_is_30(self, monkeypatch):
        df = _sample_holdings(n=60)
        monkeypatch.setattr(
            statistics_tools, "load_holdings",
            lambda ticker, start=None, end=None: (df, {}),
        )

        rows = statistics_tools.get_holdings_window("005930.KS")

        assert len(rows) == 30
