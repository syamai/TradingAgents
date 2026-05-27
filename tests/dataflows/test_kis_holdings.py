"""Unit tests for kis_holdings — compute_holdings/holdings_window/store hook."""
from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.kis_holdings import (
    SUBS_10, INFO_TOTALS, HOLDINGS_COLUMNS,
    compute_holdings, holdings_window, holdings_at,
)
from tradingagents.dataflows.kis_history_store import KisHistoryStore


def _zero_subs() -> dict:
    out = {}
    for s in SUBS_10:
        out[f"{s}_qty"] = 0
    return out


def _investor_row(date, close, **overrides) -> dict:
    """Build an investor raw row — 10 sub keys default 0. ``overrides`` override sub qty."""
    base = {"date": date, "close": close, **_zero_subs()}
    base.update(overrides)
    return base


@pytest.mark.unit
class TestComputeHoldings:
    def test_empty_input_returns_empty_with_columns(self):
        df = compute_holdings(pd.DataFrame())
        assert df.empty
        assert list(df.columns) == HOLDINGS_COLUMNS

    def test_cumsum_correctness(self):
        rows = [
            _investor_row("2026-05-01", 100, retail_qty=100),
            _investor_row("2026-05-02", 101, retail_qty=-50),
            _investor_row("2026-05-03", 102, retail_qty=200),
        ]
        df = compute_holdings(pd.DataFrame(rows))
        assert list(df["retail_net_qty"]) == [100, -50, 200]
        assert list(df["retail_cum_qty"]) == [100, 50, 250]

    def test_pct_sums_to_100(self):
        # 4 sub 모두 다른 부호로 누적
        rows = [
            _investor_row("2026-05-01", 100,
                          foreign_registered_qty=100, retail_qty=-100),
            _investor_row("2026-05-02", 101,
                          pension_qty=50, retail_qty=-50),
            _investor_row("2026-05-03", 102,
                          bank_qty=30, insurance_qty=-30),
        ]
        df = compute_holdings(pd.DataFrame(rows))
        sums = sum(df[f"{s}_pct"] for s in SUBS_10)
        # 모든 행이 정확히 100% (round 오차 0.001 이내)
        for v in sums:
            assert abs(v - 100.0) < 0.01

    def test_first_row_all_zero_pct_zero(self):
        rows = [_investor_row("2026-05-01", 100)]  # 모두 0
        df = compute_holdings(pd.DataFrame(rows))
        for s in SUBS_10:
            assert df.iloc[0][f"{s}_pct"] == 0.0

    def test_sign_preserved(self):
        rows = [
            _investor_row("2026-05-01", 100, retail_qty=-100),
            _investor_row("2026-05-02", 101, retail_qty=-200),
        ]
        df = compute_holdings(pd.DataFrame(rows))
        assert df.iloc[1]["retail_cum_qty"] == -300  # signed
        assert df.iloc[1]["retail_pct"] == 100.0  # abs 기준 영향력은 100%

    def test_price_change_pct_calculation(self):
        rows = [
            _investor_row("2026-05-01", 100, retail_qty=1),
            _investor_row("2026-05-02", 110, retail_qty=1),  # +10%
            _investor_row("2026-05-03", 99, retail_qty=1),   # -10%
        ]
        df = compute_holdings(pd.DataFrame(rows))
        assert df.iloc[0]["price_change_pct"] == 0  # 첫 행
        assert df.iloc[1]["price_change_pct"] == 10.0
        assert df.iloc[2]["price_change_pct"] == pytest.approx(-10.0, abs=0.001)

    def test_foreign_total_is_registered_plus_unregistered(self):
        rows = [
            _investor_row("2026-05-01", 100,
                          foreign_registered_qty=70, foreign_unregistered_qty=30),
            _investor_row("2026-05-02", 101,
                          foreign_registered_qty=-20, foreign_unregistered_qty=10),
        ]
        df = compute_holdings(pd.DataFrame(rows))
        # net_qty 합
        assert df.iloc[0]["foreign_net_qty"] == 100
        assert df.iloc[1]["foreign_net_qty"] == -10
        # cum_qty 합
        assert df.iloc[0]["foreign_cum_qty"] == 100
        assert df.iloc[1]["foreign_cum_qty"] == 90  # 100 + (-10)
        # pct = 등록 pct + 비등록 pct
        last = df.iloc[1]
        assert last["foreign_pct"] == pytest.approx(
            last["foreign_registered_pct"] + last["foreign_unregistered_pct"],
            abs=0.01,
        )

    def test_columns_order_matches_constant(self):
        rows = [_investor_row("2026-05-01", 100, retail_qty=1)]
        df = compute_holdings(pd.DataFrame(rows))
        assert list(df.columns) == HOLDINGS_COLUMNS


@pytest.mark.unit
class TestHoldingsWindow:
    def _sample(self):
        rows = [
            _investor_row("2026-05-01", 100, retail_qty=100, pension_qty=-100),
            _investor_row("2026-05-02", 101, retail_qty=200, pension_qty=-200),
            _investor_row("2026-05-03", 102, retail_qty=50, pension_qty=-50),
            _investor_row("2026-05-04", 103, retail_qty=-30, pension_qty=30),
        ]
        return compute_holdings(pd.DataFrame(rows))

    def test_no_window_returns_full(self):
        df = self._sample()
        out = holdings_window(df)
        # cum_qty 그대로 (baseline = 0)
        assert int(out.iloc[-1]["retail_cum_qty"]) == 100 + 200 + 50 - 30

    def test_window_baseline_subtraction(self):
        df = self._sample()
        # start=2026-05-02 → 5-01 행이 베이스라인 (retail=100)
        out = holdings_window(df, start_date="2026-05-02")
        # 5-02 행의 retail_cum_qty = 300 - 100 (baseline) = 200
        assert int(out.iloc[0]["retail_cum_qty"]) == 200
        assert int(out.iloc[-1]["retail_cum_qty"]) == 350 - 30 - 100  # = 220

    def test_window_empty_range(self):
        df = self._sample()
        out = holdings_window(df, start_date="2026-06-01")
        assert out.empty


@pytest.mark.unit
class TestHoldingsAt:
    def test_returns_row_dict(self):
        rows = [_investor_row("2026-05-01", 100, retail_qty=1)]
        df = compute_holdings(pd.DataFrame(rows))
        h = holdings_at(df, "2026-05-01")
        assert h is not None
        assert h["date"] == "2026-05-01"
        assert h["close"] == 100

    def test_returns_none_for_missing(self):
        rows = [_investor_row("2026-05-01", 100, retail_qty=1)]
        df = compute_holdings(pd.DataFrame(rows))
        assert holdings_at(df, "2026-05-99") is None

    def test_returns_none_on_empty(self):
        assert holdings_at(pd.DataFrame(), "2026-05-01") is None


@pytest.mark.unit
class TestStoreHook:
    def test_investor_write_materializes_holdings(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("parquet", "sqlite"))
        rows = [
            _investor_row("2026-05-01", 100, retail_qty=100),
            _investor_row("2026-05-02", 101, retail_qty=-50),
        ]
        store.write("005930", "investor", rows)
        h = store.read("005930", "holdings")
        assert not h.empty
        assert len(h) == 2
        assert list(h["retail_cum_qty"]) == [100, 50]

    def test_program_write_does_not_create_holdings(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("parquet", "sqlite"))
        store.write("005930", "program", [
            {"date": "2026-05-01", "close": 100, "net_qty": 1, "net_amount": 100},
        ])
        h = store.read("005930", "holdings")
        assert h.empty


@pytest.mark.unit
class TestTickersMeta:
    def test_set_and_get_roundtrip(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        store.set_ticker_metadata("005930.KS", "삼성전자", "KOSPI")
        meta = store.get_ticker_metadata("005930")
        assert meta is not None
        assert meta["ticker"] == "005930"
        assert meta["company_name"] == "삼성전자"
        assert meta["market"] == "KOSPI"
        assert meta["updated_at"]  # non-empty timestamp

    def test_upsert_updates_existing(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        store.set_ticker_metadata("005930", "Samsung", "KOSPI")
        store.set_ticker_metadata("005930", "삼성전자", "KOSPI")
        meta = store.get_ticker_metadata("005930")
        assert meta["company_name"] == "삼성전자"

    def test_get_missing_returns_none(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        assert store.get_ticker_metadata("999999") is None
