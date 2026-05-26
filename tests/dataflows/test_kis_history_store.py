"""Unit tests for KisHistoryStore (Parquet + SQLite 동시 저장)."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.kis_history_store import KisHistoryStore


@pytest.fixture
def tmp_store(tmp_path):
    return KisHistoryStore(root=tmp_path, backends=("parquet", "sqlite"))


def _investor_rows(dates, foreign_amount=None):
    foreign_amount = foreign_amount or [0] * len(dates)
    return [
        {
            "date": d, "close": 100 + i,
            "foreign_amount": foreign_amount[i],
            "foreign_qty": 0, "foreign_registered_qty": 0,
            "foreign_unregistered_qty": 0, "foreign_registered_amount": 0,
            "foreign_unregistered_amount": 0,
            "institution_qty": 0, "pension_qty": 0, "private_equity_qty": 0,
            "investment_trust_qty": 0, "securities_qty": 0, "bank_insurance_qty": 0,
            "institution_amount": 0, "pension_amount": 0, "private_equity_amount": 0,
            "investment_trust_amount": 0, "securities_amount": 0, "bank_insurance_amount": 0,
            "retail_qty": 0, "retail_amount": 0,
            "other_corp_qty": 0, "other_corp_amount": 0,
        }
        for i, d in enumerate(dates)
    ]


def _program_rows(dates):
    return [
        {"date": d, "close": 100 + i, "net_qty": i * 1000, "net_amount": i * 1_000_000}
        for i, d in enumerate(dates)
    ]


def _short_rows(dates):
    return [
        {"date": d, "close": 100 + i, "short_qty": i * 100,
         "short_volume_ratio": 1.0 + i * 0.1, "short_amount": i * 10_000,
         "short_amount_ratio": 1.0 + i * 0.1}
        for i, d in enumerate(dates)
    ]


@pytest.mark.unit
class TestWriteRead:
    def test_write_and_read_parquet(self, tmp_store):
        rows = _investor_rows(["2026-05-25", "2026-05-26", "2026-05-27"])
        n = tmp_store.write("005930.KS", "investor", rows)
        assert n == 3
        df = tmp_store.read("005930.KS", "investor", backend="parquet")
        assert len(df) == 3
        assert list(df["date"]) == ["2026-05-25", "2026-05-26", "2026-05-27"]

    def test_write_and_read_sqlite(self, tmp_store):
        rows = _program_rows(["2026-05-25", "2026-05-26"])
        tmp_store.write("005930.KS", "program", rows)
        df = tmp_store.read("005930.KS", "program", backend="sqlite")
        assert len(df) == 2
        assert df.iloc[0]["net_qty"] == 0
        assert df.iloc[1]["net_amount"] == 1_000_000

    def test_short_endpoint_floats_preserved(self, tmp_store):
        rows = _short_rows(["2026-05-25", "2026-05-26"])
        tmp_store.write("005930.KS", "short", rows)
        df_p = tmp_store.read("005930.KS", "short", backend="parquet")
        df_s = tmp_store.read("005930.KS", "short", backend="sqlite")
        assert df_p.iloc[1]["short_volume_ratio"] == pytest.approx(1.1)
        assert df_s.iloc[1]["short_volume_ratio"] == pytest.approx(1.1)

    def test_read_missing_returns_empty(self, tmp_store):
        df = tmp_store.read("NONEXIST", "investor", backend="parquet")
        assert df.empty


@pytest.mark.unit
class TestMergeAndDedup:
    def test_overlapping_writes_merge_and_dedup_by_date(self, tmp_store):
        # 1차: 2026-05-25, 26
        tmp_store.write("005930.KS", "investor",
                        _investor_rows(["2026-05-25", "2026-05-26"], [100, 200]))
        # 2차: 2026-05-26, 27 — 5-26 row 덮어쓰기, 5-27 추가
        tmp_store.write("005930.KS", "investor",
                        _investor_rows(["2026-05-26", "2026-05-27"], [999, 300]))
        df = tmp_store.read("005930.KS", "investor")
        assert len(df) == 3
        assert list(df["date"]) == ["2026-05-25", "2026-05-26", "2026-05-27"]
        # 5-26 row의 foreign_amount는 2차에서 999로 덮어써졌어야 함
        assert df.loc[df["date"] == "2026-05-26", "foreign_amount"].iloc[0] == 999

    def test_increment_then_full_read(self, tmp_store):
        # 증분 시뮬레이션
        tmp_store.write("005930.KS", "program", _program_rows([
            "2026-05-20", "2026-05-21", "2026-05-22",
        ]))
        tmp_store.write("005930.KS", "program", _program_rows([
            "2026-05-25", "2026-05-26",
        ]))
        df = tmp_store.read("005930.KS", "program")
        assert len(df) == 5
        assert list(df["date"]) == [
            "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25", "2026-05-26",
        ]


@pytest.mark.unit
class TestDateFilter:
    def test_start_end_filter(self, tmp_store):
        tmp_store.write("005930.KS", "investor", _investor_rows([
            "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23", "2026-05-24",
        ]))
        df = tmp_store.read(
            "005930.KS", "investor",
            start_date="2026-05-21", end_date="2026-05-23",
        )
        assert list(df["date"]) == ["2026-05-21", "2026-05-22", "2026-05-23"]


@pytest.mark.unit
class TestMeta:
    def test_last_date_after_write(self, tmp_store):
        tmp_store.write("005930.KS", "investor",
                        _investor_rows(["2026-05-25", "2026-05-26", "2026-05-27"]))
        assert tmp_store.last_date("005930.KS", "investor") == "2026-05-27"

    def test_last_date_none_when_no_data(self, tmp_store):
        assert tmp_store.last_date("UNKNOWN", "investor") is None

    def test_per_endpoint_last_date(self, tmp_store):
        tmp_store.write("005930.KS", "investor", _investor_rows(["2026-05-26"]))
        tmp_store.write("005930.KS", "short", _short_rows(["2026-05-20", "2026-05-21"]))
        assert tmp_store.last_date("005930.KS", "investor") == "2026-05-26"
        assert tmp_store.last_date("005930.KS", "short") == "2026-05-21"


@pytest.mark.unit
class TestBackendSelection:
    def test_parquet_only_skips_sqlite(self, tmp_path):
        s = KisHistoryStore(root=tmp_path, backends=("parquet",))
        s.write("005930.KS", "program", _program_rows(["2026-05-26"]))
        # parquet 파일 존재
        assert (tmp_path / "parquet" / "005930.KS" / "program.parquet").exists()
        # sqlite 파일 미생성
        assert not (tmp_path / "kis.db").exists()

    def test_sqlite_only_skips_parquet(self, tmp_path):
        s = KisHistoryStore(root=tmp_path, backends=("sqlite",))
        s.write("005930.KS", "program", _program_rows(["2026-05-26"]))
        assert (tmp_path / "kis.db").exists()
        assert not (tmp_path / "parquet" / "005930.KS").exists()

    def test_invalid_backend_raises(self, tmp_path):
        with pytest.raises(ValueError):
            KisHistoryStore(root=tmp_path, backends=())


@pytest.mark.unit
class TestListTickers:
    def test_lists_after_writes(self, tmp_store):
        tmp_store.write("005930.KS", "investor", _investor_rows(["2026-05-26"]))
        tmp_store.write("000660.KS", "investor", _investor_rows(["2026-05-26"]))
        assert sorted(tmp_store.list_tickers()) == ["000660.KS", "005930.KS"]

    def test_empty_initially(self, tmp_store):
        assert tmp_store.list_tickers() == []
