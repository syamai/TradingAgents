"""bok_macro.py — ECOS mock + 파싱 검증."""
from __future__ import annotations

from unittest.mock import patch, Mock

import pandas as pd

from tradingagents.dataflows import bok_macro


def _fake_response(rows):
    m = Mock()
    m.raise_for_status = lambda: None
    m.json = lambda: {"StatisticSearch": {"row": rows}}
    return m


class TestFetchBokSeries:
    def test_normal_response(self):
        rows = [
            {"TIME": "20250214", "DATA_VALUE": "3.00"},
            {"TIME": "20250215", "DATA_VALUE": "3.01"},
            {"TIME": "20250216", "DATA_VALUE": "3.05"},
        ]
        with patch.object(bok_macro, "_api_key", return_value="dummy"), \
             patch("requests.get", return_value=_fake_response(rows)):
            df = bok_macro.fetch_bok_series(
                "722Y001", "0101000", "2025-02-14", "2025-02-16", cache=False,
            )
        assert len(df) == 3
        assert df.iloc[0]["date"] == "2025-02-14"
        assert df.iloc[0]["value"] == 3.00

    def test_missing_key_empty(self):
        with patch.object(bok_macro, "_api_key", return_value=None):
            df = bok_macro.fetch_bok_series(
                "722Y001", "0101000", "2025-02-14", "2025-02-16", cache=False,
            )
        assert df.empty

    def test_error_response_empty(self):
        m = Mock()
        m.raise_for_status = lambda: None
        m.json = lambda: {"RESULT": {"CODE": "INFO-100", "MESSAGE": "fail"}}
        with patch.object(bok_macro, "_api_key", return_value="dummy"), \
             patch("requests.get", return_value=m):
            df = bok_macro.fetch_bok_series(
                "722Y001", "0101000", "2025-02-14", "2025-02-16", cache=False,
            )
        assert df.empty


class TestSpread:
    def test_spread_computation(self):
        # 각 시리즈 mock
        corp = pd.DataFrame({"date": ["2025-02-14", "2025-02-15"], "value": [4.20, 4.25]})
        tre = pd.DataFrame({"date": ["2025-02-14", "2025-02-15"], "value": [3.20, 3.20]})
        with patch.object(bok_macro, "fetch_corporate_aa_yield", return_value=corp), \
             patch.object(bok_macro, "fetch_treasury_3y", return_value=tre):
            df = bok_macro.fetch_corp_bond_spread("2025-02-14", "2025-02-15")
        assert df.iloc[0]["spread"] == 1.00
        assert df.iloc[1]["spread"] == 1.05


class TestSummarize:
    def test_all_unavailable_when_no_key(self):
        with patch.object(bok_macro, "_api_key", return_value=None):
            out = bok_macro.summarize_macro("2025-02-14", "2025-04-14")
        for k in ("base_rate", "corp_aa_yield", "treasury_3y"):
            assert out[k]["available"] is False
