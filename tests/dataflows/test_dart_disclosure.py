"""dart_disclosure.py — list.json mock + 필터 검증."""
from __future__ import annotations

from unittest.mock import patch, Mock

from tradingagents.dataflows import dart_disclosure
from tradingagents.dataflows.dart_disclosure import (
    Disclosure, filter_significant_disclosures, summarize_disclosures,
    list_disclosures,
)


def _fake_resp(status="000", items=None, total_page=1):
    m = Mock()
    m.raise_for_status = lambda: None
    m.json = lambda: {
        "status": status, "list": items or [],
        "total_page": total_page,
    }
    return m


class TestListDisclosures:
    def test_basic_parse(self):
        items = [
            {"rcept_no": "20250228800702", "corp_name": "DB손해보험",
             "report_nm": "사업보고서", "rcept_dt": "20250313", "flr_nm": "DB",
             "pblntf_ty": "A"},
            {"rcept_no": "20250228800688", "corp_name": "DB손해보험",
             "report_nm": "현금배당결정", "rcept_dt": "20250228", "flr_nm": "DB",
             "pblntf_ty": "B"},
        ]
        with patch("tradingagents.dataflows.dart_disclosure._api_key", return_value="dummy"), \
             patch("requests.get", return_value=_fake_resp(items=items)):
            out = list_disclosures("00100114", "2025-02-14", "2025-04-14")
        assert len(out) == 2
        # 날짜 오름차순 정렬
        assert out[0].rcept_dt == "2025-02-28"
        assert out[1].rcept_dt == "2025-03-13"

    def test_no_key_empty(self):
        with patch("tradingagents.dataflows.dart_disclosure._api_key", return_value=None):
            out = list_disclosures("00100114", "2025-02-14", "2025-04-14")
        assert out == []

    def test_status_013_empty(self):
        with patch("tradingagents.dataflows.dart_disclosure._api_key", return_value="dummy"), \
             patch("requests.get", return_value=_fake_resp(status="013")):
            out = list_disclosures("00100114", "2025-02-14", "2025-04-14")
        assert out == []


class TestFilter:
    def test_significant_keywords(self):
        items = [
            Disclosure("1", "DB", "사업보고서", "2025-03-13", None, "A"),
            Disclosure("2", "DB", "임원변경", "2025-04-01", None, "B"),
            Disclosure("3", "DB", "자기주식 취득 결정", "2025-03-27", None, "B"),
            Disclosure("4", "DB", "현금배당결정", "2025-02-28", None, "B"),
        ]
        sig = filter_significant_disclosures(items)
        names = {d.report_nm for d in sig}
        assert "사업보고서" in names
        assert "자기주식 취득 결정" in names
        assert "현금배당결정" in names
        assert "임원변경" not in names


class TestSummarize:
    def test_summary_structure(self):
        items = [
            Disclosure("1", "DB", "사업보고서", "2025-03-13", None, "A"),
            Disclosure("2", "DB", "자기주식소각결정", "2025-03-27", None, "B"),
        ]
        out = summarize_disclosures(items)
        assert out["total"] == 2
        assert out["significant"] == 2
        assert out["by_pblntf_ty"]["A"] == 1
        assert out["by_pblntf_ty"]["B"] == 1
