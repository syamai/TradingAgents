"""dashboard.app 의 streamlit-독립 헬퍼 단위 테스트.

UI(`st.*`) 호출은 streamlit runtime 없이 테스트 어려움 — 여기서는
순수 함수만 검증한다.
"""
from __future__ import annotations

import pytest

from dashboard.app import _filter_tickers, _make_searchbox_options


SAMPLE = [
    ("005930", "삼성전자", "KOSPI"),
    ("000660", "SK하이닉스", "KOSPI"),
    ("034220", "LG디스플레이", "KOSPI"),
    ("035720", "카카오", "KOSPI"),
    ("247540", "에코프로비엠", "KOSDAQ"),
]


@pytest.mark.unit
class TestFilterTickers:
    def test_empty_query_returns_all(self):
        assert _filter_tickers(SAMPLE, "") == SAMPLE

    def test_whitespace_query_returns_all(self):
        assert _filter_tickers(SAMPLE, "   ") == SAMPLE

    def test_match_company_name_substring(self):
        out = _filter_tickers(SAMPLE, "삼성")
        assert out == [("005930", "삼성전자", "KOSPI")]

    def test_match_ticker_code(self):
        out = _filter_tickers(SAMPLE, "034220")
        assert out == [("034220", "LG디스플레이", "KOSPI")]

    def test_match_ticker_prefix(self):
        out = _filter_tickers(SAMPLE, "0357")
        assert out == [("035720", "카카오", "KOSPI")]

    def test_case_insensitive(self):
        out = _filter_tickers(SAMPLE, "sk")
        assert out == [("000660", "SK하이닉스", "KOSPI")]
        out2 = _filter_tickers(SAMPLE, "SK")
        assert out2 == out

    def test_no_match_returns_empty(self):
        assert _filter_tickers(SAMPLE, "존재하지않는종목") == []

    def test_market_is_not_searched(self):
        # KOSPI 4종 중 회사명/코드에 'kospi' 들어간 종목 없음 → 빈 리스트
        assert _filter_tickers(SAMPLE, "KOSPI") == []

    def test_partial_match_multiple_hits(self):
        # '에코' 는 1종, 'L' 은 LG디스플레이 + (대소문자 무시로 카카오는 없음)
        out = _filter_tickers(SAMPLE, "에코")
        assert out == [("247540", "에코프로비엠", "KOSDAQ")]


@pytest.mark.unit
class TestMakeSearchboxOptions:
    def test_returns_label_value_tuples(self):
        out = _make_searchbox_options(SAMPLE, "삼성")
        assert out == [("005930 — 삼성전자 (KOSPI)", "005930")]

    def test_empty_query_returns_all_capped(self):
        out = _make_searchbox_options(SAMPLE, "")
        # SAMPLE은 5종 → cap(30) 영향 없음, 전체 반환
        assert len(out) == len(SAMPLE)
        # value 는 ticker 코드
        assert [v for _, v in out] == [t for t, _, _ in SAMPLE]

    def test_caps_at_30_results(self):
        many = [
            (f"{i:06d}", f"종목{i}", "KOSPI") for i in range(50)
        ]
        out = _make_searchbox_options(many, "")
        assert len(out) == 30

    def test_no_match_returns_empty(self):
        assert _make_searchbox_options(SAMPLE, "없는단어") == []
