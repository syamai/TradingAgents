"""collect_kis_history._resolve_window 의 endpoint별 end_date 분기 검증.

분봉(minute)은 장 마감 후 당일치를 바로 적재해야 하므로 end=오늘,
수급(investor/program/short)은 KIS가 당일분을 익일 확정하므로 end=어제(T-1).
"""
from datetime import date

import pytest

import collect_kis_history as cch


class _FakeDate:
    """date.today() 만 고정한 스텁 — 날짜 산술은 실제 date 객체로 위임."""

    @staticmethod
    def today():
        return date(2026, 6, 5)  # 금요일


class _FakeStore:
    def __init__(self, last):
        self._last = last

    def last_date(self, ticker, endpoint):
        return self._last


@pytest.fixture
def fixed_today(monkeypatch):
    monkeypatch.setattr(cch, "date", _FakeDate)


class TestResolveWindowEndDate:
    def test_minute_increment_end_is_today(self, fixed_today):
        # 분봉 last 는 "YYYY-MM-DD HH:MM:SS" 타임스탬프
        store = _FakeStore("2026-06-02 15:32:00")
        start, end = cch._resolve_window(store, "005930", "minute", 1, full=False)
        assert end == "2026-06-05"  # 오늘 = 당일치 포함
        assert start == "2026-06-03"  # last+1

    def test_supply_demand_increment_end_is_yesterday(self, fixed_today):
        store = _FakeStore("2026-06-01")
        start, end = cch._resolve_window(store, "005930", "investor", 1, full=False)
        assert end == "2026-06-04"  # 어제 (T-1)
        assert start == "2026-06-02"

    def test_minute_full_end_is_today(self, fixed_today):
        store = _FakeStore(None)
        start, end = cch._resolve_window(store, "005930", "minute", 1, full=True)
        assert end == "2026-06-05"

    def test_supply_demand_full_end_is_yesterday(self, fixed_today):
        store = _FakeStore(None)
        start, end = cch._resolve_window(store, "005930", "short", 1, full=True)
        assert end == "2026-06-04"

    def test_minute_up_to_date_through_today_skips(self, fixed_today):
        # 오늘치까지 이미 있으면 skip
        store = _FakeStore("2026-06-05 15:30:00")
        assert cch._resolve_window(store, "005930", "minute", 1, full=False) is None

    def test_minute_yesterday_stored_still_fetches_today(self, fixed_today):
        # 어제까지만 있으면 오늘치를 받으러 간다 (T-1 이었다면 None 이 됐을 경계)
        store = _FakeStore("2026-06-04 15:30:00")
        win = cch._resolve_window(store, "005930", "minute", 1, full=False)
        assert win == ("2026-06-05", "2026-06-05")

    def test_supply_demand_yesterday_stored_skips(self, fixed_today):
        # 수급은 어제까지 있으면 최신 (오늘분은 아직 미확정)
        store = _FakeStore("2026-06-04")
        assert cch._resolve_window(store, "005930", "investor", 1, full=False) is None
