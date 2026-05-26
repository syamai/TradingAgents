"""Unit tests for KIS HTTP thin client (kis_api)."""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from tradingagents.dataflows import kis_api, kis_auth


_KST = ZoneInfo("Asia/Seoul")


# === fixtures ===

@pytest.fixture
def kis_env(monkeypatch):
    """Apply KIS env vars FIRST — must run before isolated_cache so
    ``_save_cached_token`` uses the mock env (otherwise system .env's
    ``KIS_ENV=real`` leaks in and the token file gets saved under
    ``kis_oauth_token_real.json``, missing the mock-env lookup later)."""
    monkeypatch.setenv("KIS_APP_KEY", "k")
    monkeypatch.setenv("KIS_APP_SECRET", "s")
    monkeypatch.setenv("KIS_ENV", "mock")
    yield monkeypatch


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch, kis_env):
    """Isolate both auth-token cache and kis response cache to tmp_path.

    Depends on ``kis_env`` so env is set before the seed token is written
    — otherwise the seed token lands in the wrong env-specific cache file.
    """
    def _fake_cfg():
        return {"data_cache_dir": str(tmp_path)}
    monkeypatch.setattr(kis_api, "get_config", _fake_cfg)
    monkeypatch.setattr(kis_auth, "get_config", _fake_cfg)
    # Seed a valid token cache so _call doesn't hit network for auth
    kis_auth._save_cached_token({
        "access_token": "test-token",
        "issued_at": time.time() - 60,
        "expires_at": time.time() + 86340,
    })
    return tmp_path


def _mock_response(status=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json = lambda: json_body or {}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(f"{status}")
    return resp


# === market hours / cache policy ===

@pytest.mark.unit
class TestMarketOpen:
    def test_weekday_during_hours_is_open(self):
        # 2026-05-27 (Wednesday) 11:00 KST
        now = datetime(2026, 5, 27, 11, 0, tzinfo=_KST)
        assert kis_api._is_market_open(now) is True

    def test_weekday_before_open_is_closed(self):
        now = datetime(2026, 5, 27, 8, 59, tzinfo=_KST)
        assert kis_api._is_market_open(now) is False

    def test_weekday_after_close_is_closed(self):
        now = datetime(2026, 5, 27, 15, 30, tzinfo=_KST)
        assert kis_api._is_market_open(now) is False

    def test_saturday_is_closed(self):
        now = datetime(2026, 5, 30, 11, 0, tzinfo=_KST)
        assert kis_api._is_market_open(now) is False

    def test_sunday_is_closed(self):
        now = datetime(2026, 5, 31, 11, 0, tzinfo=_KST)
        assert kis_api._is_market_open(now) is False


@pytest.mark.unit
class TestNormalizeEndDate:
    """KIS investor endpoint의 'TIME LIMIT 00:00 ~ 15:40' 우회 — end=오늘 &
    cutoff 이전이면 어제로 슬라이드.
    """

    def test_today_before_cutoff_slides_to_yesterday(self):
        # KST 2026-05-27 11:00 — cutoff(15:40) 이전, end=오늘 → 어제
        now = datetime(2026, 5, 27, 11, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("20260527", now) == "20260526"

    def test_today_at_cutoff_keeps_today(self):
        # 15:40 정각 — 슬라이드 안 함
        now = datetime(2026, 5, 27, 15, 40, tzinfo=_KST)
        assert kis_api._normalize_end_date("20260527", now) == "20260527"

    def test_today_after_cutoff_keeps_today(self):
        now = datetime(2026, 5, 27, 16, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("20260527", now) == "20260527"

    def test_yesterday_kept_as_is_before_cutoff(self):
        # end=어제는 cutoff 무관 — 과거 데이터는 시간 무관 호출 가능
        now = datetime(2026, 5, 27, 1, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("20260526", now) == "20260526"

    def test_far_past_date_kept(self):
        now = datetime(2026, 5, 27, 1, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("20250101", now) == "20250101"

    def test_future_date_before_cutoff_slides(self):
        # end > today (잘못된 입력이지만) cutoff 이전이면 슬라이드 적용
        now = datetime(2026, 5, 27, 1, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("20260601", now) == "20260526"

    def test_midnight_edge(self):
        # KST 00:00 — cutoff 이전이므로 슬라이드
        now = datetime(2026, 5, 27, 0, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("20260527", now) == "20260526"

    def test_invalid_format_passthrough(self):
        now = datetime(2026, 5, 27, 11, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("bogus", now) == "bogus"

    def test_month_boundary_slides_to_prev_month(self):
        # 6월 1일 새벽 → 5월 31일로 슬라이드
        now = datetime(2026, 6, 1, 9, 0, tzinfo=_KST)
        assert kis_api._normalize_end_date("20260601", now) == "20260531"


@pytest.mark.unit
class TestShouldSkipCache:
    def test_past_date_caches(self, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        # end is yesterday in any timezone — should cache
        from datetime import date, timedelta
        past = (date.today() - timedelta(days=2)).strftime("%Y%m%d")
        assert kis_api._should_skip_cache(past) is False

    def test_today_during_market_skips(self, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        today = datetime.now(_KST).strftime("%Y%m%d")
        assert kis_api._should_skip_cache(today) is True

    def test_today_after_close_caches(self, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: False)
        today = datetime.now(_KST).strftime("%Y%m%d")
        assert kis_api._should_skip_cache(today) is False

    def test_invalid_date_skips_to_be_safe(self):
        assert kis_api._should_skip_cache("bogus") is True


# === parser helpers ===

@pytest.mark.unit
class TestParsers:
    def test_safe_int_none_zero(self):
        assert kis_api._safe_int(None) == 0
        assert kis_api._safe_int("") == 0

    def test_safe_int_signed(self):
        assert kis_api._safe_int("-12345") == -12345

    def test_safe_int_float_string(self):
        assert kis_api._safe_int("123.0") == 123

    def test_safe_int_garbage(self):
        assert kis_api._safe_int("abc") == 0

    def test_safe_float_none(self):
        assert kis_api._safe_float(None) == 0.0

    def test_safe_float_pct(self):
        assert kis_api._safe_float("12.34") == pytest.approx(12.34)

    def test_format_date_yyyymmdd(self):
        assert kis_api._format_date("20260527") == "2026-05-27"

    def test_format_date_passthrough_when_invalid(self):
        assert kis_api._format_date("bad") == "bad"
        assert kis_api._format_date("") == ""


# === _call: auth retry / error handling ===

@pytest.mark.unit
class TestCall:
    def test_success_returns_body(self, isolated_cache, kis_env):
        body = {"rt_cd": "0", "output2": []}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            assert kis_api._call("/path", "TR", {"a": "b"}) == body
            assert get.call_count == 1

    def test_401_retries_with_new_token(self, isolated_cache, kis_env):
        success_body = {"rt_cd": "0", "output2": []}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get, \
             patch("tradingagents.dataflows.kis_auth.requests.post") as post:
            get.side_effect = [
                _mock_response(401, {}),
                _mock_response(200, success_body),
            ]
            post.return_value.raise_for_status = lambda: None
            post.return_value.json = lambda: {
                "access_token": "fresh", "expires_in": 86400,
            }
            assert kis_api._call("/p", "TR", {}) == success_body
            assert get.call_count == 2
            assert post.call_count == 1  # token refresh once

    def test_403_retries_once_then_propagates(self, isolated_cache, kis_env):
        with patch("tradingagents.dataflows.kis_api.requests.get") as get, \
             patch("tradingagents.dataflows.kis_auth.requests.post") as post:
            get.side_effect = [_mock_response(403, {}), _mock_response(403, {})]
            post.return_value.raise_for_status = lambda: None
            post.return_value.json = lambda: {"access_token": "x", "expires_in": 86400}
            from requests.exceptions import HTTPError
            with pytest.raises(HTTPError):
                kis_api._call("/p", "TR", {})
            assert get.call_count == 2  # original + 1 retry

    def test_rt_cd_nonzero_raises(self, isolated_cache, kis_env):
        body = {"rt_cd": "9", "msg1": "기존 토큰 사용중"}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            with pytest.raises(RuntimeError, match="rt_cd=9"):
                kis_api._call("/p", "TR", {})

    def test_headers_include_required_keys(self, isolated_cache, kis_env):
        body = {"rt_cd": "0"}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            kis_api._call("/p", "MY_TR_ID", {})
            headers = get.call_args.kwargs["headers"]
            assert headers["authorization"] == "Bearer test-token"
            assert headers["appkey"] == "k"
            assert headers["appsecret"] == "s"
            assert headers["tr_id"] == "MY_TR_ID"
            assert headers["custtype"] == "P"


# === fetchers: response parsing + lookback trimming ===

_INVESTOR_OUTPUT2_SAMPLE = [
    {
        "stck_bsop_date": "20260527", "stck_clpr": "75300",
        "frgn_ntby_qty": "-123456", "orgn_ntby_qty": "234567", "prsn_ntby_qty": "-111111",
        "frgn_ntby_tr_pbmn": "-9302092800", "orgn_ntby_tr_pbmn": "17668994100",
        "prsn_ntby_tr_pbmn": "-8366901300",
    },
    {
        "stck_bsop_date": "20260526", "stck_clpr": "75100",
        "frgn_ntby_qty": "100000", "orgn_ntby_qty": "-50000", "prsn_ntby_qty": "-50000",
        "frgn_ntby_tr_pbmn": "7510000000", "orgn_ntby_tr_pbmn": "-3755000000",
        "prsn_ntby_tr_pbmn": "-3755000000",
    },
    # ... 30 days total typically — for test, 2 rows enough
]


@pytest.mark.unit
class TestFetchInvestorTrend:
    def test_parses_fields_and_trims(self, isolated_cache, kis_env, monkeypatch):
        # Force market open to avoid caching side effect interfering
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        body = {"rt_cd": "0", "output2": _INVESTOR_OUTPUT2_SAMPLE}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            rows = kis_api.fetch_investor_trend("005930", "2026-05-27", lookback_days=7)
            assert len(rows) == 2
            assert rows[0] == {
                "date": "2026-05-27", "close": 75300,
                "foreign_qty": -123456, "institution_qty": 234567, "retail_qty": -111111,
                "foreign_amount": -9302092800, "institution_amount": 17668994100,
                "retail_amount": -8366901300,
            }

    def test_lookback_3_truncates(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        many = [{"stck_bsop_date": f"2026052{i}"} for i in range(7)]  # 7 rows
        body = {"rt_cd": "0", "output2": many}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            rows = kis_api.fetch_investor_trend("005930", "2026-05-27", lookback_days=3)
            assert len(rows) == 3

    def test_request_params_strip_dashes(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, {"rt_cd": "0", "output2": []})
            # Past date — not subject to _normalize_end_date sliding
            kis_api.fetch_investor_trend("005930", "2026-05-20")
            params = get.call_args.kwargs["params"]
            assert params["FID_INPUT_DATE_1"] == "20260520"
            assert params["FID_INPUT_ISCD"] == "005930"
            assert params["FID_COND_MRKT_DIV_CODE"] == "J"

    def test_caches_when_market_closed(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: False)
        body = {"rt_cd": "0", "output2": _INVESTOR_OUTPUT2_SAMPLE}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            kis_api.fetch_investor_trend("005930", "2026-05-26", lookback_days=7)
            kis_api.fetch_investor_trend("005930", "2026-05-26", lookback_days=7)
            assert get.call_count == 1  # second call hit cache

    def test_skips_cache_during_market(self, isolated_cache, kis_env, monkeypatch):
        # _should_skip_cache reasons about the *requested* end_date being today.
        # _normalize_end_date may have slid an end=today to yesterday before
        # cache lookup runs — to exercise the in-market cache skip we feed a
        # future date that won't be slid (the slide only fires for end>=today;
        # here we patch _normalize_end_date as a no-op to isolate the cache logic).
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        monkeypatch.setattr(kis_api, "_normalize_end_date", lambda d, *a: d)
        today = datetime.now(_KST).strftime("%Y-%m-%d")
        body = {"rt_cd": "0", "output2": _INVESTOR_OUTPUT2_SAMPLE}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            kis_api.fetch_investor_trend("005930", today)
            kis_api.fetch_investor_trend("005930", today)
            assert get.call_count == 2

    def test_different_lookback_different_cache(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: False)
        body = {"rt_cd": "0", "output2": _INVESTOR_OUTPUT2_SAMPLE * 5}  # 10 rows
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            kis_api.fetch_investor_trend("005930", "2026-05-26", lookback_days=3)
            kis_api.fetch_investor_trend("005930", "2026-05-26", lookback_days=7)
            assert get.call_count == 2  # different lookback → different cache key

    def test_today_end_date_slides_to_yesterday_in_request(
        self, isolated_cache, kis_env, monkeypatch
    ):
        """fetcher가 호출되어 KIS에 도달할 때 FID_INPUT_DATE_1이 어제로 슬라이드됐는지."""
        from datetime import datetime as _dt
        # Freeze "now" to 2026-05-27 01:00 KST — before 15:40 cutoff
        fixed_now = _dt(2026, 5, 27, 1, 0, tzinfo=_KST)
        monkeypatch.setattr(
            "tradingagents.dataflows.kis_api.datetime",
            type("MockDT", (), {
                "now": staticmethod(lambda tz=None: fixed_now),
                "strptime": _dt.strptime,
            }),
        )
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, {"rt_cd": "0", "output2": []})
            kis_api.fetch_investor_trend("005930", "2026-05-27")
            params = get.call_args.kwargs["params"]
            assert params["FID_INPUT_DATE_1"] == "20260526"  # slid to yesterday


@pytest.mark.unit
class TestFetchProgramTrading:
    def test_parses_single_output(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        body = {"rt_cd": "0", "output": [{
            "stck_bsop_date": "20260527", "stck_clpr": "75300",
            "whol_smtn_ntby_qty": "150000",
            "whol_smtn_ntby_tr_pbmn": "11295000000",
        }]}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            rows = kis_api.fetch_program_trading("005930", "2026-05-27")
            assert rows == [{
                "date": "2026-05-27", "close": 75300,
                "net_qty": 150000, "net_amount": 11295000000,
            }]

    def test_empty_response_ok(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, {"rt_cd": "0", "output": []})
            assert kis_api.fetch_program_trading("005930", "2026-05-27") == []


@pytest.mark.unit
class TestFetchShortInterest:
    def test_parses_output2(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        body = {"rt_cd": "0", "output2": [{
            "stck_bsop_date": "20260527", "stck_clpr": "75300",
            "ssts_cntg_qty": "55555", "ssts_vol_rlim": "3.21",
            "ssts_tr_pbmn": "4181626500", "ssts_tr_pbmn_rlim": "3.05",
        }]}
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, body)
            rows = kis_api.fetch_short_interest("005930", "2026-05-20", "2026-05-27")
            assert rows == [{
                "date": "2026-05-27", "close": 75300,
                "short_qty": 55555, "short_volume_ratio": pytest.approx(3.21),
                "short_amount": 4181626500, "short_amount_ratio": pytest.approx(3.05),
            }]

    def test_passes_start_and_end(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        with patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, {"rt_cd": "0", "output2": []})
            # Past end_date — not subject to _normalize_end_date sliding
            kis_api.fetch_short_interest("005930", "2026-05-13", "2026-05-20")
            params = get.call_args.kwargs["params"]
            assert params["FID_INPUT_DATE_1"] == "20260513"
            assert params["FID_INPUT_DATE_2"] == "20260520"


# === rate limiting ===

@pytest.mark.unit
class TestRateLimit:
    def test_sleeps_between_calls(self, isolated_cache, kis_env, monkeypatch):
        monkeypatch.setattr(kis_api, "_is_market_open", lambda *a: True)
        kis_api._last_call_ts = 0.0  # reset
        with patch("tradingagents.dataflows.kis_api.time.sleep") as slp, \
             patch("tradingagents.dataflows.kis_api.requests.get") as get:
            get.return_value = _mock_response(200, {"rt_cd": "0", "output2": []})
            # Two distinct past dates → distinct cache keys, both miss cache,
            # both hit HTTP, second call must wait for rate limit
            kis_api.fetch_investor_trend("005930", "2026-05-20")
            kis_api.fetch_investor_trend("005930", "2026-05-21")
            assert slp.call_count >= 1
