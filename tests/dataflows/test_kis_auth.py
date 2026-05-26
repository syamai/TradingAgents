"""Unit tests for KIS OAuth auth + token cache."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.dataflows import kis_auth


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Isolate token cache to tmp_path by overriding get_config."""
    def _fake_cfg():
        return {"data_cache_dir": str(tmp_path)}
    monkeypatch.setattr(kis_auth, "get_config", _fake_cfg)
    return tmp_path


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ENV"):
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


@pytest.mark.unit
class TestEnvAndUrl:
    def test_default_env_is_mock(self, clean_env):
        assert kis_auth._kis_env() == "mock"

    def test_explicit_real_env(self, clean_env):
        clean_env.setenv("KIS_ENV", "real")
        assert kis_auth._kis_env() == "real"

    def test_invalid_env_raises(self, clean_env):
        clean_env.setenv("KIS_ENV", "prod")
        with pytest.raises(ValueError):
            kis_auth._kis_env()

    def test_base_url_mock(self):
        assert "openapivts" in kis_auth._base_url("mock")

    def test_base_url_real(self):
        assert kis_auth._base_url("real") == "https://openapi.koreainvestment.com:9443"


@pytest.mark.unit
class TestCredentials:
    def test_missing_raises(self, clean_env):
        with pytest.raises(kis_auth.KisCredentialError):
            kis_auth._credentials()

    def test_both_set_returns_tuple(self, clean_env):
        clean_env.setenv("KIS_APP_KEY", "k")
        clean_env.setenv("KIS_APP_SECRET", "s")
        assert kis_auth._credentials() == ("k", "s")

    def test_only_appkey_set_still_raises(self, clean_env):
        clean_env.setenv("KIS_APP_KEY", "k")
        with pytest.raises(kis_auth.KisCredentialError):
            kis_auth._credentials()


@pytest.mark.unit
class TestRefreshPolicy:
    def _token(self, issued_offset, expires_offset, now=None):
        now = now if now is not None else time.time()
        return {
            "access_token": "t",
            "issued_at": now + issued_offset,
            "expires_at": now + expires_offset,
        }

    def test_expired_token_refreshes(self):
        tok = self._token(-25 * 3600, -3600)  # 25h ago issued, expired 1h ago
        assert kis_auth._should_refresh(tok) is True

    def test_just_issued_does_not_refresh(self):
        tok = self._token(-60, 24 * 3600 - 60)  # 1 min ago, 24h-1min left
        assert kis_auth._should_refresh(tok) is False

    def test_5h_old_with_expiry_far_does_not_refresh(self):
        # 5h old, 19h left — within KIS 6h refresh lockout
        tok = self._token(-5 * 3600, 19 * 3600)
        assert kis_auth._should_refresh(tok) is False

    def test_5h_old_near_expiry_still_does_not_refresh(self):
        # 5h old but only 30min left — refresh window OK but lockout still active.
        # KIS would reject — must wait for expiry.
        tok = self._token(-5 * 3600, 1800)
        assert kis_auth._should_refresh(tok) is False

    def test_7h_old_with_expiry_far_does_not_refresh(self):
        # Past 6h lockout but expiry still 17h away — no need yet
        tok = self._token(-7 * 3600, 17 * 3600)
        assert kis_auth._should_refresh(tok) is False

    def test_7h_old_near_expiry_refreshes(self):
        # Past 6h lockout AND within 1h of expiry → refresh
        tok = self._token(-7 * 3600, 1800)
        assert kis_auth._should_refresh(tok) is True


@pytest.mark.unit
class TestTokenCacheIO:
    def test_cache_path_includes_env(self, isolated_cache, clean_env):
        clean_env.setenv("KIS_ENV", "real")
        path = kis_auth._token_cache_path()
        assert path.name == "kis_oauth_token_real.json"

    def test_cache_path_mock_default(self, isolated_cache, clean_env):
        path = kis_auth._token_cache_path()
        assert path.name == "kis_oauth_token_mock.json"

    def test_save_writes_0600(self, isolated_cache, clean_env):
        tok = {"access_token": "x", "issued_at": 0.0, "expires_at": 86400.0}
        kis_auth._save_cached_token(tok)
        path = kis_auth._token_cache_path()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    def test_load_round_trip(self, isolated_cache, clean_env):
        tok = {"access_token": "x", "issued_at": 1.0, "expires_at": 2.0}
        kis_auth._save_cached_token(tok)
        loaded = kis_auth._load_cached_token()
        assert loaded["access_token"] == "x"

    def test_load_missing_returns_none(self, isolated_cache, clean_env):
        assert kis_auth._load_cached_token() is None

    def test_load_corrupt_returns_none(self, isolated_cache, clean_env):
        path = kis_auth._token_cache_path()
        path.write_text("{not json")
        assert kis_auth._load_cached_token() is None

    def test_load_missing_field_returns_none(self, isolated_cache, clean_env):
        path = kis_auth._token_cache_path()
        path.write_text(json.dumps({"access_token": "x"}))  # no issued_at/expires_at
        assert kis_auth._load_cached_token() is None


@pytest.mark.unit
class TestGetAccessToken:
    def test_uses_cached_when_fresh(self, isolated_cache, clean_env):
        clean_env.setenv("KIS_APP_KEY", "k")
        clean_env.setenv("KIS_APP_SECRET", "s")
        now = time.time()
        kis_auth._save_cached_token({
            "access_token": "cached-token",
            "issued_at": now - 60,
            "expires_at": now + 86340,
        })
        with patch("tradingagents.dataflows.kis_auth.requests.post") as post:
            assert kis_auth.get_access_token() == "cached-token"
            assert post.call_count == 0

    def test_requests_new_when_no_cache(self, isolated_cache, clean_env):
        clean_env.setenv("KIS_APP_KEY", "k")
        clean_env.setenv("KIS_APP_SECRET", "s")
        with patch("tradingagents.dataflows.kis_auth.requests.post") as post:
            post.return_value.raise_for_status = lambda: None
            post.return_value.json = lambda: {
                "access_token": "fresh-token",
                "expires_in": 86400,
                "token_type": "Bearer",
            }
            assert kis_auth.get_access_token() == "fresh-token"
            assert post.call_count == 1
        # Verify cache was written
        cached = kis_auth._load_cached_token()
        assert cached["access_token"] == "fresh-token"

    def test_refreshes_when_expired(self, isolated_cache, clean_env):
        clean_env.setenv("KIS_APP_KEY", "k")
        clean_env.setenv("KIS_APP_SECRET", "s")
        now = time.time()
        kis_auth._save_cached_token({
            "access_token": "old-token",
            "issued_at": now - 25 * 3600,
            "expires_at": now - 3600,  # expired
        })
        with patch("tradingagents.dataflows.kis_auth.requests.post") as post:
            post.return_value.raise_for_status = lambda: None
            post.return_value.json = lambda: {
                "access_token": "new-token",
                "expires_in": 86400,
            }
            assert kis_auth.get_access_token() == "new-token"
            assert post.call_count == 1

    def test_missing_credentials_raises(self, isolated_cache, clean_env):
        with pytest.raises(kis_auth.KisCredentialError):
            kis_auth.get_access_token()

    def test_unexpected_response_raises(self, isolated_cache, clean_env):
        clean_env.setenv("KIS_APP_KEY", "k")
        clean_env.setenv("KIS_APP_SECRET", "s")
        with patch("tradingagents.dataflows.kis_auth.requests.post") as post:
            post.return_value.raise_for_status = lambda: None
            post.return_value.json = lambda: {"error": "bad"}
            with pytest.raises(RuntimeError):
                kis_auth.get_access_token()


@pytest.mark.unit
class TestInvalidate:
    def test_invalidate_deletes_cache(self, isolated_cache, clean_env):
        kis_auth._save_cached_token({
            "access_token": "x", "issued_at": 0.0, "expires_at": 86400.0
        })
        path = kis_auth._token_cache_path()
        assert path.exists()
        kis_auth.invalidate_cached_token()
        assert not path.exists()

    def test_invalidate_when_no_cache_is_noop(self, isolated_cache, clean_env):
        # Should not raise
        kis_auth.invalidate_cached_token()
