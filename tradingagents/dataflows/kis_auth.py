"""KIS(한국투자증권 OpenAPI) OAuth 인증 + 토큰 캐시.

자격증명: ``KIS_APP_KEY`` / ``KIS_APP_SECRET`` 환경변수 (DART 패턴 따름 —
default_config에 노출하지 않음).
환경 토글: ``KIS_ENV=mock|real``, 기본 ``mock``.

KIS 토큰 정책 (https://apiportal.koreainvestment.com 문서 기준):
- 발급 후 24시간 유효
- 동일 appkey로 **6시간 이내 재발급 거부** ("이미 발급된 토큰이 유효함")
- 따라서 issued_at + 6h 경과 + 만료 1h 전일 때만 refresh

캐시 파일은 mock / real 환경별로 분리 — 같은 디렉터리에서 환경을 토글하면
서로의 토큰을 덮어쓰지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Optional

import requests

from .config import get_config

logger = logging.getLogger(__name__)

_REAL_BASE = "https://openapi.koreainvestment.com:9443"
_MOCK_BASE = "https://openapivts.koreainvestment.com:29443"
_TOKEN_PATH = "/oauth2/tokenP"
_REQUEST_TIMEOUT = 15.0

_REFRESH_MIN_AGE_SEC = 6 * 3600   # KIS 신규 발급 허용 최소 간격
_REFRESH_WINDOW_SEC = 1 * 3600    # 만료 1h 전부터 refresh 후보


class KisCredentialError(RuntimeError):
    """``KIS_APP_KEY`` / ``KIS_APP_SECRET``가 설정되지 않았을 때."""


def _kis_env() -> str:
    env = (os.environ.get("KIS_ENV") or "mock").strip().lower()
    if env not in ("mock", "real"):
        raise ValueError(f"KIS_ENV must be 'mock' or 'real', got {env!r}")
    return env


def _base_url(env: Optional[str] = None) -> str:
    env = env or _kis_env()
    return _MOCK_BASE if env == "mock" else _REAL_BASE


def _credentials() -> tuple[str, str]:
    appkey = os.environ.get("KIS_APP_KEY") or ""
    appsecret = os.environ.get("KIS_APP_SECRET") or ""
    if not appkey or not appsecret:
        raise KisCredentialError(
            "KIS_APP_KEY and KIS_APP_SECRET must be set in environment"
        )
    return appkey, appsecret


def _token_cache_path(env: Optional[str] = None) -> Path:
    env = env or _kis_env()
    cfg = get_config()
    cache_dir = cfg.get("data_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".tradingagents", "cache"
    )
    os.makedirs(cache_dir, exist_ok=True)
    return Path(cache_dir) / f"kis_oauth_token_{env}.json"


def _load_cached_token(env: Optional[str] = None) -> Optional[dict]:
    path = _token_cache_path(env)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("KIS token cache load failed (%s): %s", path, exc)
        return None
    for key in ("access_token", "issued_at", "expires_at"):
        if key not in data:
            return None
    return data


def _save_cached_token(token_data: dict, env: Optional[str] = None) -> None:
    path = _token_cache_path(env)
    with path.open("w", encoding="utf-8") as f:
        json.dump(token_data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError as exc:
        # 권한 변경 실패는 치명적 아님 — 경고만 남기고 계속
        logger.warning("Could not chmod 0600 on %s: %s", path, exc)


def _should_refresh(token_data: dict, now: Optional[float] = None) -> bool:
    now = now if now is not None else time.time()
    expires_at = float(token_data["expires_at"])
    issued_at = float(token_data["issued_at"])
    if now >= expires_at:
        return True
    if (expires_at - now) < _REFRESH_WINDOW_SEC and (now - issued_at) >= _REFRESH_MIN_AGE_SEC:
        return True
    return False


def _request_new_token(env: Optional[str] = None) -> dict:
    """POST /oauth2/tokenP — 자격증명 누락 시 KisCredentialError."""
    env = env or _kis_env()
    appkey, appsecret = _credentials()
    url = _base_url(env) + _TOKEN_PATH
    payload = {
        "grant_type": "client_credentials",
        "appkey": appkey,
        "appsecret": appsecret,
    }
    resp = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()
    access_token = body.get("access_token")
    expires_in = body.get("expires_in")
    if not access_token or not expires_in:
        raise RuntimeError(f"unexpected KIS token response: {body!r}")
    now = time.time()
    return {
        "access_token": access_token,
        "issued_at": now,
        "expires_at": now + float(expires_in),
        "env": env,
    }


def get_access_token() -> str:
    """Public: 캐시 hit이면 그대로, 없거나 refresh 조건이면 신규 발급."""
    env = _kis_env()
    cached = _load_cached_token(env)
    if cached and not _should_refresh(cached):
        return cached["access_token"]
    fresh = _request_new_token(env)
    _save_cached_token(fresh, env)
    return fresh["access_token"]


def invalidate_cached_token(env: Optional[str] = None) -> None:
    """401 응답을 받은 호출자가 호출 — 다음 ``get_access_token``이 refresh."""
    path = _token_cache_path(env)
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not delete %s: %s", path, exc)
