"""DART company.json — 업종(KSIC) 기반 동종 추출.

수동 회고 §6 동종 비교(삼성/현대/메리츠/흥국)를 자동화하려면 ticker 의
업종 분류가 필요. DART ``company.json`` endpoint 는 ``induty_code`` 5자리
(KSIC) 를 반환한다.

폴백:
1. ``HARD_CODED_PEERS`` 에 있으면 우선. KSIC 가 sub-sector 까지 못 잡는 경우
   (예: 손해보험 vs 생명보험 모두 ``651``) hand-coded 이 정확.
2. 사용자 ``candidate_tickers`` 제공 시 induty_code 동일 항목 매칭.
3. 둘 다 실패 시 빈 리스트.

API: https://opendart.fss.or.kr/api/company.json?crtfc_key={K}&corp_code={CC}
응답 status="000" 시 induty_code/corp_name/stock_code 등 반환.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.dart_fundamentals import (
    _api_key, _load_corp_map, _resolve_corp_code,
)

logger = logging.getLogger(__name__)

_COMPANY_API = "https://opendart.fss.or.kr/api/company.json"
_TIMEOUT = 15.0


# === Hand-coded peers (KSIC 약점 보완) ========================================
# KSIC 5자리만으로는 sub-sector 가 안 잡히는 경우 직접 지정.
# 각 key(종목) 의 value 는 그 종목의 동종 — 종목들끼리 무관.
HARD_CODED_PEERS: dict[str, list[str]] = {
    # 손해보험 4사 — KSIC 651(보험) 만으로는 생명보험까지 섞임
    # 메리츠손해보험(000060) 은 2022 메리츠금융지주 합병으로 상장폐지 → 한화손해 000370 로 교체
    "005830": ["000810", "001450", "000370", "000540"],   # DB → 삼성화재/현대해상/한화손해/흥국화재
    # 엔터테인먼트 4사 (KOSDAQ)
    "035900": ["041510", "122870", "352820"],             # JYP → SM/YG/HYBE
    # 2차전지 셀+소재
    "373220": ["006400", "096770", "003670"],             # LG에솔 → 삼성SDI/SK이노/포스코퓨처엠
}


@dataclass
class CompanyInfo:
    ticker: str               # 6자리 종목코드
    corp_code: str            # 8자리 DART
    corp_name: str
    induty_code: Optional[str]    # KSIC 5자리
    stock_name: Optional[str]
    corp_cls: Optional[str]   # Y=유가증권 / K=코스닥 / N=코넥스 / E=기타


@dataclass
class Peer:
    ticker: str
    corp_code: str
    company_name: str
    induty_code: Optional[str]
    source: str   # "hard_coded" | "ksic" | "manual"


# === 캐시 ====================================================================
def _cache_path() -> str:
    cfg = get_config()
    cache_dir = (
        cfg.get("data_cache_dir")
        or os.path.join(os.path.expanduser("~"), ".tradingagents", "cache")
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "dart_company_info.json")


def _load_company_cache() -> dict[str, dict]:
    p = _cache_path()
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_company_cache(cache: dict[str, dict]) -> None:
    p = _cache_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except OSError as exc:
        logger.warning("dart_industry cache write failed: %s", exc)


# === company.json ===========================================================
def fetch_company_info(ticker: str, *, cache: bool = True) -> Optional[CompanyInfo]:
    """company.json 호출 → CompanyInfo.

    캐시 우선. 키 미설정 / 매핑 실패 / API 오류 → None.
    """
    key = _api_key()
    if not key:
        logger.warning("DART_API_KEY not set — fetch_company_info skip")
        return None

    cache_data = _load_company_cache() if cache else {}
    if cache and ticker in cache_data:
        c = cache_data[ticker]
        return CompanyInfo(
            ticker=c["ticker"], corp_code=c["corp_code"],
            corp_name=c["corp_name"], induty_code=c.get("induty_code"),
            stock_name=c.get("stock_name"), corp_cls=c.get("corp_cls"),
        )

    try:
        corp_code = _resolve_corp_code(ticker)
    except Exception as exc:
        logger.warning("resolve corp_code(%s) failed: %s", ticker, exc)
        return None

    try:
        resp = requests.get(
            _COMPANY_API,
            params={"crtfc_key": key, "corp_code": corp_code},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        logger.warning("company.json(%s) failed: %s", ticker, exc)
        return None

    if body.get("status") != "000":
        logger.warning(
            "company.json status %s: %s",
            body.get("status"), body.get("message"),
        )
        return None

    info = CompanyInfo(
        ticker=ticker,
        corp_code=corp_code,
        corp_name=(body.get("corp_name") or "").strip(),
        induty_code=(body.get("induty_code") or "").strip() or None,
        stock_name=(body.get("stock_name") or "").strip() or None,
        corp_cls=(body.get("corp_cls") or "").strip() or None,
    )

    if cache:
        cache_data[ticker] = asdict(info)
        _save_company_cache(cache_data)
    return info


def get_induty_code(ticker: str) -> Optional[tuple[str, str]]:
    """ticker → (induty_code, corp_name). 실패 시 None."""
    info = fetch_company_info(ticker)
    if not info or not info.induty_code:
        return None
    return info.induty_code, info.corp_name


# === peers 추출 =============================================================
def _match_induty(target: str, other: str, *, prefix_len: int = 5) -> bool:
    """KSIC 코드 prefix 매칭. 기본 5자리(완전 일치). prefix_len=3 시 대분류."""
    if not target or not other:
        return False
    n = min(prefix_len, len(target), len(other))
    return target[:n] == other[:n]


def find_peers(
    ticker: str,
    *,
    n: int = 4,
    candidate_tickers: Optional[list[str]] = None,
    induty_prefix_len: int = 5,
) -> list[Peer]:
    """동종 종목 추출.

    1. ``HARD_CODED_PEERS`` 에 있으면 그것 사용 (KSIC sub-sector 약점 보완).
    2. ``candidate_tickers`` 제공 시 induty_code 매칭 (최대 ``n`` 개).
    3. 둘 다 실패 시 빈 리스트.

    Args:
        ticker: 6자리 종목코드.
        n: 반환 개수 상한.
        candidate_tickers: 매칭 후보 (보통 holdings 종목 list).
        induty_prefix_len: KSIC prefix 길이 — 5(완전) / 3(대분류 fallback).

    Returns:
        ``Peer`` 리스트. ``source`` 필드로 매칭 방식 구분.
    """
    # 1. hand-coded fallback 우선
    if ticker in HARD_CODED_PEERS:
        out: list[Peer] = []
        cache = _load_company_cache()
        corp_map = {}
        try:
            corp_map = _load_corp_map()
        except Exception:
            pass
        for peer_ticker in HARD_CODED_PEERS[ticker][:n]:
            corp_code = corp_map.get(peer_ticker, "")
            cached = cache.get(peer_ticker, {})
            out.append(Peer(
                ticker=peer_ticker,
                corp_code=corp_code,
                company_name=cached.get("corp_name") or peer_ticker,
                induty_code=cached.get("induty_code"),
                source="hard_coded",
            ))
        return out

    # 2. candidate_tickers 매칭
    if candidate_tickers:
        target = fetch_company_info(ticker)
        if not target or not target.induty_code:
            return []
        matched: list[Peer] = []
        for cand in candidate_tickers:
            if cand == ticker:
                continue
            info = fetch_company_info(cand)
            if not info:
                continue
            if _match_induty(
                target.induty_code, info.induty_code or "",
                prefix_len=induty_prefix_len,
            ):
                matched.append(Peer(
                    ticker=info.ticker, corp_code=info.corp_code,
                    company_name=info.corp_name,
                    induty_code=info.induty_code,
                    source="ksic",
                ))
            if len(matched) >= n:
                break
        return matched

    return []


def register_peers(ticker: str, peers: list[str]) -> None:
    """수동 등록 — ``HARD_CODED_PEERS`` 런타임 갱신.

    UI 에서 ``text_input`` 으로 받은 콤마 구분 ticker 등록용.
    영구 저장은 하지 않음 (코드 수정으로 처리).
    """
    HARD_CODED_PEERS[ticker] = list(peers)
