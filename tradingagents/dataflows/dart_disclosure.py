"""DART list.json — 기간 내 공시 시계열.

기존 ``dart_fundamentals.py`` 의 ``get_dart_fundamentals(ticker, curr_date)`` 는
단일 시점 최신 분기 보고서만 반환. 회고 기간(수개월) 동안 *공시 일자별로*
무엇이 발표됐는지 확인하려면 list.json endpoint 가 필요.

API: https://opendart.fss.or.kr/api/list.json
응답: {"status": "000", "list": [{"rcept_no", "corp_name", "report_nm", "rcept_dt", ...}]}

페이징: page_no 파라미터, 한 호출 최대 100건. 본 모듈은 자동 페이지 루프.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional

import requests

from tradingagents.dataflows.dart_fundamentals import (
    _api_key, _resolve_corp_code,
)

logger = logging.getLogger(__name__)

_LIST_API = "https://opendart.fss.or.kr/api/list.json"
_TIMEOUT = 15.0
_PAGE_SIZE = 100

# pblntf_ty: 공시 유형 (A=정기공시, B=주요사항보고, C=발행공시, D=지분공시,
#                     E=기타공시, F=외부감사관련, G=펀드공시, H=자산유동화)
PBLNTF_REGULAR = "A"
PBLNTF_MAJOR = "B"


@dataclass
class Disclosure:
    rcept_no: str           # 접수번호 (14자리)
    corp_name: str
    report_nm: str
    rcept_dt: str           # YYYY-MM-DD
    flr_nm: Optional[str]   # 제출인
    pblntf_ty: Optional[str]


def _format_date(d: str) -> str:
    return d.replace("-", "")


def _parse_dart_date(s: str) -> str:
    s = (s or "").strip()
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def list_disclosures(
    corp_code: str,
    since: str, until: str,
    *, pblntf_ty: Optional[str] = None,
    max_pages: int = 20,
) -> list[Disclosure]:
    """기간 내 공시 목록.

    Args:
        corp_code: DART 8자리 고유번호 (``_resolve_corp_code`` 결과).
        since, until: ``YYYY-MM-DD``.
        pblntf_ty: ``A``(정기) / ``B``(주요사항) 등. None = 전체.
        max_pages: 페이지 상한 (안전장치).

    Returns:
        date 오름차순 정렬된 Disclosure 리스트. 키 미설정 / 오류 → 빈 리스트.
    """
    key = _api_key()
    if not key:
        logger.warning("DART_API_KEY not set — list_disclosures empty")
        return []

    out: list[Disclosure] = []
    page_no = 1
    while page_no <= max_pages:
        params = {
            "crtfc_key": key,
            "corp_code": corp_code,
            "bgn_de": _format_date(since),
            "end_de": _format_date(until),
            "page_no": page_no,
            "page_count": _PAGE_SIZE,
        }
        if pblntf_ty:
            params["pblntf_ty"] = pblntf_ty
        try:
            resp = requests.get(_LIST_API, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            logger.warning("DART list page %d failed: %s", page_no, exc)
            break

        status = body.get("status")
        if status == "013":
            # 조회된 데이타가 없습니다 — 정상 빈 결과
            break
        if status != "000":
            logger.warning(
                "DART list status %s: %s",
                status, body.get("message"),
            )
            break

        rows = body.get("list") or []
        if not rows:
            break
        for r in rows:
            out.append(Disclosure(
                rcept_no=(r.get("rcept_no") or "").strip(),
                corp_name=(r.get("corp_name") or "").strip(),
                report_nm=(r.get("report_nm") or "").strip(),
                rcept_dt=_parse_dart_date(r.get("rcept_dt") or ""),
                flr_nm=(r.get("flr_nm") or "").strip() or None,
                pblntf_ty=r.get("pblntf_ty"),
            ))
        total_page = body.get("total_page") or 1
        if page_no >= int(total_page):
            break
        page_no += 1

    out.sort(key=lambda d: d.rcept_dt)
    return out


def list_disclosures_for_ticker(
    ticker: str, since: str, until: str,
    *, pblntf_ty: Optional[str] = None,
) -> list[Disclosure]:
    """ticker(6자리) → corp_code 매핑 후 list_disclosures."""
    try:
        corp_code = _resolve_corp_code(ticker)
    except Exception as exc:
        logger.warning("dart_disclosure resolve %s failed: %s", ticker, exc)
        return []
    return list_disclosures(corp_code, since, until, pblntf_ty=pblntf_ty)


def filter_significant_disclosures(
    items: list[Disclosure],
    *, keywords: tuple[str, ...] = (
        "사업보고서", "분기보고서", "반기보고서",
        "자기주식", "자사주", "배당",
        "주요사항", "유상증자", "무상증자",
        "합병", "분할", "주식소각",
    ),
) -> list[Disclosure]:
    """report_nm 키워드 기반 의미있는 공시만 필터."""
    return [d for d in items if any(k in d.report_nm for k in keywords)]


def summarize_disclosures(items: list[Disclosure]) -> dict:
    """공시 시계열 요약 — 총 건수, 유형별, 주요 일자."""
    sig = filter_significant_disclosures(items)
    by_type: dict[str, int] = {}
    for d in items:
        ty = d.pblntf_ty or "?"
        by_type[ty] = by_type.get(ty, 0) + 1
    return {
        "total": len(items),
        "significant": len(sig),
        "by_pblntf_ty": by_type,
        "significant_items": [asdict(d) for d in sig],
    }
