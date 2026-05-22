"""DART(전자공시) API — 한국 기업의 정밀 재무제표.

엔드포인트:
  - 기업 코드 다운로드: ``opendart.fss.or.kr/api/corpCode.xml`` (zip)
  - 단일회사 전체 재무제표: ``.../api/fnlttSinglAcntAll.json``

``DART_API_KEY``가 설정되지 않은 환경에서는 호출 즉시 ``<unavailable: ...>`` 문자열을
반환해 ``yahoo_naver`` wrapper가 yfinance 결과만으로 정상 작동하게 한다.

본 모듈은 종목코드↔DART corp_code 매핑을 디스크 캐시(`data_cache_dir/dart_corp_codes.json`)에
한 번만 저장한 뒤 재사용한다. corpCode.xml은 ~6MB 정도지만 매번 다운받기엔 부담이다.
"""
from __future__ import annotations

import io
import json
import os
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from typing import Optional

import requests

from .config import get_config
from .korean_utils import to_naver_code


_CORP_API = "https://opendart.fss.or.kr/api/corpCode.xml"
_FIN_API = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
_TIMEOUT = 15.0

# reprt_code: 분기보고서 / 반기보고서 / 3분기보고서 / 사업보고서(연간)
_REPRT_QUARTERLY = ["11013", "11012", "11014", "11011"]
_REPRT_ANNUAL = ["11011"]


def _api_key() -> Optional[str]:
    return os.environ.get("DART_API_KEY") or None


def _cache_path() -> str:
    cfg = get_config()
    cache_dir = cfg.get("data_cache_dir") or os.path.join(os.path.expanduser("~"), ".tradingagents", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "dart_corp_codes.json")


def _load_corp_map() -> dict[str, str]:
    """{stock_code(6자리) → corp_code(8자리)} 매핑.

    1) 디스크 캐시가 있으면 그것을 사용 (TTL 없음 — 회사 코드는 거의 변하지 않음).
    2) 없으면 corpCode.xml(zip)을 한 번 받아 매핑을 JSON으로 저장.
    """
    cache = _cache_path()
    if os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass  # 손상된 캐시는 다시 다운로드

    key = _api_key()
    if not key:
        raise RuntimeError("DART_API_KEY not set")

    resp = requests.get(_CORP_API, params={"crtfc_key": key}, timeout=_TIMEOUT)
    resp.raise_for_status()

    mapping: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open("CORPCODE.xml") as f:
            tree = ET.parse(f)
    for item in tree.iter("list"):
        stock = (item.findtext("stock_code") or "").strip()
        corp = (item.findtext("corp_code") or "").strip()
        if stock and corp:
            mapping[stock] = corp

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return mapping


def _resolve_corp_code(ticker: str) -> str:
    stock_code = to_naver_code(ticker)
    mapping = _load_corp_map()
    corp = mapping.get(stock_code)
    if not corp:
        raise RuntimeError(f"No DART corp_code for stock {stock_code}")
    return corp


def _fetch_financials(corp_code: str, year: str, reprt_code: str, fs_div: str = "CFS") -> list[dict]:
    """단일회사 전체 재무제표. 실패 시 빈 리스트.

    fs_div='CFS' = 연결재무제표 (보통 대기업이 우선). 실패 시 OFS(별도) 재시도.
    """
    key = _api_key()
    if not key:
        return []
    for div in (fs_div, "OFS" if fs_div == "CFS" else "CFS"):
        try:
            resp = requests.get(_FIN_API, params={
                "crtfc_key": key,
                "corp_code": corp_code,
                "bsns_year": year,
                "reprt_code": reprt_code,
                "fs_div": div,
            }, timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue
        if data.get("status") == "000":
            return data.get("list", []) or []
    return []


def _latest_available(corp_code: str, curr_date: str, reprt_codes: list[str]) -> tuple[str, str, list[dict]]:
    """주어진 종류의 보고서 중 curr_date 직전까지 사용 가능한 가장 최신을 찾는다.

    Returns: (year, reprt_code, rows). 모두 실패 시 ('','',[])
    """
    curr = datetime.strptime(curr_date, "%Y-%m-%d")
    year = curr.year
    for back in range(0, 3):  # 올해 → 작년 → 재작년까지 시도
        y = str(year - back)
        for rc in reprt_codes:
            rows = _fetch_financials(corp_code, y, rc)
            if rows:
                return y, rc, rows
    return "", "", []


_REPRT_NAME = {
    "11013": "1분기보고서", "11012": "반기보고서",
    "11014": "3분기보고서", "11011": "사업보고서(연간)",
}


def _format_rows(rows: list[dict], sj_div_filter: Optional[set[str]] = None) -> str:
    """DART rows를 사람이 읽기 좋은 표 형태로 포맷."""
    if not rows:
        return "<no rows>"
    selected = [r for r in rows if not sj_div_filter or r.get("sj_div") in sj_div_filter]
    if not selected:
        return f"<no rows for {sj_div_filter}>"

    # 보고기간 라벨 (모든 행에서 동일하므로 첫 행 사용)
    first = selected[0]
    head_thstrm = first.get("thstrm_nm", "당기")
    head_frmtrm = first.get("frmtrm_nm", "전기")
    head_bfefrmtrm = first.get("bfefrmtrm_nm", "전전기")

    lines = [f"| 구분 | 계정 | {head_thstrm} | {head_frmtrm} | {head_bfefrmtrm} |"]
    lines.append("|---|---|---:|---:|---:|")
    for r in selected:
        lines.append(
            f"| {r.get('sj_nm','')} | {r.get('account_nm','')} | "
            f"{r.get('thstrm_amount','-') or '-'} | "
            f"{r.get('frmtrm_amount','-') or '-'} | "
            f"{r.get('bfefrmtrm_amount','-') or '-'} |"
        )
    return "\n".join(lines)


def _unavailable_prefix() -> Optional[str]:
    if not _api_key():
        return "<unavailable: DART_API_KEY not set>"
    return None


def _make_header(corp_code: str, year: str, reprt_code: str, ticker: str) -> str:
    return (
        f"# DART {ticker} — {year} {_REPRT_NAME.get(reprt_code, reprt_code)} "
        f"(corp_code: {corp_code})\n"
    )


# ─── Public API (yahoo_naver wrapper가 호출) ────────────────────────────────

def get_dart_fundamentals(ticker: str, curr_date: str) -> str:
    if (pref := _unavailable_prefix()):
        return pref
    corp_code = _resolve_corp_code(ticker)
    year, reprt_code, rows = _latest_available(corp_code, curr_date, _REPRT_QUARTERLY)
    if not rows:
        return f"<unavailable: no DART filings found for {ticker} as of {curr_date}>"
    # fundamentals는 전체(재무상태표 + 손익계산서 핵심) — sj_div=BS,IS 우선
    return _make_header(corp_code, year, reprt_code, ticker) + _format_rows(rows, {"BS", "IS", "CIS"})


def get_dart_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None) -> str:
    if (pref := _unavailable_prefix()):
        return pref
    corp_code = _resolve_corp_code(ticker)
    reprt_codes = _REPRT_QUARTERLY if freq == "quarterly" else _REPRT_ANNUAL
    year, rc, rows = _latest_available(corp_code, curr_date or datetime.today().strftime("%Y-%m-%d"), reprt_codes)
    if not rows:
        return f"<unavailable: no DART filings for {ticker}>"
    return _make_header(corp_code, year, rc, ticker) + _format_rows(rows, {"BS"})


def get_dart_cashflow(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None) -> str:
    if (pref := _unavailable_prefix()):
        return pref
    corp_code = _resolve_corp_code(ticker)
    reprt_codes = _REPRT_QUARTERLY if freq == "quarterly" else _REPRT_ANNUAL
    year, rc, rows = _latest_available(corp_code, curr_date or datetime.today().strftime("%Y-%m-%d"), reprt_codes)
    if not rows:
        return f"<unavailable: no DART filings for {ticker}>"
    return _make_header(corp_code, year, rc, ticker) + _format_rows(rows, {"CF"})


def get_dart_income_statement(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None) -> str:
    if (pref := _unavailable_prefix()):
        return pref
    corp_code = _resolve_corp_code(ticker)
    reprt_codes = _REPRT_QUARTERLY if freq == "quarterly" else _REPRT_ANNUAL
    year, rc, rows = _latest_available(corp_code, curr_date or datetime.today().strftime("%Y-%m-%d"), reprt_codes)
    if not rows:
        return f"<unavailable: no DART filings for {ticker}>"
    return _make_header(corp_code, year, rc, ticker) + _format_rows(rows, {"IS", "CIS"})
