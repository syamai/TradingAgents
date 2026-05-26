"""KIS HTTP thin client — 종목별 일별 수급 시계열 3종.

공개 함수:
    fetch_investor_trend(code6, end_date, lookback_days=7)
    fetch_program_trading(code6, end_date, lookback_days=7)
    fetch_short_interest(code6, start_date, end_date, lookback_days=7)

공통 동작:
- 인증 헤더 / TR_ID / 401·403 시 토큰 무효화 + 1회 재시도 (``_call``)
- Rate limit: 호출 간 50ms (20 req/s 한도)
- 응답 캐시: 장 종료(KST 15:30) 후만 저장. 장중 호출은 캐시 skip.
- 응답 필드는 분석가가 바로 읽을 수 있는 단순 키(``foreign_qty`` 등)로 정규화.

raw KIS 응답 필드는 `chk_*.py` 컬럼 매핑 기반(reference repo).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from . import kis_auth
from .config import get_config

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 15.0
_RATE_LIMIT_SLEEP_SEC = 0.05  # 20 req/s headroom
_KST = ZoneInfo("Asia/Seoul")

# investor endpoint(FHPTJ04160001)는 ``end_date == 오늘 KST`` 호출을 KST
# 15:40 이전엔 ``rt_cd=2 TIME LIMIT 00:00 ~ 15:40`` 으로 거부 (실측). 일별
# 데이터는 장 종료(15:30)+10분 이후 확정/배치되는 KIS 정책으로 추정.
# 그 시간대에는 end_date를 자동으로 어제로 슬라이드해 호출 가능하게 한다.
# program/short는 실측상 이 시간대에도 동작하지만 stale 위험 동일하므로
# 일관성 위해 3개 fetcher 모두 같은 보정 적용.
_TODAY_DATA_CONFIRM_HOUR = 15
_TODAY_DATA_CONFIRM_MIN = 40

# 확정된 endpoint (단계 0)
_TR_INVESTOR = "FHPTJ04160001"
_TR_PROGRAM = "FHPPG04650201"
_TR_SHORT = "FHPST04830000"

_URL_INVESTOR = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
_URL_PROGRAM = "/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily"
_URL_SHORT = "/uapi/domestic-stock/v1/quotations/daily-short-sale"

_last_call_ts = 0.0  # module-level sequential rate limiter (단일 프로세스 가정)


# === cache / market hours ===

def _kis_cache_dir() -> Path:
    cfg = get_config()
    base = cfg.get("data_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".tradingagents", "cache"
    )
    d = Path(base) / "kis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(code6: str, end_yyyymmdd: str, lookback_days: int, tr_id: str) -> Path:
    return _kis_cache_dir() / f"{code6}_{end_yyyymmdd}_W{lookback_days}_{tr_id}.json"


def _is_market_open(now_kst: Optional[datetime] = None) -> bool:
    """KST 평일 09:00 ≤ now < 15:30이면 장 중."""
    now_kst = now_kst if now_kst is not None else datetime.now(_KST)
    if now_kst.weekday() >= 5:  # Sat=5, Sun=6
        return False
    open_t = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
    close_t = now_kst.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now_kst < close_t


def _normalize_end_date(end_yyyymmdd: str, now_kst: Optional[datetime] = None) -> str:
    """``end_date == 오늘 KST`` & KST 15:40 이전 → 어제로 슬라이드.

    KIS investor endpoint가 새벽~장중에 오늘 데이터 호출을 거부하는 정책 우회.
    과거 날짜는 그대로. 잘못된 형식도 그대로 (호출 측이 KIS 오류로 받음).
    """
    now_kst = now_kst if now_kst is not None else datetime.now(_KST)
    try:
        end = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    except ValueError:
        return end_yyyymmdd
    today_kst = now_kst.date()
    if end < today_kst:
        return end_yyyymmdd  # 과거 데이터는 시간 무관 호출 가능
    # end >= today_kst — KST cutoff 검사
    cutoff = now_kst.replace(
        hour=_TODAY_DATA_CONFIRM_HOUR,
        minute=_TODAY_DATA_CONFIRM_MIN,
        second=0, microsecond=0,
    )
    if now_kst >= cutoff:
        return end_yyyymmdd  # cutoff 이후는 그대로
    # cutoff 이전 — 어제로 슬라이드
    yesterday = today_kst - timedelta(days=1)
    return yesterday.strftime("%Y%m%d")


def _should_skip_cache(end_yyyymmdd: str) -> bool:
    """end_date가 오늘이거나 미래이고 KST 장중이면 캐시 skip — stale 위험."""
    try:
        end = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    except ValueError:
        return True
    today_kst = datetime.now(_KST).date()
    if end >= today_kst and _is_market_open():
        return True
    return False


def _load_cache(path: Path) -> Optional[list[dict]]:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("KIS response cache load failed (%s): %s", path, exc)
        return None


def _save_cache(path: Path, data: list[dict]) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("KIS response cache save failed (%s): %s", path, exc)


# === HTTP ===

def _rate_limit_sleep() -> None:
    global _last_call_ts
    elapsed = time.time() - _last_call_ts
    if elapsed < _RATE_LIMIT_SLEEP_SEC:
        time.sleep(_RATE_LIMIT_SLEEP_SEC - elapsed)
    _last_call_ts = time.time()


def _build_headers(tr_id: str, access_token: str, appkey: str, appsecret: str) -> dict:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": tr_id,
        "custtype": "P",  # personal account
    }


def _call(url_path: str, tr_id: str, params: dict, _retry: bool = False) -> dict:
    """공통 진입 — 401/403 시 토큰 무효화 후 1회 재시도. KIS 에러코드(rt_cd != "0")는 RuntimeError."""
    _rate_limit_sleep()
    access_token = kis_auth.get_access_token()
    appkey, appsecret = kis_auth._credentials()
    url = kis_auth._base_url() + url_path
    headers = _build_headers(tr_id, access_token, appkey, appsecret)
    resp = requests.get(url, headers=headers, params=params, timeout=_REQUEST_TIMEOUT)
    if resp.status_code in (401, 403) and not _retry:
        kis_auth.invalidate_cached_token()
        return _call(url_path, tr_id, params, _retry=True)
    resp.raise_for_status()
    body = resp.json()
    rt_cd = body.get("rt_cd")
    if rt_cd is not None and rt_cd != "0":
        raise RuntimeError(f"KIS API error (rt_cd={rt_cd}): {body.get('msg1', '')}")
    return body


# === parsers ===

def _safe_int(x) -> int:
    try:
        return int(float(x or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(x) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def _format_date(yyyymmdd: str) -> str:
    if not yyyymmdd or len(yyyymmdd) != 8:
        return yyyymmdd or ""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def _strip_dashes(date_str: str) -> str:
    return date_str.replace("-", "") if date_str else ""


# === public fetchers ===

def fetch_investor_trend(code6: str, end_date: str, lookback_days: int = 7) -> list[dict]:
    """종목별 일별 투자자(외국인·기관·개인) 매매동향.

    KIS endpoint는 ``end_date`` 기준 과거 ~30일치 시계열을 반환.
    최근 ``lookback_days`` 행만 잘라 반환.

    Returns:
        list[dict] each with keys:
            date (YYYY-MM-DD), close (KRW),
            foreign_qty, institution_qty, retail_qty (수량),
            foreign_amount, institution_amount, retail_amount (KRW)
    """
    end = _normalize_end_date(_strip_dashes(end_date))
    cache_p = _cache_path(code6, end, lookback_days, _TR_INVESTOR)
    skip = _should_skip_cache(end)
    if not skip:
        cached = _load_cache(cache_p)
        if cached is not None:
            return cached
    body = _call(_URL_INVESTOR, _TR_INVESTOR, {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code6,
        "FID_INPUT_DATE_1": end,
        "FID_ORG_ADJ_PRC": "",
        "FID_ETC_CLS_CODE": "",
    })
    rows = body.get("output2") or []
    out = []
    for r in rows:
        bank_qty = _safe_int(r.get("bank_ntby_qty"))
        insu_qty = _safe_int(r.get("insu_ntby_qty"))
        bank_amt = _safe_int(r.get("bank_ntby_tr_pbmn"))
        insu_amt = _safe_int(r.get("insu_ntby_tr_pbmn"))
        out.append({
            "date": _format_date(r.get("stck_bsop_date", "")),
            "close": _safe_int(r.get("stck_clpr")),
            # 외국인 — 통합 + 등록(장기) / 비등록(단기 외국 자금) 분리
            "foreign_qty": _safe_int(r.get("frgn_ntby_qty")),
            "foreign_registered_qty": _safe_int(r.get("frgn_reg_ntby_qty")),
            "foreign_unregistered_qty": _safe_int(r.get("frgn_nreg_ntby_qty")),
            "foreign_amount": _safe_int(r.get("frgn_ntby_tr_pbmn")),
            "foreign_registered_amount": _safe_int(r.get("frgn_reg_ntby_pbmn")),
            "foreign_unregistered_amount": _safe_int(r.get("frgn_nreg_ntby_pbmn")),
            # 기관 — 통합 + 5개 sub. 종금/기타는 신호 약해 생략.
            "institution_qty": _safe_int(r.get("orgn_ntby_qty")),
            "pension_qty": _safe_int(r.get("fund_ntby_qty")),               # 연기금
            "private_equity_qty": _safe_int(r.get("pe_fund_ntby_vol")),     # 사모펀드
            "investment_trust_qty": _safe_int(r.get("ivtr_ntby_qty")),      # 투자신탁
            "securities_qty": _safe_int(r.get("scrt_ntby_qty")),            # 증권
            "bank_insurance_qty": bank_qty + insu_qty,                       # 은행 + 보험 (보수적 운용)
            "institution_amount": _safe_int(r.get("orgn_ntby_tr_pbmn")),
            "pension_amount": _safe_int(r.get("fund_ntby_tr_pbmn")),
            "private_equity_amount": _safe_int(r.get("pe_fund_ntby_tr_pbmn")),
            "investment_trust_amount": _safe_int(r.get("ivtr_ntby_tr_pbmn")),
            "securities_amount": _safe_int(r.get("scrt_ntby_tr_pbmn")),
            "bank_insurance_amount": bank_amt + insu_amt,
            # 개인 + 기타법인 (자사주 매입 가능성)
            "retail_qty": _safe_int(r.get("prsn_ntby_qty")),
            "retail_amount": _safe_int(r.get("prsn_ntby_tr_pbmn")),
            "other_corp_qty": _safe_int(r.get("etc_corp_ntby_vol")),
            "other_corp_amount": _safe_int(r.get("etc_corp_ntby_tr_pbmn")),
        })
    trimmed = out[:lookback_days]
    if not skip:
        _save_cache(cache_p, trimmed)
    return trimmed


def fetch_program_trading(code6: str, end_date: str, lookback_days: int = 7) -> list[dict]:
    """종목별 일별 프로그램매매 추이 (종목 단위 endpoint는 종합값만 제공 — 차익/비차익 구분 없음).

    Returns:
        list[dict] each with keys:
            date, close, net_qty, net_amount
            (net_qty = 전체 합계 순매수 수량, net_amount = 전체 합계 순매수 거래대금 KRW)
    """
    end = _normalize_end_date(_strip_dashes(end_date))
    cache_p = _cache_path(code6, end, lookback_days, _TR_PROGRAM)
    skip = _should_skip_cache(end)
    if not skip:
        cached = _load_cache(cache_p)
        if cached is not None:
            return cached
    body = _call(_URL_PROGRAM, _TR_PROGRAM, {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code6,
        "FID_INPUT_DATE_1": end,
    })
    rows = body.get("output") or []
    out = [{
        "date": _format_date(r.get("stck_bsop_date", "")),
        "close": _safe_int(r.get("stck_clpr")),
        "net_qty": _safe_int(r.get("whol_smtn_ntby_qty")),
        "net_amount": _safe_int(r.get("whol_smtn_ntby_tr_pbmn")),
    } for r in rows]
    trimmed = out[:lookback_days]
    if not skip:
        _save_cache(cache_p, trimmed)
    return trimmed


def fetch_short_interest(
    code6: str, start_date: str, end_date: str, lookback_days: int = 7
) -> list[dict]:
    """종목별 공매도 일별 추이.

    Returns:
        list[dict] each with keys:
            date, close,
            short_qty (공매도 체결 수량),
            short_volume_ratio (공매도 거래량 비중 %),
            short_amount (공매도 거래대금 KRW),
            short_amount_ratio (공매도 거래대금 비중 %)
    """
    start = _strip_dashes(start_date)
    end = _normalize_end_date(_strip_dashes(end_date))
    cache_p = _cache_path(code6, end, lookback_days, _TR_SHORT)
    skip = _should_skip_cache(end)
    if not skip:
        cached = _load_cache(cache_p)
        if cached is not None:
            return cached
    body = _call(_URL_SHORT, _TR_SHORT, {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code6,
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
    })
    rows = body.get("output2") or []
    out = [{
        "date": _format_date(r.get("stck_bsop_date", "")),
        "close": _safe_int(r.get("stck_clpr")),
        "short_qty": _safe_int(r.get("ssts_cntg_qty")),
        "short_volume_ratio": _safe_float(r.get("ssts_vol_rlim")),
        "short_amount": _safe_int(r.get("ssts_tr_pbmn")),
        "short_amount_ratio": _safe_float(r.get("ssts_tr_pbmn_rlim")),
    } for r in rows]
    trimmed = out[:lookback_days]
    if not skip:
        _save_cache(cache_p, trimmed)
    return trimmed
