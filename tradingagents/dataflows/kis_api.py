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

# 1분봉 (일별분봉) — 당일분봉(FHKST03010200)은 전일자 미제공이라 백필 불가.
# 일별분봉(FHKST03010230)만 FID_INPUT_DATE_1 로 과거일자 조회 가능 (단, KIS
# 서버 보관분 ~1년까지). 매일 장후 누적 저장 + 1년 이내 백필 둘 다 이걸로 한다.
_TR_MINUTE = "FHKST03010230"
_URL_MINUTE = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"

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


# ============================================================
# Range fetchers — 5년치 등 임의 기간 sliding-window 수집
# ============================================================
#
# 실측: 50ms 간격으로 burst하면 KIS가 명시적 EGW00201이 아닌 HTTP 500으로
# 거부하는 패턴 관찰 (50회 호출 중 23회 실패 = 46%). 따라서 range fetcher는
# 분석가용 _call(50ms)보다 보수적인 정책으로 분리:
#   - 호출 간격 100ms (10 req/s) — burst 회피
#   - HTTP 500 → exponential backoff (1s → 2s → 4s) 최대 3회 재시도
#   - HTTP 401/403 → 기존 _call의 토큰 재발급에 위임
#
# 페이지네이션:
#   - investor/program: 한 호출당 30 거래일. end_date를 응답의 가장 오래된
#     날짜 - 1일로 슬라이드해 백워드 진행
#   - short: 한 호출당 최대 100 거래일 (start/end 둘 다 받음). 100일 윈도우로
#     백워드 진행

_RANGE_CALL_SLEEP_SEC = 0.1     # 100ms (burst-거부 회피)
_RANGE_BACKOFF_BASE_SEC = 1.0
_RANGE_MAX_RETRIES = 3
_INVESTOR_PROGRAM_PAGE_DAYS = 30
_SHORT_PAGE_DAYS = 100


def _call_with_backoff(url_path: str, tr_id: str, params: dict) -> dict:
    """_call + HTTP 500 시 exponential backoff. 5년치 수집의 burst-거부 회복용.

    토큰 만료(401/403)는 기존 _call이 처리. 여기서는 KIS의 "soft rate limit"
    (HTTP 500으로 거부) 패턴만 흡수.
    """
    for attempt in range(_RANGE_MAX_RETRIES):
        try:
            return _call(url_path, tr_id, params)
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 500 and attempt < _RANGE_MAX_RETRIES - 1:
                wait = _RANGE_BACKOFF_BASE_SEC * (2 ** attempt)
                logger.warning(
                    "KIS 500 (attempt %d/%d), backing off %.1fs",
                    attempt + 1, _RANGE_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")  # _RANGE_MAX_RETRIES 이상 도달


def _parse_investor_row(r: dict) -> dict:
    """raw KIS row → 정규화 dict. 단일 fetcher와 range fetcher가 공유.

    KIS는 은행(``bank_ntby_qty``)과 보험(``insu_ntby_qty``)을 별도 필드로 반환.
    이전 정책은 둘을 합쳐 ``bank_insurance_*`` 로 저장했으나, 보유 분석 시 운용
    성향이 다른 두 주체를 분리해서 보는 게 정확해 별도 컬럼으로 유지한다.
    """
    return {
        "date": _format_date(r.get("stck_bsop_date", "")),
        "open": _safe_int(r.get("stck_oprc")),
        "high": _safe_int(r.get("stck_hgpr")),
        "low": _safe_int(r.get("stck_lwpr")),
        "close": _safe_int(r.get("stck_clpr")),
        "volume": _safe_int(r.get("acml_vol")),
        "foreign_qty": _safe_int(r.get("frgn_ntby_qty")),
        "foreign_registered_qty": _safe_int(r.get("frgn_reg_ntby_qty")),
        "foreign_unregistered_qty": _safe_int(r.get("frgn_nreg_ntby_qty")),
        "foreign_amount": _safe_int(r.get("frgn_ntby_tr_pbmn")),
        "foreign_registered_amount": _safe_int(r.get("frgn_reg_ntby_pbmn")),
        "foreign_unregistered_amount": _safe_int(r.get("frgn_nreg_ntby_pbmn")),
        "institution_qty": _safe_int(r.get("orgn_ntby_qty")),
        "pension_qty": _safe_int(r.get("fund_ntby_qty")),                   # 연기금
        "private_equity_qty": _safe_int(r.get("pe_fund_ntby_vol")),         # 사모펀드
        "investment_trust_qty": _safe_int(r.get("ivtr_ntby_qty")),          # 투자신탁
        "securities_qty": _safe_int(r.get("scrt_ntby_qty")),                # 증권(금융투자)
        "bank_qty": _safe_int(r.get("bank_ntby_qty")),                      # 은행
        "insurance_qty": _safe_int(r.get("insu_ntby_qty")),                 # 보험
        "institution_amount": _safe_int(r.get("orgn_ntby_tr_pbmn")),
        "pension_amount": _safe_int(r.get("fund_ntby_tr_pbmn")),
        "private_equity_amount": _safe_int(r.get("pe_fund_ntby_tr_pbmn")),
        "investment_trust_amount": _safe_int(r.get("ivtr_ntby_tr_pbmn")),
        "securities_amount": _safe_int(r.get("scrt_ntby_tr_pbmn")),
        "bank_amount": _safe_int(r.get("bank_ntby_tr_pbmn")),
        "insurance_amount": _safe_int(r.get("insu_ntby_tr_pbmn")),
        "retail_qty": _safe_int(r.get("prsn_ntby_qty")),
        "retail_amount": _safe_int(r.get("prsn_ntby_tr_pbmn")),
        "other_corp_qty": _safe_int(r.get("etc_corp_ntby_vol")),
        "other_corp_amount": _safe_int(r.get("etc_corp_ntby_tr_pbmn")),
    }


def _parse_program_row(r: dict) -> dict:
    return {
        "date": _format_date(r.get("stck_bsop_date", "")),
        "open": _safe_int(r.get("stck_oprc")),
        "high": _safe_int(r.get("stck_hgpr")),
        "low": _safe_int(r.get("stck_lwpr")),
        "close": _safe_int(r.get("stck_clpr")),
        "volume": _safe_int(r.get("acml_vol")),
        "net_qty": _safe_int(r.get("whol_smtn_ntby_qty")),
        "net_amount": _safe_int(r.get("whol_smtn_ntby_tr_pbmn")),
    }


def _parse_short_row(r: dict) -> dict:
    return {
        "date": _format_date(r.get("stck_bsop_date", "")),
        "open": _safe_int(r.get("stck_oprc")),
        "high": _safe_int(r.get("stck_hgpr")),
        "low": _safe_int(r.get("stck_lwpr")),
        "close": _safe_int(r.get("stck_clpr")),
        "volume": _safe_int(r.get("acml_vol")),
        "short_qty": _safe_int(r.get("ssts_cntg_qty")),
        "short_volume_ratio": _safe_float(r.get("ssts_vol_rlim")),
        "short_amount": _safe_int(r.get("ssts_tr_pbmn")),
        "short_amount_ratio": _safe_float(r.get("ssts_tr_pbmn_rlim")),
    }


def _parse_minute_row(r: dict, fallback_yyyymmdd: str = "") -> dict:
    """raw 일별분봉 row → 정규화 dict.

    ``date`` 는 분 단위 타임스탬프(``"YYYY-MM-DD HH:MM:SS"``). KIS 일별분봉
    output2 행은 영업일자(``stck_bsop_date``)를 누락할 수 있어 호출 시점의
    조회일자(``fallback_yyyymmdd``)로 보강한다. ``close`` 는 분봉 종가에
    해당하는 현재가(``stck_prpr``).
    """
    yyyymmdd = (r.get("stck_bsop_date") or "").strip() or fallback_yyyymmdd
    hhmmss = (r.get("stck_cntg_hour") or "").strip()
    return {
        "date": _format_datetime(yyyymmdd, hhmmss),
        "open": _safe_int(r.get("stck_oprc")),
        "high": _safe_int(r.get("stck_hgpr")),
        "low": _safe_int(r.get("stck_lwpr")),
        "close": _safe_int(r.get("stck_prpr")),
        "volume": _safe_int(r.get("cntg_vol")),
    }


def fetch_investor_trend_range(
    code6: str, start_date: str, end_date: str,
    *, progress_cb=None,
) -> list[dict]:
    """``start_date`` ~ ``end_date`` 사이 전체 일별 시계열을 sliding-window로 수집.

    한 호출당 30 거래일, end_date를 응답의 가장 오래된 날짜 - 1일로 슬라이드.
    KIS 보유 한계는 실측상 5년+ 정상. 결과는 date 오름차순 정렬.

    ``progress_cb(rows_so_far)`` 콜백을 통해 진행률 출력 가능.
    """
    return _range_paginate(
        code6, start_date, end_date,
        url_path=_URL_INVESTOR, tr_id=_TR_INVESTOR,
        extra_params={"FID_ORG_ADJ_PRC": "", "FID_ETC_CLS_CODE": ""},
        output_key="output2", parser=_parse_investor_row,
        progress_cb=progress_cb,
    )


def fetch_program_trading_range(
    code6: str, start_date: str, end_date: str,
    *, progress_cb=None,
) -> list[dict]:
    return _range_paginate(
        code6, start_date, end_date,
        url_path=_URL_PROGRAM, tr_id=_TR_PROGRAM,
        extra_params={},
        output_key="output", parser=_parse_program_row,
        progress_cb=progress_cb,
    )


def _range_paginate(
    code6: str, start_date: str, end_date: str,
    *, url_path: str, tr_id: str, extra_params: dict,
    output_key: str, parser, progress_cb=None,
) -> list[dict]:
    """investor/program 공통 — end_date 단일 파라미터 백워드 슬라이딩."""
    start_d = datetime.strptime(_strip_dashes(start_date), "%Y%m%d").date()
    end_d = datetime.strptime(_strip_dashes(end_date), "%Y%m%d").date()
    rows: list[dict] = []
    seen_dates: set[str] = set()
    cursor = end_d
    safety_iter = 0
    max_iter = 200  # 5년 * 250거래일 / 30 ≈ 42 호출, 200으로 안전 마진

    while cursor >= start_d and safety_iter < max_iter:
        safety_iter += 1
        cursor_str = cursor.strftime("%Y%m%d")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code6,
            "FID_INPUT_DATE_1": cursor_str,
            **extra_params,
        }
        body = _call_with_backoff(url_path, tr_id, params)
        time.sleep(_RANGE_CALL_SLEEP_SEC)
        page = body.get(output_key) or []
        if not page:
            break

        added = 0
        oldest_in_page = None
        for r in page:
            parsed = parser(r)
            d = parsed["date"]
            if not d:
                continue
            if d in seen_dates:
                continue
            if d < start_date or d > end_date:
                continue
            rows.append(parsed)
            seen_dates.add(d)
            added += 1
            if oldest_in_page is None or d < oldest_in_page:
                oldest_in_page = d

        if progress_cb:
            progress_cb(len(rows))

        if oldest_in_page is None or added == 0:
            break
        # 다음 cursor: 응답에서 본 가장 오래된 날짜 - 1일
        oldest_date = datetime.strptime(oldest_in_page, "%Y-%m-%d").date()
        new_cursor = oldest_date - timedelta(days=1)
        if new_cursor >= cursor:  # forward 진행 멈춤 — 안전 종료
            break
        cursor = new_cursor

    rows.sort(key=lambda r: r["date"])
    return rows


def fetch_short_interest_range(
    code6: str, start_date: str, end_date: str,
    *, progress_cb=None,
) -> list[dict]:
    """short endpoint는 start/end 둘 다 받고 한 호출당 최대 100행. 100일
    윈도우로 백워드 슬라이딩.
    """
    start_d = datetime.strptime(_strip_dashes(start_date), "%Y%m%d").date()
    end_d = datetime.strptime(_strip_dashes(end_date), "%Y%m%d").date()
    rows: list[dict] = []
    seen_dates: set[str] = set()
    cursor_end = end_d
    safety_iter = 0
    max_iter = 50  # 5년 * 250 / 100 = 12.5 → 50 안전 마진

    while cursor_end >= start_d and safety_iter < max_iter:
        safety_iter += 1
        cursor_start = max(start_d, cursor_end - timedelta(days=_SHORT_PAGE_DAYS - 1))
        body = _call_with_backoff(_URL_SHORT, _TR_SHORT, {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code6,
            "FID_INPUT_DATE_1": cursor_start.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": cursor_end.strftime("%Y%m%d"),
        })
        time.sleep(_RANGE_CALL_SLEEP_SEC)
        page = body.get("output2") or []
        if not page:
            break

        added = 0
        oldest_in_page = None
        for r in page:
            parsed = _parse_short_row(r)
            d = parsed["date"]
            if not d or d in seen_dates:
                continue
            if d < start_date or d > end_date:
                continue
            rows.append(parsed)
            seen_dates.add(d)
            added += 1
            if oldest_in_page is None or d < oldest_in_page:
                oldest_in_page = d

        if progress_cb:
            progress_cb(len(rows))

        if oldest_in_page is None or added == 0:
            break
        oldest_date = datetime.strptime(oldest_in_page, "%Y-%m-%d").date()
        new_end = oldest_date - timedelta(days=1)
        if new_end >= cursor_end:
            break
        cursor_end = new_end

    rows.sort(key=lambda r: r["date"])
    return rows


# ============================================================
# Minute bars — 1분봉 OHLCV (일별분봉 FHKST03010230)
# ============================================================
#
# 페이지네이션: 일별분봉은 한 호출당 최대 120봉. FID_INPUT_HOUR_1(조회 종료
# 시각, HHMMSS)을 15:30부터 역방향으로 슬라이드해 하루(09:00~15:30, ~380봉)를
# ~4회 호출로 모은다. range 수집은 거래일 단위로 fetch_minute_bars 를 반복.

_MINUTE_PAGE_BARS = 120          # 일별분봉 1회 최대 (실전계좌)
_MINUTE_OPEN_HHMMSS = "090000"
_MINUTE_START_CURSOR = "153000"  # 정규장 마감(종가 단일가 포함). 시간외 단일가 제외.
_MINUTE_MAX_ITER = 20            # 380/120 ≈ 4회, 20 안전 마진


def _minute_cursor_back(hhmmss: str) -> Optional[str]:
    """``HHMMSS`` 에서 1분 뺀 ``HHMMSS``. 파싱 실패 시 None."""
    try:
        t = datetime.strptime(hhmmss.zfill(6), "%H%M%S")
    except ValueError:
        return None
    return (t - timedelta(minutes=1)).strftime("%H%M%S")


def fetch_minute_bars(code6: str, target_date: str, *, market: str = "J") -> list[dict]:
    """단일 거래일 1분봉 OHLCV 전량 (일별분봉 FHKST03010230).

    ``target_date`` 는 ``YYYY-MM-DD`` 또는 ``YYYYMMDD``. 휴장일이면 빈 리스트.
    결과는 시각 오름차순, 각 dict 키: ``date`` (``YYYY-MM-DD HH:MM:SS``),
    ``open/high/low/close/volume``.

    KIS 분봉 보관 한계(~1년)를 넘은 과거일은 빈 응답 → 빈 리스트.
    """
    yyyymmdd = _strip_dashes(target_date)
    target_day = _format_date(yyyymmdd)  # "YYYY-MM-DD" — 경계 판정용
    rows: list[dict] = []
    seen_times: set[str] = set()
    cursor = _MINUTE_START_CURSOR
    for _ in range(_MINUTE_MAX_ITER):
        body = _call_with_backoff(_URL_MINUTE, _TR_MINUTE, {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_INPUT_ISCD": code6,
            "FID_INPUT_DATE_1": yyyymmdd,
            "FID_INPUT_HOUR_1": cursor,
            "FID_PW_DATA_INCU_YN": "Y",   # 과거 데이터 포함
            "FID_FAKE_TICK_INCU_YN": "N",  # 허봉 제외
        })
        time.sleep(_RANGE_CALL_SLEEP_SEC)
        page = body.get("output2") or []
        if not page:
            break
        added = 0
        crossed_prev_day = False
        oldest_hhmmss: Optional[str] = None
        for r in page:
            parsed = _parse_minute_row(r, fallback_yyyymmdd=yyyymmdd)
            ts = parsed["date"]
            if not ts:
                continue
            if ts[:10] != target_day:
                # 반일장(예: 1/2 단축개장) 경계에서 한 페이지가 직전 거래일로
                # 넘어감 — 그 종목·날은 별도 호출로 수집하므로 여기선 멈춘다.
                crossed_prev_day = True
                continue
            hhmmss = (r.get("stck_cntg_hour") or "").strip()
            if not hhmmss or hhmmss in seen_times:
                continue
            rows.append(parsed)
            seen_times.add(hhmmss)
            added += 1
            if oldest_hhmmss is None or hhmmss < oldest_hhmmss:
                oldest_hhmmss = hhmmss
        if crossed_prev_day or added == 0 or oldest_hhmmss is None:
            break
        if oldest_hhmmss <= _MINUTE_OPEN_HHMMSS:  # 장 시작 도달
            break
        new_cursor = _minute_cursor_back(oldest_hhmmss)
        if new_cursor is None or new_cursor >= cursor:  # 진행 멈춤 — 안전 종료
            break
        cursor = new_cursor
    rows.sort(key=lambda r: r["date"])
    return rows


def fetch_minute_bars_range(
    code6: str, start_date: str, end_date: str,
    *, progress_cb=None, market: str = "J",
) -> list[dict]:
    """``start_date`` ~ ``end_date`` 거래일별 1분봉을 모아 반환.

    거래일 단위로 ``fetch_minute_bars`` 를 역방향 반복. 주말은 건너뛰고
    휴장일은 빈 응답으로 자연 skip. KIS 분봉 보관(~1년)을 넘은 구간은
    빈 응답이라 호출만 낭비 — 호출 측에서 윈도우를 1년 이내로 제한 권장.
    """
    start_d = datetime.strptime(_strip_dashes(start_date), "%Y%m%d").date()
    end_d = datetime.strptime(_strip_dashes(end_date), "%Y%m%d").date()
    rows: list[dict] = []
    day = end_d
    while day >= start_d:
        if day.weekday() < 5:  # Sat=5, Sun=6 제외
            rows.extend(fetch_minute_bars(code6, day.strftime("%Y%m%d"), market=market))
            if progress_cb:
                progress_cb(len(rows))
        day -= timedelta(days=1)
    rows.sort(key=lambda r: r["date"])
    return rows


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


def _format_datetime(yyyymmdd: str, hhmmss: str) -> str:
    """``YYYYMMDD`` + ``HHMMSS`` → ``"YYYY-MM-DD HH:MM:SS"``. 형식 불량 시 ""."""
    if not yyyymmdd or len(yyyymmdd) != 8:
        return ""
    hhmmss = (hhmmss or "").zfill(6)
    if len(hhmmss) != 6:
        return ""
    return (
        f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]} "
        f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
    )


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
    out = [_parse_investor_row(r) for r in rows]
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
        "open": _safe_int(r.get("stck_oprc")),
        "high": _safe_int(r.get("stck_hgpr")),
        "low": _safe_int(r.get("stck_lwpr")),
        "close": _safe_int(r.get("stck_clpr")),
        "volume": _safe_int(r.get("acml_vol")),
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
        "open": _safe_int(r.get("stck_oprc")),
        "high": _safe_int(r.get("stck_hgpr")),
        "low": _safe_int(r.get("stck_lwpr")),
        "close": _safe_int(r.get("stck_clpr")),
        "volume": _safe_int(r.get("acml_vol")),
        "short_qty": _safe_int(r.get("ssts_cntg_qty")),
        "short_volume_ratio": _safe_float(r.get("ssts_vol_rlim")),
        "short_amount": _safe_int(r.get("ssts_tr_pbmn")),
        "short_amount_ratio": _safe_float(r.get("ssts_tr_pbmn_rlim")),
    } for r in rows]
    trimmed = out[:lookback_days]
    if not skip:
        _save_cache(cache_p, trimmed)
    return trimmed
