"""네이버 DataLab 검색어트렌드 ASVI — 한국 retail 검색 관심(Phase 2).

네이버 통합검색어트렌드 API(``/v1/datalab/search``)로 한국 종목명의 주간 검색량을
받아 ASVI = log(이번주) − log(직전 8주 중앙값) 를 계산한다(google_trends 의
한국판). 검색 = 직접 관측 가능한 retail attention 으로, 종목토론방 글수
(naver_board)와 **독립적인 두 번째 소스**다 → 둘 다 뜨면 2소스 동의(조작 방어).

snapshot-lock: DataLab ratio 는 요청마다 0–100 으로 재스케일되므로(요청 내 최대=
100), asof_date 로 한 번 저장한 값은 절대 덮어쓰지 않는다(``TrendStore.write`` 의
INSERT OR IGNORE). ASVI 는 log 비(比)라 선형 재스케일에 불변 — 단, 정밀도 보존을
위해 **종목당 1요청**(groupName=종목명)으로 각 시계열을 자기 최대=100 으로 스케일.

leadingness = L. universe(브리지가 만든 한국 후보 코드)를 주입받아 코드→한글명
으로 검색 키워드를 만든다. 자격증명 미설정·DataLab 스코프 미인증·네트워크 실패는
graceful degrade(``([], False)``) — 다음 tick 자가복구.
"""
from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Optional

import requests

from .base import LEADING, SignalRow

logger = logging.getLogger(__name__)

_SOURCE = "naver_datalab"
_METRIC = "asvi"
_API = "https://openapi.naver.com/v1/datalab/search"
_LOOKBACK_DAYS = 90  # 주간 ~13포인트(8주 baseline + 여유)
_MIN_WEEKS = 9  # 8주 중앙값 + 당주
_KST = timezone(timedelta(hours=9))  # 한국 종목 검색은 KST 일자 기준


def _today() -> str:
    return datetime.now(_KST).date().isoformat()


def _resolve_names(codes) -> dict:
    """6자리 코드 → 한글 종목명(검색 키워드). 브리지 캐시 우선, KIS 메타 보강."""
    names: dict = {}
    try:
        from ...hermes.kr_peer_bridge import KrPeerCache

        for code, info in KrPeerCache().reverse(codes).items():
            if info.get("name"):
                names[code] = info["name"]
    except Exception as exc:  # 캐시 미존재 등 — 보강으로 넘어감
        logger.debug("peer-cache 이름 조회 실패: %s", exc)
    missing = [c for c in codes if c not in names]
    if missing:
        try:
            from ..kis_history_store import KisHistoryStore

            kis = KisHistoryStore()
            for c in missing:
                meta = kis.get_ticker_metadata(c)
                if meta and meta.get("company_name"):
                    names[c] = meta["company_name"]
        except Exception as exc:
            logger.debug("KIS 이름 보강 실패: %s", exc)
    return names


def _fetch_series(keyword: str, start: str, end: str, *, headers: dict, timeout: float):
    """단일 키워드 주간 검색량 시계열(list[float]). 실패/스코프오류 시 None."""
    body = {
        "startDate": start,
        "endDate": end,
        "timeUnit": "week",
        "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}],
    }
    try:
        resp = requests.post(_API, json=body, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("datalab 요청 실패 %s: %s", keyword, exc)
        return None
    try:
        data = resp.json()
    except ValueError:
        logger.warning("datalab 비-JSON 응답 %s (status=%s)", keyword, resp.status_code)
        return None
    if "results" not in data:  # 스코프 미인증(024) 등 errorCode 응답
        logger.warning(
            "datalab 오류 %s: code=%s msg=%s",
            keyword, data.get("errorCode"), data.get("errorMessage"),
        )
        return None
    results = data.get("results") or []
    if not results:
        return []
    series = [float(p.get("ratio", 0.0)) for p in (results[0].get("data") or [])]
    return series


def collect_naver_datalab(
    market: str = "kr",
    *,
    universe: Optional[list] = None,
    asof_date: Optional[str] = None,
    lookback_days: int = _LOOKBACK_DAYS,
    timeout: float = 10.0,
    batch_sleep: float = 0.2,
) -> tuple[list[SignalRow], bool]:
    """한국 후보별 네이버 검색량 ASVI → SignalRow. universe 없으면 ([], True).

    자격증명 미설정/스코프 미인증/전 종목 실패 시 ``([], False)``(degrade).
    """
    if not universe:
        return [], True

    cid = os.environ.get("NAVER_CLIENT_ID")
    csec = os.environ.get("NAVER_CLIENT_SECRET")
    if not cid or not csec:
        logger.info("naver_datalab skip — NAVER_CLIENT_ID/SECRET 미설정")
        return [], False

    asof = asof_date or _today()
    end = asof
    start = (datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=lookback_days)).date().isoformat()
    headers = {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csec,
        "Content-Type": "application/json",
    }
    names = _resolve_names(list(universe))

    rows: list[SignalRow] = []
    any_ok = False
    for code in universe:
        keyword = names.get(code)
        if not keyword:  # 키워드 없으면 검색 불가 — skip
            continue
        series = _fetch_series(keyword, start, end, headers=headers, timeout=timeout)
        if series is None:  # 요청/스코프 실패 — 다음 종목 시도
            continue
        any_ok = True
        if len(series) < _MIN_WEEKS:
            continue
        cur = series[-1]
        base = median(series[-_MIN_WEEKS:-1])  # 직전 8주 중앙값(google_trends 와 동일 정의)
        if cur <= 0 or base <= 0:
            continue
        asvi = math.log(cur) - math.log(base)
        # ASVI(비정상 검색량)는 정의상 '평소(직전 8주 중앙값) 대비 상승'. 음수/0(검색이
        # 평소 이하)은 검색 쏠림 신호가 아니며, 토론방과의 2소스 동의로 잘못 잡히면
        # '검색도 달아오름'이라는 전제를 깬다 → 양의 ASVI 만 신호로 표면화.
        if asvi <= 0:
            continue
        rows.append(
            SignalRow(
                release_date=asof,
                asof_date=asof,
                market=market,
                entity=code,
                source=_SOURCE,
                metric=_METRIC,
                leadingness=LEADING,
                raw_value=round(cur, 2),
                abnormal_value=round(asvi, 4),
            )
        )
        time.sleep(batch_sleep)

    if not rows and not any_ok:
        return [], False  # 모든 요청 실패(스코프/네트워크)
    rows.sort(
        key=lambda r: r.abnormal_value if r.abnormal_value is not None else -1e9,
        reverse=True,
    )
    for i, r in enumerate(rows):
        r.rank = i + 1
    return rows, True
