"""Google Trends ASVI — 검색량 기반 retail attention.

pytrends 로 종목 주간 검색량(SVI)을 받아 ASVI = log(이번주 SVI) − log(직전 8주
중앙값)를 계산한다(노트: 검색 = 직접 관측 가능한 retail attention; 리테일 관심은
leading 약 2주 후 반전). 종목 universe 는 그날 이미 뜬 후보(hot_candidates)로
한정해 pytrends 429·cron 예산을 통제한다(per-keyword 라 전수 조회 금지).

snapshot-lock: Google Trends 는 매 호출 0–100 으로 재스케일되므로, asof_date 로
한 번 저장한 값은 절대 덮어쓰지 않는다(TrendStore.write 의 INSERT OR IGNORE).
이것이 백테스트 look-ahead 누출을 막는 핵심이다.

leadingness = L. pytrends 는 미유지보수·429 빈발이라 실패는 graceful degrade.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Optional

from .base import LEADING, SignalRow

logger = logging.getLogger(__name__)

_SOURCE = "google_trends"
_METRIC = "asvi"
_BATCH = 5  # pytrends build_payload 키워드 상한


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def collect_google_trends(
    *,
    market: str = "us",
    asof_date: Optional[str] = None,
    universe: Optional[list] = None,
    batch_sleep: float = 2.0,
    lookback_days: int = 120,  # 시그니처 일관성용
    timeout: float = 30.0,  # 시그니처 일관성용
) -> tuple[list[SignalRow], bool]:
    """universe 종목의 ASVI 를 수집. universe 없으면 빈(첫 tick — 실패 아님).

    반환 ``(rows, fetch_ok)``. pytrends 미설치/초기화 실패 시 ``([], False)``.
    """
    asof = asof_date or _today()
    if not universe:
        return [], True
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("pytrends 미설치 — google_trends degrade")
        return [], False
    try:
        pt = TrendReq(hl="en-US", tz=360)
    except Exception as exc:  # 네트워크/초기화 실패
        logger.warning("pytrends 초기화 실패: %s", exc)
        return [], False

    rows: list[SignalRow] = []
    any_ok = False
    for i in range(0, len(universe), _BATCH):
        batch = [str(x).upper() for x in universe[i:i + _BATCH]]
        try:
            pt.build_payload(batch, timeframe="today 12-m")  # 주간 SVI
            df = pt.interest_over_time()
        except Exception as exc:  # 429 등 — 배치 스킵, 다음 tick 자가복구
            logger.warning("pytrends batch 실패 %s: %s", batch, exc)
            continue
        if df is None or df.empty:
            continue
        any_ok = True
        for kw in batch:
            if kw not in df.columns:
                continue
            svi = df[kw].astype(float)
            if int((svi > 0).sum()) < 9:  # 8주 baseline + 당주 부족
                continue
            cur = float(svi.iloc[-1])
            base = float(svi.iloc[-9:-1].median())  # 직전 8주 중앙값
            if cur <= 0 or base <= 0:
                continue
            asvi = math.log(cur) - math.log(base)
            rows.append(
                SignalRow(
                    release_date=asof, asof_date=asof, market=market, entity=kw,
                    source=_SOURCE, metric=_METRIC, leadingness=LEADING,
                    raw_value=cur, abnormal_value=round(asvi, 4), rank=None,
                )
            )
        time.sleep(batch_sleep)

    if not rows and not any_ok:
        return [], False  # 모든 배치 실패
    rows.sort(key=lambda r: r.abnormal_value if r.abnormal_value is not None else -1e9, reverse=True)
    for rank, r in enumerate(rows, start=1):
        r.rank = rank
    return rows, True
