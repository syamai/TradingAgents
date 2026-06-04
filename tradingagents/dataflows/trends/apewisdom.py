"""ApeWisdom — 무료 Reddit/WSB mention-momentum 수집.

apewisdom.io 의 공개 API(키 불필요)는 서브레딧별 종목 언급(mention) 집계와
24시간 전 대비 값을 준다. 우리는 '24h mention momentum'(현재 24h 언급이 직전
24h 대비 얼마나 급증했나)을 ``abnormal_value`` 로 쓴다 — 자기 베이스라인 대비
'새로 생긴 관심'.

응답 형태(검증):
  {"count":..,"pages":..,"results":[{"rank":1,"ticker":"AVGO","mentions":1501,
   "mentions_24h_ago":356,...}, ...]}

leadingness = C(coincident/contrarian): 소셜 buzz 는 대개 동행하며, 화제가
절정일 때가 천장인 경우가 많다(노트). 추격이 아니라 fade/리스크 플래그용.
봇·pump 에 취약 → ``trend_rank`` 에서 ≥2 소스 동의로 1차 방어.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .base import COINCIDENT, SignalRow, _http_get_json

_API = "https://apewisdom.io/api/v1.0/filter/{filter}/page/{page}"
_SOURCE = "apewisdom"
_METRIC = "mention_momentum_24h"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def collect_apewisdom(
    *,
    market: str = "us",
    asof_date: Optional[str] = None,
    top_n: int = 20,
    pages: int = 2,
    min_mentions: int = 10,
    filter_name: str = "wallstreetbets",
    timeout: float = 10.0,
    lookback_days: int = 1,  # 시그니처 일관성용 — ApeWisdom 은 24h 고정이라 미사용
) -> tuple[list[SignalRow], bool]:
    """WSB 24h mention momentum 상위 종목을 SignalRow 로 수집.

    반환 ``(rows, fetch_ok)``. 첫 페이지부터 실패하면 ``([], False)``;
    중간 페이지 실패는 가진 만큼 ``True`` 로 반환.
    """
    asof = asof_date or _today()
    raw: list[dict] = []
    first = True
    for page in range(1, pages + 1):
        url = _API.format(filter=filter_name, page=page)
        data = _http_get_json(url, timeout=timeout)
        if data is None:
            if first:
                return [], False
            break
        first = False
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            break
        raw.extend(results)

    scored: list[tuple[float, dict]] = []
    seen: set[str] = set()
    for r in raw:
        try:
            ticker = str(r["ticker"]).upper()
            mentions = float(r["mentions"])
            prev = float(r.get("mentions_24h_ago") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if ticker in seen or mentions < min_mentions:
            continue  # 페이지 경계 중복 ticker 제거
        seen.add(ticker)
        # 자기 베이스라인(직전 24h) 대비 비정상 증가율
        momentum = (mentions - prev) / max(prev, 1.0)
        scored.append((momentum, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    rows: list[SignalRow] = []
    for rank, (momentum, r) in enumerate(scored[:top_n], start=1):
        rows.append(
            SignalRow(
                release_date=asof,
                asof_date=asof,
                market=market,
                entity=str(r["ticker"]).upper(),
                source=_SOURCE,
                metric=_METRIC,
                leadingness=COINCIDENT,
                raw_value=float(r["mentions"]),
                abnormal_value=round(momentum, 4),
                rank=rank,
            )
        )
    return rows, True
