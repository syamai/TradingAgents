"""트렌드(투자자 관심·자금 쏠림) 신호 수집 공통 기반.

각 소스 fetcher(apewisdom, finviz_unusual, ...)는 ``SignalRow`` 리스트를 만들어
``TrendStore`` 에 적재한다. 네트워크 실패는 예외를 올리지 않고 graceful degrade
(빈 리스트 + ``fetch_ok=False``) 하여, cron tick 이 죽지 않고 다음 tick 에서
자가복구하도록 한다(stocktwits.py 와 동일 규약).

``leadingness`` 태그는 리서치 노트(트렌드 연구자료.md)의 정적 휴리스틱이다 —
대부분의 attention 신호는 coincident(동행)이며, 검색·소셜 buzz 는 늦은
contrarian 신호인 경우가 많다. v1 은 측정이 아니라 per-source 고정 태그.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# leadingness 태그. L: leading(가격보다 앞섬, 드묾) / C: coincident(동행) /
# Lag: lagging(후행). 노트의 핵심 결론 = "보이는 buzz 는 대개 C/늦음".
LEADING = "L"
COINCIDENT = "C"
LAGGING = "Lag"
_VALID_LEADINGNESS = {LEADING, COINCIDENT, LAGGING}
_VALID_MARKETS = {"us", "kr"}

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_DEFAULT_TIMEOUT = 10.0


@dataclass
class SignalRow:
    """트렌드 신호 1건 = ``trend_snapshots`` 1 row.

    release_date: 수집(스냅샷)한 날 (YYYY-MM-DD)
    asof_date:    신호가 유효한 기준일. 백테스트는 ``asof_date <= date[i]`` 만 읽음.
    market:       'us' | 'kr'
    entity:       티커 ('NVDA' | '005930') — Phase1 ticker-only
    source:       'apewisdom' | 'finviz_unusual' | 'kis_retail' | ...
    metric:       'mention_momentum_24h' | 'abnormal_vol' | 'retail_crowding_z'
    raw_value:    소스 원시 크기 (감사용)
    abnormal_value: 자기 trailing baseline 대비 비정상값 (랭킹 기준)
    leadingness:  'L' | 'C' | 'Lag'
    rank:         소스 내 비정상 순위 (1 = 가장 뜨거움), 없으면 None
    """

    release_date: str
    asof_date: str
    market: str
    entity: str
    source: str
    metric: str
    leadingness: str
    entity_type: str = "ticker"  # 'ticker' | 'sector' | 'theme'
    raw_value: Optional[float] = None
    abnormal_value: Optional[float] = None
    rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.leadingness not in _VALID_LEADINGNESS:
            raise ValueError(
                f"leadingness must be one of {sorted(_VALID_LEADINGNESS)}, "
                f"got {self.leadingness!r}"
            )
        if self.market not in _VALID_MARKETS:
            raise ValueError(
                f"market must be one of {sorted(_VALID_MARKETS)}, "
                f"got {self.market!r}"
            )


def _http_get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    headers: Optional[dict] = None,
) -> Optional[dict]:
    """JSON GET. 실패(네트워크·HTTP·파싱) 시 ``None`` 을 반환하고 예외는 안 올린다."""
    hdrs = {"User-Agent": _UA, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    try:
        resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("JSON GET 실패 %s: %s", url, exc)
        return None


def _http_get_html(
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: float = _DEFAULT_TIMEOUT,
    headers: Optional[dict] = None,
) -> Optional[str]:
    """HTML GET. ``requests`` 실패(403 등) 시 ``cloudscraper`` 로 1회 재시도.

    Finviz 등 Cloudflare 보호 엔드포인트 대응. cloudscraper 미설치/재실패 시
    ``None`` 으로 조용히 degrade(다음 tick 자가복구).
    """
    hdrs = {"User-Agent": _UA}
    if headers:
        hdrs.update(headers)
    try:
        resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.warning("HTML GET 실패 %s (%s) — cloudscraper 재시도", url, exc)

    try:
        import cloudscraper
    except ImportError:
        logger.warning("cloudscraper 미설치 — %s degrade", url)
        return None
    try:
        scraper = cloudscraper.create_scraper()
        resp = scraper.get(url, params=params, headers=hdrs, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # cloudscraper 예외 타입이 다양해 광범위 캐치
        logger.warning("cloudscraper 도 실패 %s: %s", url, exc)
        return None
