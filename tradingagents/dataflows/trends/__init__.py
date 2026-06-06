"""트렌드(투자자 관심·자금 쏠림) 신호 수집 서브패키지.

각 소스 fetcher 가 ``SignalRow`` 리스트를 만들고, ``tradingagents.dataflows.
trend_store.TrendStore`` 가 SQLite 에 point-in-time 적재한다. ``SOURCE_REGISTRY``
는 ``trend_collect`` 진입점이 ``--source`` 이름으로 fetcher 를 찾는 데 쓴다.

각 fetcher 는 ``collect_*(market, lookback_days, timeout, ...) -> (rows, fetch_ok)``
규약을 따른다(stocktwits.collect_* 와 동일). ``fetch_ok=False`` 는 첫 호출부터
실패한 경우만 — 부분 수집은 가진 만큼 ``True`` 로 반환.
"""
from __future__ import annotations

from typing import Callable

from .apewisdom import collect_apewisdom
from .base import COINCIDENT, LAGGING, LEADING, SignalRow
from .finviz_unusual import collect_finviz_unusual
from .google_trends import collect_google_trends
from .naver_board import collect_naver_board
from .naver_datalab import collect_naver_datalab
from .options_os import collect_options_os
from .sector_rotation import collect_sector_rotation
from .stocktwits_delta import collect_stocktwits_delta

# source 이름 -> (collect 함수, 기본 market). trend_collect 가 --source 로 조회.
# 순서 주의: per-ticker universe 소스(google_trends/stocktwits_delta/options_os)는
# hot_candidates 를 읽으므로 apewisdom/finviz 뒤에 둔다(같은 tick 에서 후보 먼저 적재).
# naver_board/naver_datalab(kr)의 universe 는 hot_candidates 가 아니라
# kr_peer_bridge 가 만든다. naver_datalab(검색량)은 토론방과 독립인 2번째 KR 소스.
SOURCE_REGISTRY: dict[str, dict] = {
    "apewisdom": {"fn": collect_apewisdom, "market": "us"},
    "finviz_unusual": {"fn": collect_finviz_unusual, "market": "us"},
    "sector_rotation": {"fn": collect_sector_rotation, "market": "us"},
    "google_trends": {"fn": collect_google_trends, "market": "us"},
    "stocktwits_delta": {"fn": collect_stocktwits_delta, "market": "us"},
    "options_os": {"fn": collect_options_os, "market": "us"},
    "naver_board": {"fn": collect_naver_board, "market": "kr"},
    "naver_datalab": {"fn": collect_naver_datalab, "market": "kr"},
}

# universe 를 주입받는 per-ticker 소스(us=hot_candidates, kr=kr_peer_bridge)
UNIVERSE_SOURCES: frozenset[str] = frozenset(
    {"google_trends", "stocktwits_delta", "options_os", "naver_board", "naver_datalab"}
)

__all__ = [
    "SignalRow",
    "LEADING",
    "COINCIDENT",
    "LAGGING",
    "SOURCE_REGISTRY",
    "UNIVERSE_SOURCES",
    "collect_apewisdom",
    "collect_finviz_unusual",
    "collect_sector_rotation",
    "collect_google_trends",
    "collect_stocktwits_delta",
    "collect_options_os",
    "collect_naver_board",
    "collect_naver_datalab",
]
