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

# source 이름 -> (collect 함수, 기본 market). trend_collect 가 --source 로 조회.
SOURCE_REGISTRY: dict[str, dict] = {
    "apewisdom": {"fn": collect_apewisdom, "market": "us"},
    "finviz_unusual": {"fn": collect_finviz_unusual, "market": "us"},
}

__all__ = [
    "SignalRow",
    "LEADING",
    "COINCIDENT",
    "LAGGING",
    "SOURCE_REGISTRY",
    "collect_apewisdom",
    "collect_finviz_unusual",
]
