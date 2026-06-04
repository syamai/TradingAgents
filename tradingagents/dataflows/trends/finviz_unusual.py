"""Finviz — 비정상 거래량(unusual volume) 스크리너 수집.

finviz.com 의 screener unusual-volume 시그널은 평소 대비 거래량이 급증한 종목을
선별한다. 등재 자체가 비정상 거래량 = 인지(attention) 충격의 대리지표다.

Phase1 은 결과 종목 **티커 목록 + 순위**만 수집한다(``abnormal_value`` 는 순위
역수). Relative Volume 수치의 정밀 추출은 Finviz 행 레이아웃에 의존해 깨지기
쉬우므로 Phase2 로 미룬다. 결과 행은 ``table.screener_table`` 안의
``stock?t=<TICKER>`` 링크로 식별한다(``pd.read_html`` 은 스크리너 필터 드롭다운을
결과 테이블로 오인함).

leadingness = C(coincident): 거래량 급증은 대개 같은 날 attention 과 동행(노트).
Finviz 는 Cloudflare 보호라 ``_http_get_html`` 의 cloudscraper fallback 에
의존하며, 막히면 ``([], False)`` 로 조용히 degrade(다음 tick 자가복구).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup

from .base import COINCIDENT, SignalRow, _http_get_html

_URL = "https://finviz.com/screener.ashx"
_PARAMS = {"v": "111", "s": "ta_unusualvolume", "o": "-volume"}
_SOURCE = "finviz_unusual"
_METRIC = "unusual_volume_rank"
_TICKER_HREF = re.compile(r"(?:^|/)stock\?t=([A-Za-z][A-Za-z.\-]{0,5})", re.I)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _parse_tickers(html: str, top_n: int) -> list[str]:
    """결과 테이블의 종목 링크(stock?t=<TICKER>)에서 티커를 순서대로 추출."""
    soup = BeautifulSoup(html, "html.parser")
    scope = soup.select_one("table.screener_table") or soup
    seen: list[str] = []
    for a in scope.find_all("a", href=True):
        m = _TICKER_HREF.search(a["href"])
        if not m:
            continue
        t = m.group(1).upper()
        if t not in seen:
            seen.append(t)
            if len(seen) >= top_n:
                break
    return seen


def collect_finviz_unusual(
    *,
    market: str = "us",
    asof_date: Optional[str] = None,
    top_n: int = 20,
    timeout: float = 15.0,
    lookback_days: int = 1,  # 시그니처 일관성용 — Finviz 는 당일 스냅샷
) -> tuple[list[SignalRow], bool]:
    """unusual-volume 상위 종목을 SignalRow 로 수집. 실패 시 ``([], False)``."""
    asof = asof_date or _today()
    html = _http_get_html(_URL, params=_PARAMS, timeout=timeout)
    if html is None:
        return [], False
    tickers = _parse_tickers(html, top_n)
    if not tickers:
        return [], False
    return (
        [
            SignalRow(
                release_date=asof,
                asof_date=asof,
                market=market,
                entity=t,
                source=_SOURCE,
                metric=_METRIC,
                leadingness=COINCIDENT,
                raw_value=None,
                # 등재 순위 역수를 비정상 점수 대용으로(상위 = 큰 값)
                abnormal_value=float(top_n - i),
                rank=i + 1,
            )
            for i, t in enumerate(tickers)
        ],
        True,
    )
