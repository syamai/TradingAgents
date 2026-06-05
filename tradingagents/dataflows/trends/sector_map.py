"""ticker → GICS 섹터 매핑 (yfinance .info, 영속 캐시).

와치리스트 종목을 섹터로 집계해 "어느 섹터에 retail attention 이 집중되는가"를
보기 위한 **모니터링용** 매핑이다(① 통제실험 결과: attention 은 독립 alpha 가
아니므로 alpha 신호로 쓰지 않는다). yfinance .info 는 느려 JSON 캐시에 영속하고,
섹터는 거의 안 변하므로 미스만 조회한다. ETF/소형주는 'Unknown' 일 수 있다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE = Path.home() / ".tradingagents" / "trends" / "sector_map.json"


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _save_cache(d: dict) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(d, ensure_ascii=False))


def get_sectors(tickers: list) -> dict:
    """ticker→sector dict. 캐시 우선, 미스만 yfinance .info 로 조회."""
    cache = _load_cache()
    upper = [str(t).upper() for t in tickers]
    missing = [t for t in upper if t not in cache]
    if missing:
        try:
            import yfinance as yf
        except ImportError:
            return {t: cache.get(t, "Unknown") for t in upper}
        for t in missing:
            try:
                info = yf.Ticker(t).info
                cache[t] = info.get("sector") or info.get("category") or "Unknown"
            except Exception as exc:  # yfinance 예외 다양 → 광범위 캐치
                logger.warning("sector 조회 실패 %s: %s", t, exc)
                cache[t] = "Unknown"
        _save_cache(cache)
    return {t: cache.get(t, "Unknown") for t in upper}
