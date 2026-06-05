"""거시·지수 가격 시계열 — yfinance 래퍼.

KOSPI/USDKRW/미국10Y금리/WTI 같은 special symbol 의 일별 close 를 가져온다.
naver_stock.fetch_naver_ohlcv_df 는 종목 코드(6자리)만 처리해 지수 미지원이라
yfinance 로 통일.

캐시: ``~/.tradingagents/cache/market/{key}_{start}_{end}.parquet`` (1일 TTL).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# yfinance 심볼 매핑 — 별칭(key) → 실제 yf 심볼
SPECIAL_SYMBOLS: dict[str, str] = {
    "kospi":    "^KS11",
    "kospi200": "^KS200",
    "kosdaq":   "^KQ11",
    "usdkrw":   "KRW=X",
    "us_10y":   "^TNX",
    "wti":      "CL=F",
    "vix":      "^VIX",
    "sp500":    "^GSPC",
    "nasdaq":   "^IXIC",
}


def _cache_dir() -> Path:
    base = os.environ.get(
        "TRADINGAGENTS_CACHE_DIR",
        str(Path.home() / ".tradingagents" / "cache"),
    )
    path = Path(base) / "market"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(key: str, start: str, end: str) -> Path:
    return _cache_dir() / f"{key}_{start}_{end}.parquet"


def _is_historical_end(end: str) -> bool:
    """요청 범위가 완전히 과거인지 판정한다.

    백테스트/재랭크는 재현성이 우선이다. 과거 구간의 시장 데이터 캐시를 TTL 만료
    때문에 자동 갱신하면 Yahoo 수정치·부분 다운로드·FX 소스 결손으로 같은 전략의
    지표가 조용히 달라질 수 있다. 오늘을 포함한 범위만 TTL 갱신 대상이다.
    """
    try:
        return datetime.strptime(end, "%Y-%m-%d").date() < datetime.now().date()
    except ValueError:
        return False


def _cache_ttl_seconds() -> int:
    raw = os.environ.get("TRADINGAGENTS_MARKET_CACHE_TTL_SECONDS")
    if raw is None:
        return 86400
    try:
        return int(raw)
    except ValueError:
        return 86400


def _is_cache_fresh(p: Path, *, ttl_seconds: Optional[int] = None) -> bool:
    if not p.exists():
        return False
    if ttl_seconds is None:
        ttl_seconds = _cache_ttl_seconds()
    if ttl_seconds < 0:
        return True
    age = time.time() - p.stat().st_mtime
    return age < ttl_seconds


def fetch_close_series(
    key: str, start: str, end: str,
    *, cache: bool = True,
) -> pd.DataFrame:
    """key 또는 yfinance 심볼 → 일별 close 시계열.

    Args:
        key: SPECIAL_SYMBOLS 별칭(``kospi`` 등) 또는 yfinance 심볼 직접(``^KS11``).
        start, end: ``YYYY-MM-DD``.

    Returns:
        DataFrame columns ``[date, close]``. 빈 결과 시 빈 DataFrame.
    """
    symbol = SPECIAL_SYMBOLS.get(key.lower(), key)
    cache_p = _cache_path(key.lower(), start, end)
    force_refresh = os.environ.get("TRADINGAGENTS_MARKET_CACHE_REFRESH") == "1"
    use_cached_historical = cache and cache_p.exists() and _is_historical_end(end) and not force_refresh
    if cache and not force_refresh and (use_cached_historical or _is_cache_fresh(cache_p)):
        try:
            return pd.read_parquet(cache_p)
        except Exception:
            pass  # 캐시 손상 시 재fetch

    try:
        ticker = yf.Ticker(symbol)
        # yfinance end_date는 exclusive — +1일로 보정
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        raw = ticker.history(
            start=start, end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=False, prepost=False,
        )
    except Exception as exc:
        logger.warning("yfinance fetch %s failed: %s", symbol, exc)
        return pd.DataFrame(columns=["date", "close"])

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close"])

    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d")
    out = raw[["date", "Close"]].rename(columns={"Close": "close"})
    out = out.reset_index(drop=True)

    if cache:
        try:
            out.to_parquet(cache_p, index=False)
        except Exception as exc:
            logger.warning("market cache write failed: %s", exc)
    return out


def fetch_kospi(start: str, end: str) -> pd.DataFrame:
    return fetch_close_series("kospi", start, end)


def fetch_usdkrw(start: str, end: str) -> pd.DataFrame:
    return fetch_close_series("usdkrw", start, end)


def fetch_us_10y(start: str, end: str) -> pd.DataFrame:
    return fetch_close_series("us_10y", start, end)


def fetch_wti(start: str, end: str) -> pd.DataFrame:
    return fetch_close_series("wti", start, end)


def compute_relative_strength(
    target: pd.DataFrame, benchmark: pd.DataFrame,
) -> Optional[float]:
    """R/S — target 수익률 − benchmark 수익률. 단순 시작/종료 차분.

    각 DataFrame에 date 오름차순 + close 필수. 빈 시리즈는 None.
    """
    if target.empty or benchmark.empty:
        return None
    t0, t1 = float(target["close"].iloc[0]), float(target["close"].iloc[-1])
    b0, b1 = float(benchmark["close"].iloc[0]), float(benchmark["close"].iloc[-1])
    if t0 <= 0 or b0 <= 0:
        return None
    target_ret = (t1 / t0 - 1) * 100
    bench_ret = (b1 / b0 - 1) * 100
    return round(target_ret - bench_ret, 4)


def summarize_series(series: pd.DataFrame) -> dict:
    """시계열 요약 — 시작/종료/min/max/변화율."""
    if series.empty:
        return {"available": False}
    start_val = float(series["close"].iloc[0])
    end_val = float(series["close"].iloc[-1])
    return {
        "available": True,
        "start_date": str(series["date"].iloc[0]),
        "end_date": str(series["date"].iloc[-1]),
        "start": start_val,
        "end": end_val,
        "min": float(series["close"].min()),
        "max": float(series["close"].max()),
        "change_pct": round((end_val / start_val - 1) * 100, 4) if start_val > 0 else None,
        "n_days": len(series),
    }
