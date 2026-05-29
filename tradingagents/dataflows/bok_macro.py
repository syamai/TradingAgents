"""한국은행 ECOS API — 거시 시계열 (기준금리·회사채·스프레드).

환경변수 ``BOK_ECOS_API_KEY`` 미설정 시 빈 DataFrame + 로그 경고. 호출자는
``df.empty`` 또는 ``df["available"]`` 으로 분기.

ECOS API:
  https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/1/1000/{STAT}/{CYCLE}/{START}/{END}/{ITEM}
  응답: ``{"StatisticSearch": {"row": [{"TIME": "20250214", "DATA_VALUE": "3.00"}...]}}``
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"
_TIMEOUT = 15.0

# ECOS 통계표 코드 — 자주 쓰는 거시 지표
STAT_BASE_RATE = ("722Y001", "0101000")        # 한국은행 기준금리 (일)
STAT_CORP_AA = ("817Y002", "010300000")        # 회사채(AA-, 3년) 수익률 (일)
STAT_TREASURY_3Y = ("817Y002", "010200000")    # 국고채 3년 수익률 (일)
STAT_TREASURY_10Y = ("817Y002", "010210000")   # 국고채 10년 수익률 (일)


def _api_key() -> Optional[str]:
    return os.environ.get("BOK_ECOS_API_KEY") or None


def _cache_dir() -> Path:
    base = os.environ.get(
        "TRADINGAGENTS_CACHE_DIR",
        str(Path.home() / ".tradingagents" / "cache"),
    )
    path = Path(base) / "bok"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(stat: str, item: str, start: str, end: str) -> Path:
    return _cache_dir() / f"{stat}_{item}_{start}_{end}.parquet"


def _is_cache_fresh(p: Path, *, ttl_seconds: int = 86400) -> bool:
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < ttl_seconds


def _format_date(d: str) -> str:
    """YYYY-MM-DD or YYYYMMDD → YYYYMMDD."""
    return d.replace("-", "")


def fetch_bok_series(
    stat_code: str, item_code: str,
    start: str, end: str,
    *, freq: str = "D", cache: bool = True,
) -> pd.DataFrame:
    """ECOS 통계 시계열 → DataFrame ``[date, value]``.

    빈 결과 / 키 미설정 / API 오류 모두 빈 DataFrame (with logging).
    """
    cache_p = _cache_path(stat_code, item_code, _format_date(start), _format_date(end))
    if cache and _is_cache_fresh(cache_p):
        try:
            return pd.read_parquet(cache_p)
        except Exception:
            pass

    key = _api_key()
    if not key:
        logger.warning("BOK_ECOS_API_KEY not set — returning empty %s", stat_code)
        return pd.DataFrame(columns=["date", "value"])

    url = (
        f"{_ECOS_BASE}/{key}/json/kr/1/1000/{stat_code}/{freq}/"
        f"{_format_date(start)}/{_format_date(end)}/{item_code}"
    )
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        logger.warning("ECOS %s fetch failed: %s", stat_code, exc)
        return pd.DataFrame(columns=["date", "value"])

    # 정상 응답: {"StatisticSearch": {"list_total_count": N, "row": [...]}}
    # 오류: {"RESULT": {"CODE": "INFO-200", "MESSAGE": "..."}}
    if "RESULT" in body and body["RESULT"].get("CODE") not in (None, "INFO-200"):
        logger.warning(
            "ECOS error %s: %s",
            body["RESULT"].get("CODE"), body["RESULT"].get("MESSAGE"),
        )
        return pd.DataFrame(columns=["date", "value"])

    rows = (body.get("StatisticSearch") or {}).get("row") or []
    if not rows:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame([
        {
            "date": _parse_ecos_date(r.get("TIME", "")),
            "value": _safe_float(r.get("DATA_VALUE")),
        }
        for r in rows
    ])
    df = df.dropna(subset=["date", "value"]).reset_index(drop=True)

    if cache and not df.empty:
        try:
            df.to_parquet(cache_p, index=False)
        except Exception as exc:
            logger.warning("BOK cache write failed: %s", exc)
    return df


def _parse_ecos_date(s: str) -> Optional[str]:
    """ECOS TIME(YYYYMMDD or YYYYMM 등) → YYYY-MM-DD."""
    s = (s or "").strip()
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    if len(s) == 6:
        return f"{s[:4]}-{s[4:6]}-01"
    if len(s) == 4:
        return f"{s}-01-01"
    return None


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "-") else None
    except (ValueError, TypeError):
        return None


# === 편의 함수 ==================================================================

def fetch_base_rate(start: str, end: str) -> pd.DataFrame:
    return fetch_bok_series(*STAT_BASE_RATE, start, end, freq="D")


def fetch_corporate_aa_yield(start: str, end: str) -> pd.DataFrame:
    return fetch_bok_series(*STAT_CORP_AA, start, end, freq="D")


def fetch_treasury_3y(start: str, end: str) -> pd.DataFrame:
    return fetch_bok_series(*STAT_TREASURY_3Y, start, end, freq="D")


def fetch_corp_bond_spread(start: str, end: str) -> pd.DataFrame:
    """회사채AA- − 국고채3Y 스프레드.

    두 시리즈를 date 기준 outer join. 어느 한 쪽이라도 빈 결과면 빈 DF.
    """
    corp = fetch_corporate_aa_yield(start, end)
    tre = fetch_treasury_3y(start, end)
    if corp.empty or tre.empty:
        return pd.DataFrame(columns=["date", "spread"])
    merged = corp.merge(tre, on="date", how="inner", suffixes=("_corp", "_tre"))
    merged["spread"] = (merged["value_corp"] - merged["value_tre"]).round(4)
    return merged[["date", "spread"]].reset_index(drop=True)


def summarize_macro(start: str, end: str) -> dict:
    """기간 동안 BOK 거시 지표 요약 — 시작/종료/변화."""
    out: dict[str, dict] = {}
    for label, fetcher in [
        ("base_rate", fetch_base_rate),
        ("corp_aa_yield", fetch_corporate_aa_yield),
        ("treasury_3y", fetch_treasury_3y),
    ]:
        df = fetcher(start, end)
        if df.empty:
            out[label] = {"available": False}
            continue
        s, e = float(df["value"].iloc[0]), float(df["value"].iloc[-1])
        out[label] = {
            "available": True,
            "start": s, "end": e,
            "change_bp": round((e - s) * 100, 1),  # basis point
            "n_days": len(df),
            "start_date": str(df["date"].iloc[0]),
            "end_date": str(df["date"].iloc[-1]),
        }
    return out
