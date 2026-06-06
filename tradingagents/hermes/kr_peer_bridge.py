"""US 관심 쏠림 종목 → 유사 한국 종목 매핑(브리지).

미국장 마감 후 ``fade_ranking('us')`` 로 떠오른 미국 종목과 사업·테마·공급망이
유사한 한국 상장 종목을 LLM 으로 찾아, 한국 트렌드 모니터링 후보 유니버스를
만든다. US 티커당 1회 호출 + SQLite 캐시(TTL)로 비용을 무시할 수준으로 낮춘다.

모니터링 전용 — 백테스트 미연동. cron LLM-free 원칙의 예외이며, 캐시 hit 시
LLM 호출 0. 순환 import 회피를 위해 ``fade_ranking``/``KisHistoryStore`` 는
함수 내부에서 지연 import 한다.
"""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..dataflows.trend_store import TrendStore

_SIX_DIGIT = re.compile(r"^\d{6}$")


class _KrPeer(BaseModel):
    code: str = Field(description="한국 상장 종목의 6자리 코드 (예: '005930')")
    name: str = Field(description="한국 종목명 (예: '삼성전자')")
    reason: str = Field(description="유사성 근거 한 줄 (사업·테마·공급망)")


class _KrPeerList(BaseModel):
    peers: list[_KrPeer] = Field(
        description="US 종목과 사업/테마/공급망이 유사한 한국 상장 종목들"
    )


_PROMPT = (
    "당신은 한미 증시에 정통한 애널리스트다. 미국 종목 {ticker} 와 "
    "사업·테마·공급망이 유사한 **한국 상장 종목**을 최대 {n}개 고른다. "
    "단순히 같은 대분류 섹터가 아니라, 같은 제품/기술/공급망/테마로 한국 투자자가 "
    "'{ticker} 관련주'로 연상할 종목을 우선한다. 각 종목의 6자리 코드와 종목명, "
    "유사성 근거 한 줄을 제시한다. 유사 종목이 없으면 빈 리스트."
)


def _default_llm():
    from ..default_config import DEFAULT_CONFIG
    from ..llm_clients.factory import create_llm_client

    cfg = DEFAULT_CONFIG
    client = create_llm_client(
        cfg["llm_provider"], cfg["quick_think_llm"], cfg.get("backend_url")
    )
    return client.get_llm()


def _korean_name(code: str) -> Optional[str]:
    """KIS 메타에서 6자리 코드 → 종목명. 없으면 None(지연 import)."""
    try:
        from ..dataflows.kis_history_store import KisHistoryStore

        meta = KisHistoryStore().get_ticker_metadata(code)
    except Exception:
        return None
    return meta.get("company_name") if meta else None


def _clean_peers(peers: list[dict], max_peers: int) -> list[dict]:
    """6자리 형식 검증 · 중복 제거 · 개수 상한 · 종목명 보강."""
    out: list[dict] = []
    seen: set[str] = set()
    for p in peers:
        code = str(p.get("code", "")).strip()
        if not _SIX_DIGIT.match(code) or code in seen:
            continue
        seen.add(code)
        name = (p.get("name") or "").strip() or _korean_name(code) or code
        out.append({"code": code, "name": name, "reason": (p.get("reason") or "").strip()})
        if len(out) >= max_peers:
            break
    return out


class KrPeerCache:
    """US 티커 → 유사 한국 종목 매핑 캐시(SQLite). TTL 로 주기적 갱신."""

    def __init__(self, *, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else Path.home() / ".tradingagents" / "hermes"
        self.root.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kr_peer_map ("
                "  us_ticker TEXT PRIMARY KEY,"
                "  kr_json TEXT NOT NULL,"
                "  model_id TEXT,"
                "  cached_at TEXT NOT NULL"
                ")"
            )

    def _db_path(self) -> Path:
        return self.root / "kr_peer_map.db"

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self._db_path()))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, us_ticker: str, *, ttl_days: int = 90) -> Optional[list[dict]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT kr_json, cached_at FROM kr_peer_map WHERE us_ticker=?",
                (us_ticker,),
            ).fetchone()
        if row is None:
            return None
        try:
            cached_at = datetime.strptime(row[1], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
        if datetime.utcnow() - cached_at > timedelta(days=ttl_days):
            return None
        return json.loads(row[0])

    def put(self, us_ticker: str, peers: list[dict], *, model_id: str = "") -> None:
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO kr_peer_map (us_ticker, kr_json, model_id, cached_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(us_ticker) DO UPDATE SET "
                "  kr_json=excluded.kr_json, model_id=excluded.model_id, "
                "  cached_at=excluded.cached_at",
                (us_ticker, json.dumps(peers, ensure_ascii=False), model_id, now),
            )

    def reverse(self, codes) -> dict:
        """한국 코드 → {us_ticker, reason} (캐시 전체에서 역조회). digest provenance 용."""
        want = set(codes)
        rev: dict = {}
        with self._conn() as conn:
            rows = conn.execute("SELECT us_ticker, kr_json FROM kr_peer_map").fetchall()
        for us, kr_json in rows:
            for p in json.loads(kr_json):
                if p["code"] in want and p["code"] not in rev:
                    rev[p["code"]] = {
                        "us": us,
                        "reason": p.get("reason", ""),
                        "name": p.get("name", ""),
                    }
        return rev


def map_us_to_kr(
    us_tickers,
    *,
    llm=None,
    cache: Optional[KrPeerCache] = None,
    max_peers: int = 5,
    ttl_days: int = 90,
) -> dict:
    """US 티커 리스트 → ``{us_ticker: [{code, name, reason}, ...]}``.

    캐시 hit 은 LLM 호출 없이 반환, miss 만 LLM 1회 호출 후 캐시. LLM 실패한
    티커는 빈 리스트(graceful).
    """
    cache = cache or KrPeerCache()
    out: dict = {}
    pending: list[str] = []
    for t in us_tickers:
        hit = cache.get(t, ttl_days=ttl_days)
        if hit is not None:
            out[t] = hit
        else:
            pending.append(t)
    if not pending:
        return out

    llm = llm or _default_llm()
    structured = llm.with_structured_output(_KrPeerList)
    model_id = str(getattr(llm, "model", "") or "")
    for t in pending:
        try:
            res = structured.invoke(_PROMPT.format(ticker=t, n=max_peers))
            raw = [p.model_dump() for p in res.peers]
        except Exception:
            raw = []
        peers = _clean_peers(raw, max_peers)
        if peers:
            cache.put(t, peers, model_id=model_id)
        out[t] = peers
    return out


def kr_universe(
    asof_date: Optional[str] = None,
    *,
    store: Optional[TrendStore] = None,
    top_us: int = 10,
    llm=None,
    cache: Optional[KrPeerCache] = None,
    max_peers: int = 5,
) -> list:
    """미국 fade 와치리스트 상위 → 유사 한국 종목 코드(중복 제거) 유니버스."""
    from .trend_rank import fade_ranking

    store = store or TrendStore()
    fade = fade_ranking("us", asof_date=asof_date, top_n=top_us, store=store)
    if fade.empty:
        return []
    us_tickers = list(fade["entity"])
    mapping = map_us_to_kr(us_tickers, llm=llm, cache=cache, max_peers=max_peers)
    seen: list[str] = []
    for t in us_tickers:
        for p in mapping.get(t, []):
            if p["code"] not in seen:
                seen.append(p["code"])
    return seen
