"""트렌드 신호 영구 저장 — SQLite 백엔드.

``trend_snapshots`` 한 테이블, append-only. ``snapshot_hash``
(``asof_date|market|source|entity|metric`` 의 md5) UNIQUE 로 멱등하다.

**한 번 쓴 행은 절대 갱신하지 않는다(snapshot-lock).** Google Trends 소급
재스케일·벤더 정정이 과거 신호를 바꿔 look-ahead 누출을 일으키는 것을 차단한다
(point-in-time). 백테스트는 ``asof_date <= date[i]`` 만 읽는다.

스토어 패턴은 ``hermes/hypothesis_store.py`` 와 동일(``_conn`` contextmanager,
``_init_schema``, 서버측 ``fetched_at`` default).
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

from .trends.base import SignalRow


def _default_store_dir() -> Path:
    return Path.home() / ".tradingagents" / "trends"


def snapshot_hash(row: SignalRow) -> str:
    """행의 멱등 키. 같은 (기준일·시장·소스·엔티티·지표)면 같은 해시 = 무갱신."""
    key = f"{row.asof_date}|{row.market}|{row.source}|{row.entity}|{row.metric}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


class TrendStore:
    """트렌드 신호 point-in-time 스토어."""

    def __init__(self, *, root: Optional[Path] = None):
        self.root = Path(root) if root else _default_store_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _db_path(self) -> Path:
        return self.root / "trends.db"

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path()))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS trend_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_date TEXT NOT NULL,
                    asof_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    raw_value REAL,
                    abnormal_value REAL,
                    leadingness TEXT NOT NULL,
                    rank INTEGER,
                    metric TEXT NOT NULL,
                    snapshot_hash TEXT NOT NULL UNIQUE,
                    fetched_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    )
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_trend_asof_market "
                "ON trend_snapshots(asof_date, market)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_trend_rank "
                "ON trend_snapshots(asof_date, market, source, rank)"
            )

    # === Public API ===

    def write(self, rows: Sequence[SignalRow]) -> int:
        """행 적재. ``snapshot_hash`` 중복은 무시(``INSERT OR IGNORE``).

        반환 = 신규 삽입된 행 수. 기존 행은 절대 갱신하지 않는다(snapshot-lock).
        """
        if not rows:
            return 0
        new = 0
        with self._conn() as c:
            for r in rows:
                cur = c.execute(
                    """
                    INSERT OR IGNORE INTO trend_snapshots (
                        release_date, asof_date, market, entity, source,
                        raw_value, abnormal_value, leadingness, rank, metric,
                        snapshot_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r.release_date, r.asof_date, r.market, r.entity, r.source,
                        r.raw_value, r.abnormal_value, r.leadingness, r.rank,
                        r.metric, snapshot_hash(r),
                    ),
                )
                new += cur.rowcount
        return new

    def read(
        self,
        *,
        asof_date_lte: Optional[str] = None,
        market: Optional[str] = None,
        source: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> pd.DataFrame:
        """필터 조회. ``asof_date_lte`` 로 point-in-time(과거만) 읽기."""
        clauses, params = [], []
        if asof_date_lte is not None:
            clauses.append("asof_date <= ?")
            params.append(asof_date_lte)
        if market is not None:
            clauses.append("market = ?")
            params.append(market)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if entity is not None:
            clauses.append("entity = ?")
            params.append(entity)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM trend_snapshots {where} "
                f"ORDER BY asof_date, market, source, rank",
                params,
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def last_snapshot_date(self, market: str, source: str) -> Optional[str]:
        """증분 수집용 — 해당 (시장·소스)의 가장 최근 asof_date. 없으면 None."""
        with self._conn() as c:
            row = c.execute(
                "SELECT MAX(asof_date) AS d FROM trend_snapshots "
                "WHERE market=? AND source=?",
                (market, source),
            ).fetchone()
        return row["d"] if row and row["d"] else None

    def hot_list(
        self,
        market: str,
        *,
        asof_date: Optional[str] = None,
        top: int = 20,
        min_leadingness: Optional[str] = None,
    ) -> pd.DataFrame:
        """시장의 최신(또는 지정) asof_date 비정상 상위 목록."""
        with self._conn() as c:
            if asof_date is None:
                row = c.execute(
                    "SELECT MAX(asof_date) AS d FROM trend_snapshots WHERE market=?",
                    (market,),
                ).fetchone()
                asof_date = row["d"] if row else None
            if asof_date is None:
                return pd.DataFrame()
            clauses = ["market=?", "asof_date=?"]
            params: list = [market, asof_date]
            if min_leadingness is not None:
                clauses.append("leadingness=?")
                params.append(min_leadingness)
            rows = c.execute(
                f"SELECT * FROM trend_snapshots WHERE {' AND '.join(clauses)} "
                f"ORDER BY abnormal_value IS NULL, abnormal_value DESC LIMIT ?",
                params + [top],
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
