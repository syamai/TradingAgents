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
                    entity_type TEXT NOT NULL DEFAULT 'ticker',
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
            # 마이그레이션: 기존 DB 에 entity_type 추가(idempotent, kis_history_store 패턴)
            existing = {r[1] for r in c.execute("PRAGMA table_info(trend_snapshots)")}
            if "entity_type" not in existing:
                c.execute(
                    "ALTER TABLE trend_snapshots ADD COLUMN entity_type "
                    "TEXT NOT NULL DEFAULT 'ticker'"
                )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_trend_asof_market "
                "ON trend_snapshots(asof_date, market)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_trend_rank "
                "ON trend_snapshots(asof_date, market, source, rank)"
            )
            # 트렌드 필터 정책(Hermes 대화로 수정 → cron·digest 가 active 정책을 읽음).
            # market 당 active 1개. 변경 시 이전 active 를 비활성화하고 새 행 추가(이력 보존).
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS trend_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    name TEXT,
                    min_sources INTEGER,
                    rotation_peak_pct REAL,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    ),
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_trend_policy_active "
                "ON trend_policies(market, active)"
            )
            # 신규 진입 종목 심층분석(C) 캐시·기록. (market,entity,asof_date) 당 1개 —
            # 같은 종목/날 재분석 0(분석가 시간·비용 절약). 내러티브가 summary 를 인용.
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS trend_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    asof_date TEXT NOT NULL,
                    name TEXT,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    ),
                    UNIQUE(market, entity, asof_date)
                )
                """
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
                        release_date, asof_date, market, entity, entity_type, source,
                        raw_value, abnormal_value, leadingness, rank, metric,
                        snapshot_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r.release_date, r.asof_date, r.market, r.entity, r.entity_type,
                        r.source, r.raw_value, r.abnormal_value, r.leadingness, r.rank,
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

    def hot_candidates(
        self,
        market: str,
        *,
        asof_date: Optional[str] = None,
        limit: int = 10,
    ) -> list:
        """오늘(또는 지정) 등장한 개별종목 ticker 상위 — per-ticker 소스
        (google_trends/stocktwits_delta)의 universe 후보. abnormal 큰 순."""
        with self._conn() as c:
            if asof_date is None:
                row = c.execute(
                    "SELECT MAX(asof_date) AS d FROM trend_snapshots WHERE market=?",
                    (market,),
                ).fetchone()
                asof_date = row["d"] if row else None
            if asof_date is None:
                return []
            rows = c.execute(
                "SELECT entity, MAX(abnormal_value) AS a FROM trend_snapshots "
                "WHERE market=? AND asof_date=? AND entity_type='ticker' "
                "GROUP BY entity ORDER BY a IS NULL, a DESC LIMIT ?",
                (market, asof_date, limit),
            ).fetchall()
        return [r["entity"] for r in rows]

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

    # === 트렌드 필터 정책(Hermes 수정 ↔ cron 반영) ===

    def get_active_policy(self, market: str) -> Optional[dict]:
        """market 의 현재 활성 정책(없으면 None). digest·cron 이 필터값을 여기서 읽음."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM trend_policies WHERE market=? AND active=1 "
                "ORDER BY id DESC LIMIT 1",
                (market,),
            ).fetchone()
        return dict(row) if row else None

    def set_policy(
        self,
        market: str,
        *,
        min_sources: Optional[int] = None,
        rotation_peak_pct: Optional[float] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict:
        """필터 변경 → 이전 active 비활성화 + 새 active 행 추가(이력 보존).

        지정 안 한 필드는 이전 active 정책 값을 승계(부분 수정 지원).

        LLM(MCP) 경계 입력이라 검증: market 은 'us'|'kr', min_sources 는 1~10 정수.
        잘못된 값은 ValueError(쓰레기 정책이 active 불변식을 오염하지 않게).
        """
        if market not in ("us", "kr"):
            raise ValueError(f"market must be 'us' or 'kr', got {market!r}")
        if min_sources is not None:
            if not isinstance(min_sources, int) or isinstance(min_sources, bool) \
                    or not (1 <= min_sources <= 10):
                raise ValueError(f"min_sources must be int in 1..10, got {min_sources!r}")
        if rotation_peak_pct is not None:
            try:
                rotation_peak_pct = float(rotation_peak_pct)
            except (TypeError, ValueError):
                raise ValueError(f"rotation_peak_pct must be a number, got {rotation_peak_pct!r}")
            if rotation_peak_pct < 0:
                raise ValueError(f"rotation_peak_pct must be >= 0, got {rotation_peak_pct}")
        prev = self.get_active_policy(market) or {}
        merged = {
            "min_sources": min_sources if min_sources is not None else prev.get("min_sources"),
            "rotation_peak_pct": (
                rotation_peak_pct if rotation_peak_pct is not None
                else prev.get("rotation_peak_pct")
            ),
            "name": name if name is not None else prev.get("name"),
            "notes": notes if notes is not None else prev.get("notes"),
        }
        with self._conn() as c:
            c.execute(
                "UPDATE trend_policies SET active=0 WHERE market=? AND active=1",
                (market,),
            )
            cur = c.execute(
                "INSERT INTO trend_policies (market, name, min_sources, "
                "rotation_peak_pct, notes, active) VALUES (?, ?, ?, ?, ?, 1)",
                (market, merged["name"], merged["min_sources"],
                 merged["rotation_peak_pct"], merged["notes"]),
            )
            new_id = cur.lastrowid
            row = c.execute(
                "SELECT * FROM trend_policies WHERE id=?", (new_id,)
            ).fetchone()
        return dict(row)

    def list_policies(self, market: str, *, limit: int = 20) -> list:
        """market 의 정책 이력(최신순)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM trend_policies WHERE market=? ORDER BY id DESC LIMIT ?",
                (market, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # === 신규 진입 심층분석(C) 캐시 ===

    def get_analysis(self, market: str, entity: str, asof_date: str) -> Optional[dict]:
        """저장된 심층분석(없으면 None). 같은 종목/날 재분석 방지·내러티브 인용용."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM trend_analyses WHERE market=? AND entity=? AND asof_date=?",
                (market, entity, asof_date),
            ).fetchone()
        return dict(row) if row else None

    def save_analysis(
        self, market: str, entity: str, asof_date: str, summary: str,
        *, name: Optional[str] = None,
    ) -> None:
        """심층분석 요약 저장(같은 키 재저장은 갱신 — 멱등)."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO trend_analyses (market, entity, asof_date, name, summary) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(market, entity, asof_date) DO UPDATE SET "
                "  summary=excluded.summary, "
                "  name=COALESCE(excluded.name, trend_analyses.name)",  # None 이 기존 이름 미덮음
                (market, entity, asof_date, name, summary),
            )

    def latest_analysis(self, market: str, entity: str) -> Optional[dict]:
        """종목의 최신 심층분석(asof 무관). 갱신 필요(staleness) 판정·읽기용."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM trend_analyses WHERE market=? AND entity=? "
                "ORDER BY asof_date DESC, id DESC LIMIT 1",
                (market, entity),
            ).fetchone()
        return dict(row) if row else None

    def latest_analysis_asof(self, market: str) -> Optional[str]:
        """trend_analyses 의 최신 asof_date(스냅샷 날짜와 분리 — deep_dive 읽기용)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT MAX(asof_date) AS d FROM trend_analyses WHERE market=?",
                (market,),
            ).fetchone()
        return row["d"] if row and row["d"] else None

    def list_analyses(self, market: str, asof_date: str) -> list:
        """해당 날짜의 모든 심층분석(내러티브가 일괄 인용)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM trend_analyses WHERE market=? AND asof_date=? ORDER BY id",
                (market, asof_date),
            ).fetchall()
        return [dict(r) for r in rows]

    def entity_presence(self, market: str, entity: str, *, lookback_days: int = 7) -> dict:
        """종목의 와치리스트 지속성(시계열) — 최근 N일 중 며칠 등장 + 최초 등장일.

        ``{days_present, window_days, first_seen}``. '오늘 처음' vs 'N일째 지속' 판정용.
        """
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(DISTINCT asof_date) AS d, MIN(asof_date) AS first "
                "FROM trend_snapshots WHERE market=? AND entity=? "
                "AND asof_date >= date('now', '+9 hours', ?)",  # '+9h'=KST(asof_date 가 KST)
                (market, entity, f"-{int(lookback_days)} days"),
            ).fetchone()
            first_all = c.execute(
                "SELECT MIN(asof_date) AS f FROM trend_snapshots WHERE market=? AND entity=?",
                (market, entity),
            ).fetchone()
        return {
            "days_present": int(row["d"]) if row and row["d"] else 0,
            "window_days": int(lookback_days),
            "first_seen": (first_all["f"] if first_all else None),
        }

    def asof_dates(
        self, market: str, *, lt: Optional[str] = None,
        lookback_days: Optional[int] = None,
    ) -> list:
        """market 의 distinct asof_date(최신순). ``lt`` 미만·``lookback_days`` 이내 한정."""
        clauses = ["market=?"]
        params: list = [market]
        if lt is not None:
            clauses.append("asof_date < ?")
            params.append(lt)
            if lookback_days is not None:
                clauses.append("asof_date >= date(?, ?)")
                params += [lt, f"-{int(lookback_days)} days"]
        with self._conn() as c:
            rows = c.execute(
                f"SELECT DISTINCT asof_date FROM trend_snapshots "
                f"WHERE {' AND '.join(clauses)} ORDER BY asof_date DESC",
                params,
            ).fetchall()
        return [r["asof_date"] for r in rows]
