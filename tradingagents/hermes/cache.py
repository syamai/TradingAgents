"""분석가 출력 영구 캐시 — SQLite 백엔드.

PRD 모듈 #2 (AnalystOutputCache). 복합 키 (ticker, date, analyst_name,
model_id, prompt_version) 로 같은 입력에 대한 분석가 마크다운을 영구 저장.

재현성·진화 추적의 핵심:
  - **재현성**: 같은 (ticker, date, model, prompt) → 항상 같은 출력
  - **진화 추적**: model_id 업그레이드 (gemma4 → gemma5) 시 옛 row 보존,
    새 모델은 새 row → 라벨 통계로 모델 진화 품질 비교 가능
  - **비용 절감**: 캐시 hit 시 LLM 호출 0

비위치 디폴트 DB 경로: ``~/.tradingagents/hermes/cache.db`` —
``KisHistoryStore`` (``~/.tradingagents/kis_history``) 와 동일 root 의
``hermes`` 서브디렉터리. 테스트는 ``root=tmp_path`` 로 격리.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_TABLE = "analyst_outputs"


def _default_cache_dir() -> Path:
    return Path.home() / ".tradingagents" / "hermes"


class AnalystOutputCache:
    """(ticker, date, analyst, model, prompt) → 분석가 마크다운 영구 저장.

    ticker 는 정규화하지 않음 — AnalystRunner 가 받는 형식 (e.g. "005930.KS")
    그대로 키로 사용. 호출자가 일관된 형식을 쓰면 충돌 없음.
    """

    def __init__(self, *, root: Optional[Path] = None):
        self.root = Path(root) if root else _default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _db_path(self) -> Path:
        return self.root / "cache.db"

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
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    analyst_name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    report TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    PRIMARY KEY (ticker, date, analyst_name, model_id, prompt_version)
                )
            """)
            c.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_TABLE}_ticker_analyst "
                f"ON {_TABLE}(ticker, analyst_name)"
            )

    # === Public API ===

    def set(
        self,
        ticker: str,
        date: str,
        analyst_name: str,
        model_id: str,
        prompt_version: str,
        report: str,
    ) -> None:
        """같은 복합 키 row 가 있으면 덮어쓰기.

        같은 (ticker, date, analyst, model, prompt) 입력이면 같은 출력이
        나와야 결정론. 덮어쓰기는 수동 invalidate 가 아닌 *재실행* 으로 정정
        하고 싶을 때 (예: 분석가 로직 버그 수정) 사용.
        """
        with self._conn() as c:
            c.execute(
                f"""
                INSERT INTO {_TABLE}
                    (ticker, date, analyst_name, model_id, prompt_version, report)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, date, analyst_name, model_id, prompt_version)
                DO UPDATE SET report=excluded.report,
                              created_at=strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                """,
                (ticker, date, analyst_name, model_id, prompt_version, report),
            )

    def get(
        self,
        ticker: str,
        date: str,
        analyst_name: str,
        model_id: str,
        prompt_version: str,
    ) -> Optional[str]:
        """캐시 hit 면 report 문자열, miss 면 None."""
        with self._conn() as c:
            row = c.execute(
                f"""
                SELECT report FROM {_TABLE}
                WHERE ticker=? AND date=? AND analyst_name=?
                  AND model_id=? AND prompt_version=?
                """,
                (ticker, date, analyst_name, model_id, prompt_version),
            ).fetchone()
        return row["report"] if row else None

    def list(
        self,
        *,
        ticker: Optional[str] = None,
        analyst_name: Optional[str] = None,
    ) -> list[dict]:
        """필터 조건 만족하는 row 의 *메타 정보만* 반환 (report 제외).

        report 는 클 수 있어 list 결과에 포함하지 않음 (메모리·전송 비용).
        실제 report 는 ``get`` 으로 개별 fetch.
        """
        clauses, params = [], []
        if ticker is not None:
            clauses.append("ticker=?")
            params.append(ticker)
        if analyst_name is not None:
            clauses.append("analyst_name=?")
            params.append(analyst_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT ticker, date, analyst_name, model_id, prompt_version,
                       created_at
                FROM {_TABLE}
                {where}
                ORDER BY ticker, date, analyst_name
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def invalidate(
        self,
        *,
        ticker: Optional[str] = None,
        date: Optional[str] = None,
        analyst_name: Optional[str] = None,
        model_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> int:
        """주어진 필터 조합으로 row 삭제. 반환=삭제된 행 수.

        모든 필터가 None 이면 전체 삭제 — 안전 조치로 명시적 ``all=True``
        패턴은 두지 않음. 호출자가 의도 분명히 표현해야.
        """
        if not any(
            v is not None for v in
            (ticker, date, analyst_name, model_id, prompt_version)
        ):
            # 안전: 빈 필터로 전체 삭제는 금지.
            raise ValueError(
                "at least one filter required (or call delete_all if intended)"
            )
        clauses, params = [], []
        for col, val in (
            ("ticker", ticker), ("date", date), ("analyst_name", analyst_name),
            ("model_id", model_id), ("prompt_version", prompt_version),
        ):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        with self._conn() as c:
            cur = c.execute(
                f"DELETE FROM {_TABLE} WHERE {' AND '.join(clauses)}",
                params,
            )
            return cur.rowcount
