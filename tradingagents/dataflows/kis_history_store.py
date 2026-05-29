"""KIS 수급 데이터 영구 저장소 — Parquet + SQLite 동시 지원.

분석가의 응답 캐시(``~/.tradingagents/cache/kis/``)는 7일 윈도우 한 번 호출
결과를 저장하는 단기 cache. 이 모듈은 5년+ 장기 히스토리를 영구 보존.

저장 위치 (기본):
    ~/.tradingagents/kis_history/
        parquet/{ticker}/{endpoint}.parquet
        kis.db                 (SQLite, 단일 DB)
        _meta/{ticker}.json    (last_updated 등 메타)

API:
    >>> from tradingagents.dataflows.kis_history_store import KisHistoryStore
    >>> store = KisHistoryStore(backends=("parquet", "sqlite"))
    >>> store.write("005930.KS", "investor", rows)
    >>> df = store.read("005930.KS", "investor", backend="parquet")
    >>> last = store.last_date("005930.KS", "investor")  # 증분 업데이트용

증분 업데이트:
    store.last_date(...)가 반환한 날짜 다음날부터 어제까지 수집해
    store.write(...) 호출. write는 자동으로 기존 + 신규를 merge하고 중복
    제거 (date 기준).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import pandas as pd

from tradingagents.dataflows.korean_utils import to_naver_code

logger = logging.getLogger(__name__)

Endpoint = Literal["investor", "program", "short", "holdings"]
Backend = Literal["parquet", "sqlite"]

_VALID_ENDPOINTS: tuple[Endpoint, ...] = ("investor", "program", "short", "holdings")
_VALID_BACKENDS: tuple[Backend, ...] = ("parquet", "sqlite")


def _norm_ticker(ticker: str) -> str:
    """저장 키를 6자리 KIS 코드로 정규화.

    ``005930.KS`` / ``005930.KQ`` / ``005930`` 셋 다 ``005930`` 로 매핑된다.
    KIS store는 한국 종목 전용이라 비한국 ticker는 ``to_naver_code`` 가 ValueError 발생.
    """
    return to_naver_code(ticker)


def _default_history_dir() -> Path:
    base = os.environ.get("KIS_HISTORY_DIR") or os.path.join(
        os.path.expanduser("~"), ".tradingagents", "kis_history"
    )
    return Path(base)


class KisHistoryStore:
    """Parquet + SQLite 동시 저장 store.

    ``backends``로 활성화할 백엔드 선택. 둘 다 활성화하면 모든 write가 둘 다에
    저장되어 사용자가 워크플로에 맞는 쪽을 read 시 선택 가능.
    """

    def __init__(
        self,
        *,
        root: Optional[Path] = None,
        backends: Sequence[Backend] = ("parquet", "sqlite"),
    ):
        self.root = Path(root) if root else _default_history_dir()
        self.backends = tuple(b for b in backends if b in _VALID_BACKENDS)
        if not self.backends:
            raise ValueError(f"at least one backend required, got {backends}")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_meta").mkdir(exist_ok=True)
        if "parquet" in self.backends:
            (self.root / "parquet").mkdir(exist_ok=True)

    # === Paths ===

    def _parquet_path(self, ticker: str, endpoint: Endpoint) -> Path:
        # 내부 헬퍼는 호출자가 _norm_ticker 통과한 값만 넘긴다고 가정 (public method 진입점에서 정규화).
        return self.root / "parquet" / ticker / f"{endpoint}.parquet"

    def _sqlite_path(self) -> Path:
        return self.root / "kis.db"

    def _meta_path(self, ticker: str) -> Path:
        return self.root / "_meta" / f"{ticker}.json"

    # === SQLite helpers ===

    @contextmanager
    def _sqlite_conn(self):
        conn = sqlite3.connect(str(self._sqlite_path()))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_sqlite_table(self, conn: sqlite3.Connection, endpoint: Endpoint, columns: list[str]) -> None:
        """endpoint별 테이블 생성. (ticker, date)가 복합 primary key.

        기존 테이블에 누락된 컬럼이 있으면 ``ALTER TABLE ADD COLUMN`` 으로
        자동 마이그레이션 — OHLCV 5컬럼 같은 스키마 확장을 무중단으로 흡수.
        """
        def _col_type(col: str) -> str:
            if col == "date":
                return "TEXT NOT NULL"
            if col.endswith(("_ratio", "_pct")):
                return "REAL"
            return "INTEGER"

        col_defs = [f"{col} {_col_type(col)}" for col in columns]
        col_def_sql = ",\n    ".join(col_defs)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {endpoint} (
                ticker TEXT NOT NULL,
                {col_def_sql},
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{endpoint}_date ON {endpoint}(date)"
        )
        # 기존 테이블 누락 컬럼 마이그레이션
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({endpoint})").fetchall()}
        for col in columns:
            if col not in existing and col != "date":
                conn.execute(f"ALTER TABLE {endpoint} ADD COLUMN {col} {_col_type(col)}")

    # === Public API ===

    def write(self, ticker: str, endpoint: Endpoint, rows: Iterable[dict]) -> int:
        """``rows`` 를 두 백엔드에 저장. 기존 데이터와 merge, date 기준 중복 제거.

        ``ticker`` 는 ``005930.KS`` / ``005930.KQ`` / ``005930`` 셋 다 받아 6자리로 정규화.

        Returns: write 된 행 수 (merge 후 ticker 전체 행 수).
        """
        if endpoint not in _VALID_ENDPOINTS:
            raise ValueError(f"unknown endpoint: {endpoint}")
        ticker = _norm_ticker(ticker)
        rows = list(rows)
        if not rows:
            return 0
        new_df = pd.DataFrame(rows)
        if "date" not in new_df.columns:
            raise ValueError(f"rows must have 'date' column, got {list(new_df.columns)}")

        existing = self._read_parquet_if_exists(ticker, endpoint)
        if existing is not None:
            merged = pd.concat([existing, new_df], ignore_index=True)
        else:
            merged = new_df
        merged = (
            merged.drop_duplicates(subset=["date"], keep="last")
                  .sort_values("date")
                  .reset_index(drop=True)
        )

        if "parquet" in self.backends:
            self._write_parquet(ticker, endpoint, merged)
        if "sqlite" in self.backends:
            self._write_sqlite(ticker, endpoint, merged)
        self._update_meta(ticker, endpoint, merged)
        # raw investor 갱신 시 derived holdings도 자동 재계산.
        # endpoint=="holdings" 일 때는 재귀 호출 안 됨 (raw → derived 한 방향).
        if endpoint == "investor":
            self._materialize_holdings(ticker, merged)
        return len(merged)

    def read(
        self, ticker: str, endpoint: Endpoint,
        *,
        backend: Backend = "parquet",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """date 오름차순 정렬된 DataFrame 반환. 데이터 없으면 빈 DataFrame.

        ``backend`` 로 어느 백엔드에서 읽을지 선택. 기본 parquet (pandas
        친화적·압축률). ``ticker`` 는 ``write`` 와 동일 규칙으로 정규화.
        """
        if endpoint not in _VALID_ENDPOINTS:
            raise ValueError(f"unknown endpoint: {endpoint}")
        ticker = _norm_ticker(ticker)
        if backend == "parquet":
            df = self._read_parquet_if_exists(ticker, endpoint)
            if df is None:
                return pd.DataFrame()
        elif backend == "sqlite":
            df = self._read_sqlite(ticker, endpoint)
        else:
            raise ValueError(f"unknown backend: {backend}")

        if start_date is not None:
            df = df[df["date"] >= start_date]
        if end_date is not None:
            df = df[df["date"] <= end_date]
        return df.reset_index(drop=True)

    def last_date(self, ticker: str, endpoint: Endpoint) -> Optional[str]:
        """가장 최근 데이터 날짜 (YYYY-MM-DD) — 증분 업데이트 진입점.

        없으면 None. ``ticker`` 는 ``write`` 와 동일 규칙으로 정규화.
        """
        meta = self._read_meta(_norm_ticker(ticker))
        return (meta.get(endpoint) or {}).get("last_date")

    def list_tickers(self) -> list[str]:
        """저장된 종목 리스트 (parquet 기반). 모든 키는 6자리 정규형으로 반환."""
        parquet_root = self.root / "parquet"
        if not parquet_root.exists():
            return []
        keys: set[str] = set()
        for p in parquet_root.iterdir():
            if not p.is_dir():
                continue
            try:
                keys.add(_norm_ticker(p.name))
            except ValueError:
                # 정규형이 아닌 잔재 디렉토리는 무시 (마이그레이션 대상).
                continue
        return sorted(keys)

    # === backend implementations ===

    def _read_parquet_if_exists(self, ticker: str, endpoint: Endpoint) -> Optional[pd.DataFrame]:
        path = self._parquet_path(ticker, endpoint)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("parquet read failed (%s): %s", path, exc)
            return None

    def _write_parquet(self, ticker: str, endpoint: Endpoint, df: pd.DataFrame) -> None:
        path = self._parquet_path(ticker, endpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")

    def _write_sqlite(self, ticker: str, endpoint: Endpoint, df: pd.DataFrame) -> None:
        cols = [c for c in df.columns if c != "ticker"]
        with self._sqlite_conn() as conn:
            self._ensure_sqlite_table(conn, endpoint, cols)
            # Upsert: 기존 (ticker, date) 행 삭제 후 일괄 insert
            placeholders = ", ".join(["?"] * (len(cols) + 1))  # +ticker
            col_list = ", ".join(["ticker"] + cols)
            # 한 ticker만 다루므로 그 ticker의 행 모두 삭제 후 재삽입.
            conn.execute(f"DELETE FROM {endpoint} WHERE ticker = ?", (ticker,))
            rows_to_insert = [
                (ticker, *[row[c] if c in row.index else None for c in cols])
                for _, row in df.iterrows()
            ]
            conn.executemany(
                f"INSERT INTO {endpoint} ({col_list}) VALUES ({placeholders})",
                rows_to_insert,
            )

    def _read_sqlite(self, ticker: str, endpoint: Endpoint) -> pd.DataFrame:
        path = self._sqlite_path()
        if not path.exists():
            return pd.DataFrame()
        with self._sqlite_conn() as conn:
            try:
                df = pd.read_sql(
                    f"SELECT * FROM {endpoint} WHERE ticker = ? ORDER BY date",
                    conn, params=(ticker,),
                )
            except pd.errors.DatabaseError:
                return pd.DataFrame()
        if "ticker" in df.columns:
            df = df.drop(columns=["ticker"])
        return df

    # === Meta ===

    def _read_meta(self, ticker: str) -> dict:
        path = self._meta_path(ticker)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("meta read failed (%s): %s", path, exc)
            return {}

    def _update_meta(self, ticker: str, endpoint: Endpoint, df: pd.DataFrame) -> None:
        meta = self._read_meta(ticker)
        if df.empty:
            return
        meta[endpoint] = {
            "last_date": str(df["date"].max()),
            "first_date": str(df["date"].min()),
            "n_rows": int(len(df)),
        }
        path = self._meta_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # === Holdings derived (raw investor → cumsum/pct/price_change) ===

    def _materialize_holdings(self, ticker: str, investor_df: pd.DataFrame) -> None:
        """raw investor → holdings derived. ``write(endpoint='investor')`` 끝에 자동 호출.

        ticker 는 정규화된 6자리. 한 ticker 분량 investor 데이터를 받아
        ``compute_holdings`` 결과를 다시 ``write(endpoint='holdings')`` 로 영구화.
        """
        from tradingagents.dataflows.kis_holdings import compute_holdings
        derived = compute_holdings(investor_df)
        if derived.empty:
            return
        # write() 재진입 — endpoint=='holdings' 라 hook 재귀 없음.
        self.write(ticker, "holdings", derived.to_dict("records"))

    def materialize_holdings(self, ticker: str) -> int:
        """기존 raw investor 데이터에서 holdings 재계산 (CLI 일괄 마이그레이션용).

        Returns: holdings 테이블에 저장된 행 수 (0이면 raw 없음).
        """
        norm = _norm_ticker(ticker)
        raw = self.read(norm, "investor")
        if raw.empty:
            return 0
        self._materialize_holdings(norm, raw)
        return len(raw)

    # === Tickers meta (종목명·시장) ===

    def _ensure_tickers_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                market TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

    def set_ticker_metadata(
        self, ticker: str, company_name: str, market: str
    ) -> None:
        """tickers 메타 upsert. ``market`` 은 'KOSPI' 또는 'KOSDAQ'."""
        if "sqlite" not in self.backends:
            return
        from datetime import datetime
        norm = _norm_ticker(ticker)
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._sqlite_conn() as conn:
            self._ensure_tickers_table(conn)
            conn.execute(
                "INSERT INTO tickers (ticker, company_name, market, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET "
                "  company_name=excluded.company_name, "
                "  market=excluded.market, "
                "  updated_at=excluded.updated_at",
                (norm, company_name, market, now),
            )

    def get_ticker_metadata(self, ticker: str) -> Optional[dict]:
        """단일 ticker의 메타 dict 반환. 없으면 None."""
        if "sqlite" not in self.backends:
            return None
        norm = _norm_ticker(ticker)
        if not self._sqlite_path().exists():
            return None
        with self._sqlite_conn() as conn:
            try:
                row = conn.execute(
                    "SELECT ticker, company_name, market, updated_at "
                    "FROM tickers WHERE ticker = ?",
                    (norm,),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        if row is None:
            return None
        return {
            "ticker": row[0], "company_name": row[1],
            "market": row[2], "updated_at": row[3],
        }

    # === Analysis reports (raw·derived 외 분석 산출물) ===

    def _ensure_analysis_reports_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_reports (
                ticker TEXT NOT NULL,
                kind TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                payload TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (ticker, kind, as_of_date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_reports_kind "
            "ON analysis_reports(kind, generated_at)"
        )

    def write_analysis_report(
        self, ticker: str, kind: str, as_of_date: str, payload: dict,
    ) -> None:
        """분석 산출물 upsert. ``payload`` 는 JSON 직렬화 가능한 dict.

        ``kind`` 예시: ``"correlation"``. ``as_of_date`` 는 보통 분석 기간 끝
        (YYYY-MM-DD) — 같은 종목·종류·종료일이면 덮어쓰기.
        """
        if "sqlite" not in self.backends:
            return
        from datetime import datetime
        norm = _norm_ticker(ticker)
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        with self._sqlite_conn() as conn:
            self._ensure_analysis_reports_table(conn)
            conn.execute(
                "INSERT INTO analysis_reports (ticker, kind, as_of_date, "
                "  payload, generated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker, kind, as_of_date) DO UPDATE SET "
                "  payload=excluded.payload, "
                "  generated_at=excluded.generated_at",
                (norm, kind, as_of_date, payload_json, now),
            )

    def read_analysis_report(
        self, ticker: str, kind: str, as_of_date: Optional[str] = None,
    ) -> Optional[dict]:
        """단일 보고서 조회. ``as_of_date`` 미지정 시 가장 최근."""
        if "sqlite" not in self.backends:
            return None
        norm = _norm_ticker(ticker)
        if not self._sqlite_path().exists():
            return None
        with self._sqlite_conn() as conn:
            try:
                self._ensure_analysis_reports_table(conn)
                if as_of_date is None:
                    row = conn.execute(
                        "SELECT ticker, kind, as_of_date, payload, generated_at "
                        "FROM analysis_reports WHERE ticker = ? AND kind = ? "
                        "ORDER BY as_of_date DESC LIMIT 1",
                        (norm, kind),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT ticker, kind, as_of_date, payload, generated_at "
                        "FROM analysis_reports WHERE ticker = ? AND kind = ? "
                        "AND as_of_date = ?",
                        (norm, kind, as_of_date),
                    ).fetchone()
            except sqlite3.OperationalError:
                return None
        if row is None:
            return None
        return {
            "ticker": row[0], "kind": row[1], "as_of_date": row[2],
            "payload": json.loads(row[3]),
            "generated_at": row[4],
        }

    def list_analysis_reports(self, kind: Optional[str] = None) -> list[dict]:
        """저장된 보고서 목록 (ticker, kind, as_of_date, generated_at만)."""
        if "sqlite" not in self.backends or not self._sqlite_path().exists():
            return []
        sql = ("SELECT ticker, kind, as_of_date, generated_at FROM "
               "analysis_reports")
        params: tuple = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            params = (kind,)
        sql += " ORDER BY generated_at DESC"
        with self._sqlite_conn() as conn:
            try:
                self._ensure_analysis_reports_table(conn)
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
        return [
            {"ticker": r[0], "kind": r[1], "as_of_date": r[2],
             "generated_at": r[3]}
            for r in rows
        ]
