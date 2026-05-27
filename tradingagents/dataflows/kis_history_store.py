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

Endpoint = Literal["investor", "program", "short"]
Backend = Literal["parquet", "sqlite"]

_VALID_ENDPOINTS: tuple[Endpoint, ...] = ("investor", "program", "short")
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
        """endpoint별 테이블 생성. (ticker, date)가 복합 primary key."""
        col_defs = []
        for col in columns:
            if col == "date":
                col_defs.append("date TEXT NOT NULL")
            elif col.endswith("_ratio"):
                col_defs.append(f"{col} REAL")
            else:
                col_defs.append(f"{col} INTEGER")
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
