"""검증된 전략 영구 저장 — SQLite 백엔드.

``HypothesisStore`` 와 동형. 백테스트한 전략 spec(JSON) + in/out-sample 메트릭 +
게이트 통과 여부를 1 row 로 저장한다. ``spec_hash`` UNIQUE 로 동일 로직의 전략을
중복 저장하지 않는다(자율 연구 루프가 같은 변이를 재시도해도 멱등) — 호출자
신뢰 불가하므로 중복 방지를 store 가 책임진다.

게이트 미통과 전략도 ``gate_passed=0`` 으로 저장(연구 로그 + 시도 카운트 +
재시도 차단). ``list_strategies`` 가 시도/통과 카운트의 single source.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from tradingagents.hermes.strategy_spec import spec_hash, validate_spec


def _default_store_dir() -> Path:
    return Path.home() / ".tradingagents" / "hermes"


class StrategyStore:
    """전략 spec + 백테스트 메트릭 영구 저장."""

    def __init__(self, *, root: Optional[Path] = None):
        self.root = Path(root) if root else _default_store_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _db_path(self) -> Path:
        return self.root / "strategies.db"

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
            c.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_hash TEXT NOT NULL UNIQUE,
                    direction TEXT NOT NULL,
                    in_win_rate REAL, in_sharpe REAL, in_mdd REAL,
                    in_cum_return_pct REAL, in_avg_hold_days REAL,
                    in_n_trades INTEGER, in_excess_return_pct REAL,
                    out_win_rate REAL, out_sharpe REAL, out_mdd REAL,
                    out_cum_return_pct REAL, out_avg_hold_days REAL,
                    out_n_trades INTEGER, out_excess_return_pct REAL,
                    gate_passed INTEGER NOT NULL,
                    universe_size INTEGER,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    )
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_strat_gate "
                "ON strategies(gate_passed)"
            )

    # === Public API ===

    def exists(self, hash_: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM strategies WHERE spec_hash=?", (hash_,)
            ).fetchone()
        return row is not None

    def save(self, spec: dict, result: dict, *, name: Optional[str] = None):
        """전략 + 백테스트 result 저장. 반환 ``(strategy_id, is_new)``.

        ``result`` 는 ``run_universe_backtest`` 반환 dict
        (``in_sample`` / ``out_sample`` / ``gate_passed`` / ``universe_size``).
        spec_hash 중복이면 기존 id 와 ``is_new=False`` 반환(재백테스트 안 함).
        """
        validate_spec(spec)
        h = spec_hash(spec)
        in_m, out_m = result["in_sample"], result["out_sample"]
        with self._conn() as c:
            existing = c.execute(
                "SELECT id FROM strategies WHERE spec_hash=?", (h,)
            ).fetchone()
            if existing:
                return existing["id"], False
            cur = c.execute(
                """
                INSERT INTO strategies (
                    name, spec_json, spec_hash, direction,
                    in_win_rate, in_sharpe, in_mdd, in_cum_return_pct,
                    in_avg_hold_days, in_n_trades, in_excess_return_pct,
                    out_win_rate, out_sharpe, out_mdd, out_cum_return_pct,
                    out_avg_hold_days, out_n_trades, out_excess_return_pct,
                    gate_passed, universe_size
                ) VALUES (?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?)
                """,
                (
                    name or spec.get("name", "unnamed"),
                    json.dumps(spec, ensure_ascii=False),
                    h,
                    spec["direction"],
                    in_m["win_rate"], in_m["sharpe"], in_m["mdd_pct"],
                    in_m["cum_return_pct"], in_m["avg_hold_days"],
                    in_m["n_trades"], in_m["avg_excess_ret_pct"],
                    out_m["win_rate"], out_m["sharpe"], out_m["mdd_pct"],
                    out_m["cum_return_pct"], out_m["avg_hold_days"],
                    out_m["n_trades"], out_m["avg_excess_ret_pct"],
                    1 if result["gate_passed"] else 0,
                    result.get("universe_size"),
                ),
            )
            return cur.lastrowid, True

    def get(self, strategy_id: int) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM strategies WHERE id=?", (strategy_id,)
            ).fetchone()
        return _row_to_strategy(row) if row else None

    def get_by_hash(self, hash_: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM strategies WHERE spec_hash=?", (hash_,)
            ).fetchone()
        return _row_to_strategy(row) if row else None

    def list(self, *, gate_passed: Optional[bool] = None) -> list[dict]:
        """전략 메타 리스트. ``gate_passed`` 필터 (None=전체)."""
        clause, params = "", []
        if gate_passed is not None:
            clause = "WHERE gate_passed=?"
            params.append(1 if gate_passed else 0)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM strategies {clause} ORDER BY id", params
            ).fetchall()
        return [_row_to_strategy(r) for r in rows]


def _row_to_strategy(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["spec"] = json.loads(d["spec_json"])
    d["gate_passed"] = bool(d["gate_passed"])
    return d
