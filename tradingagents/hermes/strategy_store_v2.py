"""전략 저장 v2 — 별도 DB(``strategies_v2.db``), 완화 게이트 spec 영속화.

기존 ``StrategyStore`` 를 상속해 스키마/조회 로직을 재사용하되 (1) DB 파일을
분리(기존 100개의 strict-게이트 의미와 섞이지 않게), (2) 저장 시 검증을
``validate_spec_v2`` 로 교체한다. ``spec_hash`` 는 signal-agnostic(정규형 JSON
md5)이라 v2 spec 에도 그대로 사용 가능 — 중복 변이 멱등 차단 동일.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.hermes.strategy_store import StrategyStore


# v3 추가 컬럼 — 시간분할 검증 결과 + 엔진 버전(provenance).
# 기존 in_*/out_* 는 종목분할(xsec) 보조 지표로 유지, gate_passed 는 두 게이트
# (xsec AND time) 결합으로 의미 확장. 기존 행은 새 컬럼 NULL(=pre-v3).
_V3_COLUMNS = (
    ("time_in_sharpe", "REAL"),
    ("time_out_sharpe", "REAL"),
    ("time_gate_passed", "INTEGER"),
    ("engine_version", "TEXT"),
    ("time_result_json", "TEXT"),
)


class StrategyStoreV2(StrategyStore):
    """v2 전략 저장 — ``strategies_v2.db`` 분리, v2 검증 + v3 시간분할 컬럼."""

    def _db_path(self) -> Path:
        return self.root / "strategies_v2.db"

    def _init_schema(self) -> None:
        """기존 스키마 + v3 컬럼 자동 마이그레이션(멱등). ADD COLUMN 은 nullable 이라
        실행 중 구코드 저장과 호환된다."""
        super()._init_schema()
        with self._conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(strategies)")}
            for name, decl in _V3_COLUMNS:
                if name not in cols:
                    c.execute(f"ALTER TABLE strategies ADD COLUMN {name} {decl}")

    def save(
        self,
        spec: dict,
        result: dict,
        *,
        name: Optional[str] = None,
        time_result: Optional[dict] = None,
        engine_version: Optional[str] = None,
    ):
        """전략 + 백테스트 result 저장. 반환 ``(strategy_id, is_new)``.

        ``result`` 는 종목분할(xsec) 백테스트(``run_universe_backtest_v2``).
        ``time_result`` 가 주어지면(``run_time_split_validation``) 시간분할 IS/OOS
        를 함께 저장하고 ``gate_passed`` 컬럼을 **xsec AND time 결합**으로 기록한다.
        ``time_result`` 가 없으면 기존 동작(xsec 게이트만) — 하위호환.
        spec_hash 중복이면 재실행 없이 기존 id 와 ``is_new=False``.
        """
        validate_spec_v2(spec)
        h = spec_hash(spec)
        in_m, out_m = result["in_sample"], result["out_sample"]
        xsec_gate = bool(result["gate_passed"])
        if time_result is not None:
            time_gate = bool(time_result["gate_passed"])
            gate = xsec_gate and time_gate
            t_in = time_result["in_sample"]["sharpe"]
            t_out = time_result["out_sample"]["sharpe"]
            t_json = json.dumps(time_result, ensure_ascii=False)
        else:
            time_gate = gate = xsec_gate
            t_in = t_out = t_json = None
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
                    gate_passed, universe_size,
                    time_in_sharpe, time_out_sharpe, time_gate_passed,
                    engine_version, time_result_json
                ) VALUES (?,?,?,?, ?,?,?,?,?,?,?, ?,?,?,?,?,?,?, ?,?, ?,?,?,?,?)
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
                    1 if gate else 0,
                    result.get("universe_size"),
                    t_in, t_out, (1 if time_gate else 0) if time_result is not None else None,
                    engine_version, t_json,
                ),
            )
            return cur.lastrowid, True
