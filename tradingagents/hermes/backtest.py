"""백테스트 모드 — env-gated point-in-time 강제.

``HERMES_BACKTEST_AS_OF=YYYY-MM-DD`` 가 설정되면 MCP 도구가 그 시점 이하
데이터만 보도록 강제하고, 가설 저장소를 별도 root 로 격리한다. 과거 시점으로
실제 Hermes 파이프라인을 돌려 미래를 기다리지 않고 즉시 라벨링하기 위함.

env 미설정 시 모든 함수가 no-op / passthrough — 운영 동작 무영향.

강제 항목 (env 설정 시):
  - 분석가 ``date`` → AS_OF (clamp_date)
  - 통계 ``end_date`` → AS_OF 이하 (clamp_end_date)
  - ``analyst_supply_demand`` 는 라이브 KIS 대신 kis.db 슬라이스
    (install_kis_history_patch — 라이브는 과거 ~30일만 서빙)
  - 가설 저장소 root → ``HERMES_BACKTEST_STORE_ROOT``
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

# 백테스트 상태 파일 — 드라이버가 쓰고 MCP 서버가 읽는다. env 가 아니라 파일을
# 쓰는 이유: Hermes 의 ``_build_safe_env`` 가 MCP 서브프로세스로 전달되는
# 환경변수를 화이트리스트로 필터링해 ``HERMES_BACKTEST_*`` 가 차단되기 때문.
# 파일은 MCP 서버 프로세스가 spawn 방식과 무관하게 항상 읽을 수 있다.
_STATE_FILE = Path.home() / ".tradingagents" / "hermes" / "backtest_mode.json"


def _state() -> dict:
    """현재 백테스트 상태 — env 우선(단위 테스트용), 없으면 상태 파일."""
    env_as_of = os.environ.get("HERMES_BACKTEST_AS_OF", "").strip()
    if env_as_of:
        return {
            "as_of": env_as_of,
            "store_root": os.environ.get("HERMES_BACKTEST_STORE_ROOT", "").strip() or None,
            "analyst_model": os.environ.get("HERMES_BACKTEST_ANALYST_MODEL", "").strip() or None,
        }
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def write_state(*, as_of: str, store_root: str, analyst_model: Optional[str] = None) -> None:
    """드라이버가 한 셀 시작 전에 상태 파일을 쓴다."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(
        json.dumps({"as_of": as_of, "store_root": store_root, "analyst_model": analyst_model}),
        encoding="utf-8",
    )


def clear_state() -> None:
    """배치 종료 후 상태 파일 제거 — 운영 모드 복귀."""
    _STATE_FILE.unlink(missing_ok=True)


def as_of() -> Optional[str]:
    """백테스트 기준일 "YYYY-MM-DD" 또는 None (운영 모드)."""
    return _state().get("as_of") or None


def is_active() -> bool:
    return as_of() is not None


def store_root() -> Optional[Path]:
    """가설 저장소 격리 root. None 이면 HypothesisStore 디폴트 경로."""
    v = _state().get("store_root")
    return Path(v) if v else None


def analyst_model() -> Optional[str]:
    """백테스트 분석가 LLM override (느린 26b 대신 경량). None=기본값."""
    return _state().get("analyst_model") or None


def clamp_date(date: str) -> str:
    """분석가 기준일 — 백테스트 시 AS_OF 로 강제."""
    return as_of() or date


def clamp_end_date(end_date: Optional[str]) -> Optional[str]:
    """통계 end_date — 백테스트 시 AS_OF 이하로 강제 (미래 데이터 차단)."""
    ao = as_of()
    if ao is None:
        return end_date
    return ao if end_date is None else min(end_date, ao)


# === kis.db point-in-time 리더 (analyst_supply_demand 용 patch) ===

_KIS_DB = Path.home() / ".tradingagents" / "kis_history" / "kis.db"

_INVESTOR_COLS = [
    "date", "close",
    "foreign_amount", "foreign_registered_amount", "foreign_unregistered_amount",
    "institution_amount", "pension_amount", "private_equity_amount",
    "investment_trust_amount", "securities_amount", "bank_amount",
    "insurance_amount", "retail_amount", "other_corp_amount",
]


def _read_window(table: str, cols: list[str], code6: str, end_date: str, n: int) -> list[dict]:
    """``end_date`` 이하 최신 n 거래일 (newest first) 을 dict 리스트로."""
    conn = sqlite3.connect(f"file:{_KIS_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM {table} "
            f"WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT ?",
            (code6, end_date, n),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def install_kis_history_patch() -> None:
    """``analyst_supply_demand`` 의 kis_api fetcher 를 kis.db 리더로 교체.

    *무조건* 호출해도 안전 (idempotent): 패치된 fetcher 는 호출 시점에
    ``is_active()`` 를 보고, 백테스트 모드일 때만 kis.db 슬라이스를 반환하고
    아니면 원래 라이브 함수에 위임한다. 이렇게 해야 MCP 서버가 상태 파일보다
    먼저 떠도(=import 시 inactive) 이후 활성화를 호출 시점에 반영한다.

    라이브 KIS 는 과거 ~30일만 서빙 → 과거 시점은 5y kis.db 에서 슬라이스.
    분석가 코드는 수정하지 않음. 호출 키(date/close/*_amount 등)는
    supply_demand_analyst 의 formatter 기대와 동일.
    """
    from tradingagents.dataflows import kis_api

    if getattr(kis_api, "_backtest_patched", False):
        return  # 이중 패치 방지 (원본 함수 참조 보존)
    orig_investor = kis_api.fetch_investor_trend
    orig_program = kis_api.fetch_program_trading
    orig_short = kis_api.fetch_short_interest

    def fetch_investor_trend(code6, end_date, lookback_days=7):
        if not is_active():
            return orig_investor(code6, end_date, lookback_days)
        return _read_window(
            "investor", _INVESTOR_COLS, code6, clamp_end_date(end_date), lookback_days
        )

    def fetch_program_trading(code6, end_date, lookback_days=7):
        if not is_active():
            return orig_program(code6, end_date, lookback_days)
        return _read_window(
            "program", ["date", "close", "net_qty", "net_amount"],
            code6, clamp_end_date(end_date), lookback_days,
        )

    def fetch_short_interest(code6, start_date, end_date, lookback_days=7):
        if not is_active():
            return orig_short(code6, start_date, end_date, lookback_days)
        return _read_window(
            "short",
            ["date", "close", "short_qty", "short_volume_ratio",
             "short_amount", "short_amount_ratio"],
            code6, clamp_end_date(end_date), lookback_days,
        )

    kis_api.fetch_investor_trend = fetch_investor_trend
    kis_api.fetch_program_trading = fetch_program_trading
    kis_api.fetch_short_interest = fetch_short_interest
    kis_api._backtest_patched = True
