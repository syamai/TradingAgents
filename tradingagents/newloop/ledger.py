"""사전등록·재응시 원장 — 골대 이동과 시험지 마모를 막는 장치 (append-only).

규칙 (2026-06-11 확정):
  - family(가설 계열)는 채점 **전에** 가설 한 줄과 함께 등록한다.
  - family 당 제출(submission) 은 MAX_SUBMISSIONS 회 — 떨어진 걸 보고 고쳐서
    재응시를 반복하면 홀드아웃 시험지도 유출되기 때문.
  - 제출 1회에 변형(spec) 은 VARIANTS_MAX 개까지.
  - 결과 기록은 append-only: 수정·삭제 함수를 제공하지 않는다.

총 N 상한 (2026-06-12 추가):
  - "느리게 도는 것"은 다중검정 수학을 바꾸지 않는다(거짓통과 확률에 시간 변수
    없음). family당 한도·고갈정지는 *약한* 상한 — 운으로 통과하면 리셋된다.
  - 깨끗한 천장은 홀드아웃 1장당 **누적 채점 spec 수의 절대 상한**에서만 나온다.
    HOLDOUT_TRIAL_BUDGET 도달 시 어떤 family도 제출 불가 → 새 홀드아웃 수집 필요.
  - 이 누적 N 이 게이트의 동적 합격선(다중검정 보정)을 결정한다(stage1_gate).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

LEDGER_DIR = os.path.expanduser("~/.tradingagents/newloop")
LEDGER_DB = os.path.join(LEDGER_DIR, "ledger.db")

MAX_SUBMISSIONS = 2   # family 당 제출 한도 (최초 1 + 수정 재응시 1)
VARIANTS_MAX = 24     # 제출 1회당 spec 변형 상한
HOLDOUT_TRIAL_BUDGET = 400  # 홀드아웃 1장당 누적 채점 spec 절대 상한 (다중검정 천장)
# 근거: t≥3 동적 보정 하에서 N=400 합격선 t_crit≈3.66 — 진짜 우위는 여전히 통과
# 가능하되, 운 통과 누적은 봉쇄. 1/평일·~12spec 기준 ≈7주 수명. 도달 시 새 홀드아웃.


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or LEDGER_DB
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS families (
            family_key TEXT PRIMARY KEY,
            hypothesis TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            max_submissions INTEGER NOT NULL,
            submissions INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active'   -- active | exhausted | passed
        );
        CREATE TABLE IF NOT EXISTS gate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_key TEXT NOT NULL,
            submission INTEGER NOT NULL,
            strategy_ref TEXT NOT NULL,             -- 전략 식별(예: db id, spec 이름)
            gate_version TEXT NOT NULL,
            status TEXT NOT NULL,                   -- pass | fail | insufficient
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    return con


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def cumulative_trials(*, db_path: str | None = None) -> int:
    """홀드아웃에 채점된 누적 spec 수(= gate_results 전체 행). 다중검정 N."""
    con = _connect(db_path)
    return con.execute("SELECT COUNT(*) FROM gate_results").fetchone()[0]


def budget_state(*, db_path: str | None = None) -> dict:
    used = cumulative_trials(db_path=db_path)
    return {"used": used, "budget": HOLDOUT_TRIAL_BUDGET,
            "remaining": max(0, HOLDOUT_TRIAL_BUDGET - used)}


def register_family(family_key: str, hypothesis: str, *, db_path: str | None = None) -> None:
    """채점 전 사전등록. 이미 있으면 무시(가설 변경 불가 — write-once)."""
    con = _connect(db_path)
    with con:
        con.execute(
            "INSERT OR IGNORE INTO families (family_key, hypothesis, registered_at, max_submissions) "
            "VALUES (?, ?, ?, ?)",
            (family_key, hypothesis, _now(), MAX_SUBMISSIONS),
        )


def can_submit(family_key: str, *, db_path: str | None = None) -> tuple[bool, str]:
    con = _connect(db_path)
    row = con.execute(
        "SELECT submissions, max_submissions, status FROM families WHERE family_key=?",
        (family_key,),
    ).fetchone()
    if row is None:
        return False, "미등록 family — 가설과 함께 사전등록부터"
    used, mx, status = row
    if status == "passed":
        return False, "이미 통과한 family — Stage2(forward)로"
    if status == "exhausted" or used >= mx:
        return False, f"재응시 한도 소진({used}/{mx}) — 시험지 마모 방지"
    total = con.execute("SELECT COUNT(*) FROM gate_results").fetchone()[0]
    if total >= HOLDOUT_TRIAL_BUDGET:
        return False, f"홀드아웃 시험지 예산 소진({total}/{HOLDOUT_TRIAL_BUDGET}) — 새 홀드아웃 수집 필요"
    return True, "ok"


def record_submission(family_key: str, results: list[dict], *, db_path: str | None = None) -> int:
    """제출 1회 기록. results = [{strategy_ref, gate_version, status, result}, ...].

    반환: 제출 번호. 한도 초과·변형 수 초과·빈 제출은 ValueError.

    한도 강제는 조건부 UPDATE 의 rowcount 로 원자적으로 한다 — can_submit 후
    기록하는 2단계는 동시 제출에서 한도를 뚫린다(8프로세스 실측: 한도 2에
    4건 기록·제출번호 중복). BEGIN IMMEDIATE 로 쓰기 락을 선점한다.
    """
    if not results:
        raise ValueError("빈 제출 — 채점 결과 없이 슬롯을 소모할 수 없다")
    if len(results) > VARIANTS_MAX:
        raise ValueError(f"변형 {len(results)}개 > 제출당 상한 {VARIANTS_MAX}")
    con = _connect(db_path)
    try:
        con.execute("BEGIN IMMEDIATE")
        total = con.execute("SELECT COUNT(*) FROM gate_results").fetchone()[0]
        if total + len(results) > HOLDOUT_TRIAL_BUDGET:
            con.rollback()
            raise ValueError(
                f"홀드아웃 시험지 예산 초과: 누적 {total}+{len(results)} > "
                f"{HOLDOUT_TRIAL_BUDGET} — 새 홀드아웃 수집 필요"
            )
        cur = con.execute(
            "UPDATE families SET submissions = submissions + 1 "
            "WHERE family_key=? AND status='active' AND submissions < max_submissions",
            (family_key,),
        )
        if cur.rowcount == 0:
            con.rollback()
            ok, why = can_submit(family_key, db_path=db_path)
            raise ValueError(why if not ok else "제출 슬롯 점유 실패")
        sub, mx = con.execute(
            "SELECT submissions, max_submissions FROM families WHERE family_key=?",
            (family_key,),
        ).fetchone()
        for r in results:
            con.execute(
                "INSERT INTO gate_results (family_key, submission, strategy_ref, gate_version, "
                "status, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (family_key, sub, str(r["strategy_ref"]), r["gate_version"],
                 r["status"], json.dumps(r["result"], ensure_ascii=False), _now()),
            )
        passed = any(r["status"] == "pass" for r in results)
        new_status = "passed" if passed else ("exhausted" if sub >= mx else "active")
        con.execute("UPDATE families SET status=? WHERE family_key=?", (new_status, family_key))
        con.commit()
    except BaseException:
        con.rollback()
        raise
    return sub


def family_state(family_key: str, *, db_path: str | None = None) -> dict | None:
    con = _connect(db_path)
    row = con.execute(
        "SELECT family_key, hypothesis, registered_at, submissions, max_submissions, status "
        "FROM families WHERE family_key=?", (family_key,),
    ).fetchone()
    if row is None:
        return None
    keys = ["family_key", "hypothesis", "registered_at", "submissions", "max_submissions", "status"]
    return dict(zip(keys, row))
