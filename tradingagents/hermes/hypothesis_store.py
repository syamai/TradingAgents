"""가설 저장·라벨 영구 저장 — SQLite 백엔드.

PRD 모듈 #3 (HypothesisStore). Hermes 가 생성한 가설 JSON 을 검증·저장하고
사후 라벨 (KOSPI 상대 수익 / 절대 수익 / 사용자 명시) 을 부여·조회한다.

스키마:
  - ``hypothesis_records``: 가설 1 개 = 1 row.
  - ``hypothesis_labels``: 라벨 1 개 = 1 row (한 가설은 여러 라벨 시점 가능).

라벨 종류 (``label_kind``):
  - ``user_immediate``: 가설 직후 사용자 ``/feedback`` 명령
  - ``user_followup``: 라벨링 시점 이후 사용자 추가 피드백
  - ``auto_relative``: 사후 KOSPI 상대 수익 자동 라벨 (G3 LabelingScheduler)
  - ``auto_absolute``: 사후 절대 수익 자동 라벨

호출자 책임: 같은 분석 응답을 두 번 ``save`` 하면 row 가 *두 벌* 생긴다.
중복 방지는 호출자 (예: Hermes skill 자체) 가 처리.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_VALID_STANCES = {
    "bullish", "moderately_bullish", "neutral",
    "moderately_bearish", "bearish",
}
_VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}
_VALID_LABEL_KINDS = {
    "user_immediate", "user_followup",
    "auto_relative", "auto_absolute",
}


def _default_store_dir() -> Path:
    return Path.home() / ".tradingagents" / "hermes"


def _validate_record(record: dict) -> None:
    """가설 JSON 스키마 검증. 실패 시 ``ValueError``.

    검증 항목:
      - top-level 필수 필드 (ticker, as_of_date, overall_stance,
        overall_confidence, hypotheses)
      - overall_stance 5 enum
      - overall_confidence 0~1
      - hypotheses 비어있지 않음
      - 각 hypothesis 필수 필드 + direction 3 enum + confidence 0~1
        + horizon_weeks 2~8 + evidence_excerpts 비어있지 않음
    """
    required_top = {
        "ticker", "as_of_date", "overall_stance",
        "overall_confidence", "hypotheses",
    }
    missing = required_top - record.keys()
    if missing:
        raise ValueError(f"missing top-level fields: {sorted(missing)}")

    if record["overall_stance"] not in _VALID_STANCES:
        raise ValueError(
            f"overall_stance must be one of {sorted(_VALID_STANCES)}, "
            f"got {record['overall_stance']!r}"
        )
    conf = record["overall_confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        raise ValueError(f"overall_confidence must be 0~1, got {conf}")

    hyps = record["hypotheses"]
    if not isinstance(hyps, list) or not hyps:
        raise ValueError("hypotheses must be a non-empty list")

    required_hyp = {
        "id", "claim", "direction", "confidence",
        "evidence_tools", "evidence_excerpts",
        "horizon_weeks", "predicted_relative_return_pct",
    }
    for i, h in enumerate(hyps):
        missing = required_hyp - h.keys()
        if missing:
            raise ValueError(
                f"hypothesis[{i}] missing fields: {sorted(missing)}"
            )
        if h["direction"] not in _VALID_DIRECTIONS:
            raise ValueError(
                f"hypothesis[{i}].direction must be one of "
                f"{sorted(_VALID_DIRECTIONS)}, got {h['direction']!r}"
            )
        c = h["confidence"]
        if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
            raise ValueError(
                f"hypothesis[{i}].confidence must be 0~1, got {c}"
            )
        hw = h["horizon_weeks"]
        if not isinstance(hw, int) or not (2 <= hw <= 8):
            raise ValueError(
                f"hypothesis[{i}].horizon_weeks must be 2~8 integer, "
                f"got {hw}"
            )
        ex = h["evidence_excerpts"]
        if not isinstance(ex, list) or not ex:
            raise ValueError(
                f"hypothesis[{i}].evidence_excerpts must be non-empty list "
                f"(raw 수치 인용 의무)"
            )


class HypothesisStore:
    """가설·라벨 영구 저장."""

    def __init__(self, *, root: Optional[Path] = None):
        self.root = Path(root) if root else _default_store_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _db_path(self) -> Path:
        return self.root / "hypotheses.db"

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
                CREATE TABLE IF NOT EXISTS hypothesis_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    h_local_id TEXT NOT NULL,
                    overall_stance TEXT NOT NULL,
                    overall_confidence REAL NOT NULL,
                    claim TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_tools TEXT NOT NULL,
                    evidence_excerpts TEXT NOT NULL,
                    horizon_weeks INTEGER NOT NULL,
                    predicted_relative_return_pct REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    )
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_hyp_ticker_date "
                "ON hypothesis_records(ticker, as_of_date)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_hyp_horizon "
                "ON hypothesis_records(horizon_weeks)"
            )
            c.execute("""
                CREATE TABLE IF NOT EXISTS hypothesis_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id INTEGER NOT NULL,
                    label_kind TEXT NOT NULL,
                    verdict TEXT,
                    actual_return_pct REAL,
                    actual_relative_return_pct REAL,
                    reason TEXT,
                    labeled_at_date TEXT,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                    ),
                    FOREIGN KEY (hypothesis_id) REFERENCES hypothesis_records(id)
                )
            """)
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_lbl_hyp "
                "ON hypothesis_labels(hypothesis_id)"
            )

    # === Public API ===

    def save(self, record: dict) -> list[int]:
        """응답 JSON 전체 저장. 반환=저장된 hypothesis_id 들.

        검증 실패 → ``ValueError``. 호출자는 이를 잡아 사용자에게 schema
        위반을 알려야 — Hermes 가설이 schema 안 맞으면 라벨링 불가.
        """
        _validate_record(record)
        ids: list[int] = []
        with self._conn() as c:
            for h in record["hypotheses"]:
                cur = c.execute(
                    """
                    INSERT INTO hypothesis_records (
                        ticker, as_of_date, h_local_id,
                        overall_stance, overall_confidence,
                        claim, direction, confidence,
                        evidence_tools, evidence_excerpts,
                        horizon_weeks, predicted_relative_return_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["ticker"],
                        record["as_of_date"],
                        h["id"],
                        record["overall_stance"],
                        record["overall_confidence"],
                        h["claim"],
                        h["direction"],
                        h["confidence"],
                        json.dumps(h["evidence_tools"], ensure_ascii=False),
                        json.dumps(h["evidence_excerpts"], ensure_ascii=False),
                        h["horizon_weeks"],
                        h["predicted_relative_return_pct"],
                    ),
                )
                ids.append(cur.lastrowid)
        return ids

    def get(self, hypothesis_id: int) -> Optional[dict]:
        """단일 가설 조회. 없으면 None. ``evidence_*`` 는 list 로 역직렬화."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM hypothesis_records WHERE id=?",
                (hypothesis_id,),
            ).fetchone()
        return _row_to_hypothesis(row) if row else None

    def list(
        self,
        *,
        ticker: Optional[str] = None,
        as_of_date: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> list[dict]:
        """필터 조건 만족하는 가설 리스트."""
        clauses, params = [], []
        if ticker is not None:
            clauses.append("ticker=?")
            params.append(ticker)
        if as_of_date is not None:
            clauses.append("as_of_date=?")
            params.append(as_of_date)
        if direction is not None:
            clauses.append("direction=?")
            params.append(direction)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as c:
            rows = c.execute(
                f"""
                SELECT * FROM hypothesis_records
                {where}
                ORDER BY ticker, as_of_date, h_local_id, id
                """,
                params,
            ).fetchall()
        return [_row_to_hypothesis(r) for r in rows]

    def add_label(
        self,
        hypothesis_id: int,
        label_kind: str,
        *,
        verdict: Optional[str] = None,
        actual_return_pct: Optional[float] = None,
        actual_relative_return_pct: Optional[float] = None,
        reason: Optional[str] = None,
        labeled_at_date: Optional[str] = None,
    ) -> int:
        """라벨 추가. 반환=label_id.

        한 가설은 여러 라벨 가능 (즉시 user_immediate + 4 주 후 auto_relative
        + 사용자 추가 user_followup 등). label_kind 는 위 4 enum.
        """
        if label_kind not in _VALID_LABEL_KINDS:
            raise ValueError(
                f"label_kind must be one of {sorted(_VALID_LABEL_KINDS)}, "
                f"got {label_kind!r}"
            )
        with self._conn() as c:
            # FK 검증 — 존재하지 않는 hypothesis_id 면 거부.
            exists = c.execute(
                "SELECT 1 FROM hypothesis_records WHERE id=?",
                (hypothesis_id,),
            ).fetchone()
            if not exists:
                raise ValueError(f"hypothesis_id {hypothesis_id} not found")
            cur = c.execute(
                """
                INSERT INTO hypothesis_labels (
                    hypothesis_id, label_kind, verdict,
                    actual_return_pct, actual_relative_return_pct,
                    reason, labeled_at_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id, label_kind, verdict,
                    actual_return_pct, actual_relative_return_pct,
                    reason, labeled_at_date,
                ),
            )
            return cur.lastrowid

    def get_labels(self, hypothesis_id: int) -> list[dict]:
        """한 가설의 라벨 모두 (시간순)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM hypothesis_labels "
                "WHERE hypothesis_id=? ORDER BY id",
                (hypothesis_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_with_labels(self, hypothesis_id: int) -> Optional[dict]:
        """가설 + 라벨들. ``labels`` 키에 list[dict]. 없으면 None."""
        h = self.get(hypothesis_id)
        if h is None:
            return None
        h["labels"] = self.get_labels(hypothesis_id)
        return h


def _row_to_hypothesis(row: sqlite3.Row) -> dict:
    """SQLite Row → 가설 dict. JSON 컬럼 자동 역직렬화."""
    d = dict(row)
    d["evidence_tools"] = json.loads(d["evidence_tools"])
    d["evidence_excerpts"] = json.loads(d["evidence_excerpts"])
    return d
