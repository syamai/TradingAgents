"""원장 테스트 — 사전등록·재응시 한도·append-only."""

from __future__ import annotations

import pytest

from tradingagents.newloop import ledger


def _db(tmp_path):
    return str(tmp_path / "ledger.db")


def _result(status="fail", ref="db:1"):
    return {"strategy_ref": ref, "gate_version": "s1-v1", "status": status,
            "result": {"status": status}}


@pytest.mark.unit
class TestLedger:
    def test_unregistered_family_blocked(self, tmp_path):
        ok, why = ledger.can_submit("ghost", db_path=_db(tmp_path))
        assert ok is False
        assert "사전등록" in why

    def test_register_then_submit(self, tmp_path):
        db = _db(tmp_path)
        ledger.register_family("fam-a", "가설 한 줄", db_path=db)
        assert ledger.can_submit("fam-a", db_path=db)[0] is True
        sub = ledger.record_submission("fam-a", [_result()], db_path=db)
        assert sub == 1
        st = ledger.family_state("fam-a", db_path=db)
        assert st["submissions"] == 1
        assert st["status"] == "active"

    def test_exhausted_after_max_submissions(self, tmp_path):
        db = _db(tmp_path)
        ledger.register_family("fam-b", "가설", db_path=db)
        for _ in range(ledger.MAX_SUBMISSIONS):
            ledger.record_submission("fam-b", [_result()], db_path=db)
        assert ledger.family_state("fam-b", db_path=db)["status"] == "exhausted"
        ok, why = ledger.can_submit("fam-b", db_path=db)
        assert ok is False
        assert "한도 소진" in why
        with pytest.raises(ValueError):
            ledger.record_submission("fam-b", [_result()], db_path=db)

    def test_pass_marks_family_passed_and_blocks_resubmit(self, tmp_path):
        db = _db(tmp_path)
        ledger.register_family("fam-c", "가설", db_path=db)
        ledger.record_submission("fam-c", [_result("fail"), _result("pass", "db:2")], db_path=db)
        assert ledger.family_state("fam-c", db_path=db)["status"] == "passed"
        ok, why = ledger.can_submit("fam-c", db_path=db)
        assert ok is False
        assert "통과" in why

    def test_variants_cap(self, tmp_path):
        db = _db(tmp_path)
        ledger.register_family("fam-d", "가설", db_path=db)
        too_many = [_result(ref=f"db:{i}") for i in range(ledger.VARIANTS_MAX + 1)]
        with pytest.raises(ValueError):
            ledger.record_submission("fam-d", too_many, db_path=db)

    def test_hypothesis_write_once(self, tmp_path):
        db = _db(tmp_path)
        ledger.register_family("fam-e", "원래 가설", db_path=db)
        ledger.register_family("fam-e", "바꾼 가설", db_path=db)  # 무시되어야
        assert ledger.family_state("fam-e", db_path=db)["hypothesis"] == "원래 가설"

    def test_empty_submission_rejected(self, tmp_path):
        """빈 제출이 슬롯을 소모하면 안 된다 (감사 발견 보강)."""
        db = _db(tmp_path)
        ledger.register_family("fam-f", "가설", db_path=db)
        with pytest.raises(ValueError):
            ledger.record_submission("fam-f", [], db_path=db)
        assert ledger.family_state("fam-f", db_path=db)["submissions"] == 0

    def test_holdout_budget_caps_cumulative_trials(self, tmp_path, monkeypatch):
        """총 N 상한: 누적 채점 spec 이 예산을 넘으면 어떤 family 도 제출 불가.
        예산 초과 제출은 원자적으로 거부되고 슬롯·카운트를 소모하지 않는다."""
        db = _db(tmp_path)
        monkeypatch.setattr(ledger, "HOLDOUT_TRIAL_BUDGET", 3)
        ledger.register_family("fam-g", "가설", db_path=db)
        ledger.record_submission("fam-g", [_result(ref="db:1"), _result(ref="db:2")], db_path=db)
        assert ledger.cumulative_trials(db_path=db) == 2

        # 다음 제출(2개)이 2+2=4>3 → 거부, 카운트·슬롯 불변
        ledger.register_family("fam-h", "가설2", db_path=db)
        with pytest.raises(ValueError, match="예산"):
            ledger.record_submission("fam-h", [_result(ref="db:3"), _result(ref="db:4")], db_path=db)
        assert ledger.cumulative_trials(db_path=db) == 2
        assert ledger.family_state("fam-h", db_path=db)["submissions"] == 0

        # 잔여 1 — 1개짜리는 통과, 그 뒤 소진
        ledger.record_submission("fam-h", [_result(ref="db:5")], db_path=db)
        assert ledger.cumulative_trials(db_path=db) == 3
        assert ledger.budget_state(db_path=db) == {"used": 3, "budget": 3, "remaining": 0}
        ledger.register_family("fam-i", "가설3", db_path=db)
        ok, why = ledger.can_submit("fam-i", db_path=db)
        assert ok is False and "예산" in why
