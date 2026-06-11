"""lessons 자가학습 컨텍스트 테스트 — 탈락 양식 분류·고갈 선언."""

from __future__ import annotations

import pytest

from tradingagents.newloop import ledger
from tradingagents.newloop.lessons import EXHAUSTION_K, build_lessons


def _db(tmp_path):
    return str(tmp_path / "ledger.db")


def _result(status, *, t_alpha=None, beta=None, checks=None):
    return {"strategy_ref": "db:1", "gate_version": "s1-v2", "status": status,
            "result": {"status": status, "t_alpha": t_alpha, "alpha_pct": 0.1,
                       "beta": beta, "checks": checks or {}}}


@pytest.mark.unit
class TestLessons:
    def test_beta_disguise_mode_classified(self, tmp_path):
        db = _db(tmp_path)
        ledger.register_family("f1", "고변동 돌파", db_path=db)
        ledger.record_submission("f1", [_result(
            "fail", t_alpha=-1.0, beta=2.5,
            checks={"down_not_broken": False, "alpha_significant": False})], db_path=db)
        f = build_lessons(db_path=db)["families"][0]
        assert "베타 위장" in f["failure_mode"]
        assert f["best_beta"] == 2.5

    def test_exhaustion_declared_after_k_consecutive(self, tmp_path):
        db = _db(tmp_path)
        for i in range(EXHAUSTION_K):
            key = f"fam{i}"
            ledger.register_family(key, f"가설{i}", db_path=db)
            for _ in range(ledger.MAX_SUBMISSIONS):
                ledger.record_submission(key, [_result(
                    "fail", t_alpha=0.5, beta=1.0,
                    checks={"alpha_significant": False})], db_path=db)
        lessons = build_lessons(db_path=db)
        assert lessons["exhausted"] is True

    def test_pass_resets_exhaustion(self, tmp_path):
        db = _db(tmp_path)
        for i in range(EXHAUSTION_K - 1):
            key = f"fam{i}"
            ledger.register_family(key, f"가설{i}", db_path=db)
            for _ in range(ledger.MAX_SUBMISSIONS):
                ledger.record_submission(key, [_result("fail")], db_path=db)
        ledger.register_family("winner", "통과 가설", db_path=db)
        ledger.record_submission("winner", [_result("pass", t_alpha=3.5)], db_path=db)
        assert build_lessons(db_path=db)["exhausted"] is False
