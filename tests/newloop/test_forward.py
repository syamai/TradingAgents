"""Stage3 forward 관찰 오케스트레이션 테스트 — 등록 동결·경과주수·판정자격(#6).

forward 메트릭 계산(홀드아웃 백테스트)은 무겁고 통합 영역이라 monkeypatch 로 격리하고,
register/observe 의 순수 로직(신규만 추가·재등록 무시·4주 자격 경계)만 결정적으로 고정한다.
"""

from __future__ import annotations

import pytest

from tradingagents.newloop import forward

_IS = {"alpha_ann_pct": 12.0, "beta": 0.9, "t_alpha": 4.0, "n_invested_days": 80,
       "deployment": 0.5, "status": "pass", "gate_passed": True}
_FWD = {"alpha_ann_pct": 5.0, "beta": 1.0, "t_alpha": 2.1, "n_invested_days": 12,
        "deployment": 0.4, "status": "fail", "gate_passed": False}


@pytest.fixture
def _isolated(tmp_path, monkeypatch):
    """레지스트리 격리 + 무거운 로딩·백테스트를 캔드 값으로 대체."""
    monkeypatch.setattr(forward, "REGISTRY", tmp_path / "forward_registry.json")
    monkeypatch.setattr(forward, "_load_all", lambda specs: ("TK", "L", "K", None))
    monkeypatch.setattr(forward, "score", lambda *a, **k: dict(_IS))
    monkeypatch.setattr(forward, "_fwd_metrics", lambda *a, **k: dict(_FWD))
    return tmp_path


@pytest.mark.unit
class TestForwardRegister:
    def test_register_freezes_is_metrics(self, _isolated):
        n = forward.register("fam", [("spA", {"name": "spA"})], "2026-06-15T10:00:00+09:00")
        assert n == 1
        reg = forward._read()
        assert len(reg) == 1
        assert reg[0]["family"] == "fam"
        assert reg[0]["freeze_date"] == forward.SCORE_DATE_HI
        assert reg[0]["is_metrics"]["status"] == "pass"

    def test_reregister_same_ref_ignored(self, _isolated):
        forward.register("fam", [("spA", {"name": "spA"})], "2026-06-15T10:00:00+09:00")
        again = forward.register("fam", [("spA", {"name": "spA"})], "2026-06-20T10:00:00+09:00")
        assert again == 0                       # 동결 유지 — 재등록 무시
        reg = forward._read()
        assert len(reg) == 1
        assert reg[0]["registered_at"] == "2026-06-15T10:00:00+09:00"  # 최초 등록일 보존


@pytest.mark.unit
class TestForwardObserveEligibility:
    def test_below_4_weeks_not_eligible(self, _isolated):
        forward.register("fam", [("spA", {"name": "spA"})], "2026-06-15T10:00:00+09:00")
        rows = forward.observe(now_iso="2026-07-01T10:00:00+09:00")   # 16일 ≈ 2.3주
        assert len(rows) == 1
        assert rows[0]["eligible_for_decision"] is False
        assert rows[0]["fwd"]["status"] == "fail"
        assert rows[0]["is"]["status"] == "pass"

    def test_at_4_weeks_eligible(self, _isolated):
        forward.register("fam", [("spA", {"name": "spA"})], "2026-06-15T10:00:00+09:00")
        rows = forward.observe(now_iso="2026-07-13T10:00:00+09:00")   # 정확히 28일 = 4.0주
        assert rows[0]["weeks_elapsed"] == 4.0
        assert rows[0]["eligible_for_decision"] is True

    def test_empty_registry_returns_empty(self, _isolated):
        assert forward.observe(now_iso="2026-07-13T10:00:00+09:00") == []
