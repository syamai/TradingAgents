"""StrategyStore 검증 — 저장·조회·중복 차단·게이트 필터."""
from __future__ import annotations

import pytest

from tradingagents.hermes.strategy_store import StrategyStore


@pytest.fixture
def store(tmp_path):
    return StrategyStore(root=tmp_path)


def _spec(name="s1", min_days=3) -> dict:
    return {
        "spec_version": 1, "name": name, "direction": "long",
        "entry": {"all_of": [
            {"signal": "net_streak", "subject": "foreign_registered",
             "min_days": min_days, "sign": "buy"},
        ]},
        "exit": {"max_hold_days": 20, "stop_loss_pct": 5.0},
    }


def _metrics(win_rate=0.62, n_trades=60) -> dict:
    return {
        "n_trades": n_trades, "win_rate": win_rate, "avg_net_ret_pct": 1.2,
        "avg_hold_days": 8.5, "cum_return_pct": 35.0, "sharpe": 1.4,
        "mdd_pct": -12.0, "avg_excess_ret_pct": 2.1,
    }


def _result(gate_passed=True, **o) -> dict:
    return {
        "in_sample": _metrics(**o), "out_sample": _metrics(**o),
        "gate_passed": gate_passed, "universe_size": 199,
        "n_in": 139, "n_out": 60,
    }


@pytest.mark.unit
class TestSaveGet:
    def test_save_returns_new_id(self, store):
        sid, is_new = store.save(_spec(), _result())
        assert isinstance(sid, int) and is_new is True

    def test_get_round_trip(self, store):
        sid, _ = store.save(_spec(name="abc"), _result(win_rate=0.7))
        s = store.get(sid)
        assert s["name"] == "abc"
        assert s["spec"]["entry"]["all_of"][0]["subject"] == "foreign_registered"
        assert s["in_win_rate"] == 0.7
        assert s["gate_passed"] is True
        assert s["universe_size"] == 199

    def test_get_unknown_returns_none(self, store):
        assert store.get(999) is None

    def test_invalid_spec_raises(self, store):
        bad = _spec()
        bad["entry"]["all_of"][0]["min_days"] = 6  # off-grid
        with pytest.raises(ValueError):
            store.save(bad, _result())


@pytest.mark.unit
class TestDedup:
    def test_same_logic_dedups(self, store):
        sid1, new1 = store.save(_spec(name="a"), _result())
        sid2, new2 = store.save(_spec(name="b"), _result())  # 이름만 다름
        assert new1 is True and new2 is False
        assert sid1 == sid2

    def test_exists(self, store):
        from tradingagents.hermes.strategy_spec import spec_hash
        spec = _spec()
        assert store.exists(spec_hash(spec)) is False
        store.save(spec, _result())
        assert store.exists(spec_hash(spec)) is True

    def test_different_logic_distinct(self, store):
        sid1, _ = store.save(_spec(min_days=3), _result())
        sid2, new2 = store.save(_spec(min_days=5), _result())
        assert sid1 != sid2 and new2 is True

    def test_get_by_hash(self, store):
        from tradingagents.hermes.strategy_spec import spec_hash
        spec = _spec()
        sid, _ = store.save(spec, _result())
        got = store.get_by_hash(spec_hash(spec))
        assert got["id"] == sid


@pytest.mark.unit
class TestList:
    def test_list_all(self, store):
        store.save(_spec(min_days=3), _result(gate_passed=True))
        store.save(_spec(min_days=5), _result(gate_passed=False))
        assert len(store.list()) == 2

    def test_list_gate_passed_only(self, store):
        store.save(_spec(min_days=3), _result(gate_passed=True))
        store.save(_spec(min_days=5), _result(gate_passed=False))
        passed = store.list(gate_passed=True)
        assert len(passed) == 1
        assert passed[0]["gate_passed"] is True

    def test_list_failed_only(self, store):
        store.save(_spec(min_days=3), _result(gate_passed=True))
        store.save(_spec(min_days=5), _result(gate_passed=False))
        failed = store.list(gate_passed=False)
        assert len(failed) == 1
        assert failed[0]["gate_passed"] is False


@pytest.mark.unit
class TestPersistence:
    def test_separate_instances_share_db(self, tmp_path):
        s1 = StrategyStore(root=tmp_path)
        sid, _ = s1.save(_spec(), _result())
        s2 = StrategyStore(root=tmp_path)
        assert s2.get(sid)["name"] == "s1"
