"""strategy_spec 검증·해시 단위 테스트."""
from __future__ import annotations

import pytest

from tradingagents.hermes.strategy_spec import (
    spec_hash,
    validate_spec,
)


def _valid_spec(**overrides) -> dict:
    base = {
        "spec_version": 1,
        "name": "fr_streak_long",
        "direction": "long",
        "entry": {
            "all_of": [
                {"signal": "net_streak", "subject": "foreign_registered",
                 "min_days": 3, "sign": "buy"},
                {"signal": "pct_threshold", "subject": "foreign_registered",
                 "op": ">=", "value": 30.0},
            ]
        },
        "exit": {
            "signal_all_of": [
                {"signal": "net_streak", "subject": "foreign_registered",
                 "min_days": 2, "sign": "sell"},
            ],
            "stop_loss_pct": 5.0,
            "take_profit_pct": 10.0,
            "max_hold_days": 20,
        },
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestValidateSpec:
    def test_valid_spec_passes(self):
        validate_spec(_valid_spec())  # no raise

    def test_minimal_spec_passes(self):
        # exit signal 없이 가격/보유일만으로 청산.
        spec = _valid_spec(exit={"max_hold_days": 10})
        validate_spec(spec)

    def test_all_five_signal_types_valid(self):
        spec = _valid_spec(entry={"all_of": [
            {"signal": "pct_delta", "subject": "pension",
             "window": 5, "op": ">=", "value": 5.0},
            {"signal": "net_vol_ratio", "subject": "private_equity",
             "window": 10, "op": ">=", "value": 0.1},
            {"signal": "price_filter", "mode": "above_ma", "window": 20},
        ]})
        validate_spec(spec)

    def test_bad_spec_version_raises(self):
        with pytest.raises(ValueError, match="spec_version"):
            validate_spec(_valid_spec(spec_version=2))

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            validate_spec(_valid_spec(name=""))

    def test_non_long_direction_raises(self):
        with pytest.raises(ValueError, match="long"):
            validate_spec(_valid_spec(direction="short"))

    def test_too_many_entry_signals_raises(self):
        sig = {"signal": "price_filter", "mode": "above_ma", "window": 5}
        with pytest.raises(ValueError, match="all_of"):
            validate_spec(_valid_spec(entry={"all_of": [sig, sig, sig, sig]}))

    def test_empty_entry_raises(self):
        with pytest.raises(ValueError, match="all_of"):
            validate_spec(_valid_spec(entry={"all_of": []}))

    def test_too_many_exit_signals_raises(self):
        sig = {"signal": "net_streak", "subject": "retail",
               "min_days": 2, "sign": "sell"}
        spec = _valid_spec()
        spec["exit"]["signal_all_of"] = [sig, sig, sig]
        with pytest.raises(ValueError, match="signal_all_of"):
            validate_spec(spec)

    def test_unknown_subject_raises(self):
        spec = _valid_spec()
        spec["entry"]["all_of"][0]["subject"] = "institution"  # 통합 — 금지
        with pytest.raises(ValueError, match="subject"):
            validate_spec(spec)

    def test_off_grid_min_days_raises(self):
        spec = _valid_spec()
        spec["entry"]["all_of"][0]["min_days"] = 6  # grid: 2,3,4,5,7,10
        with pytest.raises(ValueError, match="min_days"):
            validate_spec(spec)

    def test_off_grid_stop_loss_raises(self):
        spec = _valid_spec()
        spec["exit"]["stop_loss_pct"] = 7.0  # grid: 3,5,8,10
        with pytest.raises(ValueError, match="stop_loss_pct"):
            validate_spec(spec)

    def test_off_grid_pct_value_raises(self):
        spec = _valid_spec()
        spec["entry"]["all_of"][1]["value"] = 35.0  # grid: 10,20,30,40,50
        with pytest.raises(ValueError, match="value"):
            validate_spec(spec)

    def test_missing_max_hold_raises(self):
        spec = _valid_spec()
        del spec["exit"]["max_hold_days"]
        with pytest.raises(ValueError, match="max_hold_days"):
            validate_spec(spec)

    def test_bad_op_raises(self):
        spec = _valid_spec()
        spec["entry"]["all_of"][1]["op"] = "=="
        with pytest.raises(ValueError, match="op"):
            validate_spec(spec)

    def test_unknown_signal_type_raises(self):
        spec = _valid_spec(entry={"all_of": [{"signal": "rsi_cross"}]})
        with pytest.raises(ValueError, match="unknown signal"):
            validate_spec(spec)

    def test_int_float_grid_equivalence(self):
        # 20 (int) == 20.0 (grid float) 허용.
        spec = _valid_spec()
        spec["entry"]["all_of"][1]["value"] = 30  # int, grid has 30.0
        validate_spec(spec)


@pytest.mark.unit
class TestSpecHash:
    def test_hash_stable(self):
        assert spec_hash(_valid_spec()) == spec_hash(_valid_spec())

    def test_hash_ignores_name(self):
        assert spec_hash(_valid_spec(name="a")) == spec_hash(_valid_spec(name="b"))

    def test_hash_order_independent_entry(self):
        a = _valid_spec()
        b = _valid_spec()
        b["entry"]["all_of"] = list(reversed(b["entry"]["all_of"]))
        assert spec_hash(a) == spec_hash(b)

    def test_hash_differs_on_logic_change(self):
        a = _valid_spec()
        b = _valid_spec()
        b["entry"]["all_of"][0]["min_days"] = 5
        assert spec_hash(a) != spec_hash(b)

    def test_hash_differs_on_stop_loss_change(self):
        a = _valid_spec()
        b = _valid_spec()
        b["exit"]["stop_loss_pct"] = 8.0
        assert spec_hash(a) != spec_hash(b)
