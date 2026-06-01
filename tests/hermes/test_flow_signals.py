"""수급 흐름-변화 신호 3종 — 진입 마스크 의미·그리드 검증.

신규 신호: flow_zscore / flow_accel / flow_divergence.
각 신호마다 (a) 손계산 가능한 합성 df 로 entry 마스크가 공식과 일치,
(b) off-grid 값에 validate_spec_v2 가 ValueError 를 던지는지 확인한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.hermes import backtest_engine_v2 as v2
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2


def _spec(entry):
    return {
        "spec_version": 2, "name": "t", "direction": "long",
        "entry": {"all_of": entry},
        "exit": {"max_hold_days": 10},
    }


# === flow_zscore ===

@pytest.mark.unit
class TestFlowZscore:
    def test_semantics(self):
        # net_qty 를 일정값(=100)으로 두면 roll(5합)=500 으로 상수 → std=0 → NaN → False.
        # 마지막에 큰 스파이크를 넣어 z 가 치솟는 행만 True 가 되는지 손계산으로 확인.
        n = 80
        nq = np.full(n, 100.0)
        nq[70] = 5000.0   # row70 에서 5일 누적이 평소(500) 대비 급증
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "close": 1000.0, "volume": 1,
            "foreign_registered_net_qty": nq,
            "retail_net_qty": 0.0, "price_change_pct": 0.0,
        })
        sig = {"signal": "flow_zscore", "subject": "foreign_registered",
               "window": 5, "lookback": 60, "min_z": 1.5}
        s = v2._eval_signal_v2(df, sig)
        # 직접 재현: roll = 5일합, mu/sd = 60일 trailing.
        x = pd.Series(nq)
        roll = x.rolling(5).sum()
        mu = roll.rolling(60).mean()
        sd = roll.rolling(60).std()
        z = (roll - mu) / sd.where(sd > 0)
        expected = (z >= 1.5)
        pd.testing.assert_series_equal(
            s.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )
        # 스파이크가 5일 누적에 남아있는 구간(70~74)에 True 가 존재.
        assert bool(s.iloc[70]) is True

    def test_nan_warmup_is_false(self):
        n = 80
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "close": 1000.0, "volume": 1,
            "foreign_registered_net_qty": np.arange(n, dtype=float),
            "retail_net_qty": 0.0, "price_change_pct": 0.0,
        })
        sig = {"signal": "flow_zscore", "subject": "foreign_registered",
               "window": 5, "lookback": 60, "min_z": 1.0}
        s = v2._eval_signal_v2(df, sig)
        # lookback 미충족 워밍업 구간(<64 행)은 NaN → False.
        assert not s.iloc[:60].any()

    def test_offgrid_min_z_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(_spec([{
                "signal": "flow_zscore", "subject": "foreign_registered",
                "window": 5, "lookback": 60, "min_z": 1.7}]))

    def test_offgrid_lookback_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(_spec([{
                "signal": "flow_zscore", "subject": "foreign_registered",
                "window": 5, "lookback": 100, "min_z": 1.5}]))


# === flow_accel ===

@pytest.mark.unit
class TestFlowAccel:
    def test_semantics(self):
        # 전반 음(-100) → 후반 양(+100). ma_short 가 ma_long 보다 먼저 양전 →
        # (ma_s - ma_l) > 0 인 첫 구간을 손계산으로 확인.
        n = 60
        nq = np.r_[np.full(30, -100.0), np.full(30, 100.0)]
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "close": 1000.0, "volume": 1,
            "foreign_registered_net_qty": nq,
            "retail_net_qty": 0.0, "price_change_pct": 0.0,
        })
        sig = {"signal": "flow_accel", "subject": "foreign_registered",
               "short": 5, "long": 20}
        s = v2._eval_signal_v2(df, sig)
        x = pd.Series(nq)
        expected = ((x.rolling(5).mean() - x.rolling(20).mean()) > 0)
        pd.testing.assert_series_equal(
            s.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )
        # 음→양 전환 직후 단기평균이 먼저 반등 → row35 즈음 True.
        assert bool(s.iloc[40]) is True
        # 전반 안정 음수 구간은 ma_s==ma_l → 차이 0 → False.
        assert bool(s.iloc[25]) is False

    def test_offgrid_long_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(_spec([{
                "signal": "flow_accel", "subject": "foreign_registered",
                "short": 5, "long": 30}]))

    def test_offgrid_short_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(_spec([{
                "signal": "flow_accel", "subject": "foreign_registered",
                "short": 4, "long": 20}]))


# === flow_divergence ===

@pytest.mark.unit
class TestFlowDivergence:
    def test_semantics(self):
        # smart(외인) 매집 + 개인 매도 동시 성립하는 행만 True.
        n = 30
        smart = np.full(n, -50.0)
        retail = np.full(n, 50.0)
        # row20: 외인 누적>0 & 개인 누적<0 (W=3 합)
        smart[18:21] = 100.0
        retail[18:21] = -100.0
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "close": 1000.0, "volume": 1,
            "foreign_registered_net_qty": smart,
            "retail_net_qty": retail, "price_change_pct": 0.0,
        })
        sig = {"signal": "flow_divergence", "subject": "foreign_registered", "window": 3}
        s = v2._eval_signal_v2(df, sig)
        sm = pd.Series(smart).rolling(3).sum()
        rt = pd.Series(retail).rolling(3).sum()
        expected = (sm > 0) & (rt < 0)
        pd.testing.assert_series_equal(
            s.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )
        assert bool(s.iloc[20]) is True
        # 평상 구간: 외인 매도 + 개인 매수 → False.
        assert bool(s.iloc[10]) is False

    def test_offgrid_window_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(_spec([{
                "signal": "flow_divergence", "subject": "foreign_registered",
                "window": 7}]))

    def test_bad_subject_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(_spec([{
                "signal": "flow_divergence", "subject": "institution",
                "window": 5}]))
