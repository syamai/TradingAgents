"""트레일링 스톱 실험 엔진 — 정합성 회귀.

불변식:
  1. trail_pct=None → 결과가 backtest_engine_v2._simulate_v2 와 **완전 동일**
     (트레일링은 순수 추가 — off 일 때 기존 동작 불변).
  2. 트레일링 의미: 보유 중 고점 대비 trail% 후퇴 시 'trailing_stop' 으로 청산.
  3. look-ahead 0: 청산 이후 미래 봉을 바꿔도 과거 트레이드 불변.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.hermes import backtest_engine_v2 as v2
from tradingagents.hermes.trailing_stop_experiment import _simulate_trailing


def _df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.normal(0, 15, n))
    close = np.clip(close, 100, None)
    nq = rng.normal(0, 1000, n)
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
        "close": close,
        "market_close": 2500 + np.cumsum(rng.normal(0, 8, n)),
        "volume": rng.integers(1e5, 1e6, n),
        "foreign_registered_net_qty": nq,
        "foreign_registered_pct": rng.normal(0, 5, n),
        "price_change_pct": np.r_[0.0, np.diff(close) / close[:-1] * 100],
    })


def _spec(exit_extra=None):
    exit_ = {"take_profit_pct": 10.0, "stop_loss_pct": 8.0, "max_hold_days": 20}
    if exit_extra:
        exit_.update(exit_extra)
    return {
        "spec_version": 2, "name": "t", "direction": "long",
        "entry": {"all_of": [{"signal": "price_drop", "window": 5, "value": 5.0}]},
        "exit": exit_,
    }


@pytest.mark.unit
class TestBaselineEquivalence:
    def test_trail_none_matches_simulate_v2(self):
        df = _df(n=200, seed=3)
        spec = _spec()
        t_trail, d_trail, a_trail, _ = _simulate_trailing(
            spec, df, kospi=None, trail_pct=None, keep_fixed_sl=True)
        t_v2, d_v2, a_v2, _ = v2._simulate_v2(spec, df, kospi=None)
        assert t_trail == t_v2
        pd.testing.assert_series_equal(d_trail, d_v2)
        pd.testing.assert_series_equal(a_trail, a_v2)


@pytest.mark.unit
class TestTrailingSemantics:
    def _crafted(self):
        """bar5 에서 진입신호(net_streak) → e=6, 이후 고점 1100 후 -11% 하락."""
        n = 20
        close = np.full(n, 1000.0)
        close[7], close[8], close[9], close[10] = 1025, 1050, 1075, 1100  # 상승→고점
        close[11:] = 979.0   # 고점 1100 대비 -11% → trail10 트리거
        nq = np.full(n, -1.0)
        nq[4] = nq[5] = 1000.0   # 연속 2일 순매수 → bar5 진입신호
        return pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "close": close, "volume": 1,
            "foreign_registered_net_qty": nq, "foreign_registered_pct": 0.0,
            "price_change_pct": 0.0,
        })

    def test_trailing_triggers_on_peak_pullback(self):
        df = self._crafted()
        spec = {
            "spec_version": 2, "name": "t", "direction": "long",
            "entry": {"all_of": [{"signal": "net_streak",
                                  "subject": "foreign_registered",
                                  "min_days": 2, "sign": "buy"}]},
            "exit": {"max_hold_days": 20},   # sl/tp 없음 — 트레일링만
        }
        trades, _, _, _ = _simulate_trailing(
            spec, df, kospi=None, trail_pct=10.0, keep_fixed_sl=True)
        assert len(trades) == 1
        tr = trades[0]
        assert tr["exit_reason"] == "trailing_stop"
        assert tr["entry_price"] == 1000.0          # close[6]
        # 고점 1100, bar11(=979) 에서 트리거 → 체결 bar12.
        assert tr["exit_date"] == df["date"].iloc[12]

    def test_no_trail_holds_to_max_hold(self):
        """동일 spec, trail_pct=None 이면 트레일링 미발동 → max_hold 로 청산."""
        df = self._crafted()
        spec = {
            "spec_version": 2, "name": "t", "direction": "long",
            "entry": {"all_of": [{"signal": "net_streak",
                                  "subject": "foreign_registered",
                                  "min_days": 2, "sign": "buy"}]},
            "exit": {"max_hold_days": 20},
        }
        trades, _, _, _ = _simulate_trailing(
            spec, df, kospi=None, trail_pct=None, keep_fixed_sl=True)
        assert trades[0]["exit_reason"] in ("max_hold", "forced_eod")


@pytest.mark.unit
class TestLookAhead:
    def test_future_change_does_not_alter_past_trade(self):
        df = _df(n=120, seed=7)
        spec = _spec()
        base, _, _, _ = _simulate_trailing(
            spec, df, kospi=None, trail_pct=10.0, keep_fixed_sl=True)
        assert base, "최소 1 트레이드 필요"
        first_exit_idx = df["date"].tolist().index(base[0]["exit_date"])

        df2 = df.copy()
        fut = slice(first_exit_idx + 5, None)   # 첫 청산 이후 봉만 교란
        df2.loc[fut, "close"] = df2.loc[fut, "close"] * 100 + 1e6
        df2.loc[fut, "foreign_registered_net_qty"] = -9e9
        after, _, _, _ = _simulate_trailing(
            spec, df2, kospi=None, trail_pct=10.0, keep_fixed_sl=True)
        assert after[0] == base[0]
