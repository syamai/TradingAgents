"""strategy v2 — 신규 신호 검증·look-ahead 회귀·완화 게이트.

핵심 불변식: 신규 신호 3종(price_drop/trend_slope/rolling_corr)이 row i 까지의
데이터로만 평가된다 — 미래 행을 바꿔도 과거 신호값이 불변이어야 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes import backtest_engine_v2 as v2
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2


def _df(n=120, seed=0):
    """합성 holdings — 신호 평가에 필요한 컬럼만."""
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


# === spec 검증 ===

@pytest.mark.unit
class TestValidateSpecV2:
    def _spec(self, entry):
        return {
            "spec_version": 2, "name": "t", "direction": "long",
            "entry": {"all_of": entry},
            "exit": {"max_hold_days": 10},
        }

    def test_price_drop_valid(self):
        validate_spec_v2(self._spec(
            [{"signal": "price_drop", "window": 5, "value": 8.0}]))

    def test_price_drop_offgrid_window_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(self._spec(
                [{"signal": "price_drop", "window": 7, "value": 8.0}]))

    def test_price_drop_offgrid_value_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(self._spec(
                [{"signal": "price_drop", "window": 5, "value": 6.0}]))

    def test_trend_slope_valid(self):
        validate_spec_v2(self._spec(
            [{"signal": "trend_slope", "subject": "pension",
              "window": 20, "direction": "up"}]))

    def test_trend_slope_bad_direction_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(self._spec(
                [{"signal": "trend_slope", "subject": "pension",
                  "window": 20, "direction": "sideways"}]))

    def test_rolling_corr_valid(self):
        validate_spec_v2(self._spec(
            [{"signal": "rolling_corr", "subject": "foreign",
              "lookback": 60, "min_r": 0.3}]))

    def test_rolling_corr_offgrid_minr_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(self._spec(
                [{"signal": "rolling_corr", "subject": "foreign",
                  "lookback": 60, "min_r": 0.25}]))

    def test_market_filter_valid(self):
        validate_spec_v2(self._spec(
            [{"signal": "market_filter", "mode": "above_ma", "window": 60}]))

    def test_market_filter_bad_mode_raises(self):
        with pytest.raises(ValueError):
            validate_spec_v2(self._spec(
                [{"signal": "market_filter", "mode": "sideways", "window": 60}]))

    def test_base_signal_still_valid(self):
        validate_spec_v2(self._spec(
            [{"signal": "net_streak", "subject": "foreign_registered",
              "min_days": 3, "sign": "buy"}]))

    def test_spec_version_1_accepted(self):
        s = self._spec([{"signal": "price_filter", "mode": "above_ma", "window": 20}])
        s["spec_version"] = 1
        validate_spec_v2(s)

    def test_spec_version_3_rejected(self):
        s = self._spec([{"signal": "price_filter", "mode": "above_ma", "window": 20}])
        s["spec_version"] = 3
        with pytest.raises(ValueError):
            validate_spec_v2(s)


# === look-ahead 회귀 (신규 3종) ===

@pytest.mark.unit
class TestLookAheadV2:
    def _assert_past_invariant(self, df, sig, probe, future):
        """probe 행의 신호값이 future(>probe) 행 변경에 불변임을 단언."""
        before = v2._eval_signal_v2(df, sig).iloc[probe]
        df2 = df.copy()
        for col in ("close", "foreign_registered_net_qty", "price_change_pct"):
            df2.loc[future, col] = df2[col].iloc[future] * 100 + 1e6
        if "market_close" in df2.columns:
            df2.loc[future, "market_close"] = df2["market_close"].iloc[future] * 100 + 1e6
        after = v2._eval_signal_v2(df2, sig).iloc[probe]
        assert before == after, f"{sig['signal']} look-ahead at {probe} via {future}"

    def test_price_drop_no_lookahead(self):
        df = _df()
        sig = {"signal": "price_drop", "window": 5, "value": 5.0}
        self._assert_past_invariant(df, sig, probe=50, future=80)

    def test_trend_slope_no_lookahead(self):
        df = _df()
        sig = {"signal": "trend_slope", "subject": "foreign_registered",
               "window": 20, "direction": "up"}
        self._assert_past_invariant(df, sig, probe=50, future=80)

    def test_rolling_corr_no_lookahead(self):
        df = _df()
        sig = {"signal": "rolling_corr", "subject": "foreign_registered",
               "lookback": 20, "min_r": 0.1}
        self._assert_past_invariant(df, sig, probe=60, future=90)

    def test_market_filter_no_lookahead(self):
        df = _df()
        sig = {"signal": "market_filter", "mode": "above_ma", "window": 20}
        self._assert_past_invariant(df, sig, probe=60, future=90)

    def test_price_drop_semantics(self):
        # 직접 구성: 5일 전 대비 -15% 하락 행 → True.
        n = 30
        close = np.full(n, 1000.0)
        close[20] = 850.0  # row20 = row15(1000) 대비 -15%
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "close": close, "volume": 1,
            "foreign_registered_net_qty": 0.0, "foreign_registered_pct": 0.0,
            "price_change_pct": 0.0,
        })
        s = v2._eval_signal_v2(df, {"signal": "price_drop", "window": 5, "value": 10.0})
        assert bool(s.iloc[20]) is True
        assert bool(s.iloc[10]) is False

    def test_market_filter_semantics(self):
        df = _df(n=40)
        df["market_close"] = [100.0] * 20 + [120.0] * 20
        s = v2._eval_signal_v2(df, {"signal": "market_filter", "mode": "above_ma", "window": 20})
        assert bool(s.iloc[25]) is True

    def test_attach_market_columns_uses_past_kospi(self):
        df = _df(n=3)[["date", "close", "volume", "foreign_registered_net_qty",
                       "foreign_registered_pct", "price_change_pct"]]
        kospi = pd.DataFrame({
            "date": ["2020-12-31", "2021-01-02"],
            "close": [100.0, 110.0],
        })
        out = v2._attach_market_columns(df, kospi)
        assert list(out["market_close"]) == [100.0, 110.0, 110.0]


# === 기존 신호 위임 ===

@pytest.mark.unit
class TestDelegation:
    def test_base_signal_matches_original(self):
        df = _df()
        sig = {"signal": "net_streak", "subject": "foreign_registered",
               "min_days": 2, "sign": "buy"}
        a = v2._eval_signal_v2(df, sig)
        b = bt._eval_signal(df, sig)
        pd.testing.assert_series_equal(a, b)


# === 완화 게이트 (win>0.50, sharpe>1.0, strict) ===

@pytest.mark.unit
class TestGateV2:
    def _m(self, win, sharpe, mdd=-10.0, n=60):
        return {"n_trades": n, "win_rate": win, "sharpe": sharpe,
                "mdd_pct": mdd, "cum_return_pct": 1.0, "avg_hold_days": 5.0,
                "avg_net_ret_pct": 1.0, "avg_excess_ret_pct": 1.0}

    def test_win_must_exceed_half_strict(self):
        assert v2.passes_single_gate_v2(self._m(0.51, 1.1)) is True
        assert v2.passes_single_gate_v2(self._m(0.50, 1.1)) is False  # strict >

    def test_sharpe_must_exceed_one_strict(self):
        assert v2.passes_single_gate_v2(self._m(0.55, 1.01)) is True
        assert v2.passes_single_gate_v2(self._m(0.55, 1.00)) is False  # strict >

    def test_mdd_and_trades_still_bind(self):
        assert v2.passes_single_gate_v2(self._m(0.6, 1.5, mdd=-25.0)) is False
        assert v2.passes_single_gate_v2(self._m(0.6, 1.5, n=40)) is False

    def test_full_gate_requires_both_and_gap(self):
        good = self._m(0.55, 1.2)
        assert v2.passes_full_gate_v2(good, dict(good)) is True
        assert v2.passes_full_gate_v2(self._m(0.65, 1.2), self._m(0.52, 1.2)) is False
        assert v2.passes_full_gate_v2(good, self._m(0.48, 1.2)) is False


# === 종목군 진입점 shape ===

@pytest.mark.unit
class TestRunUniverseV2:
    def test_returns_expected_shape(self):
        df = _df(n=200, seed=1)
        loader = lambda tk: (df.copy(), {})  # noqa: E731
        spec = {
            "spec_version": 2, "name": "t", "direction": "long",
            "entry": {"all_of": [
                {"signal": "price_drop", "window": 5, "value": 5.0}]},
            "exit": {"take_profit_pct": 5.0, "max_hold_days": 10},
        }
        out = v2.run_universe_backtest_v2(
            spec, ["005930", "000660", "035720", "005380"],
            loader=loader, kospi_fetcher=None)
        assert set(out) == {"in_sample", "out_sample", "gate_passed",
                            "portfolio_policy", "universe_size", "n_in", "n_out"}
        assert isinstance(out["gate_passed"], bool)
        assert out["universe_size"] == 4
