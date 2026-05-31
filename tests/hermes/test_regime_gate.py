"""레짐 게이트 실험 — 정합성 회귀.

불변식:
  1. augment_with_regime: entry.all_of 에 market_filter 1개를 정확히 덧붙인다
     (원본 spec 불변 — deepcopy).
  2. 게이트는 진입을 *추가로 막을 뿐* — regime 거래수 ≤ baseline 거래수.
  3. KOSPI 가 줄곧 하락(above_ma 항상 False)이면 진입 0건.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.hermes import backtest_engine_v2 as v2
from tradingagents.hermes.regime_gate_experiment import augment_with_regime


def _df(n=160, seed=0):
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.normal(0, 15, n))
    close = np.clip(close, 100, None)
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
        "close": close,
        "volume": rng.integers(1e5, 1e6, n),
        "foreign_registered_net_qty": rng.normal(0, 1000, n),
        "foreign_registered_pct": rng.normal(0, 5, n),
        "price_change_pct": np.r_[0.0, np.diff(close) / close[:-1] * 100],
    })


def _spec():
    return {
        "spec_version": 2, "name": "t", "direction": "long",
        "entry": {"all_of": [{"signal": "price_drop", "window": 5, "value": 5.0}]},
        "exit": {"take_profit_pct": 8.0, "max_hold_days": 10},
    }


def _kospi(dates, *, trend):
    """trend>0 상승 / trend<0 하락 KOSPI."""
    n = len(dates)
    return pd.DataFrame({"date": list(dates),
                         "close": 2500.0 + trend * np.arange(n)})


@pytest.mark.unit
class TestAugment:
    def test_appends_one_market_filter(self):
        spec = _spec()
        aug = augment_with_regime(spec, window=60)
        assert len(spec["entry"]["all_of"]) == 1          # 원본 불변
        assert len(aug["entry"]["all_of"]) == 2
        last = aug["entry"]["all_of"][-1]
        assert last == {"signal": "market_filter", "mode": "above_ma", "window": 60}


@pytest.mark.unit
class TestGateMonotonic:
    def test_regime_trades_le_baseline(self):
        df = _df()
        kospi = _kospi(df["date"], trend=2.0)           # 상승장
        base, _, _, _ = v2._simulate_v2(_spec(), df, kospi=kospi)
        aug = augment_with_regime(_spec(), window=20)
        gated, _, _, _ = v2._simulate_v2(aug, df, kospi=kospi)
        assert len(gated) <= len(base)

    def test_falling_market_blocks_all_entries(self):
        df = _df()
        kospi = _kospi(df["date"], trend=-2.0)           # 하락장 → above_ma 항상 False
        aug = augment_with_regime(_spec(), window=20)
        gated, _, _, _ = v2._simulate_v2(aug, df, kospi=kospi)
        assert gated == []
