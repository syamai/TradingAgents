"""backtest_engine 단위 테스트 — 신호 평가·look-ahead·체결·메트릭·게이트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.hermes.backtest_engine import (
    TX_COST_ONE_WAY,
    _attach_kospi,
    _bad_bar_mask,
    _combine,
    _eval_signal,
    _metrics,
    _simulate,
    passes_full_gate,
    passes_single_gate,
    run_backtest,
    run_universe_backtest,
    split_of,
)


def _mk_df(closes, *, fr_net=None, fr_pct=None, volume=None,
           start="2024-01-01") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    data = {
        "date": list(dates),
        "close": [float(x) for x in closes],
        "volume": list(volume) if volume is not None else [1000.0] * n,
    }
    if fr_net is not None:
        data["foreign_registered_net_qty"] = list(fr_net)
    if fr_pct is not None:
        data["foreign_registered_pct"] = list(fr_pct)
    return pd.DataFrame(data)


def _entry_streak_spec(**exit_kw) -> dict:
    exit_ = {"max_hold_days": 20}
    exit_.update(exit_kw)
    return {
        "spec_version": 1, "name": "t", "direction": "long",
        "entry": {"all_of": [
            {"signal": "net_streak", "subject": "foreign_registered",
             "min_days": 2, "sign": "buy"},
        ]},
        "exit": exit_,
    }


# === 신호 평가 ===

@pytest.mark.unit
class TestSignalEval:
    def test_net_streak_buy(self):
        df = _mk_df([10] * 8,
                    fr_net=[10, 20, 30, -5, 40, 50, 60, 70])
        sig = _eval_signal(df, {"signal": "net_streak",
                                "subject": "foreign_registered",
                                "min_days": 3, "sign": "buy"})
        assert list(sig) == [False, False, True, False,
                             False, False, True, True]

    def test_net_streak_sell(self):
        df = _mk_df([10] * 5, fr_net=[-1, -2, -3, 4, -5])
        sig = _eval_signal(df, {"signal": "net_streak",
                                "subject": "foreign_registered",
                                "min_days": 2, "sign": "sell"})
        assert list(sig) == [False, True, True, False, False]

    def test_pct_threshold(self):
        df = _mk_df([10] * 5, fr_pct=[10, 20, 30, 40, 50])
        sig = _eval_signal(df, {"signal": "pct_threshold",
                                "subject": "foreign_registered",
                                "op": ">=", "value": 30.0})
        assert list(sig) == [False, False, True, True, True]

    def test_pct_delta(self):
        df = _mk_df([10] * 5, fr_pct=[10, 10, 10, 20, 30])
        sig = _eval_signal(df, {"signal": "pct_delta",
                                "subject": "foreign_registered",
                                "window": 2, "op": ">=", "value": 5.0})
        assert list(sig) == [False, False, False, True, True]

    def test_net_vol_ratio(self):
        df = _mk_df([10] * 3, fr_net=[100, 100, 100],
                    volume=[1000, 1000, 1000])
        sig = _eval_signal(df, {"signal": "net_vol_ratio",
                                "subject": "foreign_registered",
                                "window": 2, "op": ">=", "value": 0.05})
        assert list(sig) == [False, True, True]

    def test_price_filter_above_ma(self):
        df = _mk_df([10, 20, 30])
        sig = _eval_signal(df, {"signal": "price_filter",
                                "mode": "above_ma", "window": 2})
        assert list(sig) == [False, True, True]


# === look-ahead 회귀 (핵심) ===

@pytest.mark.unit
class TestNoLookAhead:
    def test_future_change_does_not_affect_past_streak(self):
        df = _mk_df([10] * 10, fr_net=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        sig = _eval_signal(df, {"signal": "net_streak",
                                "subject": "foreign_registered",
                                "min_days": 3, "sign": "buy"})
        df2 = df.copy()
        df2.loc[7, "foreign_registered_net_qty"] = -99999  # 미래 행 변조
        sig2 = _eval_signal(df2, {"signal": "net_streak",
                                  "subject": "foreign_registered",
                                  "min_days": 3, "sign": "buy"})
        assert list(sig.iloc[:7]) == list(sig2.iloc[:7])

    def test_future_change_does_not_affect_past_vol_ratio(self):
        df = _mk_df([10] * 10, fr_net=[100] * 10, volume=[1000] * 10)
        sig = _eval_signal(df, {"signal": "net_vol_ratio",
                                "subject": "foreign_registered",
                                "window": 3, "op": ">=", "value": 0.05})
        df2 = df.copy()
        df2.loc[8, "volume"] = 1
        df2.loc[8, "foreign_registered_net_qty"] = 999999
        sig2 = _eval_signal(df2, {"signal": "net_vol_ratio",
                                  "subject": "foreign_registered",
                                  "window": 3, "op": ">=", "value": 0.05})
        assert list(sig.iloc[:8]) == list(sig2.iloc[:8])


# === 체결 시점 / 청산 트리거 ===

@pytest.mark.unit
class TestSimulateExecution:
    def test_entry_at_next_close(self):
        # streak buy min2 → 신호 첫 True at i=2 → 진입 close[3].
        df = _mk_df([100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        trades, _d, _a, _c = _simulate(_entry_streak_spec(max_hold_days=5), df)
        assert trades, "최소 1 트레이드"
        assert trades[0]["entry_price"] == 103.0
        assert trades[0]["entry_date"] == df["date"].iloc[3]

    def test_stop_loss(self):
        df = _mk_df([100, 100, 100, 100, 100, 94, 93, 93, 93, 93],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        trades, *_ = _simulate(_entry_streak_spec(stop_loss_pct=5.0), df)
        assert trades[0]["exit_reason"] == "stop_loss"
        assert trades[0]["exit_date"] == df["date"].iloc[6]

    def test_take_profit(self):
        df = _mk_df([100, 100, 100, 100, 100, 111, 112, 112, 112, 112],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        trades, *_ = _simulate(_entry_streak_spec(take_profit_pct=10.0), df)
        assert trades[0]["exit_reason"] == "take_profit"
        assert trades[0]["exit_date"] == df["date"].iloc[6]

    def test_max_hold(self):
        df = _mk_df([100] * 10, fr_net=[-1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        trades, *_ = _simulate(_entry_streak_spec(max_hold_days=5), df)
        assert trades[0]["exit_reason"] == "max_hold"

    def test_signal_exit(self):
        df = _mk_df([100] * 10,
                    fr_net=[-1, 10, 10, 10, -5, -6, -7, -8, -9, -10])
        spec = _entry_streak_spec(
            max_hold_days=60,
            signal_all_of=[{"signal": "net_streak",
                            "subject": "foreign_registered",
                            "min_days": 2, "sign": "sell"}],
        )
        trades, *_ = _simulate(spec, df)
        assert trades[0]["exit_reason"] == "signal"

    def test_forced_eod(self):
        df = _mk_df([100] * 8, fr_net=[-1, 10, 10, 10, 10, 10, 10, 10])
        trades, *_ = _simulate(_entry_streak_spec(max_hold_days=60), df)
        assert trades[0]["exit_reason"] == "forced_eod"

    def test_tx_cost_reduces_return(self):
        df = _mk_df([100, 100, 100, 100, 100, 120, 120, 120, 120, 120],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        trades, *_ = _simulate(_entry_streak_spec(take_profit_pct=15.0), df)
        assert trades[0]["net_ret_pct"] < trades[0]["gross_ret_pct"]
        assert TX_COST_ONE_WAY > 0

    def test_single_position_no_overlap(self):
        # 신호 상시 True — 진입은 직전 청산 이후만.
        df = _mk_df([100] * 30, fr_net=[10] * 30)
        trades, *_ = _simulate(_entry_streak_spec(max_hold_days=5), df)
        assert len(trades) >= 2
        for prev, cur in zip(trades, trades[1:]):
            assert cur["entry_date"] > prev["exit_date"]

    def test_close_zero_rows_filtered(self):
        # pre-IPO close=0 행은 제거 후 시뮬 (inf/NaN 방지).
        df = _mk_df([0, 0, 100, 101, 102, 103, 104, 105],
                    fr_net=[0, 0, 10, 10, 10, 10, 10, 10])
        trades, _d, _a, cdf = _simulate(_entry_streak_spec(max_hold_days=3), df)
        assert (cdf["close"] > 0).all()


# === 분할/감자 미조정 점프 가드 ===

@pytest.mark.unit
class TestSplitJumpGuard:
    def test_bad_bar_mask(self):
        m = _bad_bar_mask(np.array([100.0, 100.0, 300.0, 100.0, 100.0]))
        # idx0 기준봉 없음 / 3x up / 0.33x down 둘 다 True
        assert list(m) == [False, False, True, True, False]

    def test_entry_skips_spike_bars(self):
        # streak min2 → 신호 첫 True at i=2 → 진입 후보 close[3]=300(3x 점프).
        # 점프봉(3)과 직후 복귀봉(4)은 진입 보류 → close[5]=100 에서 진입.
        df = _mk_df([100, 100, 100, 300, 100, 100, 100, 100],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10])
        trades, *_ = _simulate(_entry_streak_spec(max_hold_days=3), df)
        assert trades, "정상 봉에서 진입해야 함"
        assert trades[0]["entry_price"] == 100.0

    def test_hold_spike_does_not_trigger_take_profit(self):
        # 보유 중 단발 점프(idx5=300)에 가짜 익절이 잡히면 안 됨.
        df = _mk_df([100, 100, 100, 100, 100, 300, 100, 100, 100, 100],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        trades, daily, _a, _c = _simulate(
            _entry_streak_spec(take_profit_pct=50.0, max_hold_days=20), df)
        assert trades[0]["exit_reason"] != "take_profit"
        # 점프·복귀 봉 일별수익 0 계상 → 비현실적 일별수익 없음.
        assert daily.abs().max() < 0.5

@pytest.mark.unit
class TestMetrics:
    def test_zero_trades(self):
        m = _metrics([], pd.Series([0.0, 0.0, 0.0]))
        assert m["n_trades"] == 0
        assert m["win_rate"] is None
        assert m["sharpe"] is None

    def test_single_trade_sharpe_none(self):
        trades = [{"net_ret_pct": 5.0, "hold_days": 3}]
        m = _metrics(trades, pd.Series([0.0, 0.01, 0.02, -0.005]))
        assert m["n_trades"] == 1
        assert m["sharpe"] is None  # SHARPE_MIN_TRADES=2 미만

    def test_win_rate_and_avg(self):
        trades = [
            {"net_ret_pct": 5.0, "hold_days": 4},
            {"net_ret_pct": -3.0, "hold_days": 2},
        ]
        m = _metrics(trades, pd.Series([0.0, 0.01, -0.005, 0.02, 0.0]))
        assert m["n_trades"] == 2
        assert m["win_rate"] == 0.5
        assert m["avg_net_ret_pct"] == 1.0
        assert m["avg_hold_days"] == 3.0
        assert m["sharpe"] is not None

    def test_mdd_negative(self):
        trades = [
            {"net_ret_pct": 1.0, "hold_days": 1},
            {"net_ret_pct": -1.0, "hold_days": 1},
        ]
        # equity: 1.1, 0.99 → drawdown 발생.
        m = _metrics(trades, pd.Series([0.1, -0.1]))
        assert m["mdd_pct"] < 0


# === KOSPI 초과수익 / 휴장일 보정 ===

@pytest.mark.unit
class TestKospiExcess:
    def test_excess_with_holiday_align(self):
        trades = [{"entry_date": "2024-01-02", "exit_date": "2024-01-10",
                   "net_ret_pct": 5.0}]
        kospi = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-09", "2024-01-11"],
            "close": [100.0, 100.0, 110.0],
        })
        _attach_kospi(trades, kospi)   # exit 2024-01-10 휴장 → 직후 01-11 사용
        assert trades[0]["kospi_ret_pct"] == 10.0
        assert trades[0]["excess_ret_pct"] == -5.0

    def test_no_kospi_yields_none(self):
        trades = [{"entry_date": "2024-01-02", "exit_date": "2024-01-10",
                   "net_ret_pct": 5.0}]
        _attach_kospi(trades, None)
        assert trades[0]["kospi_ret_pct"] is None
        assert trades[0]["excess_ret_pct"] is None


# === 종목 분할 ===

@pytest.mark.unit
class TestSplit:
    def test_deterministic_and_suffix_invariant(self):
        assert split_of("005930") == split_of("005930.KS")
        assert split_of("005930") == split_of("005930")
        assert split_of("005930") in ("in", "out")

    def test_both_buckets_appear(self):
        codes = [f"{i:06d}" for i in range(0, 200000, 1000)]
        buckets = {split_of(c) for c in codes}
        assert buckets == {"in", "out"}


# === 게이트 ===

@pytest.mark.unit
class TestGate:
    def _good(self, **o):
        m = {"n_trades": 60, "win_rate": 0.62, "sharpe": 1.5, "mdd_pct": -15.0}
        m.update(o)
        return m

    def test_single_gate_pass(self):
        assert passes_single_gate(self._good())

    def test_fail_low_win_rate(self):
        assert not passes_single_gate(self._good(win_rate=0.55))

    def test_fail_low_sharpe(self):
        assert not passes_single_gate(self._good(sharpe=1.0))

    def test_fail_deep_drawdown(self):
        assert not passes_single_gate(self._good(mdd_pct=-25.0))

    def test_fail_few_trades(self):
        assert not passes_single_gate(self._good(n_trades=40))

    def test_fail_none_sharpe(self):
        assert not passes_single_gate(self._good(sharpe=None))

    def test_full_gate_pass(self):
        assert passes_full_gate(self._good(win_rate=0.62), self._good(win_rate=0.60))

    def test_full_gate_fail_on_gap(self):
        # 양쪽 단일 게이트는 통과하나 승률 격차 0.15 > 0.10.
        assert not passes_full_gate(self._good(win_rate=0.75),
                                    self._good(win_rate=0.60))


# === run_backtest 통합 (단일 종목) ===

@pytest.mark.unit
class TestRunBacktest:
    def test_returns_metrics_and_trades(self):
        df = _mk_df([100] * 30, fr_net=[10] * 30)
        out = run_backtest(_entry_streak_spec(max_hold_days=5), df)
        assert "metrics" in out and "trades" in out
        assert out["metrics"]["n_trades"] == len(out["trades"])

    def test_invalid_spec_raises(self):
        df = _mk_df([100] * 10, fr_net=[10] * 10)
        bad = _entry_streak_spec()
        bad["entry"]["all_of"][0]["min_days"] = 6  # off-grid
        with pytest.raises(ValueError):
            run_backtest(bad, df)


# === look-ahead 잔여 신호 + 체결가 ===

@pytest.mark.unit
class TestNoLookAheadExtra:
    def test_future_change_does_not_affect_past_pct_delta(self):
        df = _mk_df([10] * 10, fr_pct=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
        sig = _eval_signal(df, {"signal": "pct_delta",
                                "subject": "foreign_registered",
                                "window": 2, "op": ">=", "value": 2.0})
        df2 = df.copy()
        df2.loc[8, "foreign_registered_pct"] = -999  # 미래 행
        sig2 = _eval_signal(df2, {"signal": "pct_delta",
                                  "subject": "foreign_registered",
                                  "window": 2, "op": ">=", "value": 2.0})
        assert list(sig.iloc[:8]) == list(sig2.iloc[:8])

    def test_future_change_does_not_affect_past_price_filter(self):
        df = _mk_df([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        sig = _eval_signal(df, {"signal": "price_filter",
                                "mode": "above_ma", "window": 3})
        df2 = df.copy()
        df2.loc[8, "close"] = 0.01  # 미래 행
        sig2 = _eval_signal(df2, {"signal": "price_filter",
                                  "mode": "above_ma", "window": 3})
        assert list(sig.iloc[:8]) == list(sig2.iloc[:8])

    def test_exit_price_is_next_close_not_trigger_close(self):
        # 트리거 봉 j=5(close=93) 의 종가가 아니라 익일 j+1=6(close=88) 로 체결.
        df = _mk_df([100, 100, 100, 100, 100, 93, 88, 88],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10])
        trades, *_ = _simulate(_entry_streak_spec(stop_loss_pct=5.0), df)
        assert trades[0]["exit_reason"] == "stop_loss"
        assert trades[0]["exit_date"] == df["date"].iloc[6]
        assert trades[0]["exit_price"] == 88.0


# === tx 곱셈 정합 (per-trade net ↔ 일별 equity) ===

@pytest.mark.unit
class TestTxReconciliation:
    def test_daily_equity_reconciles_with_per_trade_product(self):
        df = _mk_df([100, 100, 100, 100, 105, 110, 121, 118, 125, 130],
                    fr_net=[-1] + [10] * 9)
        trades, daily, _a, _c = _simulate(
            _entry_streak_spec(max_hold_days=5, stop_loss_pct=8.0,
                               take_profit_pct=15.0), df)
        assert trades, "최소 1 트레이드"
        equity = float(np.prod(1.0 + daily.to_numpy()))
        per_trade = 1.0
        for t in trades:
            per_trade *= (1 + t["net_ret_pct"] / 100.0)
        # 일별 net 곱 == per-trade net 곱 (round 4자리 누적 오차만)
        assert abs(equity - per_trade) < 1e-3

    def test_one_day_hold_single_cell_equals_net(self):
        # 진입 e=3, exit signal(pct_threshold) 이 진입 봉에서 즉발 → 1일 보유.
        # first==exit_idx 라 같은 daily 셀에 entry/exit tx 두 레그가 곱해진다.
        df = _mk_df([100, 100, 100, 100, 108, 108, 108, 108],
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, 10],
                    fr_pct=[0, 0, 0, 50, 50, 50, 50, 50])
        spec = {
            "spec_version": 1, "name": "t", "direction": "long",
            "entry": {"all_of": [
                {"signal": "net_streak", "subject": "foreign_registered",
                 "min_days": 2, "sign": "buy"},
            ]},
            "exit": {
                "signal_all_of": [
                    {"signal": "pct_threshold", "subject": "foreign_registered",
                     "op": ">=", "value": 10.0},
                ],
                "max_hold_days": 20,
            },
        }
        trades, daily, _a, _c = _simulate(spec, df)
        assert trades[0]["hold_days"] == 1
        assert trades[0]["exit_reason"] == "signal"
        assert daily.iloc[4] == pytest.approx(trades[0]["net_ret_pct"] / 100.0, abs=1e-4)


# === 청산 우선순위 (stop > take > signal > max_hold) ===

@pytest.mark.unit
class TestExitPriority:
    def _sell_exit(self):
        return [{"signal": "net_streak", "subject": "foreign_registered",
                 "min_days": 2, "sign": "sell"}]

    def test_stop_beats_signal(self):
        # j=5 에서 stop(-7%) 과 sell-streak signal 동시 → stop 우선.
        df = _mk_df([100, 100, 100, 100, 100, 93, 93, 93],
                    fr_net=[-1, 10, 10, 10, -5, -6, -7, -8])
        trades, *_ = _simulate(
            _entry_streak_spec(stop_loss_pct=5.0, max_hold_days=60,
                               signal_all_of=self._sell_exit()), df)
        assert trades[0]["exit_reason"] == "stop_loss"

    def test_take_beats_signal(self):
        # j=5 에서 take(+11%) 과 sell-streak signal 동시 → take 우선.
        df = _mk_df([100, 100, 100, 100, 100, 111, 111, 111],
                    fr_net=[-1, 10, 10, 10, -5, -6, -7, -8])
        trades, *_ = _simulate(
            _entry_streak_spec(take_profit_pct=10.0, max_hold_days=60,
                               signal_all_of=self._sell_exit()), df)
        assert trades[0]["exit_reason"] == "take_profit"

    def test_signal_beats_max_hold(self):
        # j=8 에서 held==5(max_hold) 과 sell-streak signal 동시 → signal 우선.
        df = _mk_df([100] * 10,
                    fr_net=[-1, 10, 10, 10, 10, 10, 10, -5, -6, -7])
        trades, *_ = _simulate(
            _entry_streak_spec(max_hold_days=5, signal_all_of=self._sell_exit()), df)
        assert trades[0]["exit_reason"] == "signal"


# === _combine 포트폴리오 결합 ===

@pytest.mark.unit
class TestCombine:
    def test_active_equal_weight_with_date_union(self):
        rA = pd.Series([0.0, 0.10, 0.20, 0.0],
                       index=["d1", "d2", "d3", "d4"], name="A")
        aA = pd.Series([False, True, True, False],
                       index=["d1", "d2", "d3", "d4"], name="A")
        rB = pd.Series([0.0, 0.30, 0.0],
                       index=["d2", "d3", "d4"], name="B")   # d1 부재
        aB = pd.Series([False, True, False],
                       index=["d2", "d3", "d4"], name="B")
        port = _combine([rA, rB], [aA, aB])
        assert port.loc["d1"] == 0.0          # 둘 다 비active/부재
        assert port.loc["d2"] == pytest.approx(0.10)   # A 만 active
        assert port.loc["d3"] == pytest.approx(0.25)   # (0.20+0.30)/2
        assert port.loc["d4"] == 0.0          # 둘 다 비active

    def test_empty_returns_empty(self):
        assert _combine([], []).empty


# === _simulate 경계 ===

@pytest.mark.unit
class TestSimulateBoundaries:
    def test_too_short_no_trades(self):
        df = _mk_df([100, 100], fr_net=[10, 10])
        trades, daily, active, cdf = _simulate(_entry_streak_spec(), df)
        assert trades == []
        assert len(daily) == 2 and len(active) == 2

    def test_signal_at_last_minus_one_no_entry(self):
        # 신호 첫 True 가 i==n-2 (e=n-1, 보유 봉 없음) → 진입 생략.
        df = _mk_df([100] * 5, fr_net=[-1, -1, -1, 10, 10])
        # buy streak min2 첫 True at i=4 (=n-1) — i<n-2(=3) 루프 밖 → 거래 0.
        trades, *_ = _simulate(_entry_streak_spec(max_hold_days=5), df)
        assert trades == []


# === run_universe_backtest 통합 (fake loader 주입) ===

@pytest.mark.unit
class TestRunUniverse:
    def _loader_factory(self, dfs):
        def loader(ticker):
            return dfs.get(ticker, pd.DataFrame()), {}
        return loader

    def test_split_aggregation_and_shape(self):
        # 005930/000660 = in, 035720/051910 = out (split_of 로 확인됨).
        tickers = ["005930", "000660", "035720", "051910"]
        dfs = {tk: _mk_df([100] * 30, fr_net=[10] * 30) for tk in tickers}
        res = run_universe_backtest(
            _entry_streak_spec(max_hold_days=5), tickers,
            loader=self._loader_factory(dfs), kospi_fetcher=None)
        assert res["universe_size"] == 4
        assert res["n_in"] == 2 and res["n_out"] == 2
        assert res["n_in"] + res["n_out"] == res["universe_size"]
        for split in ("in_sample", "out_sample"):
            assert {"win_rate", "sharpe", "mdd_pct", "n_trades"} <= set(res[split])
            assert res[split]["n_trades"] > 0   # 양쪽 모두 거래 발생
        assert isinstance(res["gate_passed"], bool)
        assert res["gate_passed"] is False      # 평탄가 → tx 손실, 게이트 미달

    def test_empty_df_excluded_from_universe(self):
        tickers = ["005930", "000660", "035720"]
        dfs = {"005930": _mk_df([100] * 30, fr_net=[10] * 30),
               "000660": _mk_df([100] * 30, fr_net=[10] * 30)}  # 035720 결손
        res = run_universe_backtest(
            _entry_streak_spec(max_hold_days=5), tickers,
            loader=self._loader_factory(dfs), kospi_fetcher=None)
        assert res["universe_size"] == 2   # 035720 제외

    def test_kospi_fetcher_exception_falls_back(self):
        tickers = ["005930", "035720"]
        dfs = {tk: _mk_df([100] * 30, fr_net=[10] * 30) for tk in tickers}

        def boom(start, end):
            raise RuntimeError("network down")

        res = run_universe_backtest(
            _entry_streak_spec(max_hold_days=5), tickers,
            loader=self._loader_factory(dfs), kospi_fetcher=boom)
        # 예외 폴백 → 메트릭은 계산되되 초과수익은 None.
        assert res["in_sample"]["avg_excess_ret_pct"] is None


# === 추가 게이트/KOSPI/타입 엣지 ===

@pytest.mark.unit
class TestGateMore:
    def _good(self, **o):
        m = {"n_trades": 60, "win_rate": 0.62, "sharpe": 1.5, "mdd_pct": -15.0}
        m.update(o)
        return m

    def test_full_gate_fails_when_one_split_sharpe_none(self):
        assert not passes_full_gate(self._good(), self._good(sharpe=None))

    def test_full_gate_fails_when_out_below(self):
        assert not passes_full_gate(self._good(), self._good(win_rate=0.50))


@pytest.mark.unit
class TestKospiExcessEdge:
    def test_exit_beyond_kospi_range_yields_none(self):
        trades = [{"entry_date": "2024-01-02", "exit_date": "2024-12-31",
                   "net_ret_pct": 5.0}]
        kospi = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"],
                              "close": [100.0, 101.0]})
        _attach_kospi(trades, kospi)   # exit 가 kospi 범위 밖 → None
        assert trades[0]["kospi_ret_pct"] is None
        assert trades[0]["excess_ret_pct"] is None


@pytest.mark.unit
class TestMetricTypes:
    def test_trade_and_metric_floats_are_python_float(self):
        df = _mk_df([100, 100, 100, 100, 120, 120, 120, 120, 120, 120],
                    fr_net=[-1] + [10] * 9)
        out = run_backtest(_entry_streak_spec(take_profit_pct=15.0), df)
        t = out["trades"][0]
        assert type(t["net_ret_pct"]) is float
        assert type(t["gross_ret_pct"]) is float
        assert type(out["metrics"]["avg_net_ret_pct"]) is float
