"""공정 게이트 v3 — 시장대비 초과수익(IR) 메트릭 + walk-forward 게이트 + 저장.

감사 결론(원수익 long-only 게이트가 강세장 베타를 알파로 오인)을 고친 변경의
회귀 가드. 핵심 불변식:
  - 순수 베타(시장만 따라감) → 초과수익 시계열 ≈ 0 → 게이트 탈락.
  - 시장 + 알파 → 초과수익 IR > 0.
  - walk-forward 게이트가 raw sharpe 가 아니라 초과수익 IR 로 판정.
  - 저장 시 gate_passed 는 walk-forward 공정 게이트 단일 출처(v4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes import strategy_validation as V
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2


def _frames(alpha_per_day: float, n: int = 320, n_stocks: int = 6,
            active_p: float = 0.5, seed: int = 3):
    """합성 (ret_frames, act_frames, kospi). 활성일 종목수익 = 시장수익 + alpha."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d").to_numpy()
    mclose = 2000 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    kospi = pd.DataFrame({"date": dates, "close": mclose})
    m = pd.Series(mclose, index=dates).pct_change().fillna(0.0).to_numpy()
    rets, acts = [], []
    for s in range(n_stocks):
        active = rng.random(n) < active_p
        r = np.where(active, m + alpha_per_day, 0.0)
        rets.append(pd.Series(r, index=dates, name=f"s{s}"))
        acts.append(pd.Series(active, index=dates, name=f"s{s}"))
    return rets, acts, kospi


_TRADES = [{"net_ret_pct": 1.0, "hold_days": 3, "excess_ret_pct": 0.5}] * 60


@pytest.mark.unit
class TestExcessMetric:
    def test_metrics_adds_excess_fields(self):
        rets, acts, kospi = _frames(0.0)
        port = bt._combine_korea_stock_portfolio(rets, acts)
        excess = bt._excess_daily_korea(port, acts, kospi)
        m = bt._metrics(_TRADES, port, excess_daily=excess)
        assert {"excess_sharpe", "excess_cum_return_pct", "excess_mdd_pct", "daily_excess_vol_pct"} <= set(m)

    def test_metrics_omits_excess_when_not_given(self):
        m = bt._metrics(_TRADES, pd.Series([0.0, 0.01, -0.01]))
        assert "excess_sharpe" not in m

    def test_empty_trades_with_excess_has_none_ir(self):
        m = bt._metrics([], pd.Series([0.0, 0.0]), excess_daily=pd.Series([0.0, 0.0]))
        assert m["n_trades"] == 0 and m["excess_sharpe"] is None
        assert m["daily_excess_vol_pct"] == 0.0

    def test_daily_excess_vol_is_daily_standard_deviation_pct(self):
        excess = pd.Series([0.01, -0.01, 0.02, 0.0])
        m = bt._metrics(_TRADES, pd.Series([0.0, 0.0, 0.0, 0.0]), excess_daily=excess)
        assert m["daily_excess_vol_pct"] == round(float(excess.to_numpy().std(ddof=0)) * 100, 4)

    def test_pure_beta_excess_near_zero(self):
        # 시장만 따라가면(종목수익==시장수익) 초과수익은 항등적으로 상쇄돼 0.
        rets, acts, kospi = _frames(0.0)
        port = bt._combine_korea_stock_portfolio(rets, acts)
        excess = bt._excess_daily_korea(port, acts, kospi)
        assert float(np.abs(excess.to_numpy()).max()) < 1e-8

    def test_alpha_excess_is_positive(self):
        rets, acts, kospi = _frames(0.0010)   # 하루 +0.10%p 시장초과
        port = bt._combine_korea_stock_portfolio(rets, acts)
        excess = bt._excess_daily_korea(port, acts, kospi)
        m = bt._metrics(_TRADES, port, excess_daily=excess)
        assert m["excess_sharpe"] is not None and m["excess_sharpe"] > 0

    def test_no_kospi_excess_equals_raw(self):
        # 시장 데이터 결손 시 초과수익 = raw 수익(보수적). 게이트가 베타를 못 빼면
        # 못 빼는 대로 명시적으로 raw 로 떨어진다.
        rets, acts, _ = _frames(0.0010)
        port = bt._combine_korea_stock_portfolio(rets, acts)
        excess = bt._excess_daily_korea(port, acts, None)
        assert np.allclose(excess.to_numpy(), port.to_numpy())

    def test_invested_weight_capped_at_stock_weight(self):
        # active 종목 100개여도 투입비중은 stock_weight(0.90) 상한.
        acts = [pd.Series([True, True], index=["d1", "d2"], name=f"s{i}")
                for i in range(100)]
        w = bt._invested_weight_korea(acts)
        assert float(w.max()) <= bt.KOREA_STOCK_PORTFOLIO_POLICY["stock_weight"] + 1e-12

    def test_market_returns_no_lookahead(self):
        # m[t] = close[t]/close[t-1]-1 — 미래 종가 미참조. 첫날은 0.
        idx = ["2020-01-01", "2020-01-02", "2020-01-03"]
        kospi = pd.DataFrame({"date": idx, "close": [100.0, 110.0, 99.0]})
        m = bt._market_daily_returns(kospi, idx)
        assert m.iloc[0] == 0.0
        assert abs(m.iloc[1] - 0.10) < 1e-12
        assert abs(m.iloc[2] - (-0.10)) < 1e-12


def _price_drop_spec(max_hold: int = 10) -> dict:
    return {
        "spec_version": 2, "name": "vtest", "direction": "long",
        "entry": {"all_of": [{"signal": "price_drop", "window": 5, "value": 5}]},
        "exit": {"max_hold_days": max_hold},
    }


def _osc_loader(n: int):
    import math
    dates = pd.bdate_range("2016-01-01", periods=n).strftime("%Y-%m-%d")
    closes = [100.0 + 25.0 * math.sin(i / 7.0) for i in range(n)]
    df = pd.DataFrame({"date": list(dates), "close": closes, "volume": [1000.0] * n})
    return lambda tk: (df.copy(), {})


@pytest.mark.unit
class TestWalkForwardExcessGate:
    def test_gate_keyed_on_excess_ir(self):
        # 구 KOSPI 벤치마크 경로(benchmark='kospi') — 임계 GATE_EXCESS_MIN_IR.
        r = V.run_walk_forward_validation(
            _price_drop_spec(), ["A"],
            loader=_osc_loader(1500), kospi_fetcher=lambda _s, _e: None,
            is_years=2.0, oos_years=1.0, mode="rolling", benchmark="kospi")
        assert r["gate_metric"] == "excess_ir"
        assert r["benchmark"] == "kospi"
        assert r["gate_min_ir"] == V.GATE_EXCESS_MIN_IR
        assert {"oos_excess_ir_median", "oos_excess_ir_min"} <= set(r)
        # raw sharpe 는 진단용으로 함께 보고.
        assert {"oos_sharpe_median", "oos_sharpe_min"} <= set(r)
        assert isinstance(r["gate_passed"], bool)

    def test_default_benchmark_is_survivor_pool(self):
        # 신 게이트: 기본 벤치마크=생존풀(생존편향 중립), 임계 GATE_POOL_MIN_IR.
        r = V.run_walk_forward_validation(
            _price_drop_spec(), ["A"],
            loader=_osc_loader(1500), kospi_fetcher=lambda _s, _e: None,
            is_years=2.0, oos_years=1.0, mode="rolling")
        assert r["benchmark"] == "pool"
        assert r["gate_min_ir"] == V.GATE_POOL_MIN_IR
        for w in r["windows"]:
            assert "excess_sharpe" in w["out_sample"]


def _metric_dict(sharpe, win, n=100, mdd=-5.0):
    return {
        "n_trades": n, "win_rate": win, "avg_net_ret_pct": 1.0,
        "avg_hold_days": 3.0, "cum_return_pct": 10.0, "sharpe": sharpe,
        "mdd_pct": mdd, "avg_excess_ret_pct": 0.2,
        "excess_sharpe": 0.1, "excess_cum_return_pct": 1.0, "excess_mdd_pct": -2.0,
    }


@pytest.mark.unit
class TestStoreV4:
    def _spec(self):
        return _price_drop_spec()

    def test_v4_columns_migrated(self, tmp_path):
        store = StrategyStoreV2(root=tmp_path)
        with store._conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(strategies)")}
        assert {"wf_excess_ir_median", "wf_excess_ir_min", "wf_n_windows",
                "wf_gate_passed", "wf_result_json"} <= cols

    def test_wf_pass_alone_does_not_pass_without_strict_xsec(self, tmp_path):
        store = StrategyStoreV2(root=tmp_path)
        result = {"in_sample": _metric_dict(0.3, 0.45),
                  "out_sample": _metric_dict(0.2, 0.43),
                  "gate_passed": False, "universe_size": 150}
        wf_pass = {"gate_passed": True, "oos_excess_ir_median": 0.8,
                   "oos_excess_ir_min": 0.6, "n_windows": 5}
        sid, _ = store.save(self._spec(), result, name="x", wf_result=wf_pass,
                            engine_version="testeng1")
        with store._conn() as c:
            r = c.execute("SELECT gate_passed, wf_gate_passed, wf_excess_ir_median,"
                          " wf_excess_ir_min, wf_n_windows FROM strategies WHERE id=?",
                          (sid,)).fetchone()
        # gate_passed 는 WF 통과와 xsec 품질(Sharpe/MDD/거래수)의 AND. 승률은 제외.
        assert r["gate_passed"] == 0
        assert r["wf_gate_passed"] == 1
        assert abs(r["wf_excess_ir_median"] - 0.8) < 1e-9
        assert abs(r["wf_excess_ir_min"] - 0.6) < 1e-9
        assert r["wf_n_windows"] == 5

    def test_strict_gate_ignores_win_rate_threshold(self, tmp_path):
        store = StrategyStoreV2(root=tmp_path)
        # 승률은 50% 미만이지만 Sharpe/MDD/거래수 + WF 가 모두 통과하면 통과.
        result = {"in_sample": _metric_dict(1.2, 0.45, n=100, mdd=-10.0),
                  "out_sample": _metric_dict(1.1, 0.44, n=80, mdd=-8.0),
                  "gate_passed": False, "universe_size": 150}
        wf_pass = {"gate_passed": True, "oos_excess_ir_median": 0.8,
                   "oos_excess_ir_min": 0.2, "n_windows": 5}
        sid, _ = store.save(self._spec(), result, name="winrate_ignored", wf_result=wf_pass,
                            engine_version="testeng1")
        with store._conn() as c:
            r = c.execute("SELECT gate_passed, wf_gate_passed FROM strategies WHERE id=?",
                          (sid,)).fetchone()
        assert r["gate_passed"] == 1
        assert r["wf_gate_passed"] == 1

    def test_wf_fail_records_gate_zero(self, tmp_path):
        store = StrategyStoreV2(root=tmp_path)
        result = {"in_sample": _metric_dict(1.5, 0.7),
                  "out_sample": _metric_dict(1.4, 0.7),
                  "gate_passed": True, "universe_size": 150}   # xsec 통과여도
        wf_fail = {"gate_passed": False, "oos_excess_ir_median": 0.1,
                   "oos_excess_ir_min": -0.3, "n_windows": 5}
        sid, _ = store.save(self._spec(), result, name="y", wf_result=wf_fail,
                            engine_version="testeng1")
        with store._conn() as c:
            r = c.execute("SELECT gate_passed FROM strategies WHERE id=?",
                          (sid,)).fetchone()
        assert r["gate_passed"] == 0   # wf 가 막으면 xsec 통과여도 탈락
