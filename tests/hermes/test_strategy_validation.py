"""strategy_validation 단위 테스트 — 시간 분할 경계·embargo·MinBTL·엔진버전.

실데이터 없이 fake loader 로 합성 holdings 를 주입한다(엔진의 loader 주입 패턴).
price_drop 진입만 쓰므로 close 컬럼만 있으면 되고 KOSPI 도 불필요.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.hermes import strategy_validation as V


def _mk_df(n: int, start: str = "2016-01-01") -> pd.DataFrame:
    """진동하는 종가 n일치 — price_drop 진입이 양쪽 시간구간에서 발생하도록."""
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    closes = [100.0 + 25.0 * math.sin(i / 7.0) for i in range(n)]
    return pd.DataFrame({"date": list(dates), "close": closes,
                         "volume": [1000.0] * n})


def _loader_factory(n: int):
    df = _mk_df(n)
    return lambda tk: (df.copy(), {})


def _no_kospi(_s, _e):
    return None


def _price_drop_spec(max_hold: int = 10) -> dict:
    return {
        "spec_version": 2, "name": "vtest", "direction": "long",
        "entry": {"all_of": [{"signal": "price_drop", "window": 5, "value": 5}]},
        "exit": {"max_hold_days": max_hold},
    }


@pytest.mark.unit
class TestTimeSplit:
    def test_returns_expected_shape(self):
        r = V.run_time_split_validation(
            _price_drop_spec(), ["A", "B"],
            loader=_loader_factory(400), kospi_fetcher=_no_kospi)
        assert set(r) >= {"in_sample", "out_sample", "gate_passed",
                          "universe_size", "split"}
        assert r["universe_size"] == 2
        assert isinstance(r["gate_passed"], bool)
        for m in (r["in_sample"], r["out_sample"]):
            assert set(m) >= {"n_trades", "win_rate", "sharpe", "mdd_pct"}

    def test_cutoff_is_in_sample_pct_quantile(self):
        n = 400
        r = V.run_time_split_validation(
            _price_drop_spec(), ["A"],
            loader=_loader_factory(n), kospi_fetcher=_no_kospi,
            in_sample_pct=70, embargo_days=12)
        dates = list(pd.bdate_range("2016-01-01", periods=n).strftime("%Y-%m-%d"))
        assert r["split"]["cutoff"] == dates[int(n * 70 / 100)]

    def test_oos_starts_after_embargo_gap(self):
        n, emb = 400, 12
        r = V.run_time_split_validation(
            _price_drop_spec(), ["A"],
            loader=_loader_factory(n), kospi_fetcher=_no_kospi,
            embargo_days=emb)
        dates = list(pd.bdate_range("2016-01-01", periods=n).strftime("%Y-%m-%d"))
        cut_idx = dates.index(r["split"]["cutoff"])
        # oos_start 는 cutoff 보다 정확히 embargo 거래일 뒤
        assert r["split"]["oos_start"] == dates[cut_idx + emb]
        assert r["split"]["oos_start"] > r["split"]["cutoff"]

    def test_larger_embargo_pushes_oos_later(self):
        kw = dict(loader=_loader_factory(400), kospi_fetcher=_no_kospi)
        small = V.run_time_split_validation(_price_drop_spec(), ["A"], embargo_days=5, **kw)
        large = V.run_time_split_validation(_price_drop_spec(), ["A"], embargo_days=40, **kw)
        assert large["split"]["oos_start"] > small["split"]["oos_start"]

    def test_windows_do_not_overlap(self):
        r = V.run_time_split_validation(
            _price_drop_spec(), ["A"],
            loader=_loader_factory(400), kospi_fetcher=_no_kospi)
        # IS 윈도우 끝(cutoff) <= OOS 윈도우 시작(oos_start) — 겹침 없음
        assert r["in_sample"]["_window"][1] == r["split"]["cutoff"]
        assert r["out_sample"]["_window"][0] == r["split"]["oos_start"]
        assert r["in_sample"]["_window"][1] <= r["out_sample"]["_window"][0]

    def test_too_few_dates_raises(self):
        with pytest.raises(ValueError):
            V.run_time_split_validation(
                _price_drop_spec(), ["A"],
                loader=_loader_factory(5), kospi_fetcher=_no_kospi)


@pytest.mark.unit
class TestWalkForward:
    def test_multiple_windows_and_shape(self):
        r = V.run_walk_forward_validation(
            _price_drop_spec(), ["A"],
            loader=_loader_factory(1500), kospi_fetcher=_no_kospi,
            is_years=2.0, oos_years=1.0, mode="rolling")
        assert r["n_windows"] >= 2
        assert isinstance(r["gate_passed"], bool)
        for w in r["windows"]:
            assert w["is_window"][1] <= w["oos_window"][0]  # IS 끝 <= OOS 시작


@pytest.mark.unit
class TestMinBTL:
    def test_expected_max_sharpe_monotonic(self):
        vals = [V.expected_max_sharpe(n) for n in (2, 10, 45, 100, 500)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_expected_max_sharpe_guard(self):
        assert V.expected_max_sharpe(1) == 0.0

    def test_expected_max_sharpe_ballpark(self):
        # 표준화(sr_std=1) 가정에서 N=45 는 대략 2.2 부근
        assert 2.0 < V.expected_max_sharpe(45) < 2.5

    def test_max_justifiable_trials_formula(self):
        assert V.max_justifiable_trials(5, 1.0) == int(math.exp(2.5))    # 12
        assert V.max_justifiable_trials(10, 1.0) == int(math.exp(5.0))   # 148

    def test_max_justifiable_trials_guards(self):
        assert V.max_justifiable_trials(0, 1.0) == 0
        assert V.max_justifiable_trials(5, 0.0) == 0


@pytest.mark.unit
class TestEngineVersion:
    def test_returns_8char_hex_stable(self):
        v1 = V.engine_version()
        v2 = V.engine_version()
        assert v1 == v2
        assert len(v1) == 8
        int(v1, 16)  # hex 파싱 가능
