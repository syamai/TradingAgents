"""advanced_analysis 6 종 — 정상 동작 + 산출 형식 검증.

수치 정확도(statsmodels/sklearn의 알고리즘 자체)는 검증 범위 밖.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.advanced_analysis import (
    KEY_SUBJECTS, adf_test, cointegration_test, compute_advanced_report,
    granger_test, mutual_info, render_advanced_markdown, rolling_correlation,
    rolling_correlation_summary, var_irf,
)
from tradingagents.dataflows.kis_holdings import SUBS_10, compute_holdings


def _zero_subs() -> dict:
    return {f"{s}_qty": 0 for s in SUBS_10}


def _investor_row(date: str, close: int, **overrides) -> dict:
    base = {"date": date, "close": close, **_zero_subs()}
    base.update(overrides)
    return base


def _sample(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """retail 강한 역행(r≈-1), pension 양수 시계열."""
    rng = np.random.default_rng(seed)
    close = 100.0
    rows: list[dict] = []
    for i in range(n):
        d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        delta = rng.normal(0, 1.5)
        new_close = max(close + delta, 1.0)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "close": new_close,
            "volume": 1000,
            **{f"{s}_qty": 0 for s in SUBS_10},
            "retail_qty": int(round(-100 * delta)),
            "pension_qty": int(round(50 * delta)),
        })
        close = new_close
    return compute_holdings(pd.DataFrame(rows))


# === 개별 함수 ===

@pytest.mark.unit
class TestAdf:
    def test_random_walk_is_nonstationary(self):
        rng = np.random.default_rng(0)
        rw = pd.Series(rng.normal(0, 1, 200).cumsum())
        res = adf_test(rw)
        assert res["adf_stat"] is not None
        assert res["is_stationary"] is False

    def test_iid_normal_is_stationary(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 300))
        res = adf_test(s)
        assert res["is_stationary"] is True

    def test_too_short_safe(self):
        res = adf_test(pd.Series([1, 2, 3]))
        assert res["is_stationary"] is False
        assert res["adf_stat"] is None


@pytest.mark.unit
class TestGranger:
    def test_shape_matches_max_lag(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 200))
        y = pd.Series(rng.normal(0, 1, 200))
        rows = granger_test(y, x, max_lag=4)
        assert len(rows) == 4
        for r in rows:
            assert {"lag", "f_stat", "p_value", "significant"}.issubset(r)

    def test_too_short_returns_empty(self):
        rows = granger_test(pd.Series([1, 2, 3]), pd.Series([4, 5, 6]),
                            max_lag=2)
        assert rows == []


@pytest.mark.unit
class TestVarIrf:
    def test_returns_irf_for_valid_input(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(0, 1, 200),
        })
        res = var_irf(df, ["a", "b"], horizon=5)
        assert res["order"] is not None
        # 4 trace (a->a, a->b, b->a, b->b)
        assert set(res["irf_cum"].keys()) == {"a->a", "a->b", "b->a", "b->b"}
        for vals in res["irf_cum"].values():
            assert len(vals) == 6  # horizon+1

    def test_too_short_returns_none_order(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        res = var_irf(df, ["a", "b"])
        assert res["order"] is None


@pytest.mark.unit
class TestCointegration:
    def test_random_pairs_likely_not_cointegrated(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 100).cumsum())
        y = pd.Series(rng.normal(0, 1, 100).cumsum())
        res = cointegration_test(x, y)
        # 두 독립 random walk → 보통 공적분 아님 (확률적)
        assert res["score"] is not None
        assert isinstance(res["is_cointegrated"], bool)

    def test_cointegrated_pair(self):
        # y = x + 작은 노이즈 → 강한 공적분
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 200).cumsum())
        y = x + rng.normal(0, 0.1, 200)
        res = cointegration_test(y, x)
        assert res["is_cointegrated"] is True


@pytest.mark.unit
class TestMutualInfo:
    def test_independent_low_mi(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 200))
        y = pd.Series(rng.normal(0, 1, 200))
        mi = mutual_info(x, y)
        assert mi >= 0
        assert mi < 0.5

    def test_identical_high_mi(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 200))
        mi_self = mutual_info(x, x)
        mi_indep = mutual_info(x, pd.Series(rng.normal(0, 1, 200)))
        assert mi_self > mi_indep


@pytest.mark.unit
class TestRolling:
    def test_summary_keys(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 200))
        y = pd.Series(rng.normal(0, 1, 200))
        s = rolling_correlation_summary(x, y, window=30)
        assert {"mean", "std", "min", "max", "p10", "p90"}.issubset(s)
        assert s["n"] > 0


# === 통합 ===

@pytest.mark.unit
class TestComputeAdvancedReport:
    def test_full_pipeline(self):
        df = _sample(200)
        adv = compute_advanced_report(df)
        # 모든 6 섹션 존재
        assert len(adv["adf"]) >= 2 + 2 * len(KEY_SUBJECTS)  # return+close + 주체×2
        assert len(adv["granger"]) == len(KEY_SUBJECTS)
        assert adv["var_irf"] is not None
        assert len(adv["cointegration"]) == len(KEY_SUBJECTS)
        assert len(adv["mutual_info"]) == len(KEY_SUBJECTS)
        assert len(adv["rolling_r"]) == len(KEY_SUBJECTS)

    def test_empty_input_safe(self):
        adv = compute_advanced_report(pd.DataFrame())
        assert adv["adf"] == []
        assert adv["var_irf"] is None

    def test_markdown_includes_all_sections(self):
        df = _sample(200)
        adv = compute_advanced_report(df)
        md = render_advanced_markdown(adv)
        for h in ["7-1. ADF", "7-2. Granger", "7-3. VAR",
                  "7-4. Cointegration", "7-5. Mutual",
                  "7-6. Rolling", "구현하지 않은 4 종"]:
            assert h in md
