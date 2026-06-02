"""advanced_analysis 6 종 — 정상 동작 + 산출 형식 검증.

수치 정확도(statsmodels/sklearn의 알고리즘 자체)는 검증 범위 밖.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.advanced_analysis import (
    KEY_SUBJECTS, adf_test, cointegration_test, compute_advanced_report,
    granger_test, mutual_info, pnl_attribution, render_advanced_markdown,
    rolling_correlation, rolling_correlation_summary, var_irf,
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


@pytest.mark.unit
class TestPnlAttribution:
    def _df_with_amounts(self) -> pd.DataFrame:
        # 3 거래일, retail 만 거래: +100주@100, 무거래, -50주@120
        rows = [
            _investor_row("2024-01-01", 100, retail_qty=100),
            _investor_row("2024-01-02", 110, retail_qty=0),
            _investor_row("2024-01-03", 120, retail_qty=-50),
        ]
        df = compute_holdings(pd.DataFrame(rows))
        # 거래대금 병합 — KEY_SUBJECTS 5 주체 컬럼 모두 있어야 계산됨
        for s in KEY_SUBJECTS:
            df[f"{s}_amount"] = 0.0
        df.loc[0, "retail_amount"] = 100 * 100    # +100주 @100
        df.loc[2, "retail_amount"] = -50 * 120    # -50주 @120
        return df

    def test_buy_sell_split_avg(self):
        # 순매수일: +100주@100 (amt 10000) / 순매도일: -50주@120 (amt -6000)
        by = {r["subject"]: r for r in pnl_attribution(self._df_with_amounts())}
        retail = by["retail"]
        assert retail["buy_qty"] == 100
        assert retail["buy_avg"] == 100.0           # 10000 / 100
        assert retail["sell_qty"] == 50             # 절댓값 표기
        assert retail["sell_avg"] == 120.0          # -6000 / -50
        assert retail["net_qty"] == 50              # 100 - 50
        assert retail["last_close"] == 120.0
        # pnl = 120*50 - (10000-6000) = 2000
        assert retail["pnl"] == pytest.approx(2000.0)
        # 무거래 주체 → 양 방향 None, pnl 0
        assert by["pension"]["buy_avg"] is None
        assert by["pension"]["sell_avg"] is None
        assert by["pension"]["pnl"] == pytest.approx(0.0)

    def test_realized_unrealized_split(self):
        # matched=min(100,50)=50주 왕복. 실현 = 50×(120−100)=1000.
        # 남은 순매수 50주 미실현 = 50×(120−100)=1000. 합 = pnl 2000.
        retail = {r["subject"]: r
                  for r in pnl_attribution(self._df_with_amounts())}["retail"]
        assert retail["realized_pnl"] == pytest.approx(1000.0)
        assert retail["unrealized_pnl"] == pytest.approx(1000.0)
        assert retail["unrealized_kind"] == "평가"      # net 순매수 → 보유 평가
        assert retail["realized_pnl"] + retail["unrealized_pnl"] == pytest.approx(
            retail["pnl"]
        )

    def test_net_seller_unrealized_is_opportunity(self):
        # 순매도 초과(net_qty<0) → 미실현은 보유 평가가 아니라 기회손익
        rows = [_investor_row("2024-01-01", 100, retail_qty=30),
                _investor_row("2024-01-02", 120, retail_qty=-100)]
        df = compute_holdings(pd.DataFrame(rows))
        for s in KEY_SUBJECTS:
            df[f"{s}_amount"] = 0.0
        df.loc[0, "retail_amount"] = 30 * 100         # 매수 30주 @100
        df.loc[1, "retail_amount"] = -100 * 120       # 매도 100주 @120
        retail = {r["subject"]: r for r in pnl_attribution(df)}["retail"]
        assert retail["net_qty"] == -70               # 30 - 100
        assert retail["unrealized_kind"] == "기회"

    def test_only_buy_is_all_unrealized(self):
        # 매도 없이 매수만 → 실현 0, 전부 미실현
        rows = [_investor_row("2024-01-01", 100, retail_qty=100),
                _investor_row("2024-01-02", 150, retail_qty=0)]
        df = compute_holdings(pd.DataFrame(rows))
        for s in KEY_SUBJECTS:
            df[f"{s}_amount"] = 0.0
        df.loc[0, "retail_amount"] = 100 * 100      # 100주 @100
        retail = {r["subject"]: r for r in pnl_attribution(df)}["retail"]
        assert retail["realized_pnl"] == pytest.approx(0.0)
        # 미실현 = 100×(150−100) = 5000
        assert retail["unrealized_pnl"] == pytest.approx(5000.0)

    def test_split_avgs_always_positive(self):
        # 고가 매도 > 저가 매수: 단일 net 평단이면 음수지만, 분리 평단은 둘 다 양수.
        rows = [
            _investor_row("2024-01-01", 50_000, retail_qty=100),
            _investor_row("2024-01-02", 100_000, retail_qty=-90),
        ]
        df = compute_holdings(pd.DataFrame(rows))
        for s in KEY_SUBJECTS:
            df[f"{s}_amount"] = 0.0
        df.loc[0, "retail_amount"] = 100 * 50_000     # +5,000,000
        df.loc[1, "retail_amount"] = -90 * 100_000    # -9,000,000
        retail = {r["subject"]: r for r in pnl_attribution(df)}["retail"]
        assert retail["buy_avg"] == 50_000.0
        assert retail["sell_avg"] == 100_000.0        # -9,000,000 / -90
        # net_qty=10, net_amt=-4,000,000 → pnl = 100,000*10 + 4,000,000
        assert retail["pnl"] == pytest.approx(5_000_000.0)

    def test_sorted_by_pnl_desc(self):
        rows = pnl_attribution(self._df_with_amounts())
        pnls = [r["pnl"] for r in rows]
        assert pnls == sorted(pnls, reverse=True)

    def test_empty_without_amount_columns(self):
        # holdings 단독(amount 미병합) → 빈 리스트
        df = compute_holdings(pd.DataFrame([
            _investor_row("2024-01-01", 100, retail_qty=100),
        ]))
        assert pnl_attribution(df) == []


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
                  "7-6. Rolling", "7-7. 주체별 손익", "구현하지 않은 4 종"]:
            assert h in md
