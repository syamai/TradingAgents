"""LabelingScheduler 검증.

가격 fetcher 는 결정론적 fake 함수로 주입 — yfinance 호출 없이 임계값
판정 로직을 검증.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.hermes.hypothesis_store import HypothesisStore
from tradingagents.hermes.labeling import (
    LabelingScheduler,
    horizon_threshold,
    judge_verdict,
)


# === pure 함수 검증 ===

@pytest.mark.unit
class TestHorizonThreshold:
    def test_two_weeks_returns_2pct(self):
        assert horizon_threshold(2) == 2.0

    def test_four_weeks_returns_3pct(self):
        assert horizon_threshold(4) == 3.0

    def test_eight_weeks_returns_5pct(self):
        assert horizon_threshold(8) == 5.0

    def test_three_weeks_linear_interp(self):
        assert horizon_threshold(3) == 2.5

    def test_six_weeks_linear_interp(self):
        assert horizon_threshold(6) == 4.0


@pytest.mark.unit
class TestJudgeVerdict:
    def test_bullish_above_threshold_is_right(self):
        assert judge_verdict("bullish", 3.5, 3.0) == "right"

    def test_bullish_below_threshold_is_wrong(self):
        assert judge_verdict("bullish", 2.5, 3.0) == "wrong"

    def test_bullish_negative_is_wrong(self):
        assert judge_verdict("bullish", -1.0, 3.0) == "wrong"

    def test_bearish_below_neg_threshold_is_right(self):
        assert judge_verdict("bearish", -3.5, 3.0) == "right"

    def test_bearish_positive_is_wrong(self):
        assert judge_verdict("bearish", 2.0, 3.0) == "wrong"

    def test_neutral_within_half_threshold_is_right(self):
        # 4 주 threshold=3.0, half=1.5. |1.0| ≤ 1.5 → right.
        assert judge_verdict("neutral", 1.0, 3.0) == "right"

    def test_neutral_outside_half_threshold_is_wrong(self):
        assert judge_verdict("neutral", 2.0, 3.0) == "wrong"

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            judge_verdict("uncertain", 1.0, 3.0)


# === LabelingScheduler 통합 ===


def _ticker_series(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "close": closes})


def _make_fetcher(rows: dict[str, pd.DataFrame]):
    """key → DataFrame 매핑으로부터 fetcher 함수 생성. case-insensitive."""
    norm = {k.lower(): v for k, v in rows.items()}

    def fetch(key: str, start: str, end: str) -> pd.DataFrame:
        df = norm.get(key.lower(), pd.DataFrame(columns=["date", "close"]))
        if df.empty:
            return df
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].reset_index(drop=True)
    return fetch


def _record(**overrides) -> dict:
    """라벨링 테스트용 minimal 가설 record."""
    base = {
        "ticker": "005930.KS",
        "as_of_date": "2026-05-01",
        "overall_stance": "bullish",
        "overall_confidence": 0.7,
        "hypotheses": [{
            "id": "h1",
            "claim": "test",
            "direction": "bullish",
            "confidence": 0.7,
            "evidence_tools": ["t"],
            "evidence_excerpts": ["e"],
            "horizon_weeks": 4,
            "predicted_relative_return_pct": 3.0,
        }],
    }
    base.update(overrides)
    return base


@pytest.fixture
def store(tmp_path):
    return HypothesisStore(root=tmp_path)


@pytest.mark.unit
class TestLabelHypothesis:
    def test_bullish_outperforms_kospi_labeled_right(self, store):
        rec = _record()
        rec["hypotheses"][0]["direction"] = "bullish"
        ids = store.save(rec)
        # 4 주 후 = 2026-05-29. Target +6%, KOSPI +1% → relative +5% (threshold 3%).
        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-05-29"], [100.0, 106.0],
            ),
            "kospi": _ticker_series(
                ["2026-05-01", "2026-05-29"], [3000.0, 3030.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-05-30")

        assert result["status"] == "labeled"
        assert result["verdict"] == "right"
        assert result["actual_return_pct"] == pytest.approx(6.0, abs=0.01)
        assert result["actual_relative_return_pct"] == pytest.approx(5.0, abs=0.01)
        assert result["threshold_pct"] == 3.0

    def test_bullish_underperforms_kospi_labeled_wrong(self, store):
        rec = _record()
        ids = store.save(rec)
        # Target +1%, KOSPI +2% → relative -1% (bullish 가설 → wrong).
        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-05-29"], [100.0, 101.0],
            ),
            "kospi": _ticker_series(
                ["2026-05-01", "2026-05-29"], [3000.0, 3060.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-05-30")

        assert result["verdict"] == "wrong"

    def test_too_early_returns_skipped(self, store):
        rec = _record()
        ids = store.save(rec)
        fetcher = _make_fetcher({})  # 호출 안 됨
        sched = LabelingScheduler(store, fetcher=fetcher)

        # 가설 as_of=2026-05-01, horizon=4 → target=2026-05-29.
        # today=2026-05-15 → 아직 미경과.
        result = sched.label_hypothesis(ids[0], today="2026-05-15")

        assert result["status"] == "too_early"
        assert result["target_date"] == "2026-05-29"

    def test_already_labeled_skipped(self, store):
        rec = _record()
        ids = store.save(rec)
        store.add_label(ids[0], "auto_relative", verdict="right",
                        actual_relative_return_pct=2.0)
        fetcher = _make_fetcher({})
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-05-30")

        assert result["status"] == "already_labeled"

    def test_no_target_price_returns_skip(self, store):
        rec = _record()
        ids = store.save(rec)
        fetcher = _make_fetcher({
            "kospi": _ticker_series(
                ["2026-05-01", "2026-05-29"], [3000.0, 3030.0],
            ),
            # ticker 시리즈 누락.
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-05-30")

        assert result["status"] == "no_target_price"

    def test_no_kospi_returns_skip(self, store):
        rec = _record()
        ids = store.save(rec)
        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-05-29"], [100.0, 106.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-05-30")

        assert result["status"] == "no_kospi"

    def test_holiday_target_date_advances_to_next_trading_day(self, store):
        # 2026-05-29 가 휴장일 → 2026-06-01 가 다음 거래일.
        rec = _record()
        ids = store.save(rec)
        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-06-01"], [100.0, 110.0],
            ),
            "kospi": _ticker_series(
                ["2026-05-01", "2026-06-01"], [3000.0, 3060.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-06-02")

        # 휴장일 보정으로 labeled_at_date 가 2026-06-01 로 advance.
        assert result["status"] == "labeled"
        assert result["labeled_at_date"] == "2026-06-01"

    def test_bearish_outperforms_short_labeled_right(self, store):
        rec = _record()
        rec["hypotheses"][0]["direction"] = "bearish"
        ids = store.save(rec)
        # Target -4%, KOSPI 0% → relative -4% (threshold 3%, bearish → right).
        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-05-29"], [100.0, 96.0],
            ),
            "kospi": _ticker_series(
                ["2026-05-01", "2026-05-29"], [3000.0, 3000.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-05-30")

        assert result["verdict"] == "right"

    def test_neutral_within_half_threshold_labeled_right(self, store):
        rec = _record()
        rec["hypotheses"][0]["direction"] = "neutral"
        ids = store.save(rec)
        # 4 주 threshold=3.0, half=1.5. relative +1.0% → neutral right.
        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-05-29"], [100.0, 101.0],
            ),
            "kospi": _ticker_series(
                ["2026-05-01", "2026-05-29"], [3000.0, 3000.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        result = sched.label_hypothesis(ids[0], today="2026-05-30")

        assert result["verdict"] == "right"

    def test_persists_label_with_both_returns(self, store):
        rec = _record()
        ids = store.save(rec)
        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-05-29"], [100.0, 106.0],
            ),
            "kospi": _ticker_series(
                ["2026-05-01", "2026-05-29"], [3000.0, 3030.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        sched.label_hypothesis(ids[0], today="2026-05-30")
        labels = store.get_labels(ids[0])

        assert len(labels) == 1
        assert labels[0]["label_kind"] == "auto_relative"
        assert labels[0]["verdict"] == "right"
        # 절대 수익 병기 저장 — KOSPI relative 만 아닌 actual_return_pct 도.
        assert labels[0]["actual_return_pct"] == pytest.approx(6.0, abs=0.01)
        assert labels[0]["actual_relative_return_pct"] == pytest.approx(5.0, abs=0.01)


@pytest.mark.unit
class TestRunPending:
    def test_counts_by_status(self, store):
        # 라벨 가능 가설.
        rec1 = _record(ticker="005930.KS", as_of_date="2026-05-01")
        ids1 = store.save(rec1)
        # 너무 이른 가설.
        rec2 = _record(ticker="000660.KS", as_of_date="2026-06-01")
        ids2 = store.save(rec2)
        # 이미 라벨됨.
        rec3 = _record(ticker="035420.KS", as_of_date="2026-05-01")
        ids3 = store.save(rec3)
        store.add_label(ids3[0], "auto_relative", verdict="right")

        fetcher = _make_fetcher({
            "005930.KS": _ticker_series(
                ["2026-05-01", "2026-05-29"], [100.0, 106.0],
            ),
            "kospi": _ticker_series(
                ["2026-05-01", "2026-05-29"], [3000.0, 3030.0],
            ),
        })
        sched = LabelingScheduler(store, fetcher=fetcher)

        counts = sched.run_pending(today="2026-05-30")

        assert counts["labeled"] == 1
        assert counts["too_early"] == 1
        assert counts["already_labeled"] == 1
