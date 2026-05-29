"""HypothesisStore 검증."""
from __future__ import annotations

import pytest

from tradingagents.hermes.hypothesis_store import HypothesisStore


@pytest.fixture
def store(tmp_path):
    return HypothesisStore(root=tmp_path)


def _valid_record(**overrides) -> dict:
    """PRD schema 만족하는 minimal 응답 JSON."""
    base = {
        "ticker": "005930.KS",
        "as_of_date": "2026-05-29",
        "overall_stance": "moderately_bearish",
        "overall_confidence": 0.65,
        "hypotheses": [
            {
                "id": "h1",
                "claim": "외국인 등록자 매도 → 단기 약세",
                "direction": "bearish",
                "confidence": 0.7,
                "evidence_tools": ["analyst_supply_demand"],
                "evidence_excerpts": ["외국인 등록 net_qty 60일 -120만주"],
                "horizon_weeks": 2,
                "predicted_relative_return_pct": -2.5,
            },
            {
                "id": "h2",
                "claim": "개인 역추세 흡수",
                "direction": "neutral",
                "confidence": 0.5,
                "evidence_tools": ["analyst_supply_demand", "compute_trend"],
                "evidence_excerpts": ["개인 60일 phase up, agreement 38%"],
                "horizon_weeks": 4,
                "predicted_relative_return_pct": 0.0,
            },
        ],
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestSchemaValidation:
    def test_save_valid_record_returns_ids(self, store):
        ids = store.save(_valid_record())
        assert len(ids) == 2
        assert all(isinstance(i, int) for i in ids)

    def test_missing_top_level_field_raises(self, store):
        rec = _valid_record()
        del rec["overall_stance"]
        with pytest.raises(ValueError, match="missing top-level"):
            store.save(rec)

    def test_invalid_stance_raises(self, store):
        rec = _valid_record(overall_stance="ultra_bullish")
        with pytest.raises(ValueError, match="overall_stance"):
            store.save(rec)

    def test_confidence_out_of_range_raises(self, store):
        rec = _valid_record(overall_confidence=1.5)
        with pytest.raises(ValueError, match="overall_confidence"):
            store.save(rec)

    def test_empty_hypotheses_raises(self, store):
        rec = _valid_record(hypotheses=[])
        with pytest.raises(ValueError, match="non-empty"):
            store.save(rec)

    def test_hypothesis_missing_field_raises(self, store):
        rec = _valid_record()
        del rec["hypotheses"][0]["horizon_weeks"]
        with pytest.raises(ValueError, match="horizon_weeks"):
            store.save(rec)

    def test_invalid_direction_raises(self, store):
        rec = _valid_record()
        rec["hypotheses"][0]["direction"] = "uncertain"
        with pytest.raises(ValueError, match="direction"):
            store.save(rec)

    def test_horizon_out_of_swing_range_raises(self, store):
        rec = _valid_record()
        rec["hypotheses"][0]["horizon_weeks"] = 12  # 장기 — out of swing
        with pytest.raises(ValueError, match="horizon_weeks"):
            store.save(rec)

    def test_empty_evidence_excerpts_raises(self, store):
        # raw 수치 인용 의무 — 빈 evidence 는 할루시네이션 위험.
        rec = _valid_record()
        rec["hypotheses"][0]["evidence_excerpts"] = []
        with pytest.raises(ValueError, match="evidence_excerpts"):
            store.save(rec)


@pytest.mark.unit
class TestGet:
    def test_get_returns_full_hypothesis(self, store):
        ids = store.save(_valid_record())

        h = store.get(ids[0])

        assert h["ticker"] == "005930.KS"
        assert h["h_local_id"] == "h1"
        assert h["direction"] == "bearish"
        assert h["evidence_excerpts"] == ["외국인 등록 net_qty 60일 -120만주"]
        assert h["evidence_tools"] == ["analyst_supply_demand"]

    def test_get_unknown_id_returns_none(self, store):
        assert store.get(999) is None


@pytest.mark.unit
class TestList:
    def test_list_filters_by_ticker(self, store):
        store.save(_valid_record(ticker="005930.KS"))
        store.save(_valid_record(ticker="000660.KS"))

        result = store.list(ticker="005930.KS")

        assert len(result) == 2  # 2 hypotheses per record
        assert all(h["ticker"] == "005930.KS" for h in result)

    def test_list_filters_by_direction(self, store):
        store.save(_valid_record())

        result = store.list(direction="bearish")

        assert len(result) == 1
        assert result[0]["h_local_id"] == "h1"

    def test_list_filters_by_as_of_date(self, store):
        store.save(_valid_record(as_of_date="2026-05-29"))
        store.save(_valid_record(as_of_date="2026-05-30"))

        result = store.list(as_of_date="2026-05-30")

        assert len(result) == 2
        assert all(h["as_of_date"] == "2026-05-30" for h in result)


@pytest.mark.unit
class TestLabels:
    def test_add_user_immediate_label(self, store):
        ids = store.save(_valid_record())

        label_id = store.add_label(
            ids[0], "user_immediate",
            verdict="right", reason="외국인 매도가 정말 시작됐음",
        )

        labels = store.get_labels(ids[0])
        assert len(labels) == 1
        assert labels[0]["id"] == label_id
        assert labels[0]["label_kind"] == "user_immediate"
        assert labels[0]["verdict"] == "right"

    def test_add_auto_relative_label_with_returns(self, store):
        ids = store.save(_valid_record())

        store.add_label(
            ids[0], "auto_relative",
            verdict="right",
            actual_return_pct=-3.2,
            actual_relative_return_pct=-2.8,
            labeled_at_date="2026-06-12",
        )

        labels = store.get_labels(ids[0])
        assert labels[0]["actual_relative_return_pct"] == -2.8
        assert labels[0]["labeled_at_date"] == "2026-06-12"

    def test_multiple_labels_per_hypothesis(self, store):
        ids = store.save(_valid_record())

        store.add_label(ids[0], "user_immediate", verdict="right")
        store.add_label(ids[0], "auto_relative", verdict="right",
                        actual_relative_return_pct=-2.5)
        store.add_label(ids[0], "user_followup", verdict="wrong",
                        reason="이유는 외국인이 아니라 환율")

        labels = store.get_labels(ids[0])
        assert len(labels) == 3
        kinds = [l["label_kind"] for l in labels]
        assert kinds == ["user_immediate", "auto_relative", "user_followup"]

    def test_invalid_label_kind_raises(self, store):
        ids = store.save(_valid_record())

        with pytest.raises(ValueError, match="label_kind"):
            store.add_label(ids[0], "random_kind")

    def test_label_on_unknown_hypothesis_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.add_label(999, "user_immediate", verdict="right")

    def test_get_with_labels_combines(self, store):
        ids = store.save(_valid_record())
        store.add_label(ids[0], "user_immediate", verdict="right")

        h = store.get_with_labels(ids[0])

        assert h["h_local_id"] == "h1"
        assert "labels" in h
        assert len(h["labels"]) == 1


@pytest.mark.unit
class TestPersistence:
    def test_separate_instances_share_db(self, tmp_path):
        s1 = HypothesisStore(root=tmp_path)
        ids = s1.save(_valid_record())
        s1.add_label(ids[0], "user_immediate", verdict="right")

        s2 = HypothesisStore(root=tmp_path)
        h = s2.get_with_labels(ids[0])
        assert h["h_local_id"] == "h1"
        assert len(h["labels"]) == 1
