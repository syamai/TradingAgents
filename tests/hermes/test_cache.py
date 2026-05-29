"""AnalystOutputCache 검증.

결정론 입력 + 정확값 단언. ``root=tmp_path`` 로 SQLite 격리.
"""
from __future__ import annotations

import pytest

from tradingagents.hermes.cache import AnalystOutputCache


@pytest.fixture
def cache(tmp_path):
    return AnalystOutputCache(root=tmp_path)


@pytest.mark.unit
class TestSetGetRoundTrip:
    def test_set_then_get_returns_same_report(self, cache):
        cache.set(
            "005930.KS", "2026-05-29", "supply_demand",
            "gemma4:26b-a4b", "v1", "## 보고서\n외국인 등록 +120 만주.",
        )

        result = cache.get(
            "005930.KS", "2026-05-29", "supply_demand",
            "gemma4:26b-a4b", "v1",
        )

        assert result == "## 보고서\n외국인 등록 +120 만주."

    def test_get_miss_returns_none(self, cache):
        assert cache.get(
            "005930.KS", "2026-05-29", "supply_demand",
            "gemma4:26b-a4b", "v1",
        ) is None

    def test_set_overwrites_same_composite_key(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "old")
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "new")

        assert cache.get(
            "005930.KS", "2026-05-29", "supply_demand", "m", "v1",
        ) == "new"


@pytest.mark.unit
class TestCompositeKeySeparation:
    """같은 (ticker, date) 라도 model_id / prompt_version 다르면 별도 row."""

    def test_different_model_id_stored_separately(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v1", "A")
        cache.set("005930.KS", "2026-05-29", "supply_demand", "gemma5", "v1", "B")

        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v1") == "A"
        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "gemma5", "v1") == "B"

    def test_different_prompt_version_stored_separately(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v1", "A")
        cache.set("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v2", "B")

        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v1") == "A"
        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v2") == "B"

    def test_different_analyst_stored_separately(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "A")
        cache.set("005930.KS", "2026-05-29", "sentiment", "m", "v1", "B")

        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "m", "v1") == "A"
        assert cache.get("005930.KS", "2026-05-29", "sentiment", "m", "v1") == "B"


@pytest.mark.unit
class TestList:
    def test_list_filters_by_ticker(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "A")
        cache.set("000660.KS", "2026-05-29", "supply_demand", "m", "v1", "B")

        result = cache.list(ticker="005930.KS")

        assert len(result) == 1
        assert result[0]["ticker"] == "005930.KS"

    def test_list_filters_by_analyst(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "A")
        cache.set("005930.KS", "2026-05-29", "sentiment", "m", "v1", "B")

        result = cache.list(analyst_name="supply_demand")

        assert len(result) == 1
        assert result[0]["analyst_name"] == "supply_demand"

    def test_list_excludes_report_field(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "BIG_REPORT")

        result = cache.list()

        # report 는 list 결과에 노출되지 않음 (메모리·전송 비용).
        assert "report" not in result[0]
        assert "created_at" in result[0]

    def test_list_no_filter_returns_all(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "A")
        cache.set("000660.KS", "2026-05-29", "sentiment", "m", "v1", "B")

        assert len(cache.list()) == 2


@pytest.mark.unit
class TestLargeReport:
    def test_large_markdown_round_trip_lossless(self, cache):
        # 큰 마크다운 (50KB) round-trip 무손실 확인.
        big_report = "## 섹션\n" + ("외국인 등록 net_qty +120 만주.\n" * 1500)

        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", big_report)
        result = cache.get("005930.KS", "2026-05-29", "supply_demand", "m", "v1")

        assert result == big_report
        assert len(result) > 30_000


@pytest.mark.unit
class TestInvalidate:
    def test_invalidate_by_ticker(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "A")
        cache.set("005930.KS", "2026-05-30", "supply_demand", "m", "v1", "B")
        cache.set("000660.KS", "2026-05-29", "supply_demand", "m", "v1", "C")

        deleted = cache.invalidate(ticker="005930.KS")

        assert deleted == 2
        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "m", "v1") is None
        assert cache.get("000660.KS", "2026-05-29", "supply_demand", "m", "v1") == "C"

    def test_invalidate_by_model_id_preserves_others(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v1", "A")
        cache.set("005930.KS", "2026-05-29", "supply_demand", "gemma5", "v1", "B")

        deleted = cache.invalidate(model_id="gemma4")

        assert deleted == 1
        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "gemma4", "v1") is None
        assert cache.get("005930.KS", "2026-05-29", "supply_demand", "gemma5", "v1") == "B"

    def test_invalidate_with_no_filter_raises(self, cache):
        cache.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "A")

        with pytest.raises(ValueError, match="at least one filter"):
            cache.invalidate()


@pytest.mark.unit
class TestPersistence:
    def test_separate_instances_share_db(self, tmp_path):
        c1 = AnalystOutputCache(root=tmp_path)
        c1.set("005930.KS", "2026-05-29", "supply_demand", "m", "v1", "X")

        c2 = AnalystOutputCache(root=tmp_path)
        assert c2.get(
            "005930.KS", "2026-05-29", "supply_demand", "m", "v1",
        ) == "X"
