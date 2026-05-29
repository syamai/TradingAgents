"""tradingagents/dataflows/search_aggregator.py — mock 폴백 + dedup 검증."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

from tradingagents.dataflows.search_aggregator import (
    NewsHit, search_news, _normalize_url, _normalize_title, _within_window,
    _is_noise,
)


def _hit(title, url, pub, prov="mock"):
    return NewsHit(
        title=title, url=url, published=pub, snippet=title,
        source_label=prov, raw_provider=prov,
    )


class TestNormalize:
    def test_url_strips_query(self):
        a = _normalize_url("https://EXAMPLE.com/path/?x=1&y=2#frag")
        b = _normalize_url("https://example.com/path")
        assert a == b

    def test_title_removes_particles(self):
        # "는/을/를" 같은 조사 + 공백/구두점 정규화 후 동일
        a = _normalize_title("DB손해보험은 자사주를 소각")
        b = _normalize_title("DB손해보험   자사주   소각")
        assert a == b

    def test_within_window(self):
        s = date(2025, 4, 1)
        e = date(2025, 4, 10)
        assert _within_window("2025-04-05", s, e)
        assert not _within_window("2025-03-31", s, e)
        assert not _within_window("2025-04-11", s, e)
        # 날짜 미지정은 보수적 포함
        assert _within_window(None, s, e)


class TestFallbackOrder:
    def test_first_provider_satisfies(self):
        """첫 provider 가 8건 이상 반환하면 다음 provider 미호출."""
        hits = [_hit(f"Title {i}", f"https://u/{i}", "2025-04-05") for i in range(10)]
        called = []

        def fake_first(q, s, u, *, max_results):
            called.append("first")
            return hits[:max_results]

        def fake_second(q, s, u, *, max_results):
            called.append("second")
            return []

        with patch.dict(
            "tradingagents.dataflows.search_aggregator._PROVIDER_FN",
            {"first": fake_first, "second": fake_second}, clear=True,
        ):
            out = search_news(
                "q", "2025-04-01", "2025-04-10",
                max_results=10, providers=("first", "second"),
            )
        assert "first" in called
        assert "second" not in called  # early stop
        assert len(out) == 10

    def test_fallback_when_empty(self):
        called = []

        def empty(q, s, u, *, max_results):
            called.append("empty")
            return []

        def filler(q, s, u, *, max_results):
            called.append("filler")
            return [_hit("Hello world", "https://x/1", "2025-04-05")]

        with patch.dict(
            "tradingagents.dataflows.search_aggregator._PROVIDER_FN",
            {"empty": empty, "filler": filler}, clear=True,
        ):
            out = search_news(
                "q", "2025-04-01", "2025-04-10",
                max_results=10, providers=("empty", "filler"),
            )
        assert called == ["empty", "filler"]
        assert len(out) == 1


class TestDedup:
    def test_same_url_different_title(self):
        def a(q, s, u, *, max_results):
            return [_hit("제목 A", "https://x/1?q=z", "2025-04-05")]

        def b(q, s, u, *, max_results):
            return [_hit("제목 B 다름", "https://x/1", "2025-04-05")]

        with patch.dict(
            "tradingagents.dataflows.search_aggregator._PROVIDER_FN",
            {"a": a, "b": b}, clear=True,
        ):
            out = search_news(
                "q", "2025-04-01", "2025-04-10",
                max_results=10, providers=("a", "b"),
            )
        assert len(out) == 1  # URL normalize 일치 → dedup

    def test_same_title_different_url(self):
        def a(q, s, u, *, max_results):
            return [_hit("자사주 소각", "https://x/1", "2025-04-05")]

        def b(q, s, u, *, max_results):
            return [_hit("자사주를 소각", "https://y/2", "2025-04-05")]

        with patch.dict(
            "tradingagents.dataflows.search_aggregator._PROVIDER_FN",
            {"a": a, "b": b}, clear=True,
        ):
            out = search_news(
                "q", "2025-04-01", "2025-04-10",
                max_results=10, providers=("a", "b"),
            )
        assert len(out) == 1  # 정규화 title 일치 → dedup


class TestKeysMissing:
    def test_all_skip_returns_empty(self):
        """모든 provider 가 0건 반환 → 빈 리스트 + 에러 0."""
        out = search_news(
            "q", "2025-04-01", "2025-04-10",
            max_results=10,
            providers=(),  # provider 없음
        )
        assert out == []


class TestNoiseFilter:
    def test_noise_pattern_blocked(self):
        # 광고 패턴 매칭 — 차단
        h = NewsHit(
            title="원금보장 투자 가이드 [광고]",
            url="https://example.com/x",
            published="2025-04-05", snippet="",
            source_label="x", raw_provider="x",
        )
        assert _is_noise(h)

    def test_clean_title_passes(self):
        h = NewsHit(
            title="DB손해보험 자사주 소각 결정",
            url="https://news.naver.com/x",
            published="2025-04-05", snippet="",
            source_label="x", raw_provider="x",
        )
        assert not _is_noise(h)

    def test_noise_in_snippet(self):
        # snippet 에 노이즈 패턴 — 차단
        h = NewsHit(
            title="투자 안내",
            url="https://example.com/y",
            published="2025-04-05",
            snippet="지금 가입하시면 수익률 보장 + 사은품 증정",
            source_label="x", raw_provider="x",
        )
        assert _is_noise(h)

    def test_too_short_title(self):
        h = NewsHit(
            title="x", url="https://example.com/z",
            published="2025-04-05", snippet="",
            source_label="x", raw_provider="x",
        )
        assert _is_noise(h)

    def test_noise_blocked_in_search(self):
        """search_news 통합 — noise 가 결과에서 제거되는지."""
        noise = NewsHit(
            title="[광고] 원금보장 리딩방",
            url="https://x.com/spam", published="2025-04-05",
            snippet="", source_label="x", raw_provider="x",
        )
        clean = NewsHit(
            title="실적 발표", url="https://news.com/clean",
            published="2025-04-05", snippet="",
            source_label="news", raw_provider="news",
        )

        def fake(q, s, u, *, max_results):
            return [noise, clean]

        from unittest.mock import patch
        with patch.dict(
            "tradingagents.dataflows.search_aggregator._PROVIDER_FN",
            {"p": fake}, clear=True,
        ):
            out = search_news(
                "q", "2025-04-01", "2025-04-10",
                max_results=10, providers=("p",),
            )
        assert len(out) == 1
        assert out[0].title == "실적 발표"
