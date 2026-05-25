"""Unit tests for sentiment_analyst's inline triage functions.

Triage routes raw messages through Sparse / Normal / Flood bands based on
volume. Tests verify the band routing and the resulting block format
without requiring network calls.
"""

import pytest

from tradingagents.agents.analysts.sentiment_analyst import (
    _classify_volume,
    _distribution,
    _resolve_lookback,
    _triage_naver,
    _triage_reddit,
    _triage_stocktwits,
)


def _fake_reddit_post(idx: int, score: int = 10, comments: int = 5) -> dict:
    return {
        "title": f"NVDA earnings post {idx}",
        "selftext": f"detailed analysis {idx} bullish strong gains",
        "score": score,
        "num_comments": comments,
        "created_utc": 1700000000 + idx * 3600,
        "subreddit": "wallstreetbets",
    }


def _fake_stocktwits_msg(idx: int, sentiment: str = "Bullish") -> dict:
    return {
        "body": f"NVDA going strong {idx}",
        "created_at": f"2026-05-2{idx % 7}T10:00:00Z",
        "user": {"username": f"user{idx}"},
        "entities": {"sentiment": {"basic": sentiment}},
    }


def _fake_naver_post(idx: int, up: int = 5, down: int = 1) -> dict:
    return {
        "title": f"삼성전자 급등 호재 {idx}",
        "date": f"2026.05.2{idx % 7} 10:00",
        "views": 1000 + idx * 100,
        "up": up,
        "down": down,
    }


@pytest.mark.unit
class TestVolumeClassification:
    def test_sparse_band(self):
        assert _classify_volume(0) == "sparse"
        assert _classify_volume(20) == "sparse"

    def test_normal_band(self):
        assert _classify_volume(21) == "normal"
        assert _classify_volume(80) == "normal"

    def test_flood_band(self):
        assert _classify_volume(81) == "flood"
        assert _classify_volume(500) == "flood"


@pytest.mark.unit
class TestLookbackResolution:
    def test_default_strategy_maps_to_7_days(self):
        assert _resolve_lookback({"sentiment_lookback_strategy": "default"}) == 7

    def test_daily_strategy_maps_to_3_days(self):
        assert _resolve_lookback({"sentiment_lookback_strategy": "daily"}) == 3

    def test_weekly_strategy_maps_to_14_days(self):
        assert _resolve_lookback({"sentiment_lookback_strategy": "weekly"}) == 14

    def test_none_config_falls_back_to_7(self):
        assert _resolve_lookback(None) == 7

    def test_unknown_strategy_falls_back_to_7(self):
        assert _resolve_lookback({"sentiment_lookback_strategy": "yearly"}) == 7


@pytest.mark.unit
class TestDistribution:
    def test_aggregates_label_counts(self):
        scores = [
            {"label": "positive", "oov_ratio": 0.0},
            {"label": "positive", "oov_ratio": 0.0},
            {"label": "negative", "oov_ratio": 0.0},
            {"label": "neutral", "oov_ratio": 0.0},
        ]
        d = _distribution(scores)
        assert d["total"] == 4
        assert d["pos"] == 2 and d["neg"] == 1 and d["neu"] == 1
        assert d["pos_pct"] == 50

    def test_empty_input_safe(self):
        d = _distribution([])
        assert d["total"] == 0
        assert d["pos_pct"] == 0


@pytest.mark.unit
class TestRedditTriage:
    def test_sparse_uses_verbatim_format(self):
        posts = [_fake_reddit_post(i) for i in range(10)]  # N=10 → Sparse
        out = _triage_reddit(posts, lookback_days=7, display_label="NVDA",
                             subreddits=["wallstreetbets"])
        assert "[SPARSE]" not in out  # Sparse path doesn't tag header
        assert "분포:" not in out  # Sparse → no distribution stats
        assert "NVDA earnings post 0" in out

    def test_normal_includes_distribution_and_top_k(self):
        posts = [_fake_reddit_post(i) for i in range(50)]  # N=50 → Normal
        out = _triage_reddit(posts, lookback_days=7, display_label="NVDA",
                             subreddits=["wallstreetbets"])
        assert "[NORMAL]" in out
        assert "분포:" in out
        assert "대표 글" in out

    def test_flood_includes_burst_buckets(self):
        posts = [_fake_reddit_post(i) for i in range(120)]  # N=120 → Flood
        out = _triage_reddit(posts, lookback_days=7, display_label="NVDA",
                             subreddits=["wallstreetbets"])
        assert "[FLOOD]" in out
        assert "일별 게시 수" in out

    def test_empty_posts_returns_placeholder(self):
        out = _triage_reddit([], lookback_days=7, display_label="NVDA",
                             subreddits=["wallstreetbets"])
        assert "<no Reddit posts" in out


@pytest.mark.unit
class TestStockTwitsTriage:
    def test_sparse_emits_all_messages(self):
        msgs = [_fake_stocktwits_msg(i) for i in range(10)]
        out = _triage_stocktwits(msgs, lookback_days=7, ticker="NVDA")
        assert "[SPARSE]" in out
        assert "라벨 분포" in out
        # All 10 messages present
        assert "NVDA going strong 0" in out
        assert "NVDA going strong 9" in out

    def test_normal_caps_to_top_k(self):
        msgs = [_fake_stocktwits_msg(i) for i in range(40)]
        out = _triage_stocktwits(msgs, lookback_days=7, ticker="NVDA")
        assert "[NORMAL]" in out
        assert "대표 메시지" in out

    def test_label_distribution_correct(self):
        msgs = (
            [_fake_stocktwits_msg(i, "Bullish") for i in range(6)]
            + [_fake_stocktwits_msg(i, "Bearish") for i in range(4)]
        )
        out = _triage_stocktwits(msgs, lookback_days=7, ticker="NVDA")
        assert "Bullish 6 (60%)" in out
        assert "Bearish 4 (40%)" in out

    def test_empty_returns_placeholder(self):
        out = _triage_stocktwits([], lookback_days=7, ticker="NVDA")
        assert "<no StockTwits messages" in out


@pytest.mark.unit
class TestNaverTriage:
    def test_sparse_emits_all_posts(self):
        posts = [_fake_naver_post(i) for i in range(10)]
        out = _triage_naver(posts, lookback_days=7, ticker="005930.KS")
        assert "[SPARSE]" in out
        # Sparse path lists every post
        assert "삼성전자 급등 호재 0" in out

    def test_normal_includes_korean_distribution(self):
        posts = [_fake_naver_post(i) for i in range(40)]
        out = _triage_naver(posts, lookback_days=7, ticker="005930.KS")
        assert "[NORMAL]" in out
        assert "제목 분포" in out

    def test_aggregate_ratio_computed(self):
        posts = [_fake_naver_post(i, up=10, down=2) for i in range(5)]
        out = _triage_naver(posts, lookback_days=7, ticker="005930.KS")
        # 50 up / 10 down → 83.3%
        assert "83.3%" in out

    def test_empty_returns_placeholder(self):
        out = _triage_naver([], lookback_days=7, ticker="005930.KS")
        assert "<no Naver discussion" in out
