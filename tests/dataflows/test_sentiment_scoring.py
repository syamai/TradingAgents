"""Unit tests for the sentiment scoring adapters (VADER + KNU)."""

import pytest

from tradingagents.dataflows.sentiment_scoring import (
    score_knu,
    score_text,
    score_vader,
)


@pytest.mark.unit
class TestVaderScoring:
    def test_bullish_phrase_labeled_positive(self):
        # VADER는 일반 영어 어휘 기준이라 "crushed earnings" 같은 금융 슬랭은
        # 잘못 해석한다. 명백히 긍정인 문장으로 검증.
        r = score_vader("Great news! Fantastic results, very happy with the strong gains.")
        assert r["label"] == "positive"
        assert r["compound"] > 0
        assert r["oov_ratio"] == 0.0

    def test_bearish_phrase_labeled_negative(self):
        r = score_vader("This is a disaster, terrible quarter, dumping all my shares")
        assert r["label"] == "negative"
        assert r["compound"] < 0

    def test_neutral_phrase_labeled_neutral(self):
        r = score_vader("The company reported quarterly earnings yesterday.")
        assert r["label"] == "neutral"

    def test_empty_text_returns_neutral_default(self):
        r = score_vader("")
        assert r["label"] == "neutral"
        assert r["compound"] == 0.0

    def test_result_shape_stable(self):
        r = score_vader("good news")
        assert set(r.keys()) == {"pos", "neg", "neu", "compound", "label", "oov_ratio"}


@pytest.mark.unit
class TestKnuScoring:
    def test_bullish_korean_labeled_positive(self):
        r = score_knu("삼성전자 급등 호재 신고가 돌파")
        assert r["label"] == "positive"
        assert r["compound"] > 0

    def test_bearish_korean_labeled_negative(self):
        r = score_knu("폭락 손실 적자 위기 악재")
        assert r["label"] == "negative"
        assert r["compound"] < 0

    def test_neutral_korean_labeled_neutral(self):
        r = score_knu("종목 토론실 게시판 코드")
        assert r["label"] == "neutral"
        # 모든 토큰이 사전에 없으므로 oov_ratio 1.0
        assert r["oov_ratio"] == pytest.approx(1.0)

    def test_prefix_match_handles_korean_inflection(self):
        # '급등하며' → 사전의 '급등' prefix 매치되어야 함
        r = score_knu("급등하며")
        assert r["compound"] > 0
        assert r["label"] == "positive"

    def test_oov_ratio_tracks_missing_words(self):
        # 사전에 있는 단어 1개 + 없는 단어 3개
        r = score_knu("급등 라라라 두두두 컴컴컴")
        assert 0.0 < r["oov_ratio"] < 1.0
        assert r["oov_ratio"] == pytest.approx(0.75, abs=0.01)

    def test_empty_text_returns_neutral_default(self):
        r = score_knu("")
        assert r["label"] == "neutral"


@pytest.mark.unit
class TestScoreTextRouter:
    def test_lang_ko_routes_to_knu(self):
        r = score_text("급등 호재", lang="ko")
        assert r["compound"] > 0

    def test_lang_en_routes_to_vader(self):
        r = score_text("absolutely fantastic, going to moon!", lang="en")
        assert r["label"] == "positive"

    def test_unknown_lang_falls_back_to_vader(self):
        r = score_text("great news", lang="xx")
        assert r["label"] == "positive"
