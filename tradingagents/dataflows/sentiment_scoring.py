"""Pre-LLM sentiment scoring — VADER (영문) + KNU 사전 (한국어).

`sentiment_analyst`의 triage 단계에서 호출. LLM 호출 전 로컬 점수화로 분포
통계를 만들고, Top-K 원문만 LLM에 넘겨 토큰을 절약한다. 호출 횟수 추가 없음.

Output schema (공통):
    {"pos": float, "neg": float, "neu": float,
     "compound": float, "label": "positive"|"negative"|"neutral",
     "oov_ratio": float}  # oov_ratio: 한국어만 의미, 영문은 0.0

`oov_ratio`가 0.3을 넘으면 KNU 사전 부족 — README의 KnuSentiLex 업그레이드
경로 참조.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LEXICON_PATH = Path(__file__).parent / "data" / "knu_sentiment_lexicon.json"
_OVERLAY_PATH = Path(__file__).parent / "data" / "knu_financial_overlay.json"
_KOREAN_TOKEN = re.compile(r"[가-힣]+")

# VADER 라벨 임계치 (compound 점수 기반, NLTK/공식 권장값)
_VADER_POS_THRESHOLD = 0.05
_VADER_NEG_THRESHOLD = -0.05

# KNU 라벨 임계치 (어절당 평균 polarity 기반, 자체 튜닝)
_KNU_POS_THRESHOLD = 0.15
_KNU_NEG_THRESHOLD = -0.15


@lru_cache(maxsize=1)
def _knu_lexicon() -> dict[str, int]:
    """Lazy-load KNU lexicon + finance overlay once per process.

    The base file is the full KnuSentiLex (general-emotion lexicon); the
    overlay file restores finance-specific vocabulary (급등/폭락/호재/...)
    that the general lexicon does not cover. Overlay values win on key
    collision, since the finance signal is more relevant for our domain.
    """
    merged: dict[str, int] = {}
    for path in (_LEXICON_PATH, _OVERLAY_PATH):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("KNU lexicon load failed (%s): %s", path.name, exc)
            continue
        words = raw.get("words", {})
        if isinstance(words, dict):
            merged.update(words)  # later file overrides earlier
    return merged


@lru_cache(maxsize=1)
def _vader_analyzer():
    """Lazy-construct VADER analyzer. Returns None if package missing."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        logger.warning("vaderSentiment not installed; English scoring disabled")
        return None
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=2048)
def score_vader(text: str) -> dict:
    """Score English text using VADER. Returns neutral if VADER unavailable.

    The result shape is identical to ``score_knu`` so callers can stay
    language-agnostic.
    """
    analyzer = _vader_analyzer()
    if analyzer is None or not text.strip():
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0,
                "compound": 0.0, "label": "neutral", "oov_ratio": 0.0}
    s = analyzer.polarity_scores(text)
    compound = s["compound"]
    if compound >= _VADER_POS_THRESHOLD:
        label = "positive"
    elif compound <= _VADER_NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    return {"pos": s["pos"], "neg": s["neg"], "neu": s["neu"],
            "compound": compound, "label": label, "oov_ratio": 0.0}


@lru_cache(maxsize=2048)
def score_knu(text: str) -> dict:
    """Score Korean text via KNU lexicon lookup.

    Tokenizes on Korean syllable runs (skipping digits/punctuation/Latin),
    sums polarity scores from the lexicon, and normalizes to per-token mean.
    Tracks oov_ratio = (#tokens missing from lexicon) / (#total tokens).
    """
    lex = _knu_lexicon()
    if not text.strip() or not lex:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0,
                "compound": 0.0, "label": "neutral", "oov_ratio": 0.0}

    tokens = _KOREAN_TOKEN.findall(text)
    if not tokens:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0,
                "compound": 0.0, "label": "neutral", "oov_ratio": 0.0}

    polarities = []
    oov = 0
    for tok in tokens:
        # 어절 그대로 매칭 → 실패 시 가장 긴 prefix substring 매칭 (조사·어미 흡수)
        score = lex.get(tok)
        if score is None:
            score = _longest_prefix_match(tok, lex)
        if score is None:
            oov += 1
        else:
            polarities.append(score)

    total = len(tokens)
    matched = total - oov
    oov_ratio = oov / total if total else 0.0

    if not polarities:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0,
                "compound": 0.0, "label": "neutral", "oov_ratio": oov_ratio}

    pos_n = sum(1 for p in polarities if p > 0)
    neg_n = sum(1 for p in polarities if p < 0)
    avg = sum(polarities) / len(polarities)
    # Normalize avg from [-2, +2] to [-1, +1] so compound is comparable to VADER
    compound = max(-1.0, min(1.0, avg / 2.0))

    if compound >= _KNU_POS_THRESHOLD:
        label = "positive"
    elif compound <= _KNU_NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return {
        "pos": pos_n / matched if matched else 0.0,
        "neg": neg_n / matched if matched else 0.0,
        "neu": (matched - pos_n - neg_n) / matched if matched else 0.0,
        "compound": compound,
        "label": label,
        "oov_ratio": oov_ratio,
    }


def _longest_prefix_match(token: str, lex: dict[str, int]) -> Optional[int]:
    """KNU 사전의 어휘는 원형(예: '급등')이고 토큰은 활용형('급등하며')일 수
    있다. 토큰의 앞쪽 prefix를 줄여가며 사전을 룩업해 가장 긴 매치를 반환.
    """
    for n in range(len(token), 1, -1):
        score = lex.get(token[:n])
        if score is not None:
            return score
    return None


def score_text(text: str, lang: str) -> dict:
    """Language-aware sentiment scoring router.

    ``lang`` is ``"en"`` or ``"ko"``. Anything else falls back to VADER
    (most retail social posts are English-default).
    """
    if lang == "ko":
        return score_knu(text)
    return score_vader(text)
