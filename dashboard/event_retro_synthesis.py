"""이벤트 리뷰 LLM 통합 호출 — 분류(cluster) + 자연어 단락 동시 출력.

설계:
- ``compute_event_retro`` 가 만든 report 를 받아 LLM 1회 호출 → JSON 으로
  ``{clusters, synthesis_paragraphs}`` 동시 받음.
- JSON 형식 오류 / LLM 실패 → 룰베이스 키워드 매핑으로 fallback (clusters 만).
- 분해 표(``decomposition``) 는 결정적 계산 — LLM 출력 안 따름. cluster 갱신
  후 ``_recompute_decomposition`` 으로 재계산.
- 매매 추천 금지, hedge 표현 유지 (llm_synthesis.py 와 동일 톤).

호출 흐름:
    report = compute_event_retro(ticker, start, end, enable_llm_classify=False)
    report = synthesize_event_retro(report, provider="anthropic", ...)
    # report["timeline"][i]["cluster"] 채워짐
    # report["decomposition"] 재계산
    # report["synthesis"]["paragraphs"][cluster] 채워짐
"""
from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from typing import Optional

from tradingagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = os.environ.get("EVENT_RETRO_PROVIDER", "ollama")
DEFAULT_MODEL = os.environ.get("EVENT_RETRO_MODEL", "gemma4:26b-a4b")


CLUSTER_ENUM = (
    "earnings_rating", "company_specific",
    "industry_regulation", "macro_shock", "drift",
)

CLUSTER_LABEL_KO = {
    "earnings_rating":     "실적·리포트",
    "company_specific":    "회사 고유",
    "industry_regulation": "업종·규제",
    "macro_shock":         "거시 충격",
    "drift":               "기타 drift",
}


# === 룰베이스 키워드 fallback ================================================
# LLM JSON 형식 오류 / LLM 실패 / API 키 미설정 시 사용.
# 우선순위: 더 구체적인 키워드가 위에. 같은 이벤트에 여러 키워드 매칭 시 첫 매칭.
KEYWORD_TO_CLUSTER: list[tuple[str, str]] = [
    # company_specific (자사주·배당·M&A 등 회사 고유)
    ("자사주",   "company_specific"),
    ("자기주식", "company_specific"),
    ("배당",     "company_specific"),
    ("소각",     "company_specific"),
    ("M&A",      "company_specific"),
    ("합병",     "company_specific"),
    ("CEO",      "company_specific"),
    ("경영진",   "company_specific"),
    ("주주환원", "company_specific"),
    # earnings_rating (실적·리포트)
    ("실적",     "earnings_rating"),
    ("어닝",     "earnings_rating"),
    ("영업이익", "earnings_rating"),
    ("순이익",   "earnings_rating"),
    ("목표가",   "earnings_rating"),
    ("리포트",   "earnings_rating"),
    ("증권사",   "earnings_rating"),
    ("매수의견", "earnings_rating"),
    # industry_regulation (업종·규제)
    ("손해율",   "industry_regulation"),
    ("K-ICS",    "industry_regulation"),
    ("규제",     "industry_regulation"),
    ("업황",     "industry_regulation"),
    ("자본규제", "industry_regulation"),
    # macro_shock (거시·정치·관세 등)
    ("관세",     "macro_shock"),
    ("트럼프",   "macro_shock"),
    ("FOMC",     "macro_shock"),
    ("금리",     "macro_shock"),
    ("환율",     "macro_shock"),
    ("연준",     "macro_shock"),
    ("WTI",      "macro_shock"),
    ("국고채",   "macro_shock"),
]


def _classify_by_keywords(text: str) -> Optional[str]:
    """뉴스 제목+snippet 텍스트 → cluster (룰베이스). 매칭 없으면 None."""
    if not text:
        return None
    t = text.lower()
    for kw, cluster in KEYWORD_TO_CLUSTER:
        if kw.lower() in t:
            return cluster
    return None


# macro_shock 우선 키워드 — 명확한 외생 충격만. 한국어 + 영문 둘 다.
# NAVER 는 한국어 제목, Tavily 는 영문 제목 비중이 높아 양쪽 모두 매칭 필요.
# 회사 차원에서도 쓰일 수 있는 모호한 단어("금리 인상" — 인수금융 금리 등) 는 제외.
MACRO_PRIORITY_KEYWORDS: tuple[str, ...] = (
    # 한국어
    "트럼프", "관세",
    "FOMC", "연준",
    "기준금리",       # "금리 인상" 보다 명확 — BOK/Fed 통화정책
    "환율 정책",
    "무역전쟁", "수입금지", "수출제한",
    # 영문 (Tavily 영문 매칭용)
    "trump",          # 한국 종목 회고에서 항상 미국 대통령
    "tariff",         # 보험 "premium tariff" 등 false positive 가능하나 한국 종목 회고엔 드묾
    "trade war",
    "federal reserve",
    "fed funds",
)


def _apply_keyword_overrides(report: dict, clusters: dict[str, str]) -> dict[str, str]:
    """LLM 분류 결과를 매칭 뉴스 *제목* 키워드로 보강.

    1차 — ``MACRO_PRIORITY_KEYWORDS`` 매칭 시 ``macro_shock`` 강제 (외생 충격 우선).
    2차 — 그 외 ``KEYWORD_TO_CLUSTER`` 매칭 시 해당 cluster 로 override.
    매칭 키워드 없으면 LLM 분류 유지.

    제목만 검사 — snippet 은 보강 인용/단순 본문이라 노이즈 false positive 가 큼.
    예: K-ICS 보험사 뉴스에 외신 "Bessent" 같은 무관 본문 섞여 들어와 macro 오분류
    하는 케이스 방지.

    근거: gemma 등 소형 모델이 명백한 macro 이벤트(트럼프 관세 등) 를 다른 cluster
    로 분류하는 비결정성 보완. 제목 키워드 룰베이스가 명확하면 LLM 보다 우선시.
    """
    timeline = report.get("timeline") or []
    overridden = dict(clusters)
    for e in timeline:
        date_str = e.get("date")
        if not date_str:
            continue
        # 제목만 합치기 (snippet 노이즈 회피)
        titles = [(n.get("title") or "") for n in e.get("matched_news") or []]
        title_text = " ".join(titles)
        if not title_text.strip():
            continue
        title_lower = title_text.lower()
        # 1차: macro_shock 우선
        if any(kw.lower() in title_lower for kw in MACRO_PRIORITY_KEYWORDS):
            overridden[date_str] = "macro_shock"
            continue
        # 2차: 일반 룰베이스
        rule_cluster = _classify_by_keywords(title_text)
        if rule_cluster:
            overridden[date_str] = rule_cluster
    return overridden


# === LLM 통합 호출 ===========================================================
# Sonnet 수준 답안 예시 — gemma 등 소형 모델용 few-shot. 분량/구조/인용 패턴 학습용.
_FEWSHOT_EXAMPLE = textwrap.dedent("""
[예시 답안 — earnings_rating 카테고리]
"2월 말 공시된 4분기 실적에서 매출은 전년 동기 대비 증가한 반면 영업이익과 순이익은
감소했다는 보도가 확인되며, 이익 질 저하에 대한 우려가 주가에 반영됐을 가능성이
시사된다. 4월 중순에는 IFRS17·K-ICS 체계 하 ROE 최상위 분석 리포트가 제시되어
펀더멘털 재평가 계기가 됐을 여지가 있다. 두 이벤트는 분기 실적과 회계·자본 규제
전환 효과가 시장에서 엇갈리게 반영된 구간으로 해석된다."

[예시의 패턴 — 반드시 이 구조 따라하기]
① 매칭 뉴스의 구체 사실 인용 (날짜·키워드 명시) — "2월 말 공시된 4분기 실적"
② 그 사실이 주가에 어떻게 작용했을지 해석 — "이익 질 저하 우려가 반영"
③ hedge 표현으로 단정 회피 — "시사된다", "가능성이 있다"
④ 분량 — 3~5 문장, 최소 150자
""").strip()


_SYSTEM_PROMPT = textwrap.dedent("""
당신은 한국 주식 이벤트 리뷰 분석가입니다.
사용자는 한 종목의 특정 기간 가격 이벤트와 매칭된 뉴스를 줄 것입니다.
이를 읽고 다음 두 가지를 한 번에 출력하세요.

## 1. 이벤트 분류 (clusters)
각 이벤트 일자에 대해 5 고정 클러스터 중 정확히 하나를 배정:
- "earnings_rating"     — 실적 발표·증권사 리포트·목표가 변경
- "company_specific"    — 자사주·배당·M&A·소송·경영진·주주환원
- "industry_regulation" — 업종 환경·규제·손해율·자본규제
- "macro_shock"         — 거시·금리·환율·정책·관세·정치
- "drift"               — 위 4개로 설명 안 됨 (매칭 뉴스 없거나 명확치 않음)

## 2. 클러스터별 자연어 단락 (synthesis_paragraphs)
5개 클러스터 각각에 대해 자연어 설명을 작성. **반드시 다음 형식 준수**:
- **분량 최소 4 문장 / 150자 이상**. 짧으면 안 됨. 매칭 뉴스가 풍부할 땐 5~6 문장까지 늘림.
- **작성 순서 강제** — 한 단락 안에서:
   ① 매칭 뉴스의 *구체 사실*을 1개 이상 명시적으로 언급 (날짜·기관·키워드 포함)
   ② 그 사실이 주가/시장에 어떻게 작용했을지 해석
   ③ hedge 표현 (~로 해석됨, ~의 가능성, ~시사, ~여지) 으로 결론
- **수치 출력 금지** — % 변동·금액 등 숫자 절대 출력하지 마세요. 분해 표는
  별도 결정적 계산이 처리합니다.
- 해당 클러스터에 속하는 이벤트가 없으면 "본 기간에 해당 카테고리 이벤트 없음." 한 줄.
- 매매 추천 금지. 단정 표현 금지.

{fewshot}

## 출력 형식 — JSON 만 (마크다운/설명 없이)
```json
{{
  "clusters": {{"2025-03-27": "company_specific", "2025-04-07": "macro_shock", ...}},
  "synthesis_paragraphs": {{
    "earnings_rating":     "...",
    "company_specific":    "...",
    "industry_regulation": "...",
    "macro_shock":         "...",
    "drift":               "..."
  }}
}}
```

clusters 키는 입력으로 받은 모든 이벤트 일자에 대해 정확히 하나씩.
synthesis_paragraphs 는 5 클러스터 모두 포함.
""").strip().format(fewshot=_FEWSHOT_EXAMPLE)


# 입력 컨텍스트 한도 — 모델이 인용 가능한 사실의 양을 결정
NEWS_PER_EVENT = 5        # 이벤트당 매칭 뉴스 최대 (기존 3)
SNIPPET_CHARS = 300       # 뉴스 snippet 최대 길이 (기존 120)
TITLE_CHARS = 100         # 뉴스 제목 최대 (기존 80)


def _build_user_message(report: dict) -> str:
    meta = report.get("meta", {})
    ps = report.get("period_summary") or {}
    timeline = report.get("timeline") or []
    macro = report.get("macro") or {}

    header = (
        f"종목: {meta.get('company_name') or meta.get('ticker')} ({meta.get('ticker')})\n"
        f"기간: {meta.get('start')} ~ {meta.get('end')}\n"
    )
    if ps:
        header += (
            f"기간 수익률 {ps.get('total_return_pct'):+.2f}% · "
            f"최대 낙폭 {ps.get('max_drawdown_pct'):+.2f}%\n"
        )

    # 거시 — yfinance 4개
    macro_line = []
    for k, label in [("kospi", "KOSPI"), ("usdkrw", "USD/KRW"),
                     ("us_10y", "미국 10Y"), ("wti", "WTI")]:
        v = macro.get(k, {})
        if v.get("available") and v.get("change_pct") is not None:
            macro_line.append(f"{label} {v['change_pct']:+.1f}%")
    if macro_line:
        header += "거시(같은 기간): " + " / ".join(macro_line) + "\n"

    # BOK 추가 — 회사채 AA-/국고채/기준금리 (회사 고유 / 업종 클러스터 해석에 활용)
    bok = macro.get("bok") or {}
    bok_lines = []
    for k, label in [("base_rate", "BOK 기준금리"),
                     ("corp_aa_yield", "회사채 AA-"),
                     ("treasury_3y", "국고채 3Y")]:
        v = bok.get(k) or {}
        if v.get("available"):
            ch = v.get("change_bp")
            bok_lines.append(
                f"{label} {v.get('start'):.2f}%→{v.get('end'):.2f}%"
                + (f" (Δ{ch:+.1f}bp)" if ch is not None else "")
            )
    if bok_lines:
        header += "BOK 거시: " + " / ".join(bok_lines) + "\n"

    sections = [header, "## 이벤트 + 매칭 뉴스"]
    for e in timeline:
        sigs = []
        if e.get("is_price_jump"):
            sigs.append("가격±")
        if e.get("is_volume_anomaly"):
            sigs.append("거래량↑")
        if e.get("is_period_anchor"):
            sigs.append("anchor")
        line = f"- {e['date']} ({e['pct']:+.2f}%, {' '.join(sigs) or '—'})"
        news = e.get("matched_news") or []
        if news:
            sections.append(line)
            for n in news[:NEWS_PER_EVENT]:
                title = (n.get("title") or "").strip()[:TITLE_CHARS]
                snip = (n.get("snippet") or "").strip()[:SNIPPET_CHARS]
                sections.append(f"    · {title}")
                if snip:
                    sections.append(f"      → {snip}")
        else:
            sections.append(line + "  (매칭 뉴스 없음)")

    return "\n".join(sections)


def _parse_llm_json(text: str) -> Optional[dict]:
    """LLM 출력에서 JSON 추출. ```json``` fence 도 처리."""
    if not text:
        return None
    text = text.strip()
    # fence 제거
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # 첫 { 부터 마지막 } 까지 추출
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("LLM JSON parse failed: %s", exc)
        return None


def _fallback_classify(report: dict) -> dict:
    """LLM 실패 시 룰베이스 키워드 매핑으로 cluster 할당."""
    timeline = report.get("timeline") or []
    clusters: dict[str, str] = {}
    for e in timeline:
        # 매칭 뉴스 title + snippet 합쳐 키워드 매칭
        texts = []
        for n in e.get("matched_news") or []:
            texts.append(n.get("title") or "")
            texts.append(n.get("snippet") or "")
        cluster = _classify_by_keywords(" ".join(texts))
        clusters[e["date"]] = cluster or "drift"
    return {
        "clusters": clusters,
        "synthesis_paragraphs": {
            c: "_LLM 합성 비활성 — 자연어 단락 생략._"
            for c in CLUSTER_ENUM
        },
    }


def _recompute_decomposition(
    timeline: list[dict], total_return_pct: float,
) -> dict[str, float]:
    """cluster 갱신 후 분해 재계산. event_retro._compute_decomposition 과 동일 로직."""
    sums = {c: 0.0 for c in CLUSTER_ENUM}
    for e in timeline:
        c = e.get("cluster") if e.get("cluster") in CLUSTER_ENUM else "drift"
        sums[c] += e.get("pct", 0.0)
    explained = sum(sums[c] for c in CLUSTER_ENUM if c != "drift")
    sums["drift"] = round(total_return_pct - explained, 4)
    return {c: round(v, 4) for c, v in sums.items()}


# === Chained 모드 — 분류 1회 + 클러스터별 단락 5회 ============================
# 소형 모델(gemma 등) 이 단일 호출에서 짧게 끊는 문제를 회피.
# 각 호출이 한 주제에만 집중 → 단락이 풍부해짐.

_CLASSIFY_PROMPT = textwrap.dedent("""
당신은 한국 주식 이벤트 분류기입니다.
주어진 이벤트 + 매칭 뉴스를 읽고 각 이벤트 일자에 5 고정 클러스터 중 하나를 배정.

클러스터:
- "earnings_rating"     — 실적·증권사 리포트·목표가
- "company_specific"    — 자사주·배당·M&A·소송·경영진·계열사
- "industry_regulation" — 업종 환경·규제·손해율·자본규제
- "macro_shock"         — 거시·금리·환율·정책·관세·정치
- "drift"               — 위 4개로 설명 안 됨 (매칭 뉴스 없거나 무관)

출력은 JSON 만:
```json
{"clusters": {"2025-03-27": "company_specific", "2025-04-07": "macro_shock", ...}}
```

clusters 키는 입력 모든 이벤트 일자에 대해 정확히 하나씩.
""").strip()


_PARAGRAPH_PROMPT = textwrap.dedent("""
당신은 한국 주식 이벤트 리뷰 분석가입니다.
하나의 클러스터에 속하는 이벤트와 매칭 뉴스가 주어집니다.
**이 클러스터에 대한 자연어 단락 1개만** 작성하세요.

## 형식 강제
- **분량 최소 4 문장 / 150자 이상**. 매칭 뉴스가 풍부하면 5~6 문장.
- **작성 순서** — 한 단락 안에서:
   ① 매칭 뉴스의 *구체 사실* 1개 이상 인용 (날짜·기관·키워드 명시)
   ② 그 사실이 주가/시장에 작용한 메커니즘 해석
   ③ hedge 표현으로 결론 (~로 해석됨, ~가능성, ~시사, ~여지)
- **수치 출력 금지** — %·금액 등 숫자 절대 사용 금지
- 단정 금지, 매매 추천 금지
- **이벤트가 없으면** "본 기간에 해당 카테고리 이벤트 없음." 한 줄만

{fewshot}

## 출력
JSON 이나 마크다운 헤더 없이 **단락 텍스트만** 출력. 코드블록 금지.
""").strip().format(fewshot=_FEWSHOT_EXAMPLE)


def _build_cluster_context(
    report: dict, cluster: str, classified_timeline: list[dict],
) -> str:
    """클러스터별 단락 호출용 컨텍스트 — 그 클러스터 이벤트 + 매칭 뉴스만."""
    meta = report.get("meta", {})
    ps = report.get("period_summary") or {}
    macro = report.get("macro") or {}

    header = (
        f"종목: {meta.get('company_name') or meta.get('ticker')} ({meta.get('ticker')})\n"
        f"기간: {meta.get('start')} ~ {meta.get('end')}\n"
        f"클러스터: {cluster} ({CLUSTER_LABEL_KO.get(cluster, cluster)})\n"
    )
    if ps:
        header += (
            f"기간 수익률 {ps.get('total_return_pct'):+.2f}% · "
            f"최대 낙폭 {ps.get('max_drawdown_pct'):+.2f}%\n"
        )

    macro_line = []
    for k, label in [("kospi", "KOSPI"), ("usdkrw", "USD/KRW"),
                     ("us_10y", "미국 10Y"), ("wti", "WTI")]:
        v = macro.get(k, {})
        if v.get("available") and v.get("change_pct") is not None:
            macro_line.append(f"{label} {v['change_pct']:+.1f}%")
    if macro_line:
        header += "거시: " + " / ".join(macro_line) + "\n"

    # 이 클러스터에 분류된 이벤트만
    cluster_events = [e for e in classified_timeline if e.get("cluster") == cluster]
    if not cluster_events:
        return header + "\n## 이벤트\n_본 클러스터에 속한 이벤트 없음._"

    sections = [header, "\n## 이 클러스터 이벤트 + 매칭 뉴스"]
    for e in cluster_events:
        sections.append(f"\n- **{e['date']} ({e['pct']:+.2f}%)**")
        for n in (e.get("matched_news") or [])[:NEWS_PER_EVENT]:
            title = (n.get("title") or "").strip()[:TITLE_CHARS]
            snip = (n.get("snippet") or "").strip()[:SNIPPET_CHARS]
            sections.append(f"    · {title}")
            if snip:
                sections.append(f"      → {snip}")
    return "\n".join(sections)


def _call_llm(provider: str, model: str, system: str, user: str) -> tuple[str, Optional[str]]:
    """공용 LLM 호출. (content, error). error None 이면 성공."""
    try:
        client = create_llm_client(provider, model)
        llm = client.get_llm()
        result = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        content = getattr(result, "content", str(result))
        return content or "", None
    except Exception as exc:
        return "", f"LLM 호출 실패({provider}/{model}): {type(exc).__name__}: {exc}"


def _synthesize_single(
    report: dict, provider: str, model: str,
) -> tuple[Optional[dict], Optional[str]]:
    """기존 통합 호출 — 분류 + 5 단락 동시 출력 1회."""
    user_msg = _build_user_message(report)
    content, error = _call_llm(provider, model, _SYSTEM_PROMPT, user_msg)
    if error:
        return None, error
    parsed = _parse_llm_json(content)
    if not parsed or "clusters" not in parsed:
        return None, f"LLM JSON 출력 오류. 원본 {len(content)}자: {content[:200]}"
    # 룰베이스 후처리 — 명확한 macro 키워드 등은 LLM 분류 override
    parsed["clusters"] = _apply_keyword_overrides(report, parsed["clusters"])
    return parsed, None


def _synthesize_chained(
    report: dict, provider: str, model: str,
) -> tuple[Optional[dict], Optional[str]]:
    """분류 1회 + 클러스터별 단락 5회 호출. 소형 모델용.

    각 호출이 한 주제에만 집중 → 단락이 풍부해짐. ollama 무료라 호출 5배 OK.
    실패 단계마다 부분 결과로 fallback (전체 폐기 안 함).
    """
    # 1. 분류
    user_msg = _build_user_message(report)
    content, error = _call_llm(provider, model, _CLASSIFY_PROMPT, user_msg)
    if error:
        return None, error
    parsed_cls = _parse_llm_json(content)
    if not parsed_cls or "clusters" not in parsed_cls:
        return None, f"분류 JSON 오류. 원본 {len(content)}자: {content[:200]}"

    # 룰베이스 후처리 — macro 키워드 등 명확 매칭 시 cluster override.
    # cluster별 단락 호출은 보강된 분류 기준으로 진행 (paragraph가 실제 cluster 와 매칭).
    clusters = _apply_keyword_overrides(report, parsed_cls["clusters"])

    # 2. timeline 에 cluster 임시 채우고 클러스터별 호출
    classified_timeline = []
    for e in report.get("timeline") or []:
        c = clusters.get(e["date"])
        if c not in CLUSTER_ENUM:
            c = "drift"
        classified_timeline.append({**e, "cluster": c})

    paragraphs: dict[str, str] = {}
    errors: list[str] = []
    for cluster in CLUSTER_ENUM:
        ctx = _build_cluster_context(report, cluster, classified_timeline)
        para, err = _call_llm(provider, model, _PARAGRAPH_PROMPT, ctx)
        if err:
            errors.append(f"{cluster}: {err}")
            paragraphs[cluster] = "_(단락 생성 실패)_"
            continue
        # 단락 텍스트 정리 — JSON·코드블록 잔재 제거
        para = para.strip()
        if para.startswith("```"):
            # 잔재 ```...``` 제거
            para = re.sub(r"^```[a-z]*\n?", "", para)
            para = re.sub(r"\n?```$", "", para)
        paragraphs[cluster] = para.strip() or "_(생성 안 됨)_"

    return {
        "clusters": clusters,
        "synthesis_paragraphs": paragraphs,
    }, ("; ".join(errors) if errors else None)


def _should_chain(provider: str, mode: str) -> bool:
    """mode='auto' 시 provider 기반 결정. ollama 만 chained."""
    if mode == "single":
        return False
    if mode == "chained":
        return True
    # auto
    return provider.lower() == "ollama"


def synthesize_event_retro(
    report: dict,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    mode: str = "auto",
) -> dict:
    """report → LLM 호출 → cluster + synthesis 단락 갱신.

    mode:
        - "auto": ollama provider 면 chained, 그 외 single (기본).
        - "single": 1회 통합 호출 (분류+단락 동시).
        - "chained": 분류 1회 + 클러스터별 단락 5회 (총 6회).

    Returns: 갱신된 report (새 dict). 원본 변경 안 함.
    """
    new_report = {**report}
    timeline = list(new_report.get("timeline") or [])
    if not timeline:
        return new_report

    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL
    use_chain = _should_chain(provider, mode)

    if use_chain:
        parsed, error = _synthesize_chained(new_report, provider, model)
        actual_mode = "chained"
    else:
        parsed, error = _synthesize_single(new_report, provider, model)
        actual_mode = "single"

    if not parsed:
        parsed = _fallback_classify(new_report)
        warnings = list(new_report.get("warnings") or [])
        if error:
            warnings.append(error)
        new_report["warnings"] = warnings
        method = "rule_based_fallback"
    else:
        if error:
            # 부분 실패 (chained 모드 일부 단락 실패) — warnings 만 추가, 결과는 유지
            warnings = list(new_report.get("warnings") or [])
            warnings.append(f"LLM 부분 실패: {error}")
            new_report["warnings"] = warnings
        method = f"llm_{actual_mode}"

    # 1. cluster 채우기 (timeline 갱신)
    cluster_map = parsed.get("clusters") or {}
    updated_timeline = []
    for e in timeline:
        c = cluster_map.get(e["date"])
        if c not in CLUSTER_ENUM:
            c = "drift"
        updated_timeline.append({**e, "cluster": c})
    new_report["timeline"] = updated_timeline

    # 2. 분해 재계산 (결정적, LLM 출력 안 따름)
    ps = new_report.get("period_summary") or {}
    if ps:
        new_report["decomposition"] = _recompute_decomposition(
            updated_timeline, ps.get("total_return_pct", 0.0),
        )

    # 3. synthesis 단락 — 5 enum 모두 보장
    paragraphs = parsed.get("synthesis_paragraphs") or {}
    full_paragraphs = {
        c: paragraphs.get(c) or "_(생성 안 됨)_"
        for c in CLUSTER_ENUM
    }
    new_report["synthesis"] = {
        "paragraphs": full_paragraphs,
        "provider": provider,
        "model": model,
        "method": method,
    }
    return new_report


def render_synthesis_markdown(report: dict) -> str:
    """synthesis 단락 → 마크다운 5 클러스터 섹션."""
    syn = report.get("synthesis") or {}
    paragraphs = syn.get("paragraphs") or {}
    if not paragraphs:
        return ""
    lines: list[str] = ["## 8. 클러스터별 자연어 합성", ""]
    if syn.get("method") == "rule_based_fallback":
        lines.append("_(LLM 합성 실패 — 룰베이스 분류만, 자연어 단락 생략)_")
        lines.append("")
        return "\n".join(lines)
    for c in CLUSTER_ENUM:
        lines.append(f"### {CLUSTER_LABEL_KO[c]} ({c})")
        lines.append("")
        lines.append(paragraphs.get(c, "_(생성 안 됨)_"))
        lines.append("")
    return "\n".join(lines)
