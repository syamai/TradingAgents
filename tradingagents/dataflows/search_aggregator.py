"""뉴스 검색 — 4 source 무료 우선 폴백 체인 + dedup.

폴백 순서: naver_open → naver_finance → tavily → serper → anthropic 보강.
무료(naver) 가 충분히 결과를 주면 유료 호출을 skip — 비용 최소화.

키 미설정 provider 는 자동 skip + 로그.

API 키 환경변수:
  NAVER_CLIENT_ID / NAVER_CLIENT_SECRET   (naver_open)
  TAVILY_API_KEY                          (tavily)
  SERPER_API_KEY                          (serper)
  TRADINGAGENTS_ANTHROPIC_API_KEY         (anthropic web_search; 미설정 시 ANTHROPIC_API_KEY 폴백)

모델 override:
  TRADINGAGENTS_SEARCH_ANTHROPIC_MODEL    (anthropic web_search 모델; 미설정 시 claude-sonnet-4-6)
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional, Sequence
from urllib.parse import urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)


# === 데이터 클래스 ==============================================================
@dataclass
class NewsHit:
    title: str
    url: str
    published: Optional[str]    # ISO YYYY-MM-DD 또는 None
    snippet: str                # 200자 이내
    source_label: str           # 사람 친화 라벨
    raw_provider: str           # "naver_open"|"naver_finance"|"tavily"|"serper"|"anthropic"


# === 설정 ======================================================================
DEFAULT_PROVIDER_ORDER = (
    "naver_open",
    "naver_finance",
    "tavily",
    "serper",
    "anthropic",
)
EARLY_STOP_RESULTS = 8     # 누적 8건 이상 시 다음 provider 미호출
SNIPPET_MAX = 200

# anthropic web_search 보강에 쓰는 모델. env 로 override, 미설정 시 trading-ai
# 표준 Anthropic 모델. web_search_20250305 서버 툴이 Anthropic 모델 전용이라
# deep/quick_think_llm config 를 재사용하지 않는다 (그쪽 기본값은 gpt 계열).
_ANTHROPIC_SEARCH_MODEL_ENV = "TRADINGAGENTS_SEARCH_ANTHROPIC_MODEL"
_DEFAULT_ANTHROPIC_SEARCH_MODEL = "claude-sonnet-4-6"

# === 노이즈 필터 ==============================================================
# 명백한 광고/스팸/리딩방 패턴 — 매칭 시 hit 제거. 제목+snippet 둘 다 검사.
# 보수적 — 정확히 매칭하는 것만 (false positive 최소화).
NOISE_PATTERNS: tuple[str, ...] = (
    "원금보장", "수익률 보장", "수익 보장",
    "리딩방", "텔레그램방", "오픈채팅방",
    "지금 가입", "지금 신청", "무료 가입",
    "투자대회", "수익률 대회",
    "광고)", "[광고]", "(AD)",
    "할인쿠폰", "사은품",
    "추천종목 무료", "급등주 무료",
)

# 광고/스팸 도메인 — 정확 매칭 후 차단. 뉴스 도메인은 절대 포함 안 함.
NOISE_DOMAINS: frozenset[str] = frozenset({
    "ads.naver.com", "ad.naver.com",
    "ad.daum.net",
})


def _is_noise(hit: "NewsHit") -> bool:
    """광고/스팸 hit 판별. 보수적 — 명확한 것만 차단."""
    title_snip = f"{hit.title or ''}\n{hit.snippet or ''}".lower()
    for pat in NOISE_PATTERNS:
        if pat.lower() in title_snip:
            return True
    try:
        host = urlparse(hit.url).netloc.lower()
        if host in NOISE_DOMAINS:
            return True
    except Exception:
        pass
    # 너무 짧은 제목 (3자 미만) — 거의 정상 헤드라인 아님
    if len(_strip_html(hit.title or "").strip()) < 3:
        return True
    return False


# === 공용 유틸 ==================================================================
_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_KO_PARTICLE = re.compile(r"[을를이가에서의는은와과로으로]+(?=\s|$)")


def _strip_html(s: str) -> str:
    return _HTML_TAG.sub("", s or "")


def _truncate(s: str, limit: int = SNIPPET_MAX) -> str:
    s = _WS.sub(" ", _strip_html(s)).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def _normalize_url(u: str) -> str:
    try:
        p = urlparse(u)
        # query/fragment 제거, host lowercase
        return urlunparse((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", "", ""))
    except Exception:
        return u


def _normalize_title(t: str) -> str:
    """한국어 조사 + 공백 squash + 소문자."""
    s = _strip_html(t).lower()
    s = _KO_PARTICLE.sub(" ", s)
    s = re.sub(r"[^\w가-힣]+", " ", s)
    return _WS.sub(" ", s).strip()


def _within_window(published: Optional[str], since: date, until: date) -> bool:
    if not published:
        return True  # 날짜 없으면 일단 포함 (post-filter 보수적으로)
    try:
        d = datetime.strptime(published[:10], "%Y-%m-%d").date()
    except Exception:
        return True
    return since <= d <= until


def _parse_pub_date(s: Optional[str]) -> Optional[str]:
    """다양한 형식 → ISO YYYY-MM-DD. 실패 시 None.

    지원: ISO(YYYY-MM-DD 또는 YYYY-MM-DDTHH:MM:SS), RFC 822 (+0900 / GMT).
    """
    if not s:
        return None
    s = s.strip()
    # ISO prefix
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    # RFC 822 — NAVER (+0900), Tavily (GMT) 둘 다 처리
    for fmt in ("%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


# === Provider 어댑터 ============================================================

def _search_naver_open(
    query: str, since: date, until: date, *, max_results: int,
) -> list[NewsHit]:
    cid = os.environ.get("NAVER_CLIENT_ID")
    csec = os.environ.get("NAVER_CLIENT_SECRET")
    if not (cid and csec):
        logger.info("naver_open skip — NAVER_CLIENT_ID/SECRET missing")
        return []
    try:
        # 회고용: 정확도순(sim) + 30건 over-fetch → _within_window 필터로 회고 기간 추출.
        # sort=date 는 최신순이라 1년 전 회고에 매칭 0건이 됨.
        resp = requests.get(
            "https://openapi.naver.com/v1/search/news.json",
            params={"query": query, "display": 30, "sort": "sim"},
            headers={"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec},
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        logger.warning("naver_open fetch failed: %s", exc)
        return []

    out: list[NewsHit] = []
    for item in body.get("items", []):
        pub = _parse_pub_date(item.get("pubDate"))
        if not _within_window(pub, since, until):
            continue
        out.append(NewsHit(
            title=_strip_html(item.get("title", "")),
            url=item.get("originallink") or item.get("link", ""),
            published=pub,
            snippet=_truncate(item.get("description", "")),
            source_label="Naver Open API",
            raw_provider="naver_open",
        ))
        if len(out) >= max_results:
            break
    return out


def _search_naver_finance(
    query: str, since: date, until: date,
    *, max_results: int, ticker: Optional[str],
) -> list[NewsHit]:
    if not ticker:
        return []
    try:
        from tradingagents.dataflows.naver_news import _fetch_page, _extract_items
    except Exception as exc:
        logger.warning("naver_finance import failed: %s", exc)
        return []

    out: list[NewsHit] = []
    try:
        for page in range(1, 6):     # 최대 5페이지
            soup = _fetch_page(ticker, page)
            if soup is None:
                break
            items = _extract_items(soup)
            if not items:
                break
            for it in items:
                # _extract_items 가 datetime 객체로 'date' 반환 (이미 파싱됨)
                dt = it.get("date")
                pub = dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else None
                if not _within_window(pub, since, until):
                    continue
                out.append(NewsHit(
                    title=it.get("title", ""),
                    url=it.get("link", ""),
                    published=pub,
                    snippet=_truncate(it.get("title", "")),
                    source_label="Naver 금융 종목뉴스",
                    raw_provider="naver_finance",
                ))
                if len(out) >= max_results:
                    return out
    except Exception as exc:
        logger.warning("naver_finance fetch failed: %s", exc)
    return out


def _search_tavily(
    query: str, since: date, until: date, *, max_results: int,
) -> list[NewsHit]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        logger.info("tavily skip — TAVILY_API_KEY missing")
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "max_results": min(max_results, 20),
                "topic": "news",
                "start_date": since.strftime("%Y-%m-%d"),
                "end_date": until.strftime("%Y-%m-%d"),
            },
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        logger.warning("tavily fetch failed: %s", exc)
        return []

    out: list[NewsHit] = []
    for r in body.get("results", []):
        pub = _parse_pub_date(r.get("published_date"))
        if pub and not _within_window(pub, since, until):
            continue
        out.append(NewsHit(
            title=r.get("title", ""),
            url=r.get("url", ""),
            published=pub,
            snippet=_truncate(r.get("content", "")),
            source_label="Tavily",
            raw_provider="tavily",
        ))
    return out


def _search_serper(
    query: str, since: date, until: date, *, max_results: int,
) -> list[NewsHit]:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        logger.info("serper skip — SERPER_API_KEY missing")
        return []
    try:
        # cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY
        tbs = (
            f"cdr:1,cd_min:{since.strftime('%m/%d/%Y')},"
            f"cd_max:{until.strftime('%m/%d/%Y')}"
        )
        resp = requests.post(
            "https://google.serper.dev/news",
            json={
                "q": query, "tbs": tbs,
                "num": min(max_results, 20), "hl": "ko", "gl": "kr",
            },
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        logger.warning("serper fetch failed: %s", exc)
        return []

    out: list[NewsHit] = []
    for r in body.get("news", []):
        # serper의 date는 "2 days ago" 같은 상대 표현 — 미파싱
        out.append(NewsHit(
            title=r.get("title", ""),
            url=r.get("link", ""),
            published=None,  # 상대 표현이라 정확히 파싱 어려움
            snippet=_truncate(r.get("snippet", "")),
            source_label="Serper (Google News)",
            raw_provider="serper",
        ))
    return out


def _search_anthropic(
    query: str, since: date, until: date, *, max_results: int,
) -> list[NewsHit]:
    # trading-ai 전용 키 우선, 미설정 시 Hermes 공유 키 폴백 (anthropic_client 와 동일 순서).
    key = os.environ.get("TRADINGAGENTS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.info("anthropic skip — TRADINGAGENTS_ANTHROPIC_API_KEY/ANTHROPIC_API_KEY missing")
        return []
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — pip install anthropic")
        return []

    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.environ.get(_ANTHROPIC_SEARCH_MODEL_ENV, _DEFAULT_ANTHROPIC_SEARCH_MODEL),
            max_tokens=2048,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
            }],
            messages=[{
                "role": "user",
                "content": (
                    f"기간 {since.strftime('%Y-%m-%d')} ~ {until.strftime('%Y-%m-%d')}에 "
                    f"한국어 뉴스로 '{query}' 검색해서 제목·URL·날짜 목록만 정리해줘. "
                    f"각 항목은 한 줄에 'YYYY-MM-DD | 제목 | URL' 형식으로."
                ),
            }],
        )
    except Exception as exc:
        logger.warning("anthropic web_search failed: %s", exc)
        return []

    out: list[NewsHit] = []
    # 응답 content 블록에서 텍스트 추출
    for block in msg.content:
        if getattr(block, "type", None) != "text":
            continue
        for line in (block.text or "").splitlines():
            parts = [p.strip() for p in line.split("|", 2)]
            if len(parts) < 3:
                continue
            pub, title, url = parts[0], parts[1], parts[2]
            # YYYY-MM-DD 형식 검증
            try:
                datetime.strptime(pub[:10], "%Y-%m-%d")
            except Exception:
                continue
            if not _within_window(pub, since, until):
                continue
            if not url.startswith("http"):
                continue
            out.append(NewsHit(
                title=title, url=url,
                published=pub[:10],
                snippet=_truncate(title),
                source_label="Anthropic web_search",
                raw_provider="anthropic",
            ))
            if len(out) >= max_results:
                break
    return out


_PROVIDER_FN: dict[str, callable] = {
    "naver_open": _search_naver_open,
    "naver_finance": _search_naver_finance,
    "tavily": _search_tavily,
    "serper": _search_serper,
    "anthropic": _search_anthropic,
}


# === 메인 함수 =================================================================
def search_news(
    query: str,
    since,           # date or YYYY-MM-DD str
    until,
    *,
    max_results: int = 20,
    providers: Optional[Sequence[str]] = None,
    ticker: Optional[str] = None,
) -> list[NewsHit]:
    """무료 우선 폴백 체인. dedup 후 누적 ``max_results`` 까지 반환.

    early stop: 누적 ``EARLY_STOP_RESULTS`` 건 이상이면 다음 provider 미호출.
    """
    if isinstance(since, str):
        since = datetime.strptime(since, "%Y-%m-%d").date()
    if isinstance(until, str):
        until = datetime.strptime(until, "%Y-%m-%d").date()
    providers = list(providers or DEFAULT_PROVIDER_ORDER)

    seen_url: set[str] = set()
    seen_title: set[str] = set()
    out: list[NewsHit] = []
    remaining = lambda: max_results - len(out)

    for prov in providers:
        if len(out) >= EARLY_STOP_RESULTS:
            break
        fn = _PROVIDER_FN.get(prov)
        if not fn:
            continue
        kwargs = {"max_results": remaining()}
        if prov == "naver_finance":
            kwargs["ticker"] = ticker
        try:
            hits = fn(query, since, until, **kwargs)
        except Exception as exc:
            logger.warning("provider %s failed: %s", prov, exc)
            continue
        for h in hits:
            if _is_noise(h):
                continue
            u_norm = _normalize_url(h.url)
            t_norm = _normalize_title(h.title)
            if u_norm in seen_url or t_norm in seen_title:
                continue
            seen_url.add(u_norm)
            seen_title.add(t_norm)
            out.append(h)
            if len(out) >= max_results:
                return out
    return out


def hits_to_dicts(hits: list[NewsHit]) -> list[dict]:
    return [asdict(h) for h in hits]
