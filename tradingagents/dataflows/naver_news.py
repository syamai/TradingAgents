"""네이버 금융 종목 뉴스 어댑터.

엔드포인트: ``finance.naver.com/item/news_news.naver?code={6자리}&page={n}``
응답은 EUC-KR 인코딩된 HTML 테이블. 한 페이지에 약 20개의 뉴스 항목이
``<tr class="..."><td class="title">...</td>...</tr>`` 구조로 들어있다.

LLM에 그대로 주입할 수 있도록 ``### 제목 (source: 언론사)\\n날짜\\nLink: ...``
형태의 마크다운 문자열로 변환한다.

graceful degradation: 모든 예외를 ``<unavailable: ...>`` 문자열로 변환해
``yahoo_naver`` wrapper가 다른 소스 결과는 보존하도록 한다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .config import get_config
from .korean_utils import to_naver_code

_BASE = "https://finance.naver.com/item/news_news.naver"
_HEADERS = {
    # 네이버는 UA 없이 요청 시 응답을 제한하므로 일반 브라우저로 위장.
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    "Referer": "https://finance.naver.com/",
}
_TIMEOUT = 5.0
_MAX_PAGES = 5  # 최근 약 100개 뉴스까지 조회 (페이지당 ~20개)


def _parse_naver_date(raw: str) -> Optional[datetime]:
    """네이버 날짜 문자열을 ``datetime``으로. 실패하면 None."""
    if not raw:
        return None
    raw = raw.strip().replace("\xa0", " ")
    for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _fetch_page(code: str, page: int) -> Optional[BeautifulSoup]:
    try:
        resp = requests.get(
            _BASE,
            params={"code": code, "page": page, "sm": "title_entity_id.basic"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        # 페이지마다 인코딩이 섞여있어 명시 디코딩이 불안정. BS4가 meta
        # charset을 보고 자동 추론하도록 raw bytes로 전달.
        return BeautifulSoup(resp.content, "lxml")
    except Exception:
        return None


def _extract_items(soup: BeautifulSoup) -> list[dict]:
    """뉴스 테이블에서 항목 추출. 페이지 구조 변동에 대비해 두 가지 셀렉터 시도."""
    items: list[dict] = []

    # 표준 케이스: tr.first/tr.last + td.title 구조
    rows = soup.select("table.type5 tr") or soup.select("tr")
    for tr in rows:
        title_td = tr.select_one("td.title")
        if not title_td:
            continue
        a = title_td.select_one("a")
        if not a:
            continue
        title = a.get_text(strip=True)
        link = a.get("href", "").strip()
        if link and link.startswith("/"):
            link = "https://finance.naver.com" + link

        info_td = tr.select_one("td.info")
        date_td = tr.select_one("td.date")

        publisher = info_td.get_text(strip=True) if info_td else ""
        date_raw = date_td.get_text(strip=True) if date_td else ""
        date_dt = _parse_naver_date(date_raw)

        if title:
            items.append(
                {"title": title, "publisher": publisher, "date_raw": date_raw, "date": date_dt, "link": link}
            )
    return items


def fetch_naver_news(ticker: str, start_date: str, end_date: str) -> str:
    """네이버 금융 종목 뉴스를 [start_date, end_date] 범위로 반환.

    한국 종목 가드는 wrapper(`yahoo_naver.py`)에서 처리한다. 이 함수는 한국
    종목임을 전제로 호출된다. 비한국 종목이 잘못 들어오면 ValueError.
    """
    code = to_naver_code(ticker)
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    except ValueError as e:
        return f"<unavailable: invalid date — {e}>"

    article_limit = get_config().get("news_article_limit", 20)

    collected: list[dict] = []
    for page in range(1, _MAX_PAGES + 1):
        soup = _fetch_page(code, page)
        if soup is None:
            break
        items = _extract_items(soup)
        if not items:
            break

        # 페이지가 최신 → 과거 순으로 정렬되므로 범위 이전이면 다음 페이지로 진행하지 않는다.
        all_past = True
        for it in items:
            if it["date"] is None:
                # 날짜 파싱 실패한 항목은 보수적으로 포함 (헤드라인 보존)
                collected.append(it)
                all_past = False
                continue
            if start_dt <= it["date"] <= end_dt:
                collected.append(it)
                all_past = False
            elif it["date"] > end_dt:
                # 미래 항목은 무시 (있을 일 거의 없음)
                all_past = False
        if all_past:
            break
        if len(collected) >= article_limit:
            break

    if not collected:
        return f"<no Naver news found for {ticker} between {start_date} and {end_date}>"

    # 출력 — sentiment_analyst가 yfinance_news에서 받는 형식을 따른다.
    lines = [f"## {ticker} News (Naver Finance), {start_date} ~ {end_date}", ""]
    for it in collected[:article_limit]:
        lines.append(f"### {it['title']} (source: {it['publisher'] or 'naver'})")
        if it["date_raw"]:
            lines.append(it["date_raw"])
        if it["link"]:
            lines.append(f"Link: {it['link']}")
        lines.append("")
    return "\n".join(lines)
