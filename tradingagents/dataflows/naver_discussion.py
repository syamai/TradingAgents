"""네이버 금융 종목토론실 어댑터 — Sentiment Analyst 전용 소스.

엔드포인트: ``finance.naver.com/item/board.naver?code={6자리}&page={n}``
한국 개인 투자자 정서의 핵심 채널이며, StockTwits/Reddit이 한국 종목 데이터를
거의 갖지 못하는 약점을 보완한다.

각 게시글 행에서 제목, 작성일, 조회수, 공감/비공감 수를 추출한다. 본문은
호출 비용 대비 가치가 낮아 가져오지 않는다 — 제목 + 공감/비공감 비율만으로도
정서 신호로 충분하다고 판단.

라우터에 등록하지 않고 ``sentiment_analyst.py``에서 직접 호출한다 (sentiment의
StockTwits/Reddit 패턴과 동일).
"""
from __future__ import annotations

from typing import Optional

import requests
from bs4 import BeautifulSoup

from .korean_utils import to_naver_code

_BASE = "https://finance.naver.com/item/board.naver"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    "Referer": "https://finance.naver.com/",
}
_TIMEOUT = 5.0


def _fetch_page(code: str, page: int) -> Optional[BeautifulSoup]:
    try:
        resp = requests.get(
            _BASE,
            params={"code": code, "page": page},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        # 네이버 페이지는 페이지마다 EUC-KR/CP949/UTF-8이 섞여있어 명시 디코딩이
        # 자주 깨진다. BS4가 HTML meta charset을 보고 자동 추론하도록 raw bytes 전달.
        return BeautifulSoup(resp.content, "lxml")
    except Exception:
        return None


def _safe_text(el) -> str:
    return el.get_text(strip=True) if el is not None else ""


def _safe_int(s: str) -> int:
    s = s.strip().replace(",", "")
    try:
        return int(s)
    except ValueError:
        return 0


def fetch_naver_discussion(ticker: str, limit: int = 30) -> str:
    """한국 종목 토론실 최근 게시글을 limit개까지 가져와 마크다운 문자열로 반환.

    실패 시(차단, 비한국 등) ``<unavailable: ...>`` 형식. sentiment_analyst가
    그대로 프롬프트에 주입한다.
    """
    try:
        code = to_naver_code(ticker)
    except ValueError as e:
        return f"<naver_discussion unavailable: {e}>"

    posts: list[dict] = []
    page = 1
    while len(posts) < limit and page <= 3:
        soup = _fetch_page(code, page)
        if soup is None:
            break

        # 게시판 행: table.type2 안의 tr 중 td 5개 이상인 것
        rows = soup.select("table.type2 tr")
        page_posts = 0
        for tr in rows:
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            # 실제 구조 (2026 기준): [날짜, 제목, 글쓴이, 조회, 공감, 비공감]
            title_a = tds[1].find("a")
            if not title_a:
                continue
            title = title_a.get_text(strip=True)
            if not title:
                continue
            date_raw = _safe_text(tds[0])
            view_count = _safe_int(_safe_text(tds[3]))
            upvotes = _safe_int(_safe_text(tds[4]))
            downvotes = _safe_int(_safe_text(tds[5]))
            posts.append({
                "title": title,
                "date": date_raw,
                "views": view_count,
                "up": upvotes,
                "down": downvotes,
            })
            page_posts += 1
            if len(posts) >= limit:
                break

        if page_posts == 0:
            break
        page += 1

    if not posts:
        return f"<no Naver discussion posts found for {ticker}>"

    lines = [
        f"## {ticker} Discussion (Naver Stock Board, recent {len(posts)} posts)",
        "",
        "각 게시글: 제목 / 작성일시 / 조회 / 공감(👍) / 비공감(👎)",
        "공감-비공감 비율은 한국 개인 투자자 정서 신호로 해석한다.",
        "",
    ]
    total_up = sum(p["up"] for p in posts)
    total_down = sum(p["down"] for p in posts)
    if total_up + total_down > 0:
        ratio = total_up / (total_up + total_down) * 100
        lines.append(
            f"**집계: 공감 {total_up} / 비공감 {total_down} → 공감 비율 {ratio:.1f}%**"
        )
    else:
        lines.append("**집계: 공감/비공감 데이터 부족**")
    lines.append("")

    for p in posts:
        lines.append(
            f"- [{p['date']}] {p['title']} "
            f"(views {p['views']:,} · 👍 {p['up']} · 👎 {p['down']})"
        )

    return "\n".join(lines)
