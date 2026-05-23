"""Reddit search fetcher for ticker-specific discussion posts.

Uses Reddit's public JSON endpoints (``reddit.com/r/{sub}/search.json``)
which do not require an API key. Public throughput is ~10 requests per
minute per IP, well within budget for a single agent run that queries
a handful of finance subreddits per ticker.

Returns formatted plaintext blocks ready for prompt injection. Degrades
gracefully — returns a placeholder string rather than raising, so callers
never have to special-case missing data.
"""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# Default subreddits ordered roughly by signal density for ticker-specific
# discussion. wallstreetbets has the most volume but most noise; stocks /
# investing trend more measured. Caller can override.
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")

# Korean tickers are 6-digit numeric codes ("005930.KS") that English-speaking
# Reddit users essentially never write. The actual discussion language is the
# company name. Strip these corporate-form suffixes from yfinance names to
# get a clean search phrase ("Samsung Electronics Co., Ltd." → "Samsung
# Electronics"). Order matters — longer suffixes first.
_NAME_SUFFIXES = (
    " Co., Ltd.", " Co., Ltd", " Corporation", " Corp.", " Inc.",
    ", Ltd.", ", Ltd",
)


@lru_cache(maxsize=64)
def _resolve_search_term(ticker: str) -> Optional[str]:
    """Map a ticker to the query Reddit users actually use.

    Returns the ticker itself for US/etc. tickers (cashtag-style search works).
    For Korean tickers (``.KS``/``.KQ``) the numeric code is useless on Reddit,
    so we resolve to the company's English name via yfinance metadata.
    Returns ``None`` when resolution is required but failed, so the caller
    can skip the search rather than pollute Reddit with a junk query.

    Cached per ticker so repeated analyses in the same process don't replay
    the slow yfinance ``.info`` lookup.
    """
    from tradingagents.dataflows.korean_utils import is_korean_ticker
    if not is_korean_ticker(ticker):
        return ticker
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
    except Exception as exc:
        logger.warning("Could not resolve English name for %s: %s", ticker, exc)
        return None
    name = info.get("longName") or info.get("shortName")
    if not name:
        return None
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip() or None


def _fetch_subreddit(
    query: str,
    sub: str,
    limit: int,
    timeout: float,
) -> list[dict]:
    qs = urlencode({
        "q": query,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",  # last 7 days
        "limit": limit,
    })
    url = _API.format(sub=sub, qs=qs)
    # Reddit's anti-bot heuristic 403s requests that omit the standard browser
    # headers (Accept-Language, Accept-Encoding) even when User-Agent and IP
    # are otherwise fine. urllib doesn't add these by default; curl does,
    # which is why the same URL works under curl but failed here previously.
    req = Request(url, headers={
        "User-Agent": _UA,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "identity",
    })
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("Reddit fetch failed for r/%s · %s: %s", sub, query, exc)
        return []
    children = (payload.get("data") or {}).get("children") or []
    return [c.get("data", {}) for c in children if isinstance(c, dict)]


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 1.0,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    ``inter_request_delay`` keeps us under Reddit's public rate limit
    (~10 req/min per IP) even if the caller queries many subreddits.
    Default 1.0s (was 0.4s) — the lower value was occasionally tripping
    Reddit's anti-bot heuristic into returning HTTP 403 on bursts.
    """
    query = _resolve_search_term(ticker)
    if query is None:
        return (
            f"<reddit skipped — could not resolve a usable search term for {ticker.upper()} "
            f"(Korean ticker without English company name in yfinance metadata)>"
        )

    display_label = ticker.upper() if query == ticker else f'{ticker.upper()} ("{query}")'

    blocks = []
    total_posts = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(query, sub, limit_per_sub, timeout)
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {display_label} in the past 7 days>")
            continue

        lines = [f"r/{sub} — {len(posts)} recent posts mentioning {display_label}:"]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score", 0)
            comments = p.get("num_comments", 0)
            created = p.get("created_utc")
            created_str = (
                time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            )
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{created_str} · {score:>4}↑ · {comments:>3}c] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        return (
            f"<no Reddit posts found mentioning {display_label} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
        )
    return "\n\n".join(blocks)
