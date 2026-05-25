"""StockTwits public symbol-stream fetcher.

StockTwits exposes a per-symbol message stream at
``api.stocktwits.com/api/2/streams/symbol/{ticker}.json`` that requires no
API key, no OAuth, and no registration. Each message includes a
user-labeled sentiment field (``Bullish``/``Bearish``/null), the message
body, timestamp, and posting user.

The endpoint returns ~30 most-recent messages and a ``cursor`` object;
passing ``max=<cursor.max>`` walks backward in time, enabling date-windowed
fetches. We cap page count to bound rate-limit exposure.

The function is deliberately self-contained: short timeout, graceful
degradation on any HTTP or parse failure, and a string return type so
the calling agent gets a uniform interface regardless of whether the
network call succeeded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"


def _parse_created_at(s: str) -> Optional[datetime]:
    """Parse StockTwits' ISO-8601 timestamp ('2026-05-24T10:23:00Z')."""
    if not s:
        return None
    try:
        # Python 3.11+ handles trailing Z natively, 3.10 needs replacement
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fetch_page(ticker: str, max_id: Optional[int], timeout: float) -> Optional[dict]:
    """Fetch a single page; returns the parsed JSON dict or None on failure."""
    url = _API.format(ticker=ticker.upper())
    if max_id is not None:
        url = f"{url}?{urlencode({'max': max_id})}"
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("StockTwits fetch failed for %s (max=%s): %s", ticker, max_id, exc)
        return None


def collect_stocktwits_messages(
    ticker: str,
    limit: Optional[int] = None,
    lookback_days: int = 7,
    timeout: float = 10.0,
    max_pages: int = 5,
) -> tuple[list[dict], bool]:
    """Raw collection — used by triage. Returns ``(messages, fetch_ok)``.

    ``fetch_ok`` is False only when the first page errored; partial pages
    after a mid-walk failure still return what was gathered so far with
    fetch_ok=True.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    collected: list[dict] = []
    max_id: Optional[int] = None
    first_page = True
    window_exhausted = False

    for _ in range(max_pages):
        data = _fetch_page(ticker, max_id, timeout)
        if data is None:
            if first_page:
                return [], False
            break  # partial result is better than nothing
        first_page = False

        messages = data.get("messages") if isinstance(data, dict) else None
        if not messages:
            break

        for m in messages:
            created = _parse_created_at(m.get("created_at", ""))
            if created is not None and created < cutoff:
                window_exhausted = True
                break
            collected.append(m)
            if limit is not None and len(collected) >= limit:
                window_exhausted = True
                break

        if window_exhausted:
            break

        cursor = data.get("cursor") or {}
        if not cursor.get("more"):
            break
        next_max = cursor.get("max")
        if next_max is None or next_max == max_id:
            break  # defensive: don't loop on a stale cursor
        max_id = next_max

    return collected, True


def fetch_stocktwits_messages(
    ticker: str,
    limit: Optional[int] = None,
    lookback_days: int = 7,
    timeout: float = 10.0,
    max_pages: int = 5,
) -> str:
    """Fetch StockTwits messages within ``lookback_days`` for ``ticker``.

    Walks the message stream backward using the ``cursor.max`` field. Stops
    when any of the following is true:

      * a message older than ``lookback_days`` is seen (window exhausted),
      * the API reports no more pages (``cursor.more == False``),
      * ``max_pages`` reached (rate-limit safety, typically 5 × 30 = 150),
      * ``limit`` messages have been collected (when caller caps total).

    Returns a formatted plaintext block ready for prompt injection. Always
    returns a string — placeholder on failure, no exceptions surface.
    """
    messages, fetch_ok = collect_stocktwits_messages(
        ticker, limit, lookback_days, timeout, max_pages,
    )
    if not fetch_ok:
        return f"<stocktwits unavailable for ${ticker.upper()}>"
    if not messages:
        return f"<no StockTwits messages found for ${ticker.upper()} in the past {lookback_days} days>"
    return _format_messages(messages, ticker, lookback_days)


def _format_messages(messages: list[dict], ticker: str, lookback_days: int) -> str:
    lines = []
    bullish = bearish = unlabeled = 0
    for m in messages:
        created = m.get("created_at", "")
        user = (m.get("user") or {}).get("username", "?")
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        body = (m.get("body") or "").replace("\n", " ").strip()
        if len(body) > 280:
            body = body[:280] + "…"

        if sentiment == "Bullish":
            bullish += 1
            tag = "Bullish"
        elif sentiment == "Bearish":
            bearish += 1
            tag = "Bearish"
        else:
            unlabeled += 1
            tag = "no-label"
        lines.append(f"[{created} · @{user} · {tag}] {body}")

    total = bullish + bearish + unlabeled
    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0
    summary = (
        f"Bullish: {bullish} ({bull_pct}%) · "
        f"Bearish: {bearish} ({bear_pct}%) · "
        f"Unlabeled: {unlabeled} · "
        f"Total: {total} messages in the past {lookback_days} days"
    )
    return summary + "\n\n" + "\n".join(lines)
