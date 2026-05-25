"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked, runs an in-process triage stage that adapts to data
volume, and injects the result into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing
  4. Naver discussion    — Korean retail board (Korean tickers only)

Triage policy (sources 2–4): given the raw message count N within the
date window,
  * Sparse  (N ≤ 20):    pass all messages through verbatim (no regression)
  * Normal  (20 < N ≤ 80): pre-score sentiment distribution + Top-K by
                          engagement
  * Flood   (N > 80):    distribution + time-bucket burst + Top-K
Pre-scoring uses VADER (English) or the KNU lexicon (Korean) locally —
no extra LLM calls.

The agent does not use tool-calling; the data is in the prompt from
turn 0. The LLM produces the sentiment report in a single invocation.

See: https://github.com/TauricResearch/TradingAgents/issues/557
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.korean_utils import is_korean_ticker
from tradingagents.dataflows.naver_discussion import collect_naver_discussion
from tradingagents.dataflows.reddit import DEFAULT_SUBREDDITS, collect_reddit_posts
from tradingagents.dataflows.sentiment_scoring import score_text
from tradingagents.dataflows.stocktwits import collect_stocktwits_messages

# Triage 경계 — 1주차 초안. NVDA·005930.KS·소형주 실측 후 조정.
_SPARSE_MAX = 20
_NORMAL_MAX = 80
_TOP_K = 10

# 윈도우 길이 정책 (sentiment_lookback_strategy → days)
_LOOKBACK_TABLE = {"default": 7, "daily": 3, "weekly": 14}


def _resolve_lookback(config: Optional[dict]) -> int:
    """Map ``sentiment_lookback_strategy`` to a concrete days value."""
    if not config:
        return 7
    strategy = config.get("sentiment_lookback_strategy", "default")
    return _LOOKBACK_TABLE.get(strategy, 7)


def _classify_volume(n: int) -> str:
    if n <= _SPARSE_MAX:
        return "sparse"
    if n <= _NORMAL_MAX:
        return "normal"
    return "flood"


def _distribution(scores: Iterable[dict]) -> dict:
    """Aggregate per-message score dicts into a label distribution."""
    pos = neg = neu = total = 0
    oov_sum = 0.0
    for s in scores:
        total += 1
        label = s.get("label", "neutral")
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1
        oov_sum += s.get("oov_ratio", 0.0)
    return {
        "total": total,
        "pos": pos,
        "neg": neg,
        "neu": neu,
        "pos_pct": round(100 * pos / total) if total else 0,
        "neg_pct": round(100 * neg / total) if total else 0,
        "neu_pct": round(100 * neu / total) if total else 0,
        "oov_ratio_avg": (oov_sum / total) if total else 0.0,
    }


def _burst_buckets(timestamps: list[datetime], window_days: int) -> str:
    """Render a one-line per-day bucket count over the lookback window.

    Useful for spotting traffic bursts (e.g., earnings day spikes).
    """
    if not timestamps:
        return ""
    buckets: dict[str, int] = {}
    for ts in timestamps:
        key = ts.strftime("%Y-%m-%d")
        buckets[key] = buckets.get(key, 0) + 1
    parts = [f"{d}: {c}" for d, c in sorted(buckets.items())]
    return "일별 게시 수: " + " · ".join(parts)


# ---------- Per-source triage formatters ----------------------------------

def _triage_reddit(posts: list[dict], lookback_days: int, display_label: str,
                   subreddits: Iterable[str]) -> str:
    """Triage + render Reddit posts. Empty list → placeholder."""
    if not posts:
        return (
            f"<no Reddit posts found mentioning {display_label} across "
            f"{', '.join(f'r/{s}' for s in subreddits)} in the past {lookback_days} days>"
        )

    band = _classify_volume(len(posts))
    if band == "sparse":
        return _format_reddit_full(posts, lookback_days, display_label, subreddits)

    # Normal / Flood: distribution + top-K
    scored = []
    for p in posts:
        text = (p.get("title") or "") + " " + (p.get("selftext") or "")
        scored.append({**p, "_score": score_text(text, lang="en")})
    dist = _distribution(s["_score"] for s in scored)

    header = [
        f"r/{','.join(subreddits)} — {len(posts)} posts mentioning {display_label} "
        f"in the past {lookback_days} days [{band.upper()}]",
        f"분포: Bullish {dist['pos_pct']}% · Bearish {dist['neg_pct']}% · "
        f"Neutral {dist['neu_pct']}% (N={dist['total']})",
    ]
    if band == "flood":
        ts = [datetime.utcfromtimestamp(p["created_utc"]) for p in posts if p.get("created_utc")]
        burst = _burst_buckets(ts, lookback_days)
        if burst:
            header.append(burst)

    # Top-K by engagement (upvotes + comments × 2)
    top = sorted(
        scored,
        key=lambda p: (p.get("score") or 0) + 2 * (p.get("num_comments") or 0),
        reverse=True,
    )[:_TOP_K]

    header.append("")
    header.append(f"대표 글 (engagement 상위 {len(top)}건):")
    for p in top:
        title = (p.get("title") or "").replace("\n", " ").strip()
        score = p.get("score", 0)
        comments = p.get("num_comments", 0)
        sub = p.get("subreddit", "?")
        created = p.get("created_utc")
        created_str = (
            datetime.utcfromtimestamp(created).strftime("%Y-%m-%d") if created else "?"
        )
        body = (p.get("selftext") or "").replace("\n", " ").strip()
        if len(body) > 240:
            body = body[:240] + "…"
        header.append(
            f"- [{created_str} · r/{sub} · {score:>4}↑ · {comments:>3}c] {title}"
            + (f"\n    body excerpt: {body}" if body else "")
        )
    return "\n".join(header)


def _format_reddit_full(posts: list[dict], lookback_days: int, display_label: str,
                        subreddits: Iterable[str]) -> str:
    """Sparse path — original verbatim format (no regression from pre-triage)."""
    blocks = []
    for sub in subreddits:
        sub_posts = [p for p in posts if p.get("subreddit") == sub]
        if not sub_posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {display_label} in the past {lookback_days} days>")
            continue
        lines = [f"r/{sub} — {len(sub_posts)} recent posts mentioning {display_label}:"]
        for p in sub_posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score", 0)
            comments = p.get("num_comments", 0)
            created = p.get("created_utc")
            created_str = (
                datetime.utcfromtimestamp(created).strftime("%Y-%m-%d") if created else "?"
            )
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{created_str} · {score:>4}↑ · {comments:>3}c] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _triage_stocktwits(messages: list[dict], lookback_days: int, ticker: str) -> str:
    if not messages:
        return f"<no StockTwits messages found for ${ticker.upper()} in the past {lookback_days} days>"

    # StockTwits has user-labeled Bullish/Bearish tags — use those as the
    # primary distribution signal and only VADER-score for tie-breaking.
    bullish = bearish = unlabeled = 0
    for m in messages:
        sentiment = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        if sentiment == "Bullish":
            bullish += 1
        elif sentiment == "Bearish":
            bearish += 1
        else:
            unlabeled += 1
    total = bullish + bearish + unlabeled
    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0

    band = _classify_volume(len(messages))
    header = [
        f"StockTwits ${ticker.upper()} — {total} messages in the past {lookback_days} days [{band.upper()}]",
        f"라벨 분포: Bullish {bullish} ({bull_pct}%) · Bearish {bearish} ({bear_pct}%) · Unlabeled {unlabeled}",
    ]

    if band == "sparse":
        # Emit all messages
        items = messages
    else:
        # Flood: include burst summary
        if band == "flood":
            ts: list[datetime] = []
            for m in messages:
                from tradingagents.dataflows.stocktwits import _parse_created_at
                parsed = _parse_created_at(m.get("created_at", ""))
                if parsed:
                    ts.append(parsed)
            burst = _burst_buckets(ts, lookback_days)
            if burst:
                header.append(burst)
        # Top-K — StockTwits has no engagement field, so just take the most
        # recent K (they're already returned in reverse-chronological order).
        items = messages[:_TOP_K]
        header.append("")
        header.append(f"대표 메시지 (최신 {len(items)}건):")

    if band == "sparse":
        header.append("")
    for m in items:
        created = m.get("created_at", "")
        user = (m.get("user") or {}).get("username", "?")
        entities = m.get("entities") or {}
        sentiment = (entities.get("sentiment") or {}).get("basic") if isinstance(entities.get("sentiment"), dict) else None
        body = (m.get("body") or "").replace("\n", " ").strip()
        if len(body) > 280:
            body = body[:280] + "…"
        tag = sentiment or "no-label"
        header.append(f"[{created} · @{user} · {tag}] {body}")
    return "\n".join(header)


def _triage_naver(posts: list[dict], lookback_days: int, ticker: str) -> str:
    if not posts:
        return f"<no Naver discussion posts found for {ticker} in the past {lookback_days} days>"

    band = _classify_volume(len(posts))
    total_up = sum(p["up"] for p in posts)
    total_down = sum(p["down"] for p in posts)
    if total_up + total_down > 0:
        ratio = total_up / (total_up + total_down) * 100
        agg_line = f"**집계: 공감 {total_up} / 비공감 {total_down} → 공감 비율 {ratio:.1f}%**"
    else:
        agg_line = "**집계: 공감/비공감 데이터 부족**"

    header = [
        f"## {ticker} Discussion (Naver Stock Board, {len(posts)} posts, past {lookback_days}d) [{band.upper()}]",
        "",
        "각 게시글: 제목 / 작성일시 / 조회 / 공감(👍) / 비공감(👎)",
        agg_line,
        "",
    ]

    if band == "sparse":
        items = posts
    else:
        # KNU 점수화로 분포 추가
        scored = [{**p, "_score": score_text(p.get("title", ""), lang="ko")} for p in posts]
        dist = _distribution(s["_score"] for s in scored)
        header.append(
            f"제목 분포: 긍정 {dist['pos_pct']}% · 부정 {dist['neg_pct']}% · 중립 {dist['neu_pct']}% "
            f"(N={dist['total']}, KNU 미수록 비율 {dist['oov_ratio_avg']:.0%})"
        )
        if band == "flood":
            # date is "YYYY.MM.DD HH:MM" — parse with regex like the collector
            from tradingagents.dataflows.naver_discussion import _parse_date
            ts = [d for d in (_parse_date(p.get("date", "")) for p in posts) if d]
            burst = _burst_buckets(ts, lookback_days)
            if burst:
                header.append(burst)
        # Top-K by engagement (views + 5 × up + 5 × down)
        top = sorted(
            posts,
            key=lambda p: (p.get("views") or 0) + 5 * ((p.get("up") or 0) + (p.get("down") or 0)),
            reverse=True,
        )[:_TOP_K]
        items = top
        header.append("")
        header.append(f"대표 글 (engagement 상위 {len(items)}건):")

    for p in items:
        header.append(
            f"- [{p['date']}] {p['title']} "
            f"(views {p['views']:,} · 👍 {p['up']} · 👎 {p['down']})"
        )
    return "\n".join(header)


# ---------- LangGraph node ------------------------------------------------

def create_sentiment_analyst(llm, config: Optional[dict] = None):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, runs in-process triage,
    injects the result into the prompt as structured blocks, and produces
    a sentiment report in a single LLM call.

    ``config`` is read for ``sentiment_lookback_strategy`` (default/daily/
    weekly). When omitted, falls back to the 7-day default — keeping older
    call sites (no config arg) backward compatible.
    """
    lookback_days = _resolve_lookback(config)

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")
        instrument_context = build_instrument_context(ticker)

        # News block is unchanged — get_news already supports date windows.
        news_block = get_news.func(ticker, start_date, end_date)

        # StockTwits + Reddit: collect raw, then triage in-process.
        stocktwits_raw, st_ok = collect_stocktwits_messages(ticker, lookback_days=lookback_days)
        if not st_ok:
            stocktwits_block = f"<stocktwits unavailable for ${ticker.upper()}>"
        else:
            stocktwits_block = _triage_stocktwits(stocktwits_raw, lookback_days, ticker)

        query, reddit_raw = collect_reddit_posts(ticker, lookback_days=lookback_days)
        if query is None:
            reddit_block = (
                f"<reddit skipped — could not resolve a usable search term for {ticker.upper()} "
                f"(Korean ticker without English company name in yfinance metadata)>"
            )
        else:
            display_label = ticker.upper() if query == ticker else f'{ticker.upper()} ("{query}")'
            reddit_block = _triage_reddit(reddit_raw, lookback_days, display_label, DEFAULT_SUBREDDITS)

        # 한국 종목은 네이버 종목토론실로 한국어 retail 채널을 보완.
        naver_discussion_block: Optional[str] = None
        if is_korean_ticker(ticker):
            err, naver_raw = collect_naver_discussion(ticker, lookback_days=lookback_days)
            if err is not None:
                naver_discussion_block = f"<naver_discussion unavailable: {err}>"
            else:
                naver_discussion_block = _triage_naver(naver_raw, lookback_days, ticker)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            naver_discussion_block=naver_discussion_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # No bind_tools — the data is already in the prompt; a single LLM
        # call produces the report directly.
        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "sentiment_report": result.content,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    lookback_days: int,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    naver_discussion_block: Optional[str] = None,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    sources_count = "four" if naver_discussion_block else "three"

    naver_section = ""
    if naver_discussion_block:
        naver_section = f"""
### Naver Stock Board — 한국 개인 투자자 토론 (한국 종목 전용)
StockTwits/Reddit이 한국 종목 데이터를 거의 갖지 않는 약점을 보완하는 채널.
각 게시글의 공감(👍)/비공감(👎) 카운트가 정서 신호의 핵심이며, 집계된 공감 비율은
강한 retail sentiment 지표다.

<start_of_naver_discussion>
{naver_discussion_block}
<end_of_naver_discussion>
"""

    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date} ({lookback_days}-day window), drawing on {sources_count} complementary data sources that have already been collected for you.

The social-media blocks below have been **adaptively triaged** based on volume — when many messages were collected, you see a pre-computed sentiment distribution + the most engaged posts rather than every post verbatim. The [SPARSE]/[NORMAL]/[FLOOD] tag on each block tells you which mode was used.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past {lookback_days} days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body. The label distribution at the top is computed from the full sample.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past {lookback_days} days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>
{naver_section}
## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this caveat explicitly. If the sources are silent on a given subreddit, say so. In NORMAL/FLOOD modes, base your overall direction on the distribution stats first and the dispayed posts second — the displayed posts are a representative sample, not the full universe.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output

Produce a sentiment report covering, in order:

1. **Overall sentiment direction** — Bullish / Bearish / Neutral / Mixed — with a brief confidence note based on data quality and sample size.
2. **Source-by-source breakdown** — what each source (news / StockTwits / Reddit / Naver if present) is telling you, with specific evidence (cite message counts, ratios, notable posts).
3. **Divergences, alignments, and key narratives** across sources.
4. **Catalysts and risks** surfaced by the data.
5. **Markdown table** at the end summarizing key sentiment signals, their direction, source, and supporting evidence.

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm, config: Optional[dict] = None):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm, config)
