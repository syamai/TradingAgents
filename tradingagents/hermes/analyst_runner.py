"""5 분석가를 LangGraph state 없이 단독 호출 가능하게 wrap.

trading-ai 의 분석가 노드 팩토리(``create_*_analyst``)는 LangGraph state 를
받는 클로저를 반환한다. Hermes 통합 시 LangGraph 전체 그래프를 띄우지 않고
한 분석가만 호출하려면 *minimal state dict* 만 구성해 노드에 직접 전달하면
된다.

PRD 모듈 #1 (AnalystRunner). 분석가 코드는 *수정하지 않고* 외부 wrapper 만
제공한다 — trading-ai 의 LangGraph 트레이딩 흐름 회귀 0 보장.
"""
from __future__ import annotations

from typing import Literal

from tradingagents.agents.analysts.fundamentals_analyst import (
    create_fundamentals_analyst,
)
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.sentiment_analyst import (
    create_sentiment_analyst,
)
from tradingagents.agents.analysts.supply_demand_analyst import (
    create_supply_demand_analyst,
)

AnalystName = Literal[
    "supply_demand", "sentiment", "market", "news", "fundamentals",
]

# 분석가 이름 → 노드 팩토리 매핑.
_FACTORIES = {
    "supply_demand": create_supply_demand_analyst,
    "sentiment": create_sentiment_analyst,
    "market": create_market_analyst,
    "news": create_news_analyst,
    "fundamentals": create_fundamentals_analyst,
}

# 분석가 이름 → state 반환 dict 의 보고서 키. 분석가마다 다름.
_REPORT_KEYS = {
    "supply_demand": "supply_demand_report",
    "sentiment": "sentiment_report",
    "market": "market_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def run_analyst(
    ticker: str,
    date: str,
    analyst_name: AnalystName,
    *,
    llm,
) -> str:
    """분석가 한 명을 LangGraph state 없이 단독 호출.

    한국/비한국 ticker 분기는 분석가 자체 로직에 위임 — 예: supply_demand 는
    비한국 ticker 면 LLM 호출 없이 ``<not applicable: ...>`` 한 줄 반환.

    args:
        ticker: 종목 코드 (예: "005930.KS").
        date: 분석 기준일 "YYYY-MM-DD".
        analyst_name: 5 분석가 중 하나.
        llm: LangChain ``BaseChatModel`` 호환 객체 (``prompt | llm`` 가능).

    return:
        분석가 마크다운 보고서 문자열.

    raises:
        ValueError: 지원 안 하는 ``analyst_name``.
    """
    if analyst_name not in _FACTORIES:
        raise ValueError(
            f"unsupported analyst: {analyst_name!r}. "
            f"valid: {sorted(_FACTORIES)}"
        )

    factory = _FACTORIES[analyst_name]
    node = factory(llm)
    state = {
        "company_of_interest": ticker,
        "trade_date": date,
        "messages": [],
    }
    result = node(state)
    return result[_REPORT_KEYS[analyst_name]]


def list_analysts() -> list[str]:
    """지원하는 분석가 이름 리스트 (사전순)."""
    return sorted(_FACTORIES)
