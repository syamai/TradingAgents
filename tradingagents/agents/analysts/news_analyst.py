from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
    should_use_prefetch_mode,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        if should_use_prefetch_mode():
            return _run_prefetch(state, llm)
        return _run_tool_calling(state, llm)

    return news_analyst_node


def _run_tool_calling(state, llm):
    current_date = state["trade_date"]
    asset_type = state.get("asset_type", "stock")
    asset_label = "company" if asset_type == "stock" else "asset"
    instrument_context = build_instrument_context(state["company_of_interest"], asset_type)

    tools = [get_news, get_global_news]

    system_message = (
        f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
        + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        + get_language_instruction()
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK; another assistant with different tools"
                " will help where you left off. Execute what you can to make progress."
                " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}"
                "For your reference, the current date is {current_date}. {instrument_context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt = prompt.partial(current_date=current_date)
    prompt = prompt.partial(instrument_context=instrument_context)

    chain = prompt | llm.bind_tools(tools)
    result = chain.invoke(state["messages"])

    report = ""
    if len(result.tool_calls) == 0:
        report = result.content

    return {"messages": [result], "news_report": report}


def _run_prefetch(state, llm):
    current_date = state["trade_date"]
    ticker = state["company_of_interest"]
    asset_type = state.get("asset_type", "stock")
    asset_label = "company" if asset_type == "stock" else "asset"
    instrument_context = build_instrument_context(ticker, asset_type)

    start_date = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"<unavailable: {e}>"

    ticker_news = _safe(get_news.func, ticker, start_date, current_date)
    global_news = _safe(get_global_news.func, current_date)  # look_back_days/limit는 config 기본값

    system_message = f"""You are a news researcher tasked with analyzing recent news and trends.

Two pre-fetched news data blocks below cover (a) {asset_label}-specific news for the past 7 days and (b) global macroeconomic news. Do NOT call any tools. Analyze the data already in this prompt and produce a comprehensive news report relevant for trading and macroeconomics.

### {asset_label.capitalize()}-Specific News ({start_date} ~ {current_date})
<start_of_ticker_news>
{ticker_news}
<end_of_ticker_news>

### Global Macro News (configured lookback window)
<start_of_global_news>
{global_news}
<end_of_global_news>

## Your Task

Write a comprehensive news/macro report covering:
1. Key {asset_label}-specific developments and their potential price impact.
2. Macro/geopolitical/policy themes that could influence the broader market.
3. Cross-references: how does macro news affect this particular instrument?
4. Catalysts to watch and risks ahead.
5. Append a Markdown table at the end summarizing key news items, their topic, source date, and trading implication.

Provide specific, actionable insights with supporting evidence to help traders make informed decisions.{get_language_instruction()}"""

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
    prompt = prompt.partial(current_date=current_date)
    prompt = prompt.partial(instrument_context=instrument_context)

    chain = prompt | llm
    result = chain.invoke(state["messages"])

    return {"messages": [result], "news_report": result.content}
