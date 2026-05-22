from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
    should_use_prefetch_mode,
)
from tradingagents.dataflows.config import get_config


# auto/prefetch 모드에서 전부 가져올 지표. 시스템 메시지의 원본 11개 + vwma.
_PREFETCH_INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "macd", "macds", "macdh",
    "rsi",
    "boll", "boll_ub", "boll_lb", "atr",
    "vwma",
]

_INDICATOR_REFERENCE = """Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."""


def create_market_analyst(llm):

    def market_analyst_node(state):
        if should_use_prefetch_mode():
            return _run_prefetch(state, llm)
        return _run_tool_calling(state, llm)

    return market_analyst_node


def _run_tool_calling(state, llm):
    """기존 ReAct 멀티턴 방식 — tool calling 가능한 모델용."""
    current_date = state["trade_date"]
    asset_type = state.get("asset_type", "stock")
    instrument_context = build_instrument_context(
        state["company_of_interest"], asset_type
    )

    tools = [get_stock_data, get_indicators]

    system_message = (
        "You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:\n\n"
        + _INDICATOR_REFERENCE
        + "\n\n- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
        + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
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

    return {"messages": [result], "market_report": report}


def _run_prefetch(state, llm):
    """사전 fetch 방식 — tool calling 미지원 모델/서버용."""
    current_date = state["trade_date"]
    ticker = state["company_of_interest"]
    asset_type = state.get("asset_type", "stock")
    instrument_context = build_instrument_context(ticker, asset_type)

    start_date = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    # 데이터 사전 fetch — .func로 @tool 데코레이터 우회.
    try:
        stock_block = get_stock_data.func(ticker, start_date, current_date)
    except Exception as e:
        stock_block = f"<unavailable: get_stock_data failed — {e}>"

    indicator_blocks = []
    for ind in _PREFETCH_INDICATORS:
        try:
            value = get_indicators.func(ticker, ind, current_date)
        except Exception as e:
            value = f"<unavailable: {e}>"
        indicator_blocks.append(f"### {ind}\n{value}")
    indicators_block = "\n\n".join(indicator_blocks)

    system_message = f"""You are a trading assistant tasked with analyzing financial markets.

You have been provided with **pre-fetched OHLCV price data** and **all technical indicators below**. Do NOT call any tools. Analyze the data already in this prompt and produce a comprehensive technical analysis report.

## Indicator Reference (for interpretation)
{_INDICATOR_REFERENCE}

## Pre-fetched Data

### Price Data (OHLCV, last ~30 days)
<start_of_price_data>
{stock_block}
<end_of_price_data>

### Technical Indicators
<start_of_indicators>
{indicators_block}
<end_of_indicators>

## Your Task

1. Identify the most relevant indicators for current market conditions and explain why (mention at least 5 of the 12 provided).
2. Describe trend direction, momentum, volatility, and volume signals using specific values from the data.
3. Highlight any divergences, crossovers, or extreme readings.
4. Provide specific, actionable insights with supporting evidence from the data.
5. Append a Markdown table at the end summarizing key signals (indicator name, current reading, signal, interpretation).

{get_language_instruction()}"""

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

    return {"messages": [result], "market_report": result.content}
