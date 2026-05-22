from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_language_instruction,
    should_use_prefetch_mode,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        if should_use_prefetch_mode():
            return _run_prefetch(state, llm)
        return _run_tool_calling(state, llm)

    return fundamentals_analyst_node


def _run_tool_calling(state, llm):
    current_date = state["trade_date"]
    instrument_context = build_instrument_context(state["company_of_interest"])

    tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]

    system_message = (
        "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
        + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
        + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
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

    return {"messages": [result], "fundamentals_report": report}


def _run_prefetch(state, llm):
    current_date = state["trade_date"]
    ticker = state["company_of_interest"]
    instrument_context = build_instrument_context(ticker)

    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"<unavailable: {e}>"

    fundamentals_block = _safe(get_fundamentals.func, ticker, current_date)
    balance_block = _safe(get_balance_sheet.func, ticker, "quarterly", current_date)
    cashflow_block = _safe(get_cashflow.func, ticker, "quarterly", current_date)
    income_block = _safe(get_income_statement.func, ticker, "quarterly", current_date)

    system_message = f"""You are a researcher tasked with analyzing fundamental information about a company.

The four financial-data blocks below have been **pre-fetched** for you. Do NOT call any tools. Analyze the data already in this prompt and produce a comprehensive fundamentals report.

### Company Fundamentals Overview
<start_of_fundamentals>
{fundamentals_block}
<end_of_fundamentals>

### Balance Sheet (quarterly)
<start_of_balance_sheet>
{balance_block}
<end_of_balance_sheet>

### Cash Flow Statement (quarterly)
<start_of_cashflow>
{cashflow_block}
<end_of_cashflow>

### Income Statement (quarterly)
<start_of_income_statement>
{income_block}
<end_of_income_statement>

## Your Task

Write a comprehensive fundamentals report covering:
1. Company profile / business overview
2. Key financial ratios (margin, growth, leverage, liquidity) with specific numbers from the data
3. Cash flow quality and capital allocation
4. Notable trends quarter-over-quarter or year-over-year
5. Red flags or strengths that inform a trading decision
6. Append a Markdown table at the end summarizing key fundamental signals.

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

    return {"messages": [result], "fundamentals_report": result.content}
