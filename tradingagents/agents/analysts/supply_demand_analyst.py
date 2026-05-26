"""Supply/Demand analyst — KIS 수급 데이터로 한국 종목의 단기 가격 압력 분석.

한국 종목(.KS/.KQ)에서 가장 강한 단기 시그널 중 하나가 수급:
  1. 종목별 일별 투자자 매매동향 (외국인 / 기관 / 개인 순매수)
  2. 종목별 일별 프로그램매매 추이
  3. 종목별 공매도 일별 추이

KIS(한국투자증권) OpenAPI를 사용. ``KIS_APP_KEY`` / ``KIS_APP_SECRET`` 환경변수 필요.
``KIS_ENV=mock|real``로 모의/실전 endpoint 전환.

설계:
  - 비한국 ticker → ``<not applicable: ...>`` 한 줄 리포트 (다운스트림 가드와 호환)
  - 한국 ticker → 3개 fetcher를 ``_safe()`` 로 감싸 호출 (개별 실패는 unavailable 마커)
  - mode 무관 prefetch (sentiment_analyst 선례). tool calling 미지원 모델도 동작.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows import kis_api
from tradingagents.dataflows.kis_auth import KisCredentialError
from tradingagents.dataflows.korean_utils import is_korean_ticker, to_naver_code

_LOOKBACK_DAYS = 7
_NA_NON_KR = (
    "<not applicable: non-KR ticker — supply/demand analysis covers "
    "KOSPI/KOSDAQ only>"
)


def create_supply_demand_analyst(llm):
    """LangGraph 노드 팩토리. ``sentiment_analyst``와 동일하게 mode 무관 prefetch."""

    def supply_demand_analyst_node(state):
        ticker = state["company_of_interest"]
        if not is_korean_ticker(ticker):
            # 비한국 종목: LLM 호출 없이 즉시 N/A. 다운스트림은 ``startswith("<")``
            # 로 가드해 이 문자열을 prompt에 포함시키지 않는다.
            return {"messages": [], "supply_demand_report": _NA_NON_KR}

        return _run_prefetch(state, llm)

    return supply_demand_analyst_node


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KisCredentialError as e:
        return f"<unavailable: KIS credentials not configured ({e})>"
    except Exception as e:  # noqa: BLE001 — fetcher 실패가 분석가 전체를 깨뜨리면 안 됨
        return f"<unavailable: {e}>"


def _run_prefetch(state, llm):
    ticker = state["company_of_interest"]
    end_date = state["trade_date"]
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    code6 = to_naver_code(ticker)
    instrument_context = build_instrument_context(ticker)

    investor_rows = _safe(
        kis_api.fetch_investor_trend, code6, end_date, lookback_days=_LOOKBACK_DAYS
    )
    program_rows = _safe(
        kis_api.fetch_program_trading, code6, end_date, lookback_days=_LOOKBACK_DAYS
    )
    short_rows = _safe(
        kis_api.fetch_short_interest, code6, start_date, end_date,
        lookback_days=_LOOKBACK_DAYS,
    )

    investor_block = _format_investor(investor_rows)
    program_block = _format_program(program_rows)
    short_block = _format_short(short_rows)

    system_message = _build_system_message(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        investor_block=investor_block,
        program_block=program_block,
        short_block=short_block,
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

    chain = prompt | llm
    result = chain.invoke(state["messages"])
    return {"messages": [result], "supply_demand_report": result.content}


# ---------- block formatters ----------

def _format_investor(rows) -> str:
    """투자자 매매동향 list[dict] → text 블록. 실패 마커 문자열은 그대로 통과."""
    if isinstance(rows, str):
        return rows
    if not rows:
        return "<no investor-trend rows returned — possibly holiday window or KIS mock environment>"
    lines = [f"({len(rows)} rows, newest first)"]
    lines.append(
        "date | close | foreign_qty | inst_qty | retail_qty | "
        "foreign_amount_KRW | inst_amount_KRW | retail_amount_KRW"
    )
    for r in rows:
        lines.append(
            f"{r['date']} | {r['close']:,} | "
            f"{r['foreign_qty']:+,} | {r['institution_qty']:+,} | {r['retail_qty']:+,} | "
            f"{r['foreign_amount']:+,} | {r['institution_amount']:+,} | {r['retail_amount']:+,}"
        )
    return "\n".join(lines)


def _format_program(rows) -> str:
    if isinstance(rows, str):
        return rows
    if not rows:
        return "<no program-trading rows returned>"
    lines = [f"({len(rows)} rows, newest first)"]
    lines.append("date | close | net_qty | net_amount_KRW")
    for r in rows:
        lines.append(
            f"{r['date']} | {r['close']:,} | "
            f"{r['net_qty']:+,} | {r['net_amount']:+,}"
        )
    return "\n".join(lines)


def _format_short(rows) -> str:
    if isinstance(rows, str):
        return rows
    if not rows:
        return "<no short-interest rows returned>"
    lines = [f"({len(rows)} rows, newest first)"]
    lines.append(
        "date | close | short_qty | vol_ratio_pct | short_amount_KRW | amount_ratio_pct"
    )
    for r in rows:
        lines.append(
            f"{r['date']} | {r['close']:,} | "
            f"{r['short_qty']:,} | {r['short_volume_ratio']:.2f} | "
            f"{r['short_amount']:,} | {r['short_amount_ratio']:.2f}"
        )
    return "\n".join(lines)


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    investor_block: str,
    program_block: str,
    short_block: str,
) -> str:
    return f"""You are a Korean-market supply/demand (수급) analyst. Produce a comprehensive supply/demand report for {ticker} covering {start_date} → {end_date} using the pre-fetched KIS (Korea Investment & Securities OpenAPI) data below.

Each block is a markdown-friendly table (one row per trading day). Rows are typically newest first; check the dates. Quantities (`*_qty`) are in shares; amounts (`*_amount_KRW`) are in Korean won. A positive number is net buying; negative is net selling.

## Data sources (pre-fetched, in this prompt)

### Block 1 — 종목별 일별 투자자 매매동향 (Daily investor flow)
외국인(foreign) vs 기관(institution) vs 개인(retail) 순매수 추이. 외국인·기관의 누적 순매수 방향은 한국 시장 단기 가격 압력의 가장 강한 지표 중 하나.

<start_of_investor_trend>
{investor_block}
<end_of_investor_trend>

### Block 2 — 종목별 일별 프로그램매매 추이 (Daily program-trading flow)
종목 단위 종합값(차익/비차익 합계). 단기 트레이딩 알고리즘·인덱스 차익 수급의 방향성.

<start_of_program_trading>
{program_block}
<end_of_program_trading>

### Block 3 — 종목별 공매도 일별 추이 (Daily short selling)
공매도 체결 수량·금액과 전체 거래대금 대비 비중. 비중 상승 + 가격 하락 동반은 매도 압력 강화, 비중 상승 + 가격 횡보는 short squeeze 잠재력.

<start_of_short_interest>
{short_block}
<end_of_short_interest>

## How to analyze this data (best practices)

1. **외국인 순매수 방향성** — 연속 매수/매도 일수, 일일 평균 규모. 한국 시장에서 외국인 7일 연속 매도는 자체로 약세 신호.
2. **외국인 vs 기관 vs 개인 정합/괴리** — 셋이 같은 방향이면 강한 추세, 외국인·기관이 한 방향이고 개인이 반대면 retail이 추격당하는 패턴.
3. **프로그램 매매가 수급을 견인하는지** — 종목 시가총액 대비 net_amount 크기로 판단. KOSPI200 편입 종목은 인덱스 자금 효과로 프로그램 매매 영향이 큼.
4. **공매도 비중 추이** — 거래량 대비 공매도 비중 10% 이상 + 누적 잔고 증가 = 약세론 가시화. 비중 하락은 short covering 가능성.
5. **데이터 한계 인정** — 한 블록이 ``<unavailable: ...>``이거나 ``<no ... rows>``이면 그 신호는 평가할 수 없음. 명시적으로 보고서에 언급.
6. **다른 분석가와의 결합 신호로 프레이밍** — 수급은 단기 가격 압력의 한 입력일 뿐, 가격 예측이 아니다. 펀더멘털·기술적 신호와 결합되어야 함.

## Output

다음 순서로 한국어 보고서 작성:

1. **요약 (Overall direction)** — 단기 수급 방향성을 Net Bullish / Net Bearish / Mixed / Neutral 중 하나로 명시 + 데이터 품질에 따른 confidence note.
2. **외국인·기관·개인 흐름 분석** — 누적 순매수 방향, 일일 규모, 정합/괴리 패턴, 특이일.
3. **프로그램 매매 분석** — 종합 방향과 규모, 외국인 흐름과의 관계.
4. **공매도 추세 분석** — 비중 추이, 누적 잔고 시그널, short squeeze 또는 매도 압력 강화 가능성.
5. **요약 표 (Markdown)** — 4섹션(외국인/기관/프로그램/공매도) × 방향·규모·신호 강도(약/중/강)·근거 한 줄.

{get_language_instruction()}"""
