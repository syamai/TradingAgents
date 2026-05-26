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
    """투자자 매매동향 list[dict] → 표 3개 (외국인 / 기관 sub / 개인+기타법인).

    실패 마커 문자열은 그대로 통과. 수량은 종가와 함께 봐야 의미 있어 금액(KRW)
    위주로 표시 — 사모/연기금 등 sub 주체 식별이 핵심 가치.
    """
    if isinstance(rows, str):
        return rows
    if not rows:
        return "<no investor-trend rows returned — possibly holiday window or KIS mock environment>"

    out = [f"({len(rows)} rows, newest first) — amounts in KRW, signed (+ buying / − selling)"]

    out.append("\n[표 1] 외국인 (총합 / 등록 = 장기 / 비등록 = 단기 외국 자금)")
    out.append("date | close | foreign_total | foreign_registered | foreign_unregistered")
    for r in rows:
        out.append(
            f"{r['date']} | {r['close']:,} | "
            f"{r['foreign_amount']:+,} | "
            f"{r['foreign_registered_amount']:+,} | "
            f"{r['foreign_unregistered_amount']:+,}"
        )

    out.append("\n[표 2] 기관 sub-분류 (연기금 = 안정 매수, 사모 = 단기 알고리즘, 투신·증권·은행+보험 = 추세 추종)")
    out.append("date | inst_total | pension | private_equity | investment_trust | securities | bank+insurance")
    for r in rows:
        out.append(
            f"{r['date']} | "
            f"{r['institution_amount']:+,} | "
            f"{r['pension_amount']:+,} | "
            f"{r['private_equity_amount']:+,} | "
            f"{r['investment_trust_amount']:+,} | "
            f"{r['securities_amount']:+,} | "
            f"{r['bank_insurance_amount']:+,}"
        )

    out.append("\n[표 3] 개인 + 기타법인 (자사주 매입 가능성)")
    out.append("date | retail | other_corp")
    for r in rows:
        out.append(
            f"{r['date']} | {r['retail_amount']:+,} | {r['other_corp_amount']:+,}"
        )

    return "\n".join(out)


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
3개 sub-표로 분리:
- **표 1 (외국인)**: 등록(장기 보유 외국인) vs 비등록(단기 외국 자금). 비등록이 강하게 매수하면 단기 투기, 등록이 매수하면 장기 신뢰.
- **표 2 (기관 sub-분류)**: 연기금(pension, 가장 안정적·장기적), 사모펀드(private_equity, 단기 알고리즘적·변동성 주도), 투자신탁(investment_trust)·증권(securities)·은행+보험(bank+insurance)은 추세 추종 성향.
- **표 3 (개인 + 기타법인)**: 기타법인(other_corp) 매수는 자사주 매입 가능성으로 강세 신호.

기관 sub 정합/괴리가 단기 가격 압력의 핵심 — 예: 연기금 매수 + 사모 매도 = 장기·단기 시각차, 사모 + 외국인비등록 동시 매수 = 단기 모멘텀 강함.

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

1. **외국인 등록 vs 비등록** — 등록자 매수는 장기 신뢰, 비등록 매수는 단기 외국 자금 유입. 둘이 같은 방향이면 강한 신호, 괴리면 단기/장기 시각차.
2. **기관 sub-분류 패턴** — 연기금 매수 = 안정적 강세 신호 (가장 보수적·장기적 주체). 사모 매도 + 외국인비등록 매도 = 단기 차익 실현·약세. 사모 + 비등록 동시 매수 = 단기 모멘텀 강함. 투신·증권·은행+보험은 추세 추종이라 방향성 확인 보조.
3. **기타법인 매수** — 자사주 매입 또는 대주주·계열사 매수일 가능성, 강세 신호.
4. **3대 주체 정합/괴리** — 외국인+기관 동방향 + 개인 반대 = retail이 추격당하는 패턴, 추세 강함. 외국인+개인 동방향 + 기관 반대 = 기관이 차익 실현 중.
5. **프로그램 매매가 수급을 견인하는지** — 종목 시가총액 대비 net_amount 크기로 판단. KOSPI200 편입 종목은 인덱스 자금 효과로 프로그램 매매 영향이 큼.
6. **공매도 비중 추이** — 거래량 대비 공매도 비중 10% 이상 + 누적 잔고 증가 = 약세론 가시화. 비중 하락은 short covering 가능성.
7. **데이터 한계 인정** — 한 블록이 ``<unavailable: ...>``이거나 ``<no ... rows>``이면 그 신호는 평가할 수 없음. 명시적으로 보고서에 언급.
8. **다른 분석가와의 결합 신호로 프레이밍** — 수급은 단기 가격 압력의 한 입력일 뿐, 가격 예측이 아니다. 펀더멘털·기술적 신호와 결합되어야 함.

## Output

Produce a comprehensive supply/demand report covering, in order:

1. **Overall direction** — Net Bullish / Net Bearish / Mixed / Neutral, with a brief confidence note based on data quality.
2. **Foreign flow analysis** — registered vs unregistered split, long-term vs short-term foreign capital interpretation.
3. **Institutional sub-breakdown** — pension / private equity / investment trust / securities / bank+insurance individual directions and what their combined pattern signals (e.g., pension buying while PE selling = long/short horizon divergence).
4. **Retail and other-corp flow** — including any self-buyback signal from other_corp.
5. **Program-trading analysis** — overall direction and magnitude, relationship to foreign flow.
6. **Short-selling trend analysis** — volume-ratio movement, cumulative-balance signal, short-squeeze or sell-pressure escalation potential.
7. **Markdown summary table** — at least 6 rows (foreign-total / pension / private-equity / other-institutions / program / short) × direction · scale · signal strength (weak/medium/strong) · one-line evidence.

{get_language_instruction()}"""
