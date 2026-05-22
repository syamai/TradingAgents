"""yfinance + 네이버/DART 통합 vendor.

한국 종목(`.KS`/`.KQ` 또는 6자리 코드)이면 두 소스를 모두 호출해 합친 결과를
반환하고, 그 외 종목이면 yfinance 결과만 위임한다. 한쪽 소스가 실패해도 다른
쪽 결과는 보존한다 (graceful degradation).

라우터(`interface.py`)의 `VENDOR_METHODS`에 `"yahoo_naver"` 키로 등록된다.
모든 함수는 LLM 프롬프트에 그대로 주입할 수 있는 문자열을 반환한다.
"""
from __future__ import annotations

from typing import Callable

from .korean_utils import combine_sources, is_korean_ticker


def _safe(fn: Callable, *args, label: str = "", **kwargs) -> str:
    """vendor 함수를 호출하되 예외는 `<unavailable: ...>` 문자열로 변환.

    한 소스가 실패해도 다른 소스 결과는 살려서 LLM에 전달하기 위한 wrapper.
    """
    try:
        result = fn(*args, **kwargs)
        if result is None or (isinstance(result, str) and not result.strip()):
            return f"<unavailable{(': ' + label) if label else ''}: empty response>"
        return result
    except Exception as exc:  # noqa: BLE001 — 라우터로 전파하지 않음
        return f"<unavailable{(': ' + label) if label else ''}: {type(exc).__name__}: {exc}>"


# ---------------------------------------------------------------------------
# core_stock_apis
# ---------------------------------------------------------------------------
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    from .y_finance import get_YFin_data_online

    if not is_korean_ticker(symbol):
        return get_YFin_data_online(symbol, start_date, end_date)

    from .naver_stock import fetch_naver_ohlcv

    yahoo = _safe(get_YFin_data_online, symbol, start_date, end_date, label="yfinance")
    naver = _safe(fetch_naver_ohlcv, symbol, start_date, end_date, label="naver")
    return combine_sources(
        yahoo_block=yahoo, korean_block=naver, korean_label="Naver", method="get_stock_data"
    )


# ---------------------------------------------------------------------------
# technical_indicators
# ---------------------------------------------------------------------------
def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
    from .y_finance import get_stock_stats_indicators_window

    if not is_korean_ticker(symbol):
        return get_stock_stats_indicators_window(symbol, indicator, curr_date, look_back_days)

    from .naver_indicator import compute_indicator_from_naver

    yahoo = _safe(
        get_stock_stats_indicators_window, symbol, indicator, curr_date, look_back_days,
        label="yfinance",
    )
    naver = _safe(
        compute_indicator_from_naver, symbol, indicator, curr_date, look_back_days,
        label="naver",
    )
    return combine_sources(
        yahoo_block=yahoo, korean_block=naver, korean_label="Naver", method="get_indicators"
    )


# ---------------------------------------------------------------------------
# news_data
# ---------------------------------------------------------------------------
def get_news(ticker: str, start_date: str, end_date: str) -> str:
    from .yfinance_news import get_news_yfinance

    if not is_korean_ticker(ticker):
        return get_news_yfinance(ticker, start_date, end_date)

    from .naver_news import fetch_naver_news

    yahoo = _safe(get_news_yfinance, ticker, start_date, end_date, label="yfinance")
    naver = _safe(fetch_naver_news, ticker, start_date, end_date, label="naver")
    return combine_sources(
        yahoo_block=yahoo, korean_block=naver, korean_label="Naver", method="get_news"
    )


def get_global_news(curr_date: str, look_back_days=None, limit=None) -> str:
    # 글로벌 매크로 뉴스는 종목 무관 — 한국 소스 결합 가치가 낮아 yfinance만 사용.
    from .yfinance_news import get_global_news_yfinance

    return get_global_news_yfinance(curr_date, look_back_days, limit)


# ---------------------------------------------------------------------------
# fundamental_data
# ---------------------------------------------------------------------------
def get_fundamentals(ticker: str, curr_date: str) -> str:
    from .y_finance import get_fundamentals as get_yf_fundamentals

    if not is_korean_ticker(ticker):
        return get_yf_fundamentals(ticker, curr_date)

    from .dart_fundamentals import get_dart_fundamentals

    yahoo = _safe(get_yf_fundamentals, ticker, curr_date, label="yfinance")
    dart = _safe(get_dart_fundamentals, ticker, curr_date, label="dart")
    return combine_sources(
        yahoo_block=yahoo, korean_block=dart, korean_label="DART", method="get_fundamentals"
    )


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    from .y_finance import get_balance_sheet as get_yf_balance_sheet

    if not is_korean_ticker(ticker):
        return get_yf_balance_sheet(ticker, freq, curr_date)

    from .dart_fundamentals import get_dart_balance_sheet

    yahoo = _safe(get_yf_balance_sheet, ticker, freq, curr_date, label="yfinance")
    dart = _safe(get_dart_balance_sheet, ticker, freq, curr_date, label="dart")
    return combine_sources(
        yahoo_block=yahoo, korean_block=dart, korean_label="DART", method="get_balance_sheet"
    )


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    from .y_finance import get_cashflow as get_yf_cashflow

    if not is_korean_ticker(ticker):
        return get_yf_cashflow(ticker, freq, curr_date)

    from .dart_fundamentals import get_dart_cashflow

    yahoo = _safe(get_yf_cashflow, ticker, freq, curr_date, label="yfinance")
    dart = _safe(get_dart_cashflow, ticker, freq, curr_date, label="dart")
    return combine_sources(
        yahoo_block=yahoo, korean_block=dart, korean_label="DART", method="get_cashflow"
    )


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    from .y_finance import get_income_statement as get_yf_income

    if not is_korean_ticker(ticker):
        return get_yf_income(ticker, freq, curr_date)

    from .dart_fundamentals import get_dart_income_statement

    yahoo = _safe(get_yf_income, ticker, freq, curr_date, label="yfinance")
    dart = _safe(get_dart_income_statement, ticker, freq, curr_date, label="dart")
    return combine_sources(
        yahoo_block=yahoo, korean_block=dart, korean_label="DART", method="get_income_statement"
    )


# ---------------------------------------------------------------------------
# insider_transactions (한국 소스 없음 — yfinance 단독)
# ---------------------------------------------------------------------------
def get_insider_transactions(ticker: str, curr_date: str) -> str:
    from .y_finance import get_insider_transactions as get_yf_insider

    return get_yf_insider(ticker, curr_date)
