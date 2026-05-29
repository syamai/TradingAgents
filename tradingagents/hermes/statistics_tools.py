"""통계 분석 도구를 Hermes 가 단독 호출 가능하게 wrap.

dashboard 의 ``correlation_analysis`` / ``trend_analysis`` / ``advanced_analysis``
는 결정론적 함수형 모듈. Hermes 통합 시 ticker + date 범위만 받아 holdings
로드 + 분석 dict 반환하는 *얇은 인터페이스* 만 제공한다.

PRD G2 단계 — 4 함수 노출: correlation / trend / advanced / holdings_window.
"""
from __future__ import annotations

from typing import Optional

from dashboard.advanced_analysis import compute_advanced_report
from dashboard.correlation_analysis import compute_correlation_report
from dashboard.holdings_chart import load_holdings
from dashboard.trend_analysis import compute_trend_report


def compute_correlation(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """ticker 의 KIS holdings 시계열에 대한 상관 분석 리포트.

    args:
        ticker: 종목 코드 (예: "005930.KS").
        start_date: 분석 시작 "YYYY-MM-DD". ``None`` 이면 전체 히스토리.
        end_date: 분석 종료 "YYYY-MM-DD". ``None`` 이면 최신 데이터까지.

    return:
        ``compute_correlation_report`` dict (JSON 직렬화 가능). ticker 메타
        (``company_name``, ``market``) 가 결과에 포함된다. 데이터 없는 ticker
        는 ``n_days=0`` 의 빈 sections 리포트.
    """
    df, meta = load_holdings(ticker, start=start_date, end=end_date)
    return compute_correlation_report(
        df,
        ticker=ticker,
        company_name=meta.get("company_name"),
        market=meta.get("market"),
    )


def compute_trend(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """ticker 추세 분석 — 적응 윈도우 + phase 분할 + 9 주체 동행성 랭킹.

    args:
        ticker: 종목 코드 (예: "005930.KS").
        start_date: 분석 시작 "YYYY-MM-DD". ``None`` 이면 전체 히스토리.
        end_date: 분석 종료 "YYYY-MM-DD". ``None`` 이면 최신 데이터까지.

    return:
        ``compute_trend_report`` dict. ``subjects`` 는 |agreement-50| 상위
        ``top_n`` (디폴트 5) 주체의 phase 상세, ``concordance_ranking`` 은
        9 주체 전체 동행성 랭킹. 데이터 없는 ticker 는 ``n_days=0`` 의 빈 dict.
    """
    df, _meta = load_holdings(ticker, start=start_date, end=end_date)
    # compute_trend_report 는 빈 df 에서 foreign 합 컬럼 부재로 KeyError.
    # 결손 시그널을 정상 응답으로 변환 — Hermes 가 catch 안 해도 안전.
    if df.empty:
        return {"n_days": 0, "subjects": {}, "concordance_ranking": []}
    return compute_trend_report(df)


def compute_advanced(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """ticker 정교 분석 — ADF / Granger / VAR-IRF / cointegration / mutual info / rolling r.

    args:
        ticker: 종목 코드.
        start_date / end_date: ``None`` 이면 전체.

    return:
        ``compute_advanced_report`` dict. ``n_days<50`` 이면 빈 sections.
        통계 결과는 인과 아님 — 인용 시 "인과 ≠ 상관" 명시 필요.
    """
    df, _meta = load_holdings(ticker, start=start_date, end=end_date)
    return compute_advanced_report(df)


def get_holdings_window(
    ticker: str,
    days: int = 30,
) -> list[dict]:
    """ticker 의 holdings 최근 ``days`` 거래일 raw 시계열.

    가공된 통계가 아닌 *원본 수치* 가 필요한 경우 (예: "마지막 5 거래일의
    외국인 등록 net_qty 정확히 인용") 용도. 큰 윈도우는 토큰 폭증 — 디폴트
    30 일 권장.

    args:
        ticker: 종목 코드.
        days: 최근 거래일 수 (디폴트 30).

    return:
        list[dict] (각 행 = 1 거래일). 데이터 없는 ticker 는 빈 리스트.
        ``date`` 필드는 ISO 문자열 형식.
    """
    df, _meta = load_holdings(ticker)
    if df.empty:
        return []
    tail = df.tail(max(0, days)).copy()
    if "date" in tail.columns:
        tail["date"] = tail["date"].astype(str)
    return tail.to_dict(orient="records")
