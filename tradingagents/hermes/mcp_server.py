"""trading-ai → Hermes MCP 서버.

stdio transport 로 Hermes (또는 다른 MCP client) 에 도구를 노출한다.
PRD G4 단계 — 13 도구:
  분석가 (5):
    - ``analyst_supply_demand`` : KIS 수급 (한국 전용)
    - ``analyst_sentiment``     : 뉴스/소셜 sentiment (한국·미국 모두)
    - ``analyst_market``        : 기술 지표 (한국·미국)
    - ``analyst_news``          : 거시·뉴스 (한국·미국)
    - ``analyst_fundamentals``  : 펀더멘털 (한국 미적용 — N/A 가능)
  통계 (4):
    - ``compute_correlation``   : 6 섹션 상관 (concurrent/lag/level 등)
    - ``compute_trend``         : 추세 phase + 9 주체 동행성 랭킹
    - ``compute_advanced``      : ADF/Granger/VAR-IRF/cointegration
    - ``get_holdings_window``   : 최근 N 거래일 raw 시계열
  가설/라벨 (4):
    - ``save_analysis``         : 가설 JSON 영구 저장 (HypothesisStore)
    - ``add_feedback``          : /feedback 명령 — 즉시/사후 사용자 라벨
    - ``list_hypotheses``       : 누적 가설 조회 (필터: ticker/as_of_date)
    - ``get_hypothesis_with_labels`` : 가설 + 라벨 join 조회

실행:
    uv run python -m tradingagents.hermes.mcp_server

환경변수 ``HERMES_ANALYST_PROVIDER`` / ``HERMES_ANALYST_MODEL`` /
``HERMES_ANALYST_BASE_URL`` 로 분석가 LLM override. 디폴트는 로컬 Ollama
``gemma4:26b-a4b`` — trading-ai 의 ``analyst_mode="auto"`` 가 gemma 모델명을
보고 prefetch 모드로 자동 전환한다.
"""
from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Optional

from mcp.server.fastmcp import FastMCP

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes import backtest as _bt
from tradingagents.hermes.analyst_runner import run_analyst
from tradingagents.hermes.backtest_engine import run_universe_backtest
from tradingagents.hermes.hypothesis_store import HypothesisStore
from tradingagents.hermes.statistics_tools import (
    compute_advanced as _compute_advanced,
    compute_correlation as _compute_correlation,
    compute_trend as _compute_trend,
    get_holdings_window as _get_holdings_window,
)
from tradingagents.hermes.strategy_spec import spec_hash, validate_spec
from tradingagents.hermes.strategy_store import StrategyStore
from tradingagents.llm_clients.factory import create_llm_client

# 사용자 피드백 시점 분기 임계값 — 가설 as_of_date 와 today 차이가 이 일수
# 이내면 ``user_immediate`` (즉시 직관), 초과면 ``user_followup`` (사후 추가).
FEEDBACK_IMMEDIATE_THRESHOLD_DAYS = 14

# analyst_supply_demand 의 kis_api fetcher 를 무조건 patch (idempotent).
# 패치된 함수는 호출 시점에 백테스트 활성 여부를 보고 kis.db / 라이브를 분기 —
# MCP 서버가 백테스트 상태 파일보다 먼저 떠도 동적으로 반영. 운영 시 무영향.
_bt.install_kis_history_patch()


def _hypothesis_store() -> HypothesisStore:
    """HypothesisStore — 호출 시점에 root 결정 (lru_cache 금지).

    백테스트 상태 파일이 MCP 서버 기동 이후 쓰일 수 있으므로 캐시하면 안 됨.
    백테스트 모드면 격리 root, 아니면 디폴트 (운영 db).
    """
    return HypothesisStore(root=_bt.store_root())


def _strategy_store() -> StrategyStore:
    """StrategyStore — 디폴트 경로(``~/.tradingagents/hermes/strategies.db``).

    전략 연구는 시간 만기 대기가 없어 백테스트 격리(store_root)가 불필요.
    """
    return StrategyStore()


@lru_cache(maxsize=512)
def _load_holdings_cached(ticker: str):
    """ticker holdings 캐시 — 연구 루프가 같은 종목군을 반복 백테스트하므로."""
    from dashboard.holdings_chart import load_holdings
    df, _meta = load_holdings(ticker)
    return df


def _engine_loader(ticker: str):
    return _load_holdings_cached(ticker), {}


def _universe() -> list[str]:
    """백테스트 대상 종목군 — KIS 히스토리에 적재된 전체 종목(6자리)."""
    return KisHistoryStore().list_tickers()


DEFAULT_ANALYST_PROVIDER = os.environ.get("HERMES_ANALYST_PROVIDER", "ollama")
DEFAULT_ANALYST_MODEL = os.environ.get(
    "HERMES_ANALYST_MODEL", "gemma4:26b-a4b",
)
DEFAULT_ANALYST_BASE_URL = os.environ.get(
    "HERMES_ANALYST_BASE_URL", "http://localhost:11434/v1",
)


@lru_cache(maxsize=4)
def _analyst_llm_for(model: str):
    """모델명별 분석가 LLM (lazy, 모델당 1번).

    MCP 서버 import 시점에 LLM 인스턴스를 만들면 Ollama 미기동 환경에서
    import 자체가 실패 → 첫 도구 호출 시점까지 지연. 모델별로 캐시해 백테스트
    override(경량 모델)가 캐시된 기본 모델에 가려지지 않게 한다.
    """
    client = create_llm_client(
        DEFAULT_ANALYST_PROVIDER, model, base_url=DEFAULT_ANALYST_BASE_URL,
    )
    return client.get_llm()


def _analyst_llm():
    """현재 분석가 LLM — 백테스트면 경량 override, 아니면 기본 모델."""
    return _analyst_llm_for(_bt.analyst_model() or DEFAULT_ANALYST_MODEL)


mcp = FastMCP("trading-ai-hermes")


@mcp.tool()
def analyst_supply_demand(ticker: str, date: str) -> str:
    """KIS 수급 데이터 기반 한국 종목 단기 가격 압력 분석.

    한국 종목(.KS/.KQ) 전용. 비한국 ticker 면 "<not applicable>" 한 줄 반환.

    args:
        ticker: 종목 코드 (예: "005930.KS").
        date: 분석 기준일 "YYYY-MM-DD". 직전 7일 lookback.

    returns:
        한국어 마크다운 보고서. 데이터 결손 블록 (KIS API 실패 등) 은
        "<unavailable: ...>" 마커로 명시.
    """
    return run_analyst(ticker, _bt.clamp_date(date), "supply_demand", llm=_analyst_llm())


@mcp.tool()
def analyst_sentiment(ticker: str, date: str) -> str:
    """뉴스·소셜 sentiment 분석 (StockTwits/Reddit + 한국 종목은 Naver 종목토론실).

    한국·미국 종목 모두 적용. 한국 종목은 Naver 종목토론실로 한국어 retail
    채널 보완.

    args:
        ticker: 종목 코드.
        date: 분석 기준일 "YYYY-MM-DD".

    returns:
        sentiment 점수·근거 마크다운 보고서.
    """
    return run_analyst(ticker, _bt.clamp_date(date), "sentiment", llm=_analyst_llm())


@mcp.tool()
def analyst_market(ticker: str, date: str) -> str:
    """기술 지표 + 가격·거래량 추세 분석.

    한국·미국 종목 모두 적용. KOSPI/KOSDAQ 종목은 yfinance 데이터로
    가격·거래량 시계열 분석.

    args:
        ticker: 종목 코드.
        date: 분석 기준일 "YYYY-MM-DD".

    returns:
        기술 지표 (MA/RSI/MACD/볼린저) + 추세 해석 마크다운.
    """
    return run_analyst(ticker, _bt.clamp_date(date), "market", llm=_analyst_llm())


@mcp.tool()
def analyst_news(ticker: str, date: str) -> str:
    """거시·종목별 뉴스 분석 (Finnhub/Google News).

    한국·미국 종목 모두 적용. 한국 종목은 한국 거시·정책 뉴스 보강.

    args:
        ticker: 종목 코드.
        date: 분석 기준일 "YYYY-MM-DD".

    returns:
        뉴스 요약 + 시장 영향 해석 마크다운.
    """
    return run_analyst(ticker, _bt.clamp_date(date), "news", llm=_analyst_llm())


@mcp.tool()
def analyst_fundamentals(ticker: str, date: str) -> str:
    """펀더멘털 분석 — *주의: 한국 종목 미적용*.

    SimFin 데이터 기반 미국 종목 펀더멘털 분석가. 한국 종목은 SimFin 미커버라
    출력이 빈약하거나 부정확할 수 있음. 한국 종목 펀더멘털은 DART 기반 별도
    파이프라인이 더 적합 (Phase 2 검토).

    args:
        ticker: 종목 코드.
        date: 분석 기준일 "YYYY-MM-DD".

    returns:
        펀더멘털 분석 마크다운. 한국 종목이면 "데이터 결손" 가능.
    """
    return run_analyst(ticker, _bt.clamp_date(date), "fundamentals", llm=_analyst_llm())


@mcp.tool()
def compute_correlation(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """ticker 의 KIS holdings 시계열에 대한 상관 분석 리포트.

    6 섹션 (cumulative / concurrent / up_down / lag / regime / level).
    LLM 호출 없음 — 결정론적 통계 계산만. 같은 (ticker, 기간) 입력이면 항상
    동일한 출력.

    args:
        ticker: 종목 코드 (KisHistoryStore 에 holdings 데이터 있어야).
        start_date: 분석 시작 "YYYY-MM-DD". None=전체 히스토리.
        end_date: 분석 종료. None=최신 데이터.

    returns:
        dict (JSON 직렬화 가능). ``n_days=0`` 이면 데이터 없음.
    """
    return _compute_correlation(ticker, start_date, _bt.clamp_end_date(end_date))


@mcp.tool()
def compute_trend(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """추세 분석 — 적응 윈도우 + phase 분할 (up/down/flat) + 9 주체 동행성 랭킹.

    LLM 호출 없음. 같은 입력 = 같은 출력.

    args:
        ticker: 종목 코드.
        start_date / end_date: ``None`` 이면 전체.

    returns:
        ``subjects`` (top-N 주체 phase 상세) + ``concordance_ranking`` (9 주체
        동행성 |agreement-50| 랭킹) + adaptive 윈도우 입력.
    """
    return _compute_trend(ticker, start_date, _bt.clamp_end_date(end_date))


@mcp.tool()
def compute_advanced(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """정교 통계 분석 — ADF / Granger / VAR-IRF / cointegration / mutual info / rolling r.

    LLM 호출 없음. 통계 결과는 *인과 아님* — 인용 시 "인과 ≠ 상관" 명시 필수.
    ``n_days<50`` 이면 빈 sections.

    args:
        ticker: 종목 코드.
        start_date / end_date: ``None`` 이면 전체.

    returns:
        6 섹션 dict.
    """
    return _compute_advanced(ticker, start_date, _bt.clamp_end_date(end_date))


@mcp.tool()
def get_holdings_window(ticker: str, days: int = 30) -> list[dict]:
    """ticker 의 최근 ``days`` 거래일 raw holdings 시계열.

    가공 통계가 아닌 원본 수치가 필요할 때 (예: "마지막 5 거래일의 외국인
    등록 net_qty 정확히 인용") 사용. 큰 윈도우는 토큰 폭증 — 디폴트 30 일.

    args:
        ticker: 종목 코드.
        days: 최근 거래일 수 (디폴트 30).

    returns:
        list[dict] (한 행 = 1 거래일). 빈 리스트면 데이터 없음.
    """
    return _get_holdings_window(ticker, days, end_date=_bt.as_of())


# === 가설/라벨 영구 저장 도구 (G4) ===


@mcp.tool()
def save_analysis(record: dict) -> dict:
    """가설 JSON 영구 저장 — Phase 5 응답 출력 직후 호출.

    ``record`` 는 ``stock-analyst`` skill 의 출력 스키마와 동일:
      ``{ticker, as_of_date, overall_stance, overall_confidence,
         hypotheses: [{id, claim, direction, confidence, evidence_tools,
                       evidence_excerpts, horizon_weeks,
                       predicted_relative_return_pct}, ...]}``

    schema 검증 실패 시 ValueError. 저장 성공 시 반환된 ``hypothesis_ids``
    로 후속 ``/feedback`` / ``/labels`` 명령에서 참조한다.

    returns:
        ``{"hypothesis_ids": [int, ...], "h_local_id_map": {"h1": id1, ...}}``
    """
    ids = _hypothesis_store().save(record)
    h_local_ids = [h["id"] for h in record["hypotheses"]]
    return {
        "hypothesis_ids": ids,
        "h_local_id_map": dict(zip(h_local_ids, ids)),
    }


@mcp.tool()
def get_us_trend_watchlist(market: str = "us") -> str:
    """미국 트렌드 와치리스트 — 섹터 로테이션(방향) + 종목 fade(군집 경고) + 알림.

    투자자 관심(investor attention) 6소스를 종합한 모니터링 다이제스트:
    검색(Google Trends ASVI=비정상 검색량)·소셜(ApeWisdom mention-momentum·StockTwits)
    ·비정상 거래량(Finviz)·옵션 O/S(Option-to-Stock volume ratio, 옵션/주식 거래량
    비율)·섹터 로테이션(RS-momentum=섹터 상대강도 모멘텀).

    핵심 해석: 개별종목 attention 은 대개 늦은 contrarian 이라 **추격이 아닌 페이드
    /리스크 플래그** 용도다(통제실험상 독립 alpha 아님 → 모니터링용). 섹터 로테이션이
    durable 한 방향, 종목 fade(≥2소스 동시 과열)가 군집/천장 경고. cron 이 매일
    적재한 최신 스냅샷을 읽어 텍스트로 반환한다. LLM 호출 없음.

    args:
        market: 'us' (현재 US 만 지원).
    returns:
        다이제스트 텍스트(🔔알림 + 🔄섹터 로테이션 + 📊종목 와치리스트 + 📍관심 집중 섹터).
    """
    from tradingagents.hermes.trend_rank import format_digest

    return format_digest(market)


@mcp.tool()
def add_feedback(
    hypothesis_id: int,
    verdict: str,
    reason: Optional[str] = None,
    label_kind: Optional[str] = None,
) -> dict:
    """사용자 피드백 라벨 추가 — ``/feedback h<id> right|wrong reason="..."``.

    ``label_kind`` 미지정 시 가설 ``as_of_date`` 와 today 차이로 자동 분기:
      - 차이 ≤ 14 일 → ``user_immediate`` (즉시 직관 피드백)
      - 차이 > 14 일 → ``user_followup`` (사후 추가 피드백)

    args:
        hypothesis_id: DB id (``save_analysis`` 반환 또는 ``list_hypotheses``).
        verdict: ``"right"`` / ``"wrong"`` / ``"neutral"``.
        reason: 한국어 사유 (옵션).
        label_kind: ``user_immediate`` / ``user_followup`` (옵션 — 자동 분기).

    returns:
        ``{"label_id": int, "label_kind": str}``.
    """
    store = _hypothesis_store()
    if label_kind is None:
        h = store.get(hypothesis_id)
        if h is None:
            raise ValueError(f"hypothesis_id {hypothesis_id} not found")
        as_of = datetime.strptime(h["as_of_date"], "%Y-%m-%d")
        days_elapsed = (datetime.utcnow() - as_of).days
        label_kind = (
            "user_immediate"
            if days_elapsed <= FEEDBACK_IMMEDIATE_THRESHOLD_DAYS
            else "user_followup"
        )
    label_id = store.add_label(
        hypothesis_id, label_kind, verdict=verdict, reason=reason,
    )
    return {"label_id": label_id, "label_kind": label_kind}


@mcp.tool()
def list_hypotheses(
    ticker: Optional[str] = None,
    as_of_date: Optional[str] = None,
    direction: Optional[str] = None,
) -> list[dict]:
    """누적 가설 메타 리스트 — ``/labels`` 또는 ``/feedback`` ID 조회용.

    args:
        ticker: 필터 (예: ``"005930.KS"``).
        as_of_date: 필터 (``YYYY-MM-DD``).
        direction: ``bullish`` / ``bearish`` / ``neutral`` 필터.

    returns:
        list[dict] 각 항목: ``id``, ``ticker``, ``as_of_date``,
        ``h_local_id``, ``direction``, ``confidence``, ``horizon_weeks``,
        ``predicted_relative_return_pct``, ``claim``, ``created_at``.
        라벨은 포함되지 않음 — ``get_hypothesis_with_labels`` 로 개별 조회.
    """
    return _hypothesis_store().list(
        ticker=ticker, as_of_date=as_of_date, direction=direction,
    )


@mcp.tool()
def get_hypothesis_with_labels(hypothesis_id: int) -> Optional[dict]:
    """가설 한 건 + 부여된 모든 라벨.

    라벨 종류 (``label_kind``):
      - ``user_immediate``: 가설 직후 사용자 직관 피드백
      - ``user_followup``: 사후 사용자 추가 피드백
      - ``auto_relative``: KOSPI 상대 수익 자동 라벨 (LabelingScheduler)
      - ``auto_absolute``: 절대 수익 자동 라벨

    args:
        hypothesis_id: DB id.

    returns:
        가설 dict + ``labels`` 리스트. 없으면 None.
    """
    return _hypothesis_store().get_with_labels(hypothesis_id)


# === 전략 백테스트·저장 도구 (수급 룰 자율 연구) ===


@mcp.tool()
def backtest_strategy(spec: dict, universe: Optional[list[str]] = None) -> dict:
    """파라미터화 수급 룰 전략을 종목군에 백테스트 (탐색용 — 저장 안 함).

    ``spec`` 은 strategy_spec 스키마(entry/exit 신호 + 손절/익절/보유일).
    종목군을 해시로 in/out-sample 분할해 각각 집계하고 채택 게이트를 판정한다.
    look-ahead 0 (신호 row i → 체결 close[i+1]), 거래비용 편도 0.015% 반영.

    args:
        spec: 전략 spec dict. 스키마 위반 시 ValueError (그리드 밖 값 등).
        universe: 종목 코드 리스트 (예: ["005930", "000660"]).
            None 이면 KIS 히스토리 전체 종목.

    returns:
        ``{in_sample: {win_rate, sharpe, mdd_pct, cum_return_pct, n_trades, ...},
           out_sample: {...}, gate_passed: bool, universe_size, n_in, n_out}``.
        게이트(승률≥0.60·샤프≥1.2·MDD≥-20%·거래≥50·in/out 격차≤10%p)는 in/out
        양쪽 충족 시에만 True.
    """
    validate_spec(spec)
    tickers = universe if universe else _universe()
    return run_universe_backtest(spec, tickers, loader=_engine_loader)


@mcp.tool()
def save_strategy(spec: dict, name: Optional[str] = None) -> dict:
    """전략을 *내부 재백테스트* 후 영구 저장 — 메트릭 무결성 보장.

    전달된 메트릭을 신뢰하지 않고 spec 으로 직접 재백테스트해 저장한다(LLM 이
    성능을 위조 저장하는 것 방지). 동일 로직(spec_hash) 전략이 이미 있으면
    재실행 없이 기존 id 를 ``duplicate`` 로 반환. 게이트 미통과 전략도 저장
    (연구 로그 + 시도 카운트 + 재시도 차단).

    args:
        spec: 전략 spec dict. 스키마 위반 시 ValueError.
        name: 사람이 읽는 식별자 (옵션 — spec.name 사용).

    returns:
        신규: ``{strategy_id, duplicate: False, gate_passed, in_sample,
        out_sample, universe_size}``. 중복: ``{strategy_id, duplicate: True,
        gate_passed}``.
    """
    validate_spec(spec)
    store = _strategy_store()
    existing = store.get_by_hash(spec_hash(spec))
    if existing is not None:
        return {
            "strategy_id": existing["id"],
            "duplicate": True,
            "gate_passed": existing["gate_passed"],
        }
    result = run_universe_backtest(spec, _universe(), loader=_engine_loader)
    sid, _is_new = store.save(spec, result, name=name)
    return {
        "strategy_id": sid,
        "duplicate": False,
        "gate_passed": result["gate_passed"],
        "in_sample": result["in_sample"],
        "out_sample": result["out_sample"],
        "universe_size": result["universe_size"],
    }


@mcp.tool()
def list_strategies(gate_passed: Optional[bool] = None) -> list[dict]:
    """저장된 전략 메타 리스트 — 연구 루프 진행/통과 카운트의 single source.

    args:
        gate_passed: True 면 채택(게이트 통과) 전략만, False 면 미통과만,
            None 이면 전체 시도.

    returns:
        list[dict] 각 항목: ``id``, ``name``, ``spec``, ``direction``,
        ``gate_passed``, in/out-sample 메트릭 컬럼, ``universe_size``,
        ``created_at``.
    """
    return _strategy_store().list(gate_passed=gate_passed)


def main():
    """stdio transport 진입점."""
    mcp.run()


if __name__ == "__main__":
    main()
