"""이벤트 리뷰(Event Retro) — 가격·이벤트·뉴스·재무·거시·동종 통합 분석.

수동 회고 보고서를 자동 생성하는 오케스트레이션 모듈.
1주차 MVP: PeriodSummary + 이벤트 탐지 + 그룹화 + 골격 render.
2주차에 검색·거시·DART 통합. 3주차에 동종·LLM 분류+합성.

설계 원칙: trend_analysis 와 같은 dict / render 분리. 부분 실패 graceful
degradation (warnings 누적). statsmodels 의존 없음.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.market_history import (
    fetch_kospi, fetch_usdkrw, fetch_us_10y, fetch_wti,
    fetch_close_series,
    summarize_series, compute_relative_strength,
)
from tradingagents.dataflows.bok_macro import summarize_macro
from tradingagents.dataflows.dart_disclosure import (
    list_disclosures_for_ticker, filter_significant_disclosures,
)
from tradingagents.dataflows.dart_industry import (
    find_peers, fetch_company_info, Peer,
)
from tradingagents.dataflows.search_aggregator import (
    search_news, hits_to_dicts, NewsHit,
)


# === 클러스터 enum (5 고정) ====================================================
# 회고마다 비교 가능하도록 enum 고정. LLM 분류는 이 중 하나로만.
CLUSTER_ENUM = (
    "earnings_rating",      # 실적 발표·증권사 리포트
    "company_specific",     # 자사주·배당·M&A·소송·경영진
    "industry_regulation",  # 업종 환경·규제·손해율
    "macro_shock",          # 거시·금리·환율·정책·관세
    "drift",                # 잔차 — 위 4개로 설명 안 되는 부분
)

CLUSTER_LABEL_KO = {
    "earnings_rating":     "실적·리포트",
    "company_specific":    "회사 고유",
    "industry_regulation": "업종·규제",
    "macro_shock":         "거시 충격",
    "drift":               "기타 drift",
}

# === 이벤트 탐지 임계값 ========================================================
PRICE_JUMP_PCT = 3.0     # |pct| ≥ 3% → 이벤트
VOL_Z_THRESHOLD = 2.0    # 거래량 20일 rolling z-score ≥ 2.0 → 이벤트
VOL_ROLLING_WINDOW = 20
GROUP_GAP_DAYS = 3       # 인접 ±3일 이벤트는 1 그룹

# 이벤트 ↔ 뉴스 매칭 윈도우 (사건 일자 ± 일)
MATCH_DAYS_BEFORE = 2
MATCH_DAYS_AFTER = 3

# 검색 쿼리 — 이벤트당 2개 (회사명+날짜 / 산업키워드+월)
MAX_SEARCH_RESULTS_PER_QUERY = 10


# === 데이터 클래스 ==============================================================
@dataclass
class PeriodSummary:
    start_date: str
    start_close: float
    high_date: str
    high_close: float
    low_date: str
    low_close: float
    end_date: str
    end_close: float
    total_return_pct: float
    max_drawdown_pct: float


@dataclass
class EventDay:
    date: str
    close: float
    pct: float
    volume: Optional[int]
    z_volume: Optional[float]
    is_price_jump: bool
    is_volume_anomaly: bool
    is_period_anchor: bool   # start/end/high/low 자동 이벤트
    cluster: Optional[str] = None      # LLM 분류 결과 (3주차)
    matched_news: list = field(default_factory=list)


@dataclass
class EventGroup:
    """인접 ±3일 이벤트 묶음 — 쿼리 비용 절감용."""
    start_date: str
    end_date: str
    members: list[str]       # 이벤트 일자들
    label: str               # "2025-04-07 ~ 2025-04-10" 형식


# === 기간 요약 ==================================================================
def compute_period_summary(df: pd.DataFrame) -> PeriodSummary:
    """가격 df → PeriodSummary. df 는 date 오름차순, close 보유.

    max_drawdown_pct: 고점→저점 낙폭 (음수).
    total_return_pct: 시작가 → 종료가 변화율.
    """
    start = df.iloc[0]
    end = df.iloc[-1]
    hi_idx = df["close"].idxmax()
    lo_idx = df["close"].idxmin()
    hi = df.loc[hi_idx]
    lo = df.loc[lo_idx]
    total_ret = (end["close"] / start["close"] - 1) * 100
    mdd = (lo["close"] / hi["close"] - 1) * 100
    return PeriodSummary(
        start_date=str(start["date"]),
        start_close=float(start["close"]),
        high_date=str(hi["date"]),
        high_close=float(hi["close"]),
        low_date=str(lo["date"]),
        low_close=float(lo["close"]),
        end_date=str(end["date"]),
        end_close=float(end["close"]),
        total_return_pct=round(float(total_ret), 4),
        max_drawdown_pct=round(float(mdd), 4),
    )


# === 이벤트 탐지 ================================================================
def _detect_event_days(
    df: pd.DataFrame,
    *,
    price_jump_pct: float = PRICE_JUMP_PCT,
    vol_z: float = VOL_Z_THRESHOLD,
    vol_window: int = VOL_ROLLING_WINDOW,
) -> list[EventDay]:
    """이벤트 일자 union: ±N% 변동 + 거래량 z-score + start/end/high/low.

    거래량 컬럼이 없으면 (backfill 전 데이터) vol_z 이벤트는 skip.
    """
    df = df.sort_values("date").reset_index(drop=True)
    pct = df["close"].pct_change() * 100

    has_volume = "volume" in df.columns and df["volume"].notna().any()
    if has_volume:
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(vol_window, min_periods=5).mean()
        vol_sd = vol.rolling(vol_window, min_periods=5).std()
        z_vol = (vol - vol_ma) / vol_sd.where(vol_sd > 0, other=np.nan)
    else:
        vol = pd.Series([np.nan] * len(df))
        z_vol = pd.Series([np.nan] * len(df))

    hi_idx = df["close"].idxmax()
    lo_idx = df["close"].idxmin()
    anchor_idx = {0, len(df) - 1, int(hi_idx), int(lo_idx)}

    events: list[EventDay] = []
    for i, row in df.iterrows():
        p = float(pct.iloc[i]) if not pd.isna(pct.iloc[i]) else 0.0
        zv = float(z_vol.iloc[i]) if not pd.isna(z_vol.iloc[i]) else None
        is_jump = abs(p) >= price_jump_pct
        is_vol = zv is not None and abs(zv) >= vol_z
        is_anchor = i in anchor_idx
        if not (is_jump or is_vol or is_anchor):
            continue
        events.append(EventDay(
            date=str(row["date"]),
            close=float(row["close"]),
            pct=round(p, 4),
            volume=int(row["volume"]) if has_volume and pd.notna(row["volume"]) else None,
            z_volume=round(zv, 2) if zv is not None else None,
            is_price_jump=is_jump,
            is_volume_anomaly=is_vol,
            is_period_anchor=is_anchor,
        ))
    return events


def _group_events(
    events: list[EventDay], *, gap_days: int = GROUP_GAP_DAYS,
) -> list[EventGroup]:
    """인접 ±gap_days 이벤트를 1 그룹으로 묶음 (쿼리 비용 절감).

    같은 그룹 = 첫 이벤트 일자와의 차이가 ``gap_days`` 이내인 모든 후속.
    """
    if not events:
        return []
    sorted_evts = sorted(events, key=lambda e: e.date)
    groups: list[EventGroup] = []
    cur_members: list[str] = [sorted_evts[0].date]
    cur_start = sorted_evts[0].date
    for ev in sorted_evts[1:]:
        d_prev = datetime.strptime(cur_members[-1], "%Y-%m-%d").date()
        d_curr = datetime.strptime(ev.date, "%Y-%m-%d").date()
        if (d_curr - d_prev).days <= gap_days:
            cur_members.append(ev.date)
        else:
            groups.append(EventGroup(
                start_date=cur_start,
                end_date=cur_members[-1],
                members=list(cur_members),
                label=(cur_start if len(cur_members) == 1
                       else f"{cur_start} ~ {cur_members[-1]}"),
            ))
            cur_members = [ev.date]
            cur_start = ev.date
    groups.append(EventGroup(
        start_date=cur_start,
        end_date=cur_members[-1],
        members=list(cur_members),
        label=(cur_start if len(cur_members) == 1
               else f"{cur_start} ~ {cur_members[-1]}"),
    ))
    return groups


# === 분해 계산 (룰베이스, 결정적) ==============================================
def _compute_decomposition(
    events: list[EventDay], total_return_pct: float,
) -> dict[str, float]:
    """클러스터별 이벤트 day pct 합산 + drift 잔차.

    각 이벤트의 ``cluster`` 가 None 이면 drift 로 합산. 5 enum 모두 보장.
    drift = total_return − Σ(다른 4개 cluster).
    """
    sums: dict[str, float] = {c: 0.0 for c in CLUSTER_ENUM}
    for ev in events:
        c = ev.cluster if ev.cluster in CLUSTER_ENUM else "drift"
        sums[c] += ev.pct
    # drift 는 잔차 — 다른 4개 외 모든 변동 (이벤트가 아닌 일자 변동 포함)
    explained = sum(sums[c] for c in CLUSTER_ENUM if c != "drift")
    sums["drift"] = round(total_return_pct - explained, 4)
    return {c: round(v, 4) for c, v in sums.items()}


# === 쿼리 빌더 + 매칭 ===========================================================
def _build_search_queries(
    group: EventGroup, company_name: str, industry_keyword: Optional[str],
) -> list[str]:
    """그룹당 2 쿼리: 회사명+날짜, 산업키워드+월. 산업키워드 없으면 회사명만."""
    base_name = company_name or group.start_date
    queries: list[str] = [f"{base_name} {group.label}"]
    if industry_keyword:
        ym = group.start_date[:7]  # YYYY-MM
        queries.append(f"{industry_keyword} {ym}")
    return queries


def _query_window(
    group: EventGroup, before: int = MATCH_DAYS_BEFORE, after: int = MATCH_DAYS_AFTER,
) -> tuple[str, str]:
    """그룹 검색 윈도우 — [start-before, end+after] YYYY-MM-DD."""
    from datetime import timedelta
    s = datetime.strptime(group.start_date, "%Y-%m-%d").date() - timedelta(days=before)
    e = datetime.strptime(group.end_date, "%Y-%m-%d").date() + timedelta(days=after)
    return s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")


def _match_news_to_events(
    events: list[EventDay], hits: list[NewsHit],
    *, before: int = MATCH_DAYS_BEFORE, after: int = MATCH_DAYS_AFTER,
) -> None:
    """각 EventDay 의 ``matched_news`` 에 published 윈도우 안 hit 채움. in-place."""
    from datetime import timedelta
    for ev in events:
        ev_dt = datetime.strptime(ev.date, "%Y-%m-%d").date()
        lo = ev_dt - timedelta(days=before)
        hi = ev_dt + timedelta(days=after)
        for h in hits:
            if not h.published:
                continue
            try:
                pd_dt = datetime.strptime(h.published, "%Y-%m-%d").date()
            except ValueError:
                continue
            if lo <= pd_dt <= hi:
                ev.matched_news.append({
                    "title": h.title, "url": h.url,
                    "published": h.published, "snippet": h.snippet,
                    "source": h.source_label, "provider": h.raw_provider,
                })


# === 오케스트레이션 =============================================================
def _gather_search(
    groups: list[EventGroup], company_name: str, industry_keyword: Optional[str],
    ticker: str,
) -> tuple[list[NewsHit], list[str]]:
    """모든 그룹 × 2 쿼리 병렬 호출. dedup 후 누적.

    1 그룹당 직렬이면 N 그룹 × 2 쿼리 × ~5초 → 60초 초과. 병렬화 필수.
    """
    tasks: list[tuple[str, str, str]] = []
    for g in groups:
        queries = _build_search_queries(g, company_name, industry_keyword)
        since, until = _query_window(g)
        for q in queries:
            tasks.append((q, since, until))

    def run_one(task):
        q, s, u = task
        try:
            return search_news(
                q, s, u,
                max_results=MAX_SEARCH_RESULTS_PER_QUERY,
                ticker=ticker,
            ), None
        except Exception as exc:
            return [], f"검색 실패 ({q}): {type(exc).__name__}: {exc}"

    seen: set[str] = set()
    accumulated: list[NewsHit] = []
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for hits, err in pool.map(run_one, tasks):
            if err:
                warnings.append(err)
                continue
            for h in hits:
                if h.url in seen:
                    continue
                seen.add(h.url)
                accumulated.append(h)
    return accumulated, warnings


def _gather_macro(start: str, end: str) -> tuple[dict, list[str]]:
    """yfinance + BOK 거시 시계열 요약."""
    warnings: list[str] = []
    out: dict[str, dict] = {}
    for label, fetcher in [
        ("kospi", fetch_kospi),
        ("usdkrw", fetch_usdkrw),
        ("us_10y", fetch_us_10y),
        ("wti", fetch_wti),
    ]:
        try:
            df = fetcher(start, end)
            out[label] = summarize_series(df)
        except Exception as exc:
            warnings.append(f"{label} 수집 실패: {exc}")
            out[label] = {"available": False}
    try:
        out["bok"] = summarize_macro(start, end)
    except Exception as exc:
        warnings.append(f"BOK 거시 수집 실패: {exc}")
        out["bok"] = {}
    return out, warnings


def _yf_suffix(market: Optional[str]) -> str:
    """KIS market 코드 → yfinance 접미. KOSDAQ=KQ, 그 외 KS."""
    if (market or "").upper() == "KOSDAQ":
        return ".KQ"
    return ".KS"


def _compute_peer_stats(
    df: pd.DataFrame, target_return_pct: float,
) -> Optional[dict]:
    """peer 가격 시리즈 → {수익률, 낙폭, 반등, vs_target_rs}."""
    if df.empty:
        return None
    closes = df["close"].astype(float)
    if closes.iloc[0] <= 0:
        return None
    start_c = float(closes.iloc[0])
    end_c = float(closes.iloc[-1])
    min_c = float(closes.min())
    max_c = float(closes.max())
    ret = (end_c / start_c - 1) * 100
    # max DD = 고점 → 저점, 단 고점이 저점보다 시간상 앞일 때만 유효.
    # 단순 min/max 로 추정 (정확한 cumulative drawdown 은 아니지만 회고 용도엔 충분).
    drawdown = (min_c / max_c - 1) * 100 if max_c > 0 else 0.0
    rebound = (end_c / min_c - 1) * 100 if min_c > 0 else 0.0
    return {
        "return_pct": round(ret, 4),
        "drawdown_pct": round(drawdown, 4),
        "rebound_pct": round(rebound, 4),
        "vs_target_rs": round(target_return_pct - ret, 4),
    }


def _gather_peers(
    ticker: str, start: str, end: str,
    target_return_pct: float,
    *,
    peers_override: Optional[list[str]] = None,
    n: int = 4,
    market: Optional[str] = None,
) -> tuple[list[dict], list[str]]:
    """peers 자동/수동 추출 + 각 peer OHLCV 수집 + 비교표.

    Returns: ([peer_dict, ...], warnings).
    """
    warnings: list[str] = []

    # 1. peer 후보 결정
    if peers_override:
        peer_tickers = list(peers_override)
        source = "manual"
    else:
        peer_objs = find_peers(ticker, n=n)
        peer_tickers = [p.ticker for p in peer_objs]
        source = peer_objs[0].source if peer_objs else "none"
        if not peer_tickers:
            warnings.append(
                f"동종 자동 추출 실패 — HARD_CODED_PEERS/{ticker} 미등록 + "
                "candidate_tickers 미제공. 수동 등록 필요."
            )
            return [], warnings

    # 2. 각 peer 가격 + 메타 수집
    rows: list[dict] = []
    suffix = _yf_suffix(market)
    for pt in peer_tickers:
        # 회사명: DART company_info 캐시 우선, 실패 시 ticker
        try:
            info = fetch_company_info(pt)
            name = info.corp_name if info else pt
            peer_market = (info.corp_cls if info else None)
            # corp_cls Y=유가증권/K=코스닥
            peer_suffix = ".KQ" if peer_market == "K" else ".KS"
        except Exception:
            name = pt
            peer_suffix = suffix

        try:
            df = fetch_close_series(f"{pt}{peer_suffix}", start, end)
        except Exception as exc:
            warnings.append(f"{pt} 가격 수집 실패: {exc}")
            continue
        stats = _compute_peer_stats(df, target_return_pct)
        if not stats:
            warnings.append(f"{pt} 가격 데이터 부족 (yfinance {pt}{peer_suffix})")
            continue
        rows.append({
            "ticker": pt,
            "company_name": name,
            "source": source,
            **stats,
        })
    return rows, warnings


def _gather_disclosures(
    ticker: str, start: str, end: str,
) -> tuple[list[dict], list[str]]:
    """DART 공시 시계열."""
    warnings: list[str] = []
    try:
        items = list_disclosures_for_ticker(ticker, start, end)
        sig = filter_significant_disclosures(items)
        return [
            {
                "rcept_dt": d.rcept_dt, "report_nm": d.report_nm,
                "rcept_no": d.rcept_no, "pblntf_ty": d.pblntf_ty,
            }
            for d in sig
        ], warnings
    except Exception as exc:
        warnings.append(f"DART 공시 수집 실패: {exc}")
        return [], warnings


def compute_event_retro(
    ticker: str, start: str, end: str,
    *,
    store=None,
    enable_search: bool = True,         # 2주차: 기본 On
    enable_macro: bool = True,
    enable_dart: bool = True,
    enable_peers: bool = False,          # 3주차에서 On
    enable_llm_classify: bool = False,   # 3주차에서 On
    industry_keyword: Optional[str] = None,
    peers_override: Optional[list[str]] = None,
) -> dict:
    """이벤트 리뷰 dict 반환.

    2주차: 가격·이벤트·검색·거시·DART 시계열. 4 branches ThreadPoolExecutor 병렬.
    3주차에 peers·LLM 분류 추가.
    """
    warnings: list[str] = []
    df, meta = load_holdings(ticker, start=start, end=end, store=store)
    if df.empty:
        return {
            "meta": {"ticker": ticker, "start": start, "end": end},
            "warnings": [f"holdings 데이터 없음 ({ticker})"],
            "period_summary": None, "timeline": [], "decomposition": {},
        }

    df = df[df["close"] > 0].sort_values("date").reset_index(drop=True)
    company_name = meta.get("company_name") or ""

    period = compute_period_summary(df)
    events = _detect_event_days(df)
    groups = _group_events(events)

    if "volume" not in df.columns or df["volume"].isna().all():
        warnings.append(
            "거래량 컬럼 누락 — backfill 전 데이터. 거래량 z-score 이벤트 skip."
        )

    # 4 branches 병렬 — 부분 실패 격리
    search_hits: list[NewsHit] = []
    macro: dict = {}
    disclosures: list[dict] = []
    peer_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures: dict = {}
        if enable_search:
            futures["search"] = pool.submit(
                _gather_search, groups, company_name, industry_keyword, ticker,
            )
        if enable_macro:
            futures["macro"] = pool.submit(_gather_macro, start, end)
        if enable_dart:
            futures["dart"] = pool.submit(_gather_disclosures, ticker, start, end)
        if enable_peers:
            futures["peers"] = pool.submit(
                _gather_peers, ticker, start, end,
                period.total_return_pct,
                peers_override=peers_override,
                market=meta.get("market"),
            )

        for name, fut in futures.items():
            try:
                # search 가 그룹별 6 병렬화돼도 N 그룹 × 2 쿼리에 따라 시간 변동.
                # 안전 마진 180s — Tavily/NAVER 평균 5-10s.
                res = fut.result(timeout=180)
            except Exception as exc:
                warnings.append(
                    f"{name} branch 실패: {type(exc).__name__}: {exc}"
                )
                continue
            if name == "search":
                search_hits, w = res
                warnings.extend(w)
            elif name == "macro":
                macro, w = res
                warnings.extend(w)
            elif name == "dart":
                disclosures, w = res
                warnings.extend(w)
            elif name == "peers":
                peer_rows, w = res
                warnings.extend(w)

    # 이벤트 ↔ 뉴스 매칭
    if search_hits:
        _match_news_to_events(events, search_hits)

    decomposition = _compute_decomposition(events, period.total_return_pct)

    # KOSPI 기준 R/S 자동 (peers 없이도). 동종 peer 결과는 별도 리스트.
    peers_summary: list[dict] = []
    if macro.get("kospi", {}).get("available"):
        kospi_change = macro["kospi"].get("change_pct")
        target_change = period.total_return_pct
        if kospi_change is not None:
            peers_summary.append({
                "ticker": "KOSPI", "company_name": "KOSPI 지수", "source": "index",
                "return_pct": kospi_change,
                "drawdown_pct": None,
                "rebound_pct": None,
                "vs_target_rs": round(target_change - kospi_change, 4),
            })
    # 3주차: 동종 peer 결과 추가
    peers_summary.extend(peer_rows)

    return {
        "meta": {
            "ticker": ticker,
            "company_name": company_name,
            "market": meta.get("market") or "",
            "start": start, "end": end,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "period_summary": asdict(period),
        "timeline": [asdict(e) for e in events],
        "groups": [asdict(g) for g in groups],
        "decomposition": decomposition,
        "search_hits": hits_to_dicts(search_hits),
        "fundamentals_timeline": disclosures,
        "macro": macro,
        "peers": peers_summary,
        "synthesis": {},
        "sources": [
            {"title": h.title, "url": h.url, "source": h.source_label}
            for h in search_hits if h.url
        ],
        "warnings": warnings,
    }


# === 마크다운 렌더 ==============================================================
def _fmt_pct(p: float) -> str:
    return f"{p:+.2f}%"


def _fmt_int(n: int) -> str:
    return f"{n:+,}" if n else "0"


def render_event_retro_markdown(report: dict) -> str:
    """이벤트 리뷰 dict → 마크다운. 1주차는 기간 요약 + 타임라인 + 분해 표만."""
    meta = report["meta"]
    ps = report.get("period_summary")
    lines: list[str] = []

    # 0. 헤더
    title = (
        f"# 이벤트 리뷰 — {meta.get('company_name') or meta['ticker']} "
        f"({meta['ticker']})"
    )
    lines.append(title)
    lines.append("")
    lines.append(f"**기간**: {meta['start']} ~ {meta['end']}  |  "
                 f"**생성**: {meta['generated_at']}")
    lines.append("")

    # 경고
    if report.get("warnings"):
        lines.append("> ⚠️ " + " / ".join(report["warnings"]))
        lines.append("")

    if ps is None:
        lines.append("_데이터 부족으로 회고 생성 불가._")
        return "\n".join(lines)

    # 1. 기간 요약
    lines.append("## 0. 기간 요약")
    lines.append("")
    lines.append("| | 일자 | 종가 | 비고 |")
    lines.append("|---|---|---:|---|")
    lines.append(f"| 시작 | {ps['start_date']} | {ps['start_close']:,.0f} | — |")
    lines.append(f"| **고점** | {ps['high_date']} | **{ps['high_close']:,.0f}** | — |")
    lines.append(f"| **저점** | {ps['low_date']} | **{ps['low_close']:,.0f}** | — |")
    lines.append(f"| 종료 | {ps['end_date']} | {ps['end_close']:,.0f} | — |")
    lines.append("")
    lines.append(
        f"**기간 수익률**: {_fmt_pct(ps['total_return_pct'])}  |  "
        f"**최대 낙폭(고점→저점)**: {_fmt_pct(ps['max_drawdown_pct'])}"
    )
    lines.append("")

    # 2. 타임라인 (이벤트 + 클러스터)
    timeline = report.get("timeline") or []
    lines.append(f"## 1. 주요 이벤트 타임라인 ({len(timeline)}건)")
    lines.append("")
    if not timeline:
        lines.append("_이벤트 0건._")
    else:
        lines.append("| 일자 | 종가 | 변동률 | 거래량 z | 클러스터 | 분류 사유 |")
        lines.append("|---|---:|---:|---:|---|---|")
        for e in timeline:
            zv = e.get("z_volume")
            zv_str = f"{zv:+.2f}" if zv is not None else "—"
            cluster = e.get("cluster") or "—"
            cluster_label = (
                CLUSTER_LABEL_KO.get(cluster, cluster) if cluster != "—" else "—"
            )
            sigs = []
            if e["is_price_jump"]:
                sigs.append("가격 ±3%")
            if e["is_volume_anomaly"]:
                sigs.append("거래량 z↑")
            if e["is_period_anchor"]:
                sigs.append("anchor")
            sig_str = " · ".join(sigs) if sigs else "—"
            lines.append(
                f"| {e['date']} | {e['close']:,.0f} | {_fmt_pct(e['pct'])} | "
                f"{zv_str} | {cluster_label} | {sig_str} |"
            )
    lines.append("")

    # 3. 그룹 (쿼리 단위 — 정보 제공만)
    groups = report.get("groups") or []
    if groups:
        lines.append(f"## 3. 이벤트 그룹 (인접 ±{GROUP_GAP_DAYS}일 묶음, {len(groups)}그룹)")
        lines.append("")
        for g in groups:
            lines.append(f"- **{g['label']}** ({len(g['members'])}건)")
        lines.append("")

    # 4. 매칭된 뉴스 (이벤트별) — 헤더는 마크다운 H2, 본문은 <details> 로 접기.
    # <summary> 안에 ## 을 넣으면 마크다운 파싱이 안 돼 평문으로 렌더된다.
    # 헤더는 <details> 밖에, 접기 control 만 안에.
    timeline_with_news = [e for e in timeline if e.get("matched_news")]
    if timeline_with_news:
        lines.append(
            f"## 2-A. 이벤트별 매칭 뉴스 ({len(timeline_with_news)}건 이벤트)"
        )
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>클릭하여 펼치기 / 접기</summary>")
        lines.append("")
        for e in timeline_with_news:
            lines.append(f"### {e['date']} ({_fmt_pct(e['pct'])})")
            for n in e["matched_news"][:5]:
                pub = n.get("published") or "?"
                src = n.get("source") or n.get("provider", "")
                lines.append(f"- **{pub}** ({src}) — [{n['title']}]({n['url']})")
                snip = n.get("snippet") or ""
                if snip:
                    lines.append(f"  > {snip[:120]}")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    # 5. 거시 시계열
    macro = report.get("macro") or {}
    if any(macro.get(k, {}).get("available") for k in ("kospi", "usdkrw", "us_10y", "wti")):
        lines.append("## 4. 거시 지표 (같은 기간)")
        lines.append("")
        lines.append("| 지표 | 시작 | 종료 | 변화 |")
        lines.append("|---|---:|---:|---:|")
        labels = {"kospi": "KOSPI", "usdkrw": "USD/KRW",
                  "us_10y": "미국 10Y", "wti": "WTI"}
        for k, label in labels.items():
            v = macro.get(k, {})
            if not v.get("available"):
                continue
            lines.append(
                f"| {label} | {v['start']:,.2f} | {v['end']:,.2f} | "
                f"{_fmt_pct(v.get('change_pct') or 0)} |"
            )
        lines.append("")
        bok = macro.get("bok") or {}
        bok_rows = [
            (label, bok.get(k))
            for k, label in [("base_rate", "BOK 기준금리"),
                              ("corp_aa_yield", "회사채 AA-"),
                              ("treasury_3y", "국고채 3Y")]
        ]
        bok_rows = [(l, v) for l, v in bok_rows if v and v.get("available")]
        if bok_rows:
            lines.append("**BOK 거시**:")
            for label, v in bok_rows:
                lines.append(
                    f"- {label}: {v['start']:.2f}% → {v['end']:.2f}% "
                    f"(Δ {v.get('change_bp', 0):+.1f}bp)"
                )
            lines.append("")

    # 6. 동종 (peers) + KOSPI
    peers = report.get("peers") or []
    if peers:
        lines.append("## 5. 시장 + 동종 비교")
        lines.append("")
        lines.append("| 비교 대상 | 수익률 | 낙폭 | 반등 | vs 본 종목 R/S |")
        lines.append("|---|---:|---:|---:|---:|")
        for p in peers:
            dd = p.get("drawdown_pct")
            rb = p.get("rebound_pct")
            dd_str = _fmt_pct(dd) if dd is not None else "—"
            rb_str = _fmt_pct(rb) if rb is not None else "—"
            lines.append(
                f"| {p['company_name']} ({p['ticker']}) | "
                f"{_fmt_pct(p['return_pct'])} | {dd_str} | {rb_str} | "
                f"{_fmt_pct(p['vs_target_rs'])} |"
            )
        lines.append("")

    # 7. DART 공시
    discs = report.get("fundamentals_timeline") or []
    if discs:
        lines.append(f"## 6. DART 공시 ({len(discs)}건 — 주요 공시만)")
        lines.append("")
        lines.append("| 공시일 | 보고서명 | 접수번호 |")
        lines.append("|---|---|---|")
        for d in discs:
            lines.append(
                f"| {d['rcept_dt']} | {d['report_nm']} | `{d['rcept_no']}` |"
            )
        lines.append("")

    # 8. 분해 표
    decomp = report.get("decomposition") or {}
    if decomp:
        lines.append("## 7. 분해 (클러스터별 변동 기여)")
        lines.append("")
        lines.append("| 클러스터 | 변동 기여 |")
        lines.append("|---|---:|")
        for c in CLUSTER_ENUM:
            v = decomp.get(c, 0.0)
            lines.append(f"| {CLUSTER_LABEL_KO[c]} | {_fmt_pct(v)} |")
        total = sum(decomp.get(c, 0.0) for c in CLUSTER_ENUM)
        lines.append(f"| **합계** | **{_fmt_pct(total)}** |")
        lines.append("")
        lines.append(
            "_분해 수치는 결정적 계산_ (이벤트 day pct 합산 + drift 잔차). "
            "**클러스터 분류와 자연어 설명은 3주차 LLM 통합 호출 후 추가.**"
        )
        lines.append("")

    return "\n".join(lines)
