"""트렌드 신호 랭킹 — 개별종목 fade(군집/과열) 점수.

노트의 핵심: 개별종목 attention 은 대개 늦은 contrarian 신호다. 따라서 여러
소스에서 동시에 '뜨거운' 종목을 군집(crowding) 후보 = **fade/리스크 플래그**로
표면화한다(추격 매수 신호가 아님).

조작·봇 방어로 **≥2개 독립 소스 동의**를 요구한다 — 단일 소스 스파이크는 무시.
소스 간 abnormal 스케일이 달라(momentum vs rel-volume) 소스 내 순위를 점수로
환산해 합산한다(Phase1 단순화; 교차소스 정규화 z 는 Phase2).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..dataflows.trend_store import TrendStore

MIN_SOURCES = 2
ROTATION_PEAK_PCT = 10.0  # 섹터 RS-momentum 과열 임계(%) — peak-chase 경고


def _rank_score(rank, top_n: int) -> float:
    """소스 내 순위(1=최상위)를 0~1 점수로. 순위 없으면 0."""
    if rank is None or pd.isna(rank):
        return 0.0
    return max(0.0, (top_n - int(rank) + 1) / top_n)


def _agg_leadingness(tags) -> str:
    """기여 소스 중 가장 보수적인 태그(Lag > C > L). fade 는 본질적으로 경고."""
    s = set(tags)
    for t in ("Lag", "C", "L"):
        if t in s:
            return t
    return "C"


def fade_ranking(
    market: str,
    *,
    asof_date: Optional[str] = None,
    top_n: int = 20,
    min_sources: int = MIN_SOURCES,
    store: Optional[TrendStore] = None,
) -> pd.DataFrame:
    """시장의 최신(또는 지정) asof_date 에서 ticker 별 fade_score.

    여러 소스에 동시 등장(≥min_sources)한 종목만 남긴다. 반환 컬럼:
    ``entity, fade_score, n_sources, sources, leadingness, asof_date``.
    """
    store = store or TrendStore()
    snap = store.hot_list(market, asof_date=asof_date, top=10_000)
    if not snap.empty and "entity_type" in snap.columns:
        snap = snap[snap["entity_type"] == "ticker"]  # fade 는 개별종목만(섹터/테마 제외)
    empty_cols = [
        "entity", "fade_score", "n_sources", "sources", "leadingness", "asof_date",
    ]
    if snap.empty:
        return pd.DataFrame(columns=empty_cols)

    asof = snap["asof_date"].iloc[0]
    snap = snap.copy()
    snap["rank_score"] = snap["rank"].map(lambda r: _rank_score(r, top_n))

    agg = []
    for entity, g in snap.groupby("entity"):
        srcs = sorted(set(g["source"]))
        if len(srcs) < min_sources:
            continue
        agg.append(
            {
                "entity": entity,
                "fade_score": round(float(g["rank_score"].sum()), 4),
                "n_sources": len(srcs),
                "sources": ",".join(srcs),
                "leadingness": _agg_leadingness(g["leadingness"].tolist()),
                "asof_date": asof,
            }
        )
    out = pd.DataFrame(agg, columns=empty_cols)
    if out.empty:
        return out
    return (
        out.sort_values("fade_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def rotation_ranking(
    market: str,
    *,
    asof_date: Optional[str] = None,
    top_n: int = 11,
    store: Optional[TrendStore] = None,
) -> pd.DataFrame:
    """섹터 로테이션 — RS-momentum 상위(Improving→Leading). entity_type='sector'.

    반환 컬럼: ``entity, rs_momentum, rank, leadingness, asof_date``.
    """
    store = store or TrendStore()
    df = store.read(market=market, source="sector_rotation")
    cols = ["entity", "rs_momentum", "rank", "leadingness", "asof_date"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    asof = asof_date or df["asof_date"].max()
    df = df[df["asof_date"] == asof].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={"abnormal_value": "rs_momentum"})
    return (
        df.sort_values("rs_momentum", ascending=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )


def rotation_alerts(
    market: str,
    *,
    asof_date: Optional[str] = None,
    store: Optional[TrendStore] = None,
) -> list:
    """섹터 로테이션 고급 알림 — RS-momentum 양전환(Improving 진입) + 과열(peak-chase).

    양전환(전일 ≤0 → 오늘 >0)은 전일 데이터가 누적돼야 감지된다. 과열(최상위
    섹터 RS-momentum ≥ ROTATION_PEAK_PCT)은 당일만으로 판정(노트: flow 정점·이미
    Leading 추격 금지). 반환=알림 문자열 리스트(없으면 빈).
    """
    store = store or TrendStore()
    df = store.read(market=market, source="sector_rotation")
    if df.empty:
        return []
    dates = sorted(df["asof_date"].unique())
    today = asof_date or dates[-1]
    t = df[df["asof_date"] == today].set_index("entity")["abnormal_value"]
    if t.empty:
        return []
    alerts: list = []
    # 1) 양전환 = Improving→Leading 진입 (전일 비교)
    prevs = [d for d in dates if d < today]
    if prevs:
        p = df[df["asof_date"] == prevs[-1]].set_index("entity")["abnormal_value"]
        for e in t.index:
            if e in p.index and float(p[e]) <= 0 < float(t[e]):
                alerts.append(
                    f"📈 {e} 섹터 RS-momentum 양전환({float(p[e]):+.1f}%→{float(t[e]):+.1f}%)"
                    " — Improving→Leading 진입"
                )
    # 2) peak-chase: 최상위 섹터 과열 → 추격 주의
    top = t.sort_values(ascending=False)
    if not top.empty and float(top.iloc[0]) >= ROTATION_PEAK_PCT:
        alerts.append(
            f"⚠️ {top.index[0]} 섹터 과열({float(top.iloc[0]):+.1f}%) — 추격 주의(peak-chase)"
        )
    return alerts


def attention_by_sector(
    market: str,
    *,
    asof_date: Optional[str] = None,
    top_n: int = 10,
    store: Optional[TrendStore] = None,
    fade=None,
) -> dict:
    """와치리스트(fade) 종목을 GICS 섹터로 집계 — '어느 섹터에 관심 집중'.

    **모니터링용**(① 통제실험: attention 은 독립 alpha 아님 → alpha 주장 안 함).
    반환 ``{sector: [tickers]}``. 섹터 매핑은 yfinance(캐시). 'Unknown' 은 제외하지
    않고 그대로 반환(호출부에서 표시 정책 결정).
    """
    if fade is None:
        fade = fade_ranking(market, asof_date=asof_date, top_n=top_n, store=store)
    if fade.empty:
        return {}
    from ..dataflows.trends.sector_map import get_sectors

    secs = get_sectors(list(fade["entity"]))
    out: dict = {}
    for ent in fade["entity"]:
        out.setdefault(secs.get(str(ent).upper(), "Unknown"), []).append(ent)
    return out


def format_digest(
    market: str,
    *,
    asof_date: Optional[str] = None,
    top_n: int = 10,
    store: Optional[TrendStore] = None,
) -> str:
    """Telegram 일일 다이제스트 — 고급 알림 + 섹터 로테이션(방향) + 종목 fade(군집 경고)."""
    store = store or TrendStore()
    mkt = market.upper()
    fade = fade_ranking(
        market, asof_date=asof_date, top_n=top_n, store=store, min_sources=MIN_SOURCES
    )
    rot = rotation_ranking(market, asof_date=asof_date, top_n=3, store=store)
    alerts = rotation_alerts(market, asof_date=asof_date, store=store)
    if fade.empty and rot.empty and not alerts:
        return f"[트렌드 {mkt}] 신호 없음."

    lines: list[str] = []
    # 고급 알림 먼저 — 전환/과열(important)
    if alerts:
        lines.append(f"🔔 알림 [{mkt}]")
        lines.extend(f"  {a}" for a in alerts)
        lines.append("")
    # 섹터 로테이션 — durable·방향(노트: 테마로 방향, 종목으로 군집 회피)
    if not rot.empty:
        lines.append(f"🔄 섹터 로테이션 [{mkt}] (RS-momentum, Improving→Leading)")
        for _, r in rot.iterrows():
            lines.append(f"  {r['entity']} {r['rs_momentum']:+.1f}%")
        lines.append("")
    # 개별종목 fade 와치리스트 — 군집/천장 경고
    if not fade.empty:
        asof = fade["asof_date"].iloc[0]
        lines.append(f"📊 종목 와치리스트 [{mkt}] {asof}")
        lines.append("(여러 소스 동시 과열 = 군집/천장 경고 — 추격 아닌 페이드/리스크 플래그)")
        for i, row in fade.iterrows():
            flag = "🟢" if row["leadingness"] == "L" else "⚠️"
            lines.append(
                f"{i + 1}. {flag} {row['entity']}  score={row['fade_score']} "
                f"({row['n_sources']}소스: {row['sources']}, {row['leadingness']})"
            )
        # 관심 집중 섹터(모니터링용 — alpha 아님). 종목 多 섹터 순.
        secmap = attention_by_sector(market, store=store, fade=fade)
        named = {s: tks for s, tks in secmap.items() if s != "Unknown"}
        if named:
            lines.append("")
            lines.append("📍 관심 집중 섹터(모니터링):")
            for sec, tks in sorted(named.items(), key=lambda x: -len(x[1])):
                lines.append(f"  {sec}: {', '.join(tks)}")
    return "\n".join(lines)
