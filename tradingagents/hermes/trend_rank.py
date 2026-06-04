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


def format_digest(
    market: str,
    *,
    asof_date: Optional[str] = None,
    top_n: int = 10,
    store: Optional[TrendStore] = None,
) -> str:
    """Telegram 일일 다이제스트 텍스트(fade 와치리스트)."""
    df = fade_ranking(
        market, asof_date=asof_date, top_n=top_n, store=store, min_sources=MIN_SOURCES
    )
    mkt = market.upper()
    if df.empty:
        return f"[트렌드 {mkt}] 다중소스 동의 종목 없음(≥{MIN_SOURCES}소스)."
    asof = df["asof_date"].iloc[0]
    lines = [
        f"📊 트렌드 와치리스트 [{mkt}] {asof}",
        "(여러 소스 동시 과열 = 군집/천장 경고 — 추격 아닌 페이드/리스크 플래그)",
        "",
    ]
    for i, row in df.iterrows():
        flag = "🟢" if row["leadingness"] == "L" else "⚠️"
        lines.append(
            f"{i + 1}. {flag} {row['entity']}  score={row['fade_score']} "
            f"({row['n_sources']}소스: {row['sources']}, {row['leadingness']})"
        )
    return "\n".join(lines)
