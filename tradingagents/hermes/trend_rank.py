"""트렌드 신호 랭킹 — 개별종목 fade(군집/과열) 점수.

노트의 핵심: 개별종목 attention 은 대개 늦은 contrarian 신호다. 따라서 여러
소스에서 동시에 '뜨거운' 종목을 군집(crowding) 후보 = **fade/리스크 플래그**로
표면화한다(추격 매수 신호가 아님).

조작·봇 방어로 **≥2개 독립 소스 동의**를 요구한다 — 단일 소스 스파이크는 무시.
소스 간 abnormal 스케일이 달라(momentum vs rel-volume) 소스 내 순위를 점수로
환산해 합산한다(Phase1 단순화; 교차소스 정규화 z 는 Phase2).
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from ..dataflows.trend_store import TrendStore

MIN_SOURCES = 2
ROTATION_PEAK_PCT = 10.0  # 섹터 RS-momentum 과열 임계(%) — peak-chase 경고


def _kr_min_sources() -> int:
    """한국 fade 통과 최소 독립 소스 수(런타임 env 제어).

    한국은 단일 소스(종목토론방)로 출시 → 기본 1. 네이버 DataLab 검색량(독립
    2번째 소스)을 앱에 인증해 라이브로 들어오면 ``TREND_KR_MIN_SOURCES=2`` 로
    플립해 '2소스 독립 동의'만 통과(조작 방어 승격). 기본 1 은 라이브를 비우지
    않기 위함. import 시점이 아니라 호출 시점에 읽어 cron env 변경이 즉시 반영
    되고, 잘못된 값(빈문자열·비정수)은 1 로 폴백해 import 를 깨지 않는다.
    """
    try:
        return int(os.getenv("TREND_KR_MIN_SOURCES", "1"))
    except (ValueError, TypeError):
        return 1


def _resolve_min_sources(market: str, store) -> int:
    """fade 통과 최소 소스 수: active 정책 > (KR)env > 기본(KR 1·US 2).

    Hermes 대화로 ``update_trend_filter`` 가 저장한 정책을 cron·digest 가 즉시
    반영한다(영속). 정책 없으면 기존 동작(KR env, US MIN_SOURCES) 유지.
    """
    try:
        pol = store.get_active_policy(market)
        if pol and pol.get("min_sources") is not None:
            return int(pol["min_sources"])
    except Exception:
        pass
    return _kr_min_sources() if market == "kr" else MIN_SOURCES


def _resolve_rotation_peak(market: str, store) -> float:
    """섹터 과열 임계(%): active 정책 > 기본 ROTATION_PEAK_PCT. (섹터=US 전용)"""
    try:
        pol = store.get_active_policy(market)
        if pol and pol.get("rotation_peak_pct") is not None:
            return float(pol["rotation_peak_pct"])
    except Exception:
        pass
    return ROTATION_PEAK_PCT

# 다이제스트 표시용 한글 라벨(내부 식별자는 영문 유지, 출력만 풀어 씀)
_QUADRANT_KO = {"Leading": "주도", "Improving": "개선", "Weakening": "약화", "Lagging": "부진"}
_LEAD_MOVE = {
    "C": "가격과 같이 움직임",
    "L": "가격보다 먼저 움직임",
    "Lag": "가격보다 뒤늦게 움직임",
}


def _kr_price(code: str) -> Optional[dict]:
    """6자리 코드 → 주가·5일 등락(yfinance, .KS→.KQ 폴백). 실패/NaN 시 None.

    상폐·거래정지 종목은 NaN 을 낼 수 있어 모든 값을 유한값으로 가드한다.
    """
    import math

    try:
        import yfinance as yf
    except Exception:
        return None

    def _fin(x: float) -> bool:
        return isinstance(x, float) and math.isfinite(x)

    for suf in (".KS", ".KQ"):
        try:
            h = yf.Ticker(f"{code}{suf}").history(period="1mo")
        except Exception:
            continue
        if h is None or h.empty or len(h) < 2:
            continue
        close = h["Close"].astype(float)
        vol = h["Volume"].astype(float)
        cur = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        if not _fin(cur) or cur <= 0:  # 상폐/이상치 → 다음 접미사 시도
            continue
        chg1 = (cur / prev - 1) * 100 if _fin(prev) and prev > 0 else 0.0
        base5 = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
        chg5 = (cur / base5 - 1) * 100 if _fin(base5) and base5 > 0 else 0.0
        vlast = float(vol.iloc[-1])
        vlast = vlast if _fin(vlast) else 0.0
        vavg = float(vol.iloc[-20:].mean()) if len(vol) >= 5 else float(vol.mean())
        vol_ratio = (vlast / vavg) if _fin(vavg) and vavg > 0 else 0.0
        return {
            "price": round(cur),
            "change_1d_pct": round(chg1, 1),
            "change_5d_pct": round(chg5, 1),
            "volume": int(vlast),
            "volume_ratio": round(vol_ratio, 1),
        }
    return None


def _divergence_verdict(change_5d_pct: float) -> str:
    """관심(fade)은 양만 본다 → 주가 5일 등락으로 '관심 vs 주가' 괴리를 판정해
    종목마다 다른 결론을 낸다(관심↑·주가↓ 면 페이드=천장 경고 강화)."""
    if change_5d_pct <= -3:
        return f"관심 뜨거운데 주가 5일 {change_5d_pct:+.1f}% — 페이드(천장) 경고 강화"
    if change_5d_pct >= 5:
        return f"관심·주가 5일 {change_5d_pct:+.1f}% 동반 급등 — 추격 과열 주의"
    return f"관심 떴지만 주가 5일 {change_5d_pct:+.1f}% 보합 — 관망"

# 소스 metric → 사람이 읽는 근거 포맷. fade 점수의 '왜'(동시 과열의 실제 신호값)를 명시.
_EVIDENCE_FMT = {
    "mention_momentum_24h": lambda r: f"레딧 언급 {float(r['abnormal_value']):+.0%}(24시간)",
    "unusual_volume_rank": lambda r: f"거래량 급증 #{int(r['rank'])}",
    "asvi": lambda r: f"검색량 급증 {float(r['abnormal_value']):+.2f}",
    "os_ratio": lambda r: f"옵션 거래 쏠림 {float(r['abnormal_value']):.3f}",
    "social_volume": lambda r: f"소셜 글 {int(r['raw_value'])}건",
    "board_volume": lambda r: f"종목토론방 글 {int(r['raw_value'])}건",
    "board_sentiment": lambda r: _fmt_board_sentiment(r),
}


def _fmt_board_sentiment(r) -> str:
    """공감/비공감 순도(abnormal_value ∈ [-1,1]) → 분위기 한 줄(방향, 크라우딩 아님)."""
    tilt = float(r["abnormal_value"])
    pct = abs(tilt) * 100
    if tilt > 0.05:
        return f"토론방 분위기 긍정 우세(공감 +{pct:.0f}%)"
    if tilt < -0.05:
        return f"토론방 분위기 부정 우세(비공감 +{pct:.0f}%)"
    return "토론방 분위기 중립"

# 각 신호의 대표 학술 출처(노트 기반) — 신호별 방법론에 가장 직접적인 논문으로 정밀 매핑.
_SOURCE_REF = {
    # ApeWisdom = Reddit/WSB 소셜 언급 herding → Robinhood/메임주 herding 직접 연구
    "mention_momentum_24h": "Barber-Huang-Odean-Schwarz 2022",
    # 비정상 거래량 = Barber-Odean 의 '단일 최고 attention 지표'
    "unusual_volume_rank": "Barber-Odean 2008",
    # 검색량 attention
    "asvi": "Da-Engelberg-Gao 2011",
    # 옵션/주식 거래량 비율(O/S)
    "os_ratio": "Johnson-So 2012",
    # StockTwits 메시지 감성/볼륨
    "social_volume": "Divernois-Filipovic 2024",
    # 네이버 종목토론방 글수 = 인터넷 주식게시판 herding
    "board_volume": "Antweiler-Frank 2004",
}


def _source_evidence(g) -> list:
    """종목의 소스별 [신호값 · 기여도 · 학술 근거] 근거 리스트('왜 과열인지')."""
    ev = []
    g2 = g.sort_values("rank_score", ascending=False) if "rank_score" in g.columns else g
    for _, r in g2.iterrows():
        fmt = _EVIDENCE_FMT.get(r["metric"])
        try:
            sig = fmt(r) if fmt else str(r["source"])
        except (ValueError, TypeError, KeyError):
            sig = str(r["source"])
        # 감성 틸트는 *방향*(크라우딩 기여 아님) → 기여도·학술 없이 단독 표기.
        if r["metric"] == "board_sentiment":
            ev.append(sig)
            continue
        parts = [sig]
        sc = r.get("rank_score")
        if sc is not None and not pd.isna(sc):
            parts.append(f"기여도 {float(sc):.2f}")
        ref = _SOURCE_REF.get(r["metric"])
        if ref:  # 전체 인용(Barber-Huang-Odean-Schwarz 2022) → 대표저자+연도(Barber 2022)
            short = f"{ref.split('-')[0]} {ref.split()[-1]}"
            parts.append(f"학술 {short}")
        ev.append(" · ".join(parts))
    return ev


def _fade_tier(n_sources: int) -> str:
    """소스 동의 수 → 관심 쏠림 등급(많은 곳이 동시에 가리킬수록 강함)."""
    if n_sources >= 4:
        return "🔴 관심 매우 쏠림"
    if n_sources == 3:
        return "🟠 관심 많이 쏠림"
    return "🟡 관심 쏠림 시작"


def _rrg_quadrant(rs_ratio, rs_mom_pct) -> str:
    """RRG 사분면 — 상대강도 레벨(RS-ratio 100 중심) × 모멘텀 부호."""
    strong = rs_ratio is not None and not pd.isna(rs_ratio) and float(rs_ratio) >= 100.0
    rising = rs_mom_pct is not None and not pd.isna(rs_mom_pct) and float(rs_mom_pct) > 0
    if strong and rising:
        return "Leading"
    if not strong and rising:
        return "Improving"
    if strong and not rising:
        return "Weakening"
    return "Lagging"


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
        "entity", "fade_score", "tier", "n_sources", "sources",
        "leadingness", "evidence", "asof_date",
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
                "tier": _fade_tier(len(srcs)),
                "n_sources": len(srcs),
                "sources": ",".join(srcs),
                "leadingness": _agg_leadingness(g["leadingness"].tolist()),
                "evidence": _source_evidence(g),
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
    cols = ["entity", "rs_momentum", "rs_ratio", "quadrant", "rank", "leadingness", "asof_date"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    asof = asof_date or df["asof_date"].max()
    df = df[df["asof_date"] == asof].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={"abnormal_value": "rs_momentum", "raw_value": "rs_ratio"})
    df["quadrant"] = df.apply(
        lambda r: _rrg_quadrant(r.get("rs_ratio"), r["rs_momentum"]), axis=1
    )
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
                    f"📈 {e} 업종이 약세→강세 전환({float(p[e]):+.1f}%→{float(t[e]):+.1f}%)"
                    " — 개선→주도 진입"
                )
    # 2) peak-chase: 최상위 섹터 과열 → 추격 주의
    peak = _resolve_rotation_peak(market, store)
    top = t.sort_values(ascending=False)
    if not top.empty and float(top.iloc[0]) >= peak:
        alerts.append(
            f"⚠️ {top.index[0]} 업종 과열({float(top.iloc[0]):+.1f}%) — 추격 주의"
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


def _kr_provenance(codes, *, store=None) -> dict:
    """한국 코드 → {us, reason, name} 연결근거(US fade 와의 브리지 캐시 역조회)."""
    try:
        from .kr_peer_bridge import KrPeerCache

        return KrPeerCache().reverse(codes)
    except Exception:
        return {}


def format_digest(
    market: str,
    *,
    asof_date: Optional[str] = None,
    top_n: int = 10,
    store: Optional[TrendStore] = None,
    enrich_price: bool = False,
) -> str:
    """Telegram 일일 다이제스트 — 알림 + 업종 흐름 + 과열 주목 종목.

    KR 은 각 종목에 '어떤 미국 종목에서 연결됐나'(연결근거)를 함께 표기한다.
    업종 흐름은 미국 전용이라 KR 에선 데이터가 없어 자동으로 빠진다.

    ``enrich_price=True`` 면 KR 종목의 '신호 성격' 줄을 주가 5일 괴리 기반
    **판정**(페이드 경고/추격 과열/관망)으로 바꾼다 — 종목마다 결론이 달라진다.
    yfinance 네트워크 호출이 들어가므로 cron 다이제스트에서만 켜고, 즉시 응답하는
    MCP 패스스루(get_kr_trend_watchlist)·테스트는 기본 off 로 둔다.
    """
    store = store or TrendStore()
    mkt = market.upper()
    is_kr = market == "kr"
    fade = fade_ranking(
        market, asof_date=asof_date, top_n=top_n, store=store,
        min_sources=_resolve_min_sources(market, store),
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
    # 업종 흐름 — 주도/개선/약화/부진 + 근거(시장 대비 강도, 상승 속도)
    if not rot.empty:
        lines.append(
            f"🔄 업종 흐름 [{mkt}] (주도/개선/약화/부진 · 학술 Moskowitz-Grinblatt 1999)"
        )
        for _, r in rot.iterrows():
            ratio = r.get("rs_ratio")
            ratio_s = (
                f"시장 대비 강도 {float(ratio):.0f}(100=평균)"
                if ratio is not None and ratio == ratio
                else "시장 대비 강도 정보 없음"
            )
            q_ko = _QUADRANT_KO.get(r["quadrant"], r["quadrant"])
            lines.append(
                f"  [{q_ko}] {r['entity']} — {ratio_s}, 상승 속도 {r['rs_momentum']:+.1f}%"
            )
        lines.append("")
    # 과열 주목 종목 — 등급 + 근거(점수가 왜 그런지 = 어떤 정보원이 무엇을 보였나)
    if not fade.empty:
        asof = fade["asof_date"].iloc[0]
        prov = _kr_provenance(list(fade["entity"]), store=store) if is_kr else {}
        lines.append(f"📊 과열 주목 종목 [{mkt}] {asof}")
        lines.append(
            "(미국에서 달아오른 종목의 한국 짝 — 토론방이 달아오르기 시작)"
            if is_kr
            else "(여러 곳에서 동시에 화제 = 과열 경고 — 따라 사는 신호 아님)"
        )
        for i, row in fade.iterrows():
            n = int(row["n_sources"])
            code = row["entity"]
            pv = prov.get(code, {})
            label = f"{pv.get('name') or code}({code})" if is_kr else code
            lines.append(
                f"{i + 1}. {row.get('tier', '')} {label}  "
                f"쏠림점수 {row['fade_score']} · {n}곳 동의"
            )
            if is_kr and pv.get("us"):
                reason = (pv.get("reason") or "").strip()
                if len(reason) > 28:  # Telegram 가독성 — 첫 구절만
                    reason = reason[:28] + "…"
                rs = f" · {reason}" if reason else ""
                lines.append(f"   연결: 🇺🇸{pv['us']}{rs} → 한국 짝")
            ev = row.get("evidence")
            if isinstance(ev, list) and ev:
                lines.append("   근거:")
                for e in ev:
                    lines.append(f"     · {e}")
            verdict = None
            if enrich_price and is_kr:
                px = _kr_price(code)
                if px:
                    verdict = _divergence_verdict(px["change_5d_pct"])
            if verdict:
                lines.append(f"   → 판정: {verdict}")
            else:
                move = _LEAD_MOVE.get(row["leadingness"], "가격과 같이 움직임")
                tail = f"{move}(추격 위험) — 과열 주의" if n >= 3 else f"{move} — 막 달아오르는 중"
                lines.append(f"   → 신호 성격: {tail}")
        # 관심 집중 섹터(US 전용 — yfinance 섹터). KR 은 연결근거가 대신함.
        if not is_kr:
            secmap = attention_by_sector(market, store=store, fade=fade)
            named = {s: tks for s, tks in secmap.items() if s != "Unknown"}
            if named:
                lines.append("")
                lines.append("📍 관심 집중 섹터(모니터링):")
                for sec, tks in sorted(named.items(), key=lambda x: -len(x[1])):
                    lines.append(f"  {sec}: {', '.join(tks)}")
    return "\n".join(lines)
