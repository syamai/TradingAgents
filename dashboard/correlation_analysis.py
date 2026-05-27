"""주가 변동 ↔ 주체별 순매수/누적 보유량 상관 분석.

사용 기법:
  1. Pearson product-moment correlation — 동시 r + p-value (Fisher z 정규근사)
  2. Cross-correlation function (CCF) — lag -3..+3 의 r
  3. Sub-period / regime-split correlation — 전체·연도 구간 비교
  4. Conditional mean — 상승일/하락일 평균 net_qty
  5. Compositional analysis — 5년 누적 매수 주체 비중
  6. Level correlation — cum_qty 수준 vs 종가 수준

산출은 JSON 직렬화 가능한 dict. SQLite payload 저장과 streamlit 렌더에 공용.
"""
from __future__ import annotations

from datetime import datetime
from math import erf, log, sqrt
from typing import Optional

import numpy as np
import pandas as pd

from dashboard.holdings_chart import SUBJECT_LABELS, SUBJECTS_ORDER
from tradingagents.dataflows.kis_holdings import SUBS_10

LAG_OFFSETS: tuple[int, ...] = (-3, -1, 0, 1, 3)
KEY_SUBJECTS_FOR_LAG: tuple[str, ...] = (
    "foreign", "pension", "securities", "private_equity", "retail",
)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r + two-sided p-value (Fisher z 정규근사). NaN 자동 제거."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return 0.0, 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0 - 1e-12:
        return r, 0.0
    z = 0.5 * log((1 + r) / (1 - r))
    se = 1.0 / sqrt(n - 3)
    # two-sided p via standard normal CDF
    p = 2.0 * (1.0 - 0.5 * (1 + erf(abs(z / se) / sqrt(2))))
    return r, float(p)


def _sig_marker(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _interpret_r(r: float) -> str:
    if r > 0.30:
        return "↑↑ 강한 동조"
    if r > 0.10:
        return "↑ 동조"
    if r < -0.30:
        return "↓↓ 강한 역행"
    if r < -0.10:
        return "↓ 역행"
    return "≈ 약함"


def _lag_corr(net: pd.Series, ret: pd.Series, k: int) -> float:
    """lag k의 r. k>0 → net 이 ret 을 k일 선도. k<0 → net 이 ret 을 k일 후행."""
    net_a = net.to_numpy(dtype=float)
    ret_a = ret.to_numpy(dtype=float)
    if k == 0:
        r, _ = pearson(net_a, ret_a)
    elif k > 0:
        r, _ = pearson(net_a[:-k], ret_a[k:])
    else:  # k < 0
        r, _ = pearson(net_a[-k:], ret_a[:k])
    return r


def compute_correlation_report(
    df: pd.DataFrame,
    *,
    ticker: str,
    company_name: Optional[str] = None,
    market: Optional[str] = None,
) -> dict:
    """holdings DataFrame → 상관 분석 리포트 dict (JSON 직렬화 가능).

    df 는 ``KisHistoryStore.read(ticker, "holdings")`` 반환물 형태를 기대.
    빈 df 도 안전(빈 sections 반환).
    """
    out: dict = {
        "ticker": ticker,
        "company_name": company_name,
        "market": market,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": None,
        "n_days": 0,
        "price": None,
        "cumulative": [],
        "concurrent": [],
        "up_down": [],
        "lag": [],
        "regime": [],
        "level": [],
    }
    if df.empty:
        return out

    df = df.sort_values("date").reset_index(drop=True)
    ret = df["price_change_pct"].astype(float)

    p0 = float(df["close"].iloc[0])
    p1 = float(df["close"].iloc[-1])
    out["window"] = {"start": df["date"].iloc[0], "end": df["date"].iloc[-1]}
    out["n_days"] = int(len(df))
    out["price"] = {
        "start": p0,
        "end": p1,
        "total_return_pct": round((p1 - p0) / p0 * 100.0, 2) if p0 else 0.0,
        "daily_std_pct": round(float(ret.std()), 3),
        "max_pct": round(float(ret.max()), 2),
        "min_pct": round(float(ret.min()), 2),
    }

    # [1] 5년 누적 매수 (compositional)
    for s in SUBJECTS_ORDER:
        cum = int(df[f"{s}_cum_qty"].iloc[-1])
        pct = float(df[f"{s}_pct"].iloc[-1])
        out["cumulative"].append({
            "subject": s,
            "label": SUBJECT_LABELS[s],
            "cum_qty": cum,
            "pct": round(pct, 4),
            "is_info_total": s == "foreign",
        })

    # [2] 동시 상관 (Pearson r) — net_qty(t) vs ret(t)
    concurrent = []
    for s in SUBJECTS_ORDER:
        net = df[f"{s}_net_qty"].astype(float)
        r, p = pearson(net.to_numpy(), ret.to_numpy())
        concurrent.append({
            "subject": s, "label": SUBJECT_LABELS[s],
            "r": round(r, 4), "p_value": p,
            "significance": _sig_marker(p),
            "interpretation": _interpret_r(r),
        })
    concurrent.sort(key=lambda x: abs(x["r"]), reverse=True)
    out["concurrent"] = concurrent

    # [3] 상승일/하락일 평균 net_qty
    up_mask = df["price_change_pct"] > 0
    dn_mask = df["price_change_pct"] < 0
    flat_n = int(((~up_mask) & (~dn_mask)).sum())
    up_down_meta = {
        "n_up": int(up_mask.sum()), "n_down": int(dn_mask.sum()),
        "n_flat": flat_n,
    }
    up_down: list = []
    for s in SUBJECTS_ORDER:
        col = f"{s}_net_qty"
        u = float(df.loc[up_mask, col].mean()) if up_mask.any() else 0.0
        d = float(df.loc[dn_mask, col].mean()) if dn_mask.any() else 0.0
        diff = u - d
        if u > 0 and d < 0:
            tag = "accumulating_up"  # 상승일 매수, 하락일 매도 — 추세 추종
        elif u < 0 and d > 0:
            tag = "counter_trend"    # 상승일 매도, 하락일 매수 — 역행
        else:
            tag = "mixed"
        up_down.append({
            "subject": s, "label": SUBJECT_LABELS[s],
            "up_mean": round(u, 1), "down_mean": round(d, 1),
            "diff": round(diff, 1),
            "pattern": tag,
        })
    out["up_down"] = up_down
    out["up_down_meta"] = up_down_meta

    # [4] Lag (CCF)
    lag_rows = []
    for s in KEY_SUBJECTS_FOR_LAG:
        if s not in SUBJECTS_ORDER:
            continue
        net = df[f"{s}_net_qty"].astype(float)
        cells = {f"t{k:+d}" if k != 0 else "t0":
                 round(_lag_corr(net, ret, k), 4) for k in LAG_OFFSETS}
        lag_rows.append({"subject": s, "label": SUBJECT_LABELS[s], "lags": cells})
    out["lag"] = lag_rows

    # [5] Regime split
    windows = [
        ("전체", None, None),
        ("2021-22", "2021-01-01", "2022-12-31"),
        ("2023-24", "2023-01-01", "2024-12-31"),
        ("2025-", "2025-01-01", None),
    ]
    regime_rows = []
    for s in KEY_SUBJECTS_FOR_LAG:
        cells: dict = {}
        for name, st, ed in windows:
            sub = df.copy()
            if st:
                sub = sub[sub["date"] >= st]
            if ed:
                sub = sub[sub["date"] <= ed]
            if len(sub) < 30:
                cells[name] = None
                continue
            net = sub[f"{s}_net_qty"].astype(float).to_numpy()
            rr = sub["price_change_pct"].astype(float).to_numpy()
            r, _ = pearson(net, rr)
            cells[name] = round(r, 4)
        regime_rows.append({"subject": s, "label": SUBJECT_LABELS[s], "windows": cells})
    out["regime"] = regime_rows

    # [6] Level correlation — cum_qty vs close
    close = df["close"].astype(float).to_numpy()
    level_rows = []
    for s in SUBJECTS_ORDER:
        cum = df[f"{s}_cum_qty"].astype(float).to_numpy()
        if cum.std() == 0:
            level_rows.append({"subject": s, "label": SUBJECT_LABELS[s], "r": 0.0})
            continue
        r = float(np.corrcoef(cum, close)[0, 1])
        level_rows.append({"subject": s, "label": SUBJECT_LABELS[s],
                           "r": round(r, 4)})
    out["level"] = level_rows

    return out


# === 마크다운 렌더 (옵시디언·CLI 공용) ======================================

def render_markdown(report: dict) -> str:
    """correlation report dict → 마크다운 문자열."""
    lines: list[str] = []
    name = report.get("company_name") or "(no name)"
    ticker = report["ticker"]
    market = report.get("market") or "-"
    n = report["n_days"]
    if n == 0:
        return f"# {name} ({ticker})\n\n데이터 없음.\n"

    w = report["window"]
    pr = report["price"]

    lines.append(f"# {name} ({ticker}) — 주가 변동 ↔ 주체별 수급 상관 분석")
    lines.append("")
    lines.append(f"- **시장**: {market}")
    lines.append(f"- **기간**: {w['start']} ~ {w['end']} ({n}일)")
    lines.append(f"- **종가**: {pr['start']:,.0f}원 → {pr['end']:,.0f}원 "
                 f"({pr['total_return_pct']:+.1f}%)")
    lines.append(f"- **변동성**: 일별 σ = {pr['daily_std_pct']:.2f}%, "
                 f"최대 {pr['max_pct']:+.1f}% / 최저 {pr['min_pct']:+.1f}%")
    lines.append(f"- **생성**: {report['generated_at']}")
    lines.append("")

    # [1] 누적
    lines.append("## 1. 5년 누적 매수·매도 (cum_qty 최종)")
    lines.append("")
    lines.append("| 주체 | 누적 순매수 | 비중 |")
    lines.append("|---|---:|---:|")
    for r in report["cumulative"]:
        if r["is_info_total"]:
            note = " *(정보용)*"
            pct_s = "—"
        else:
            note = ""
            pct_s = f"{r['pct']:.2f}%"
        sign = "🟢" if r["cum_qty"] > 0 else "🔴" if r["cum_qty"] < 0 else "⚪"
        lines.append(f"| {sign} {r['label']}{note} | {r['cum_qty']:+,} | {pct_s} |")
    lines.append("")

    # [2] 동시 상관
    lines.append("## 2. 동시 상관 — `price_change_pct(t) ↔ net_qty(t)`")
    lines.append("")
    lines.append("| 주체 | Pearson r | p-value | 해석 |")
    lines.append("|---|---:|---:|---|")
    for r in report["concurrent"]:
        p = r["p_value"]
        p_s = f"{p:.2e}" if p > 0 else "0"
        sig = r["significance"]
        lines.append(f"| {r['label']} | {r['r']:+.4f} | {p_s} | "
                     f"{r['interpretation']} {sig} |")
    lines.append("")

    # [3] 상승/하락
    meta = report["up_down_meta"]
    lines.append(f"## 3. 상승일 vs 하락일 평균 순매수 "
                 f"(상승 {meta['n_up']}일 · 하락 {meta['n_down']}일 · "
                 f"보합 {meta['n_flat']}일)")
    lines.append("")
    lines.append("| 주체 | 상승일 평균 | 하락일 평균 | 차이 | 패턴 |")
    lines.append("|---|---:|---:|---:|---|")
    pat_label = {
        "accumulating_up": "추세 추종",
        "counter_trend": "역행 매매",
        "mixed": "혼합",
    }
    for r in report["up_down"]:
        lines.append(f"| {r['label']} | {r['up_mean']:+,.0f} | {r['down_mean']:+,.0f} | "
                     f"{r['diff']:+,.0f} | {pat_label[r['pattern']]} |")
    lines.append("")

    # [4] Lag
    lines.append("## 4. Lag 분석 (CCF) — 수급이 가격을 선도? 후행?")
    lines.append("")
    keys = [f"t{k:+d}" if k != 0 else "t0" for k in LAG_OFFSETS]
    lines.append("| 주체 | " + " | ".join(keys) + " |")
    lines.append("|---|" + "---:|" * len(keys))
    for r in report["lag"]:
        cells = [f"{r['lags'][k]:+.3f}" for k in keys]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("- `t=0` 동시. `t+k` 수급이 k일 *선도*. `t-k` 수급이 k일 *후행*(chasing).")
    lines.append("")

    # [5] Regime
    lines.append("## 5. Regime — 기간별 동시 상관 변화")
    lines.append("")
    if report["regime"]:
        win_keys = list(report["regime"][0]["windows"].keys())
        lines.append("| 주체 | " + " | ".join(win_keys) + " |")
        lines.append("|---|" + "---:|" * len(win_keys))
        for r in report["regime"]:
            cells = []
            for k in win_keys:
                v = r["windows"][k]
                cells.append("n/a" if v is None else f"{v:+.3f}")
            lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
        lines.append("")

    # [6] Level
    lines.append("## 6. 누적 수준 상관 — `r(cum_qty, close)` *(트렌드 영향 큼 — 참고)*")
    lines.append("")
    lines.append("| 주체 | r |")
    lines.append("|---|---:|")
    for r in report["level"]:
        lines.append(f"| {r['label']} | {r['r']:+.3f} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**해석 주의**")
    lines.append("- r 은 **선형 동조**이지 인과가 아님. 공통 요인으로 둘 다 움직여도 r 이 커진다.")
    lines.append("- p-value 는 표본 크기에 민감 — 1222일이면 |r|≈0.06만 넘어도 p<0.05. "
                 "의미는 r 의 *크기* 자체로 판단.")
    lines.append("- Lag 가 0 에 가깝다는 건 수급이 가격을 *1일 이상* 선도/후행하지 않는다는 "
                 "약한 증거. 인트라데이 영향과 즉시 chasing 둘 다 가능.")
    return "\n".join(lines)
