"""누적 보유 추세 분석 — 일별 Granger·CCF가 못 잡는 중기 시그널.

배경:
  advanced_analysis 의 Granger·CCF·MI 는 모두 *일별 차분* 기반이라
  단기 노이즈 안 선형 예측만 검정한다. 외국인 누적 보유가 1년에 걸쳐
  단조 증가 → 1년에 걸쳐 단조 감소 같은 *추세 전환*은 일별 단위에서
  보이지 않는다 (035900 JYP Ent. 사례: 외국인 cum 2022-08~2023-07 매집
  → 2023-07~2026-01 분산, 주가 +129% → -57.5%).

분석:
  1. ``detect_phases`` — 60일 rolling slope 부호 변화로 누적 시리즈를
     trend phase 로 분할 (up/down/flat).
  2. ``phase_regression`` — 각 phase 안에서 ``close ~ cum_qty`` OLS.
     r·slope·R² 로 phase 내 동행성 정량화.
  3. ``compute_absorbers`` — phase 안 zero-sum 흡수자 분석: 주체 A 가
     N주 매도 시 누가 받았는지 (다른 9 주체의 net 변화).
  4. ``trend_concordance`` — 60일 추세 기울기 부호 일치율: 주체 누적의
     60일 기울기 부호와 종가 60일 기울기 부호가 같은 일자 비율.

모든 함수 JSON 직렬화 가능 dict/list 반환. statsmodels 의존 없음 — numpy 만.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from dashboard.holdings_chart import SUBJECT_LABELS
from tradingagents.dataflows.kis_holdings import SUBS_10


# 추세 분석은 가장 영향력 큰 두 주체에 집중 — 너무 많은 phase 표가 가독성↓
KEY_TREND_SUBJECTS: tuple[str, ...] = ("foreign", "retail")

# Phase detection 파라미터
SLOPE_WINDOW = 60          # 추세 기울기 산정 윈도우 (≈ 3개월 거래일)
MIN_PHASE_DAYS = 60        # 최소 phase 길이 — 그 미만은 인접 phase로 병합
FLAT_THRESHOLD = 0.05      # |Δ| < (전체 cum 절대값 max) × 0.05 이면 flat

# Concordance 윈도우
CONCORDANCE_WINDOW = 60


def _ensure_foreign_combined(df: pd.DataFrame) -> pd.DataFrame:
    """``foreign_cum_qty`` 컬럼 보장 — 등록 + 비등록 합."""
    if "foreign_cum_qty" in df.columns:
        return df
    df = df.copy()
    df["foreign_cum_qty"] = (
        df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
    )
    df["foreign_net_qty"] = (
        df["foreign_registered_net_qty"] + df["foreign_unregistered_net_qty"]
    )
    return df


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """각 시점의 직전 ``window`` 일 OLS 기울기 (np.polyfit deg=1).

    첫 ``window - 1`` 일은 NaN. 윈도우가 series 길이보다 길면 전부 NaN.
    """
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    xs = np.arange(window, dtype=float)
    xs_mean = xs.mean()
    xs_var = ((xs - xs_mean) ** 2).sum()
    for i in range(window - 1, n):
        ys = values[i - window + 1: i + 1]
        ys_mean = ys.mean()
        cov = ((xs - xs_mean) * (ys - ys_mean)).sum()
        out[i] = cov / xs_var
    return out


def detect_phases(
    cum_series: pd.Series,
    window: int = SLOPE_WINDOW,
    min_days: int = MIN_PHASE_DAYS,
    flat_threshold: float = FLAT_THRESHOLD,
) -> list[tuple[int, int, str]]:
    """누적 시리즈를 추세 phase 로 분할.

    알고리즘:
      1. ``window`` 일 rolling OLS 기울기
      2. 기울기 부호(+/-/0) 변화 지점을 boundary 후보로
      3. ``min_days`` 미만 phase 는 인접 phase 로 좌측 병합
      4. phase trend: 누적 |Δ| 가 전체 |max| 의 ``flat_threshold`` 미만 → flat

    return: ``[(start_idx, end_idx, trend)]`` — end exclusive, trend ∈ {up,down,flat}.
    """
    values = np.asarray(cum_series.values, dtype=float)
    n = len(values)
    if n < window + min_days:
        delta = values[-1] - values[0]
        trend = "up" if delta > 0 else "down" if delta < 0 else "flat"
        return [(0, n, trend)]

    slopes = _rolling_slope(values, window)
    sign = np.sign(slopes)
    # NaN 영역(첫 window-1) 은 첫 valid 부호로 forward-fill
    first_valid = window - 1
    if np.isnan(sign[first_valid]):
        sign[:] = 0
    else:
        sign[:first_valid] = sign[first_valid]

    # raw boundaries: 부호 변화 지점
    raw = [0]
    for i in range(1, n):
        if sign[i] != sign[i - 1]:
            raw.append(i)
    raw.append(n)

    # min_days 미만 phase 병합: 짧은 phase 는 이전 boundary 에 흡수 (left-merge)
    merged: list[int] = [raw[0]]
    for b in raw[1:]:
        if b - merged[-1] >= min_days:
            merged.append(b)
    if merged[-1] != n:
        merged[-1] = n  # 마지막 phase 강제 종료

    abs_scale = float(np.abs(values).max()) or 1.0
    phases: list[tuple[int, int, str]] = []
    for i in range(len(merged) - 1):
        s, e = merged[i], merged[i + 1]
        delta = values[e - 1] - values[s]
        if abs(delta) < abs_scale * flat_threshold:
            trend = "flat"
        else:
            trend = "up" if delta > 0 else "down"
        phases.append((s, e, trend))
    return phases


def phase_regression(
    df: pd.DataFrame, start: int, end: int, cum_col: str,
) -> dict:
    """Phase 안에서 ``close ~ cum_qty`` 단순 OLS.

    return:
      ``{pearson_r, r_squared, slope_per_million_shares, intercept, n}``.
    """
    seg = df.iloc[start:end]
    x = seg[cum_col].values.astype(float)
    y = seg["close"].values.astype(float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {
            "pearson_r": None, "r_squared": None,
            "slope_per_million_shares": None, "intercept": None, "n": len(x),
        }
    r = float(np.corrcoef(x, y)[0, 1])
    # OLS via covariance
    x_centered = (x - x.mean()) / 1e6  # millions of shares
    slope, intercept = np.polyfit(x_centered, y, 1)
    return {
        "pearson_r": round(r, 4),
        "r_squared": round(r * r, 4),
        "slope_per_million_shares": round(float(slope), 1),
        "intercept": round(float(intercept), 1),
        "n": len(x),
    }


def compute_absorbers(
    df: pd.DataFrame, start: int, end: int, focal_subject: str,
) -> list[dict]:
    """Phase 안 zero-sum 흡수자 분석.

    focal 의 phase 내 누적 변화 부호와 *반대* 방향으로 움직인 다른 주체
    리스트 (절대량 큰 순). focal 매도 → 흡수자(매수), focal 매수 → 공급자(매도).

    return: ``[{subject, label, delta, role: "absorber"|"supplier"|"same_side"}]``
    """
    seg = df.iloc[start:end]
    focal_col = f"{focal_subject}_cum_qty"
    focal_delta = float(seg[focal_col].iloc[-1] - seg[focal_col].iloc[0])
    focal_sign = np.sign(focal_delta)

    out: list[dict] = []
    for s in SUBS_10:
        if s == focal_subject or s.startswith(focal_subject + "_"):
            continue  # foreign 분석 시 foreign_registered / _unregistered 제외
        if focal_subject == "foreign" and s in ("foreign_registered", "foreign_unregistered"):
            continue
        col = f"{s}_cum_qty"
        if col not in df.columns:
            continue
        delta = float(seg[col].iloc[-1] - seg[col].iloc[0])
        if abs(delta) < 1000:
            continue
        if focal_sign != 0 and np.sign(delta) != focal_sign:
            role = "absorber" if focal_sign < 0 else "supplier"
        else:
            role = "same_side"
        out.append({
            "subject": s,
            "label": SUBJECT_LABELS.get(s, s),
            "delta": int(delta),
            "role": role,
        })
    out.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return out


def trend_concordance(
    df: pd.DataFrame, cum_col: str, window: int = CONCORDANCE_WINDOW,
) -> dict:
    """주체 누적 60일 기울기 부호 ↔ 종가 60일 기울기 부호 일치율.

    동일 부호 일자 비율을 % 로. ``flat``(기울기 ≈ 0) 일자는 제외 분모에서.
    """
    cum = df[cum_col].values.astype(float)
    close = df["close"].values.astype(float)
    cum_slope = _rolling_slope(cum, window)
    close_slope = _rolling_slope(close, window)
    valid = ~(np.isnan(cum_slope) | np.isnan(close_slope))
    cum_sign = np.sign(cum_slope[valid])
    close_sign = np.sign(close_slope[valid])
    non_flat = (cum_sign != 0) & (close_sign != 0)
    if non_flat.sum() == 0:
        return {"agreement_pct": None, "n": 0, "window": window}
    agree = (cum_sign[non_flat] == close_sign[non_flat]).sum()
    return {
        "agreement_pct": round(float(agree) / float(non_flat.sum()) * 100, 1),
        "n": int(non_flat.sum()),
        "window": window,
    }


def compute_trend_report(
    df: pd.DataFrame,
    subjects: tuple[str, ...] = KEY_TREND_SUBJECTS,
    window: int = SLOPE_WINDOW,
    min_days: int = MIN_PHASE_DAYS,
) -> dict:
    """추세 분석 통합 보고서.

    각 ``subjects`` 에 대해 phase 분할 + phase별 회귀 + 흡수자 + 동행성.

    return:
      ``{generated_at, window, min_phase_days, subjects: {<name>: {label,
      phases: [{id, start_date, end_date, n_days, trend, cum_change,
      price_start, price_end, price_change_pct, regression, absorbers}],
      trend_concordance}}}``
    """
    df = _ensure_foreign_combined(df).reset_index(drop=True)
    out: dict = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slope_window_days": window,
        "min_phase_days": min_days,
        "n_days": len(df),
        "subjects": {},
    }
    for subject in subjects:
        cum_col = f"{subject}_cum_qty"
        if cum_col not in df.columns:
            continue
        phases_raw = detect_phases(df[cum_col], window=window, min_days=min_days)
        phases_out = []
        for i, (s, e, trend) in enumerate(phases_raw, start=1):
            reg = phase_regression(df, s, e, cum_col)
            absorbers = compute_absorbers(df, s, e, subject) if trend != "flat" else []
            phases_out.append({
                "id": f"P{i}",
                "start_idx": s,
                "end_idx": e,
                "start_date": str(df["date"].iloc[s]),
                "end_date": str(df["date"].iloc[e - 1]),
                "n_days": e - s,
                "trend": trend,
                "cum_change": int(df[cum_col].iloc[e - 1] - df[cum_col].iloc[s]),
                "cum_start": int(df[cum_col].iloc[s]),
                "cum_end": int(df[cum_col].iloc[e - 1]),
                "price_start": float(df["close"].iloc[s]),
                "price_end": float(df["close"].iloc[e - 1]),
                "price_change_pct": round(
                    float(df["close"].iloc[e - 1] / df["close"].iloc[s] - 1) * 100, 1
                ),
                "regression": reg,
                "absorbers": absorbers[:5],  # top 5
            })
        out["subjects"][subject] = {
            "label": SUBJECT_LABELS.get(subject, subject),
            "phases": phases_out,
            "trend_concordance": trend_concordance(df, cum_col),
        }
    return out


# === 자동 해석 + 마크다운 렌더 ==============================================

_TREND_ICON = {"up": "📈", "down": "📉", "flat": "➡️"}
_TREND_LABEL_KO = {"up": "매집(상승추세)", "down": "분산(하락추세)", "flat": "보합"}


def _fmt_int(n: int) -> str:
    return f"{n:+,}" if n else "0"


def render_trend_markdown(report: dict) -> str:
    """추세 분석 보고서 마크다운 렌더링 — 자동 해석 포함."""
    lines: list[str] = []
    lines.append("## 8. 추세 분석 — 누적 보유 phase 분할")
    lines.append("")
    lines.append(
        f"분석 윈도우 {report['slope_window_days']}일 기울기 부호 변화로 phase 분할 · "
        f"최소 phase 길이 {report['min_phase_days']}일"
    )
    lines.append("")
    for subject, info in report["subjects"].items():
        lines.append(f"### 8-{subject} — {info['label']}")
        lines.append("")
        conc = info["trend_concordance"]
        if conc.get("agreement_pct") is not None:
            lines.append(
                f"**60일 추세 동행성**: {info['label']} 누적의 60일 기울기 부호와 "
                f"종가 60일 기울기 부호가 **{conc['agreement_pct']}%** 일치 "
                f"(n={conc['n']}일)"
            )
            lines.append("")
        if not info["phases"]:
            lines.append("_phase 분할 불가 (시리즈 짧음)_")
            lines.append("")
            continue
        # phase 요약 표
        lines.append("| Phase | 기간 | 일수 | 추세 | 누적 Δ | 주가 Δ | phase 내 r |")
        lines.append("|---|---|---:|:--:|---:|---:|---:|")
        for p in info["phases"]:
            icon = _TREND_ICON.get(p["trend"], "")
            r_str = f"{p['regression']['pearson_r']:+.3f}" if p['regression']['pearson_r'] is not None else "—"
            lines.append(
                f"| {p['id']} | {p['start_date']} ~ {p['end_date']} | {p['n_days']} | "
                f"{icon} {_TREND_LABEL_KO[p['trend']]} | {_fmt_int(p['cum_change'])} | "
                f"{p['price_change_pct']:+.1f}% | {r_str} |"
            )
        lines.append("")
        # phase 별 흡수자
        for p in info["phases"]:
            if p["trend"] == "flat" or not p["absorbers"]:
                continue
            role_word = "흡수" if p["trend"] == "down" else "공급"
            top = p["absorbers"][:3]
            top_str = " · ".join(
                f"{a['label']} {_fmt_int(a['delta'])}" for a in top
            )
            lines.append(
                f"- **{p['id']} 주요 {role_word}자** (zero-sum 반대 방향): {top_str}"
            )
        lines.append("")
        # 자동 해석
        interp = _interpret_subject_trend(subject, info)
        if interp:
            lines.append(f"> **자동 해석**: {interp}")
            lines.append("")
    return "\n".join(lines)


def _interpret_subject_trend(subject: str, info: dict) -> str:
    """한 주체의 phase 패턴 + 동행성을 한 줄로 요약."""
    phases = info["phases"]
    conc = info["trend_concordance"]
    if not phases:
        return ""
    label = info["label"]
    parts: list[str] = []
    # 가장 큰 변화 phase
    biggest = max(phases, key=lambda p: abs(p["cum_change"]))
    parts.append(
        f"{label} 추세 phase {len(phases)}개 분할 — 최대 변화는 {biggest['id']} "
        f"({biggest['start_date']}~{biggest['end_date']}, "
        f"누적 {_fmt_int(biggest['cum_change'])}주, 주가 {biggest['price_change_pct']:+.1f}%, "
        f"phase 내 r={biggest['regression']['pearson_r']}). "
    )
    # 동행성
    if conc.get("agreement_pct") is not None:
        agree = conc["agreement_pct"]
        if agree >= 70:
            parts.append(
                f"60일 추세 동행성 {agree}% — 강한 추세 동조. "
                f"**누적 추세가 바뀌면 중기 가격 추세도 같이 바뀌는 신호**."
            )
        elif agree >= 55:
            parts.append(f"60일 추세 동행성 {agree}% — 중간 동조.")
        else:
            parts.append(
                f"60일 추세 동행성 {agree}% — 약함. {label} 추세만으로 가격 추세 예측 어려움."
            )
    # 흡수자 패턴 (가장 큰 phase 기준)
    if biggest["trend"] != "flat" and biggest["absorbers"]:
        top = biggest["absorbers"][0]
        word = "흡수" if biggest["trend"] == "down" else "공급"
        parts.append(
            f"{biggest['id']} 의 주요 {word}자는 **{top['label']}** ({_fmt_int(top['delta'])}주)."
        )
    return " ".join(parts)
