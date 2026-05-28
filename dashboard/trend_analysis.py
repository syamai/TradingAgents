"""누적 보유 추세 분석 — 일별 Granger·CCF가 못 잡는 중기 시그널.

배경:
  advanced_analysis 의 Granger·CCF·MI 는 모두 *일별 차분* 기반이라
  단기 노이즈 안 선형 예측만 검정한다. 외국인 누적 보유가 1년에 걸쳐
  단조 증가 → 1년에 걸쳐 단조 감소 같은 *추세 전환*은 일별 단위에서
  보이지 않는다 (035900 JYP Ent. 사례: 외국인 cum 2022-08~2023-07 매집
  → 2023-07~2026-01 분산, 주가 +129% → -57.5%).

분석:
  1. ``detect_phases`` — ``window``일 rolling slope 부호 변화로 누적 시리즈를
     trend phase 로 분할 (up/down/flat). ``window`` 는 ``adaptive_window`` 가
     종목별 변동성으로 자동 산정 (기본 ``BASE_WINDOW=60``일, 범위 30~120일).
  2. ``phase_regression`` — 각 phase 안에서 ``close ~ cum_qty`` OLS.
     r·slope·R² 로 phase 내 동행성 정량화.
  3. ``compute_absorbers`` — phase 안 zero-sum 흡수자 분석: 주체 A 가
     N주 매도 시 누가 받았는지 (다른 9 주체의 net 변화).
  4. ``trend_concordance`` — ``window``일 추세 기울기 부호 일치율: 주체 누적의
     ``window``일 기울기 부호와 종가 ``window``일 기울기 부호가 같은 일자 비율.

모든 함수 JSON 직렬화 가능 dict/list 반환. statsmodels 의존 없음 — numpy 만.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from dashboard.holdings_chart import SUBJECT_LABELS
from tradingagents.dataflows.kis_holdings import SUBS_10


# 9 주체 모두 분석 후 동행성 기준 상위 N 선별 — 미리 주체를 정하지 않음.
# 외국인 등록/비등록은 통합(foreign)으로 합산, 그 외 7개 sub + 통합 = 9 주체.
ALL_TREND_SUBJECTS: tuple[str, ...] = (
    "foreign",
    "pension", "private_equity", "investment_trust",
    "securities", "bank", "insurance",
    "retail", "other_corp",
)

# 동행성 |agreement_pct - 50| 가 큰 상위 N 주체만 phase 상세 분석 표시
TOP_N_TREND_SUBJECTS = 5

# Phase detection 파라미터 — 적응 윈도우 baseline
BASE_WINDOW = 60               # adaptive window 기준값
MIN_WINDOW = 30                # 최소 윈도우 (변동성 큰 종목)
MAX_WINDOW = 120               # 최대 윈도우 (변동성 작은 종목)
REF_PRICE_SIGMA = 0.02         # 기준 일별 수익률 σ (2%) — score=1 의 기준
REF_HOLDINGS_CV = 1.5          # 기준 외국인 net_qty 변동계수 — score=1 의 기준
FLAT_THRESHOLD = 0.05          # |Δ| < (전체 cum 절대값 max) × 0.05 이면 flat


def adaptive_window(
    df: pd.DataFrame,
    base: int = BASE_WINDOW,
    min_w: int = MIN_WINDOW,
    max_w: int = MAX_WINDOW,
) -> int:
    """가격 변동성 + 외국인 보유 변동성으로 적응 윈도우 산정.

    - 가격 σ: 일별 수익률 표준편차. 변동성 큰 종목은 짧은 윈도우 (추세 빨리 변함).
    - 보유 σ: 외국인 통합 net_qty 변동계수 (std / mean(|net|)).
      매매 빈도/강도가 큰 종목은 짧은 윈도우.

    두 지표를 reference 값 (``REF_PRICE_SIGMA``, ``REF_HOLDINGS_CV``) 으로
    정규화해 평균 → score. score=1 이면 ``base`` 일, score 클수록 짧음.

    반환은 ``[min_w, max_w]`` 로 clamp.
    """
    if "close" not in df.columns or len(df) < 30:
        return base

    price_sigma = df["close"].pct_change().std()
    if price_sigma is None or pd.isna(price_sigma) or price_sigma <= 0:
        price_sigma = REF_PRICE_SIGMA

    fn = df.get("foreign_registered_net_qty", pd.Series(dtype=float)).fillna(0) + \
         df.get("foreign_unregistered_net_qty", pd.Series(dtype=float)).fillna(0)
    abs_mean = float(fn.abs().mean()) if len(fn) > 0 else 0.0
    if abs_mean > 0:
        holdings_cv = float(fn.std()) / abs_mean
    else:
        holdings_cv = REF_HOLDINGS_CV

    # 두 지표 정규화 평균 → score
    score = (price_sigma / REF_PRICE_SIGMA + holdings_cv / REF_HOLDINGS_CV) / 2
    # 극단값 clamp 후 base 에 반비례
    score = max(0.5, min(4.0, score))
    window = int(round(base / score))
    return max(min_w, min(max_w, window))


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
    window: int = BASE_WINDOW,
    min_days: int = BASE_WINDOW,
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


def phase_avg_price(
    df: pd.DataFrame, start: int, end: int, subject: str, side: str,
) -> Optional[float]:
    """Phase 내 평균 거래가격 — net_qty 가중 close 평균.

    side="buy": net_qty>0 일자만 사용 (매집 평균 매수가)
    side="sell": net_qty<0 일자만 사용 (분산 평균 매도가)
    side="all": 모든 일자 |net_qty| 가중 (보합 등 — 매수+매도 통합 평균가)
    """
    net_col = f"{subject}_net_qty"
    if net_col not in df.columns or "close" not in df.columns:
        return None
    seg = df.iloc[start:end]
    net = seg[net_col].values.astype(float)
    close = seg["close"].values.astype(float)
    if side == "all":
        weight = np.abs(net)
        cl = close
    elif side == "buy":
        mask = net > 0
        weight = net[mask]
        cl = close[mask]
    else:
        mask = net < 0
        weight = -net[mask]
        cl = close[mask]
    total = float(weight.sum())
    if total <= 0:
        return None
    return float((cl * weight).sum() / total)


def phase_net_value(
    df: pd.DataFrame, start: int, end: int, subject: str,
) -> Optional[float]:
    """Phase 내 순매수 거래대금 — Σ(close × net_qty).

    net_qty 가 매수일 양수/매도일 음수이므로 합치면 매수 + 매도 − 자동 처리.
    매집 phase: 양수 (순매수), 분산 phase: 음수 (순매도), 보합: ≈0.
    """
    net_col = f"{subject}_net_qty"
    if net_col not in df.columns or "close" not in df.columns:
        return None
    seg = df.iloc[start:end]
    net = seg[net_col].values.astype(float)
    close = seg["close"].values.astype(float)
    return float((close * net).sum())


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
    df: pd.DataFrame, cum_col: str, window: int = BASE_WINDOW,
) -> dict:
    """주체 누적 ``window``일 기울기 부호 ↔ 종가 ``window``일 기울기 부호 일치율.

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
    subjects: tuple[str, ...] = ALL_TREND_SUBJECTS,
    top_n: int = TOP_N_TREND_SUBJECTS,
    window: Optional[int] = None,
    min_days: Optional[int] = None,
) -> dict:
    """추세 분석 통합 보고서.

    9 주체 모두 분석 후 동행성 ``|agreement_pct - 50|`` 기준으로 상위
    ``top_n`` 주체만 phase 상세 표시. 9 주체 전체 동행성 랭킹은
    ``concordance_ranking`` 키에 별도 보관.

    선별 기준 ``|agreement - 50|`` 은 동행(>50%)·역행(<50%) 둘 다 큰 절댓값을
    가질수록 "추세 연관성이 명확"한 신호. 50% 근처는 가격 추세와 무관.

    ``window`` / ``min_days`` 가 ``None`` 이면 ``adaptive_window`` 로
    종목별 자동 계산 — 가격 σ + 외국인 net_qty 변동계수 결합.

    return:
      ``{generated_at, window, min_phase_days, top_n, window_adaptive,
      adaptive_inputs, subjects, concordance_ranking}``
    """
    df = _ensure_foreign_combined(df).reset_index(drop=True)
    # 상장 전(close=0) 행 제거 + inf 정리 — 적응 윈도우 계산 + phase 분석 정상화
    if "close" in df.columns:
        df = df[df["close"] > 0].reset_index(drop=True)
        df = df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    # 적응 윈도우 — 명시 안 되면 종목별 자동
    adaptive_used = window is None
    if window is None:
        window = adaptive_window(df)
    if min_days is None:
        min_days = window
    # 적응 산정에 쓰인 입력 (보고용)
    price_sigma = df["close"].pct_change().std() if "close" in df.columns else None
    fn = df.get("foreign_registered_net_qty", pd.Series(dtype=float)).fillna(0) + \
         df.get("foreign_unregistered_net_qty", pd.Series(dtype=float)).fillna(0)
    abs_mean = float(fn.abs().mean()) if len(fn) > 0 else 0.0
    holdings_cv = float(fn.std()) / abs_mean if abs_mean > 0 else None

    out: dict = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slope_window_days": window,
        "min_phase_days": min_days,
        "n_days": len(df),
        "top_n": top_n,
        "window_adaptive": adaptive_used,
        "adaptive_inputs": {
            "price_sigma": round(float(price_sigma), 5) if price_sigma else None,
            "foreign_net_cv": round(holdings_cv, 3) if holdings_cv else None,
            "ref_price_sigma": REF_PRICE_SIGMA,
            "ref_holdings_cv": REF_HOLDINGS_CV,
        },
        "subjects": {},
        "concordance_ranking": [],
    }

    all_info: dict[str, dict] = {}
    for subject in subjects:
        cum_col = f"{subject}_cum_qty"
        if cum_col not in df.columns:
            continue
        phases_raw = detect_phases(df[cum_col], window=window, min_days=min_days)
        phases_out = []
        cum_nv = 0.0  # subject 내 phase 순서대로 순매수 거래대금 누적
        for i, (s, e, trend) in enumerate(phases_raw, start=1):
            reg = phase_regression(df, s, e, cum_col)
            absorbers = compute_absorbers(df, s, e, subject) if trend != "flat" else []
            side_for_price = {"up": "buy", "down": "sell", "flat": "all"}[trend]
            avg_price = phase_avg_price(df, s, e, subject, side_for_price)
            net_value = phase_net_value(df, s, e, subject)
            cum_nv += net_value if net_value is not None else 0.0
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
                "avg_price": round(avg_price, 1) if avg_price is not None else None,
                "net_value": int(net_value) if net_value is not None else None,
                "cum_net_value": int(cum_nv),
                "regression": reg,
                "absorbers": absorbers[:5],
            })
        all_info[subject] = {
            "label": SUBJECT_LABELS.get(subject, subject),
            "phases": phases_out,
            "trend_concordance": trend_concordance(df, cum_col, window=window),
        }

    # 동행성 |agreement - 50| 큰 순 정렬
    def _distance(info: dict) -> float:
        agree = info["trend_concordance"].get("agreement_pct")
        return abs((agree if agree is not None else 50.0) - 50.0)

    ranked = sorted(all_info.items(), key=lambda kv: _distance(kv[1]), reverse=True)

    # 전체 동행성 랭킹 (9 주체)
    out["concordance_ranking"] = [
        {
            "subject": s,
            "label": info["label"],
            "agreement_pct": info["trend_concordance"].get("agreement_pct"),
            "distance_from_50": round(_distance(info), 1),
        }
        for s, info in ranked
    ]

    # phase 상세는 상위 N 만
    out["subjects"] = {s: info for s, info in ranked[:top_n]}
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
    w = report["slope_window_days"]
    adaptive_note = ""
    if report.get("window_adaptive"):
        inp = report.get("adaptive_inputs", {})
        ps = inp.get("price_sigma")
        cv = inp.get("foreign_net_cv")
        adaptive_note = (
            f" (**적응 윈도우** — 가격 σ={ps*100:.2f}%/일, 외국인 net_qty CV="
            f"{cv:.2f} 기반 자동 산정)"
            if ps is not None and cv is not None else " (적응 윈도우)"
        )
    lines.append(
        f"9 주체 모두 분석 후 추세 동행성 ``|일치율 − 50%|`` 기준으로 상위 "
        f"{report.get('top_n', TOP_N_TREND_SUBJECTS)} 주체만 phase 상세 표시. "
        f"50% 근처는 가격 추세와 무관, 70%↑ 동행 / 30%↓ 역행이 명확한 신호. "
        f"분석 윈도우 **{w}일**{adaptive_note}, 최소 phase 길이 "
        f"{report['min_phase_days']}일."
    )
    lines.append("")
    # 9 주체 동행성 랭킹 표
    if report.get("concordance_ranking"):
        lines.append(f"### 8-0. 9 주체 {w}일 추세 동행성 랭킹")
        lines.append("")
        lines.append("| 순위 | 주체 | 일치율 | \\|일치율−50\\| | 신호 |")
        lines.append("|---:|---|---:|---:|---|")
        for rank, r in enumerate(report["concordance_ranking"], start=1):
            agree = r["agreement_pct"]
            agree_str = f"{agree:.1f}%" if agree is not None else "—"
            if agree is None:
                signal = "—"
            elif agree >= 70:
                signal = "📈 강한 동행"
            elif agree >= 60:
                signal = "↗ 약한 동행"
            elif agree <= 30:
                signal = "📉 강한 역행"
            elif agree <= 40:
                signal = "↘ 약한 역행"
            else:
                signal = "・ 무관"
            star = " ⭐" if rank <= report.get("top_n", TOP_N_TREND_SUBJECTS) else ""
            lines.append(
                f"| {rank}{star} | {r['label']} | {agree_str} | "
                f"{r['distance_from_50']} | {signal} |"
            )
        lines.append("")
        lines.append(
            f"⭐ 상위 {report.get('top_n', TOP_N_TREND_SUBJECTS)} 주체에 대해 "
            f"아래 phase 상세·흡수자 분석."
        )
        lines.append("")
    for subject, info in report["subjects"].items():
        lines.append(f"### 8-{subject} — {info['label']}")
        lines.append("")
        conc = info["trend_concordance"]
        if conc.get("agreement_pct") is not None:
            cw = conc.get("window", w)
            lines.append(
                f"**{cw}일 추세 동행성**: {info['label']} 누적의 {cw}일 기울기 부호와 "
                f"종가 {cw}일 기울기 부호가 **{conc['agreement_pct']}%** 일치 "
                f"(n={conc['n']}일)"
            )
            lines.append("")
        if not info["phases"]:
            lines.append("_phase 분할 불가 (시리즈 짧음)_")
            lines.append("")
            continue
        # phase 요약 표 — 평균 거래가격(trend 별 가중치: 매집=매수가/분산=매도가/보합=|net|가중),
        # 순매수 거래대금(Σ close × net_qty, 매수+/매도−), 누적금(subject 내 phase 순서대로 누적)
        lines.append(
            "| Phase | 기간 | 일수 | 추세 | 누적 Δ | 평균 거래가격 | 순매수 거래대금 | 누적금 | 주가 Δ | phase 내 r |"
        )
        lines.append("|---|---|---:|:--:|---:|---:|---:|---:|---:|---:|")
        for p in info["phases"]:
            icon = _TREND_ICON.get(p["trend"], "")
            r_str = f"{p['regression']['pearson_r']:+.3f}" if p['regression']['pearson_r'] is not None else "—"
            avg = p.get("avg_price")
            avg_str = f"{avg:,.0f}원" if avg is not None else "—"
            nv = p.get("net_value")
            nv_str = f"{nv/1e8:+,.0f}억" if nv is not None else "—"
            cnv = p.get("cum_net_value")
            cnv_str = f"{cnv/1e8:+,.0f}억" if cnv is not None else "—"
            lines.append(
                f"| {p['id']} | {p['start_date']} ~ {p['end_date']} | {p['n_days']} | "
                f"{icon} {_TREND_LABEL_KO[p['trend']]} | {_fmt_int(p['cum_change'])} | "
                f"{avg_str} | {nv_str} | {cnv_str} | {p['price_change_pct']:+.1f}% | {r_str} |"
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
        cw = conc.get("window", 60)
        if agree >= 70:
            parts.append(
                f"{cw}일 추세 동행성 {agree}% — 강한 추세 동조. "
                f"**누적 추세가 바뀌면 중기 가격 추세도 같이 바뀌는 신호**."
            )
        elif agree >= 55:
            parts.append(f"{cw}일 추세 동행성 {agree}% — 중간 동조.")
        else:
            parts.append(
                f"{cw}일 추세 동행성 {agree}% — 약함. {label} 추세만으로 가격 추세 예측 어려움."
            )
    # 흡수자 패턴 (가장 큰 phase 기준)
    if biggest["trend"] != "flat" and biggest["absorbers"]:
        top = biggest["absorbers"][0]
        word = "흡수" if biggest["trend"] == "down" else "공급"
        parts.append(
            f"{biggest['id']} 의 주요 {word}자는 **{top['label']}** ({_fmt_int(top['delta'])}주)."
        )
    return " ".join(parts)
