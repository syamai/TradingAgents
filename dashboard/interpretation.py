"""상관/정교한 분석 결과 dict → 자연어 해석 텍스트.

streamlit info/markdown · CLI 마크다운 · 옵시디언 보고서 공용.
규칙 기반(LLM 미사용). 보수적 표현 — 단정짓지 않고 "시사한다·약한 증거"
같은 표현으로 통계적 한계 함께 전달.
"""
from __future__ import annotations

from typing import Optional


# === 상관 분석 6 섹션 =======================================================

def interpret_cumulative(report: dict) -> str:
    rows = [r for r in report.get("cumulative", []) if not r["is_info_total"]]
    if not rows:
        return "해석 불가 — 데이터 없음."
    pos = sorted([r for r in rows if r["cum_qty"] > 0],
                 key=lambda r: r["cum_qty"], reverse=True)[:2]
    neg = sorted([r for r in rows if r["cum_qty"] < 0],
                 key=lambda r: r["cum_qty"])[:2]
    pr = report.get("price") or {}
    direction = (
        f"가격은 {pr.get('total_return_pct', 0):+.1f}% 변동했고, "
        if pr else ""
    )
    buyers = " · ".join(f"{r['label']} {r['cum_qty']:+,} ({r['pct']:.1f}%)"
                        for r in pos) or "없음"
    sellers = " · ".join(f"{r['label']} {r['cum_qty']:+,} ({r['pct']:.1f}%)"
                         for r in neg) or "없음"
    return (
        f"{direction}그 기간 동안 **순매수 상위**는 {buyers}, "
        f"**순매도 상위**는 {sellers}. 비중(%)은 10 sub `|cum_qty|` 합 분모 — "
        f"비중이 크다는 건 *영향력*이 크다는 의미, 매수/매도 방향은 부호 참조."
    )


def interpret_concurrent(report: dict) -> str:
    rows = report.get("concurrent", [])
    if not rows:
        return "해석 불가 — 데이터 없음."
    strong = [r for r in rows if abs(r["r"]) >= 0.30]
    weak = [r for r in rows if abs(r["r"]) < 0.10]
    top_pos = max(rows, key=lambda r: r["r"])
    top_neg = min(rows, key=lambda r: r["r"])

    parts = []
    if top_pos["r"] > 0.10:
        parts.append(f"가장 강한 *동조*는 **{top_pos['label']}** (r={top_pos['r']:+.3f})")
    if top_neg["r"] < -0.10:
        parts.append(f"가장 강한 *역행*은 **{top_neg['label']}** (r={top_neg['r']:+.3f})")
    if not parts:
        return "11 주체 모두 |r|<0.1 — 일별 매매와 그날 수익률의 선형 관계 약함."

    extra = ""
    if len(strong) >= 4:
        extra = " 강한 상관(|r|≥0.3) 주체가 다수 → 주체별 매매가 그날 가격 방향과 잘 정렬되는 종목."
    elif len(weak) >= 8:
        extra = " 대부분 주체가 약한 상관 → 시장 분위기·외생 요인이 강하게 작용."

    return (
        " · ".join(parts) + ". p-value 는 표본 1,200+ 일이라 |r|≈0.06 만 넘어도 "
        "p<0.05 — 통계적 유의보다 *크기* 자체로 판단." + extra
    )


def interpret_up_down(report: dict) -> str:
    rows = report.get("up_down", [])
    if not rows:
        return "해석 불가 — 데이터 없음."
    counter = [r for r in rows if r["pattern"] == "counter_trend"]
    accumulating = [r for r in rows if r["pattern"] == "accumulating_up"]
    biggest = max(rows, key=lambda r: abs(r["diff"]))

    parts = []
    if counter:
        names = " · ".join(r["label"] for r in counter)
        parts.append(f"**역행 매매**(상승일 매도/하락일 매수): {names}")
    if accumulating:
        names = " · ".join(r["label"] for r in accumulating[:5])
        more = "" if len(accumulating) <= 5 else f" 외 {len(accumulating)-5}"
        parts.append(f"**추세 추종**(상승일 매수/하락일 매도): {names}{more}")
    parts.append(
        f"강도 가장 큰 주체는 **{biggest['label']}** "
        f"(상승일 평균 {biggest['up_mean']:+,.0f} vs 하락일 {biggest['down_mean']:+,.0f}, "
        f"차이 {biggest['diff']:+,.0f})"
    )
    return " · ".join(parts) + "."


def interpret_lag(report: dict) -> str:
    rows = report.get("lag", [])
    if not rows:
        return "해석 불가 — 데이터 없음."

    leads: list[str] = []   # t+k 에서 큰 r → 수급이 가격 선도
    chasings: list[str] = []  # t-k 에서 큰 r → 가격이 수급 선도(chasing)
    THRESH = 0.15
    for r in rows:
        for k_str, v in r["lags"].items():
            if k_str == "t0":
                continue
            if abs(v) < THRESH:
                continue
            if k_str.startswith("t+"):
                leads.append(f"{r['label']}({k_str} r={v:+.2f})")
            elif k_str.startswith("t-"):
                chasings.append(f"{r['label']}({k_str} r={v:+.2f})")

    if not leads and not chasings:
        return (
            "모든 lag t±k 에서 |r|<0.15 → **수급이 가격을 1일 이상 선도하지 않고, "
            "가격도 수급을 선도하지 않음**. 동시 r 만 강하다는 건 인트라데이 영향 "
            "또는 즉시 chasing 둘 다 가능 — 1 일+ 단위 단기 신호로는 사용 어려움."
        )
    parts = []
    if leads:
        parts.append(f"**수급 → 가격 선도**: {' · '.join(leads[:3])}")
    if chasings:
        parts.append(f"**가격 → 수급 chasing**: {' · '.join(chasings[:3])}")
    return " · ".join(parts) + " (|r|≥0.15 만 표시)."


def interpret_regime(report: dict) -> str:
    rows = report.get("regime", [])
    if not rows:
        return "해석 불가 — 데이터 없음."

    changes: list[str] = []
    for r in rows:
        w = r["windows"]
        keys = list(w.keys())
        if len(keys) < 2:
            continue
        first_valid = next((w[k] for k in keys[1:] if w[k] is not None), None)
        last_valid = w.get(keys[-1])
        if first_valid is None or last_valid is None:
            continue
        diff = last_valid - first_valid
        if abs(diff) >= 0.10:
            arrow = "↑" if diff > 0 else "↓"
            changes.append(
                f"**{r['label']}** {arrow} "
                f"({first_valid:+.2f} → {last_valid:+.2f}, Δ {diff:+.2f})"
            )
    if not changes:
        return "기간별 동시 상관이 안정적(|Δ|<0.10) — 5년 동안 매매 패턴이 큰 변화 없음."
    return (
        "기간별 상관 변화가 큰 주체(|Δ|≥0.10): "
        + " · ".join(changes)
        + ". 시장 환경 변화나 주체 전략 변화 신호 — Cointegration·Rolling r 결과와 함께 확인."
    )


def interpret_level(report: dict) -> str:
    rows = report.get("level", [])
    if not rows:
        return "해석 불가 — 데이터 없음."
    high = sorted(rows, key=lambda r: abs(r["r"]), reverse=True)[:3]
    parts = " · ".join(f"{r['label']} r={r['r']:+.2f}" for r in high)
    return (
        f"종가 수준과 누적 보유의 상관 — 상위: {parts}. "
        "⚠️ 두 시계열 모두 비정상(추세 보유)이라 *spurious correlation* 위험. "
        "양수면 '누적이 늘수록 가격↑'으로 보이지만, **둘 다 시간에 따라 단조 변화**하면 "
        "관계 없이도 r 이 커진다. 7-4 Cointegration ✓ 일 때만 신뢰 가능."
    )


# === 정교한 분석 6 섹션 (advanced) =========================================

def interpret_adf(adv: dict) -> str:
    rows = adv.get("adf", [])
    if not rows:
        return "해석 불가."
    nonstat = [r for r in rows if r["is_stationary"] is False and r["adf_stat"] is not None]
    stat = [r for r in rows if r["is_stationary"] is True]
    header = f"정상(stationary) 시리즈 {len(stat)} 개 · 비정상 {len(nonstat)} 개"
    # 보통 return 은 정상, close·cum_qty 는 비정상이 기대값
    ret_row = next((r for r in rows if r["series"].startswith("return")), None)
    close_row = next((r for r in rows if r["series"] == "close"), None)
    diag = []
    if ret_row:
        diag.append(
            "return 은 " + ("✓ 정상" if ret_row["is_stationary"] else "✗ 비정상(이례)")
        )
    if close_row:
        diag.append(
            "close 는 " + ("✗ 비정상(예상대로)" if not close_row["is_stationary"] else "✓ 정상(이례)")
        )
    return header + " · " + " · ".join(diag) + (
        ". 비정상 시리즈는 직접 회귀·상관에 spurious 위험 — 차분 후 분석 또는 "
        "Cointegration(7-4) 으로 별도 검증."
    )


def interpret_granger(adv: dict) -> str:
    rows = adv.get("granger", [])
    if not rows:
        return "해석 불가."
    lead: list[str] = []
    chase: list[str] = []
    for r in rows:
        sig_fwd = [x for x in r["net_causes_return"] if x["significant"]]
        sig_bwd = [x for x in r["return_causes_net"] if x["significant"]]
        if sig_fwd:
            lags = ",".join(str(x["lag"]) for x in sig_fwd[:3])
            lead.append(f"{r['label']} (lag {lags})")
        if sig_bwd:
            lags = ",".join(str(x["lag"]) for x in sig_bwd[:3])
            chase.append(f"{r['label']} (lag {lags})")

    parts = []
    if lead:
        parts.append(f"**매매 → 가격 예측** 유의(p<0.05): {' · '.join(lead)}")
    else:
        parts.append("**매매 → 가격 예측** 유의 주체 없음 — 일별 매매로 다음날 가격 예측 어려움")
    if chase:
        parts.append(f"**가격 → 매매 (chasing)** 유의: {' · '.join(chase)}")
    else:
        parts.append("**가격 → 매매 chasing** 유의 없음")
    return " · ".join(parts) + (
        ". Granger 인과는 *선형 예측력*이지 진짜 인과 아님."
    )


def interpret_var_irf(adv: dict) -> str:
    irf = adv.get("var_irf") or {}
    if not irf or irf.get("order") is None:
        return "VAR fit 실패 — 해석 불가."
    cum = irf.get("irf_cum") or {}
    # 각 주체 → return 의 t=10 누적 효과 비교
    effects = []
    for key, vals in cum.items():
        if not key.endswith("->return") or key == "return->return":
            continue
        if not vals:
            continue
        last = vals[-1]  # horizon 끝 (보통 t=10)
        subject = key.replace("_net->return", "")
        effects.append((subject, last))
    if not effects:
        return f"VAR 차수 {irf['order']} 적합, 그러나 IRF 추출 실패."
    effects.sort(key=lambda x: abs(x[1]), reverse=True)
    top = effects[:3]
    desc = " · ".join(f"`{s}` {v:+.4g}" for s, v in top)
    biggest_label = top[0][0]
    biggest_val = top[0][1]
    direction = "끌어올리는" if biggest_val > 0 else "끌어내리는"
    return (
        f"VAR 차수(AIC) = **{irf['order']}**, horizon {irf['horizon']}일. "
        f"각 주체 매매 1-σ 충격의 누적 IRF (return 단위, 부호=방향): {desc}. "
        f"가장 큰 영향은 **{biggest_label}** — 가격을 {direction} 방향. "
        "값의 *절대 크기*는 변수 단위 차이(return %, net_qty 주)로 직접 비교 의미 작음 — "
        "**부호와 지속성** 위주 해석 권장. Cholesky orthogonalized IRF는 변수 순서"
        "(return 을 가장 외생적으로 두는 순서)에 의존."
    )


def interpret_cointegration(adv: dict) -> str:
    rows = adv.get("cointegration", [])
    if not rows:
        return "해석 불가."
    ok = [r for r in rows if r["is_cointegrated"]]
    nope = [r for r in rows if r["is_cointegrated"] is False and r["score"] is not None]
    if ok:
        names = " · ".join(f"{r['label']} (p={r['p_value']:.3f})" for r in ok)
        rest = ""
        if nope:
            rest = (
                f" 나머지 {len(nope)} 주체는 비공적분 — 누적과 가격의 동조는 "
                "공통 추세 효과(spurious) 가능."
            )
        return (
            f"**공적분 ✓ 주체**: {names}. 이 주체는 `cum_qty` 와 `close` 가 "
            "장기적으로 함께 움직임 → level r 신뢰 가능, 누적이 가격의 균형 "
            f"수준 결정에 기여한다는 약한 증거.{rest}"
        )
    return (
        "공적분 ✓ 주체 없음 — 모든 핵심 주체의 누적과 가격은 *추세 동조성*만 보이고 "
        "장기 균형 관계는 검정에서 기각되지 않음. **6. Level correlation 의 큰 r 은 "
        "spurious 가능성 우선 의심**."
    )


def interpret_mi(adv: dict) -> str:
    rows = adv.get("mutual_info", [])
    if not rows:
        return "해석 불가."
    top = rows[0]  # 이미 큰 순 정렬
    if top["mi"] < 0.02:
        return (
            "모든 주체 MI<0.02 — 일별 매매와 수익률의 비선형 의존성도 약함. "
            "Pearson r 결과가 사실상 정보의 전부."
        )
    parts = " · ".join(f"{r['label']} MI={r['mi']:.3f}" for r in rows[:3])
    return (
        f"비선형 포함 정보량 상위: {parts}. "
        f"**{top['label']}** 의 MI 가 가장 크며, 같은 주체의 Pearson |r| 도 비교해 "
        "(r 작은데 MI 크면 비선형 신호) 점검. MI 는 nat 단위 — 한 종목 내 상대 비교용."
    )


def interpret_rolling(adv: dict) -> str:
    rows = adv.get("rolling_r", [])
    if not rows:
        return "해석 불가."
    valid = [r for r in rows if r["mean"] is not None]
    if not valid:
        return "rolling window 채울 데이터 부족."
    most_stable = min(valid, key=lambda r: r["std"])
    most_volatile = max(valid, key=lambda r: r["std"])
    swings = [r for r in valid if (r["max"] - r["min"]) > 0.6]
    parts = [
        f"가장 **안정적** 상관: **{most_stable['label']}** "
        f"(mean {most_stable['mean']:+.2f}, std {most_stable['std']:.2f})",
        f"가장 **변동 큰** 상관: **{most_volatile['label']}** "
        f"(std {most_volatile['std']:.2f}, [{most_volatile['min']:+.2f}, {most_volatile['max']:+.2f}])",
    ]
    if swings:
        names = ", ".join(r["label"] for r in swings)
        parts.append(
            f"60일 윈도우 안에서 부호까지 뒤집힌 주체: {names} — "
            "단기 regime 변화 큼"
        )
    return " · ".join(parts) + "."
