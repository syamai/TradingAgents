"""정교한 시계열 분석 — Pearson r 의 한계를 넘는 6 종 기법.

기본 가이드:
  1. ADF (Augmented Dickey-Fuller) — 정상성 검정. cum_qty·close 는 보통
     비정상이라 level r 의 spurious 위험을 정량화.
  2. Granger causality — X 가 Y 의 미래값 *예측*에 통계적으로 기여하는가.
     양방향(net_qty → return / return → net_qty)으로 chasing 분리.
  3. VAR + Impulse Response — 다변량 lag 회귀로 충격이 누적적으로 미치는
     효과 추적.
  4. Engle-Granger Cointegration — cum_qty 와 close 의 *장기 균형*.
  5. Mutual Information — 비선형 의존성(Pearson 은 선형만).
  6. Rolling correlation — DCC-GARCH 의 단순판. 시간에 따라 변하는 r.

각 함수는 JSON 직렬화 가능한 dict/list 반환. statsmodels + sklearn 의존.
"""
from __future__ import annotations

import contextlib
import io
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from dashboard.holdings_chart import SUBJECT_LABELS
from tradingagents.dataflows.kis_holdings import SUBS_10

# 무거운 분석은 핵심 주체로 제한 — 11 주체 전체는 결과 표가 너무 커진다.
KEY_SUBJECTS: tuple[str, ...] = (
    "foreign", "pension", "securities", "private_equity", "retail",
)

VAR_HORIZON = 10           # IRF 누적 효과 추적 일수
ROLLING_WINDOW = 60        # rolling Pearson 윈도우 (≈ 3 개월 거래일)
GRANGER_MAX_LAG = 5        # Granger F-test 최대 lag


def _clean_series(s) -> pd.Series:
    """NaN + ±inf 제거. statsmodels 함수 입력 전 필수.

    ``pct_change()`` 로 만든 returns 가 종가 0/매우 작은 값 다음 큰 값 케이스
    에서 ±inf 를 생성 — dropna() 만으로는 안 잡힘.
    """
    return (
        pd.Series(s)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )


# === 1. ADF 정상성 검정 =====================================================

def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller. H0: 비정상(unit root).

    return: ``{"adf_stat", "p_value", "is_stationary", "n_obs"}``.
    값이 짧거나 상수면 ``is_stationary=False``.
    """
    s = _clean_series(series)
    if len(s) < 20 or s.std() == 0:
        return {"adf_stat": None, "p_value": None,
                "is_stationary": False, "n_obs": len(s)}
    try:
        # autolag='AIC' — 보통 디폴트. regression='c' (상수항).
        stat, p, *_ = adfuller(s.values, autolag="AIC", regression="c")
    except (ValueError, np.linalg.LinAlgError):
        return {"adf_stat": None, "p_value": None,
                "is_stationary": False, "n_obs": len(s)}
    return {
        "adf_stat": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "is_stationary": bool(p < 0.05),
        "n_obs": int(len(s)),
    }


# === 2. Granger Causality ===================================================

def granger_test(y: pd.Series, x: pd.Series, max_lag: int = GRANGER_MAX_LAG) -> list[dict]:
    """X 가 Y 의 lag 1..max_lag 미래값 예측에 기여? (선형, F-test)

    H0: X 는 Y 를 Granger-cause 하지 않음. p<0.05 면 기각 → 예측 기여.

    statsmodels 의 grangercausalitytests 는 ``DataFrame([Y, X])`` 를 받아
    "X causes Y" 를 테스트. 반환: lag별 dict.
    """
    y = _clean_series(y)
    x = _clean_series(x)
    pair = (
        pd.concat([y, x], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(pair) < max_lag + 10 or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
        return []
    try:
        # statsmodels 가 verbose 무시하고 stdout 에 결과 표를 출력 — suppress.
        with contextlib.redirect_stdout(io.StringIO()):
            res = grangercausalitytests(pair.values, maxlag=max_lag)
    except (ValueError, np.linalg.LinAlgError, KeyError):
        return []
    out: list[dict] = []
    for lag, payload in res.items():
        f_stat, p_val, _df_num, _df_den = payload[0]["ssr_ftest"]
        out.append({
            "lag": int(lag),
            "f_stat": round(float(f_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
        })
    return out


# === 3. VAR + Impulse Response ===============================================

def var_irf(
    df: pd.DataFrame, cols: list[str], horizon: int = VAR_HORIZON,
    max_lag: int = 8,
) -> dict:
    """VAR fit (AIC 로 차수 선택) → 직교화(orthogonalized) 누적 IRF.

    한 변수의 1-σ *구조*충격(Cholesky 직교화)이 다른 변수에 ``horizon`` 일 동안
    누적적으로 미치는 효과. 충격이 1σ 로 표준화돼 변수 단위(예: net_qty 주 vs
    return %)가 달라도 비교 가능 — 단 응답값 자체는 응답변수 단위(여기선 return %).

    key ``"{src}->{tgt}"`` = **src 1σ 충격 → tgt 누적 반응**. Cholesky 직교화는
    ``cols`` 순서에 의존: 앞쪽 변수일수록 외생적이라, 뒤 변수의 충격에 동시(t=0)
    반응이 0 으로 식별된다(예: cols=["return", ...] 면 net→return 의 t0=0, 지연
    효과만 잡힘 = Granger 와 정합한 *선행* 측정).

    return:
      ``{"order": p, "cols": cols, "irf_cum": {f"{src}->{tgt}": [v_0..v_horizon]}}``
    """
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < (max_lag + 5) * len(cols) or any(sub[c].std() == 0 for c in cols):
        return {"order": None, "cols": cols, "irf_cum": {}}

    # VAR 은 정상성 가정 → 변동률·net_qty 같은 stationary 시리즈를 받는 게 안전.
    # 호출자가 적절히 차분 처리.
    try:
        model = VAR(sub.values)
        order = model.select_order(maxlags=max_lag).aic
        order = max(1, int(order or 1))
        results = model.fit(order)
        irf = results.irf(horizon)
    except (ValueError, np.linalg.LinAlgError):
        return {"order": None, "cols": cols, "irf_cum": {}}

    # 직교화 누적 IRF. statsmodels 규약: cum[step, response, impulse].
    # key "src->tgt"(src 충격→tgt 반응) = response=tgt(j), impulse=src(i) → cum[:, j, i].
    # (구버그: irf.cum_effects[:, i, j] = 비직교·단위충격 + 방향 반전 → 단위 혼재로 무의미)
    cum = irf.orth_cum_effects  # shape (horizon+1, k, k)
    out_irf: dict = {}
    for i, src in enumerate(cols):
        for j, tgt in enumerate(cols):
            key = f"{src}->{tgt}"
            out_irf[key] = [round(float(v), 5) for v in cum[:, j, i]]
    return {"order": int(order), "cols": cols, "irf_cum": out_irf,
            "horizon": int(horizon)}


# === 3b. 공선성 교정 선행 예측 검정 =========================================

def _bh_survivors(pvals: list[float], alpha: float = 0.05) -> list[int]:
    """Benjamini-Hochberg(FDR). 생존 p-value 의 인덱스 리스트 반환."""
    pv = np.asarray(pvals, dtype=float)
    order = np.argsort(pv)
    m = len(pv)
    keep: set = set()
    for i, idx in enumerate(order):
        if pv[idx] <= alpha * (i + 1) / m:
            keep = set(order[: i + 1].tolist())
    return sorted(keep)


def forward_lead_test(
    df: pd.DataFrame,
    subjects: tuple[str, ...] = KEY_SUBJECTS,
    *,
    numeraire: str = "retail",
    horizon: int = 1,
    hac_lags: int = 5,
    alpha: float = 0.05,
) -> dict:
    """제로섬 공선성을 교정한 선행 예측 검정 — naive 다변량 통제의 대체.

    투자자유형 net_qty 는 시장청산상 합≈0 이라 전부 동시 투입하면 다중공선성
    (개인·외국인 VIF 가 수십대로 폭증)으로 SE 가 팽창해, 진짜 선행 신호도
    비유의로 죽는다('신호 없음'과 '식별 불가'를 구분 못함). ``numeraire``
    (기본 '개인' retail)를 제외해 VIF 를 낮춘 뒤
    ``ret(t+horizon) ~ Σ_s net_s(t)[z]`` 를 HAC(Newey-West) 로 적합하고,
    각 주체의 VIF·계수·BH-FDR 생존을 반환한다.

    df 는 holdings(``close``,``volume``,``price_change_pct``,``{s}_net_qty``).
    ``close>0 & volume>0 & price_change_pct`` 유한 행만 사용.

    return: ``{"n", "numeraire", "horizon", "vif": {s: v}, "rows":
              [{subject, b, t, p}], "fdr_survivors": [s...]}``
    """
    cols = [s for s in subjects if s != numeraire]
    out: dict = {"n": 0, "numeraire": numeraire, "horizon": int(horizon),
                 "vif": {}, "rows": [], "fdr_survivors": []}
    if df.empty or not cols:
        return out
    d = df.sort_values("date").reset_index(drop=True)
    d = d[(d["close"] > 0) & (d["volume"] > 0)].replace(
        [np.inf, -np.inf], np.nan).dropna(
        subset=["price_change_pct"]).reset_index(drop=True)
    need = [f"{s}_net_qty" for s in cols]
    if len(d) < 50 or any(c not in d.columns for c in need):
        return out
    ret = d["price_change_pct"].astype(float)
    Z: dict = {}
    for s in cols:
        v = d[f"{s}_net_qty"].astype(float).to_numpy()
        sd = v.std()
        if sd == 0:
            return out
        Z[s] = (v - v.mean()) / sd
    X = pd.DataFrame(Z)
    y = ret.shift(-horizon)
    m = pd.concat([y.rename("_y"), X], axis=1).dropna()
    if len(m) < 50:
        return out
    Xc = sm.add_constant(m[cols])
    try:
        res = sm.OLS(m["_y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        vifs = {cols[i]: float(variance_inflation_factor(Xc.values, i + 1))
                for i in range(len(cols))}
    except (ValueError, np.linalg.LinAlgError):
        return out
    pvals = [float(res.pvalues[s]) for s in cols]
    keep = _bh_survivors(pvals, alpha)
    out["n"] = int(len(m))
    out["vif"] = {s: round(v, 2) for s, v in vifs.items()}
    out["rows"] = [{"subject": s, "b": round(float(res.params[s]), 4),
                    "t": round(float(res.tvalues[s]), 3),
                    "p": round(float(res.pvalues[s]), 5)} for s in cols]
    out["fdr_survivors"] = [cols[i] for i in keep]
    return out


# === 4. Cointegration (Engle-Granger) =======================================

def cointegration_test(y: pd.Series, x: pd.Series) -> dict:
    """H0: 두 시계열은 공적분되지 않음. p<0.05 면 장기 균형 존재.

    cum_qty 와 close 같은 비정상 시계열이 *함께 움직이는지* 정식 검정.
    공적분이면 spurious regression 위험이 낮아지고 level r 도 해석 가능.
    """
    y = _clean_series(y)
    x = _clean_series(x)
    n = min(len(y), len(x))
    if n < 30 or y.iloc[:n].std() == 0 or x.iloc[:n].std() == 0:
        return {"score": None, "p_value": None, "is_cointegrated": False, "n": n}
    try:
        score, p, _crit = coint(y.iloc[:n].values, x.iloc[:n].values)
    except (ValueError, np.linalg.LinAlgError):
        return {"score": None, "p_value": None, "is_cointegrated": False, "n": n}
    return {
        "score": round(float(score), 4),
        "p_value": round(float(p), 6),
        "is_cointegrated": bool(p < 0.05),
        "n": int(n),
    }


# === 5. Mutual Information ==================================================

def mutual_info(x: pd.Series, y: pd.Series, random_state: int = 0) -> float:
    """X 와 Y 사이의 상호정보량 (KSG 기반, sklearn).

    Pearson r 이 잡지 못하는 비선형 의존성. 0 = 독립, 클수록 의존.
    스케일은 데이터에 의존 — 한 종목 내 상대 비교용.
    """
    x = _clean_series(x)
    y = _clean_series(y)
    pair = (
        pd.concat([x, y], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(pair) < 20 or pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
        return 0.0
    X = pair.iloc[:, 0].values.reshape(-1, 1)
    Y = pair.iloc[:, 1].values
    mi = mutual_info_regression(X, Y, random_state=random_state)[0]
    return round(float(mi), 6)


# === 6. Rolling correlation (DCC 대체) =======================================

def rolling_correlation(
    x: pd.Series, y: pd.Series, window: int = ROLLING_WINDOW,
) -> pd.Series:
    """rolling Pearson r — 시간에 따른 상관 변화."""
    return pd.Series(x).rolling(window).corr(pd.Series(y))


def rolling_correlation_summary(
    x: pd.Series, y: pd.Series, window: int = ROLLING_WINDOW,
) -> dict:
    """rolling r 시리즈의 요약(평균/표준편차/min/max) + 50/90 백분위."""
    r = rolling_correlation(x, y, window).dropna()
    if r.empty:
        return {"mean": None, "std": None, "min": None, "max": None,
                "window": window, "n": 0}
    return {
        "mean": round(float(r.mean()), 4),
        "std": round(float(r.std()), 4),
        "min": round(float(r.min()), 4),
        "max": round(float(r.max()), 4),
        "p10": round(float(r.quantile(0.10)), 4),
        "p90": round(float(r.quantile(0.90)), 4),
        "window": int(window),
        "n": int(len(r)),
    }


# === 7. 주체별 손익 근사 (구간 mark-to-market) ==============================

PNL_AMOUNT_COLS: tuple[str, ...] = tuple(f"{s}_amount" for s in KEY_SUBJECTS)


def pnl_attribution(df: pd.DataFrame) -> list[dict]:
    """주체별 구간 손익 근사 — 순매수/순매도 분리 평단 + mark-to-market.

    표시 구간 [start, end] 의 일별 net 흐름을 *순매수일*(net_qty>0)과
    *순매도일*(net_qty<0)로 나눠 각 방향의 거래량가중 평단을 따로 낸다::

        순매수 평단 = Σ(순매수일 net_amt) / Σ(순매수일 net_qty)   (매집 평균가)
        순매도 평단 = Σ(순매도일 net_amt) / Σ(순매도일 net_qty)   (분산 평균가)

    각 버킷 안에서는 수량·대금 부호가 일관(순매수일=둘다+, 순매도일=둘다−)이라
    두 평단 모두 **양수**로 나오고, 단일 net 평단의 음수/가격범위 이탈 왜곡이
    사라진다. 두 평단 차이(매도−매수)가 왕복 거래의 수익성 방향을 보여준다.

    손익은 실현/미실현으로 분해한다. 왕복으로 맞물린 수량은 실현, 남은 순포지션은
    현재가 평가(미실현)::

        net_qty = 순매수일 + 순매도일 수량 합 (구간 순포지션 변화)
        net_amt = 순매수일 + 순매도일 대금 합 (구간 순투입 현금, 매수 +)
        matched = min(순매수량, 순매도량)              (왕복 청산 수량)
        실현손익  = matched × (순매도평단 − 순매수평단)
        미실현손익 = net_qty × (현재가 − 기준단가)
                    순매수 잔여(net_qty>0)→매수평단 기준, **평가손익**(실제 보유)
                    순매도 초과(net_qty<0)→매도평단 기준, **기회손익**(보유 아님,
                      현재가 대비 매도 타이밍 평가. 현재가 따라 변하는 MTM)
        pnl = 실현 + 미실현 = last_close × net_qty − net_amt  (항등식)

    ``unrealized_kind`` 로 두 경우를 구분("평가"/"기회"/None) — 호출자가 라벨 분기.

    ``{s}_amount`` 는 ``close`` 와 같은 단위(원)로 들어온다고 가정(호출자가 KIS
    백만원 단위를 ×1e6 환산 — `dashboard.app._load_holdings_with_amounts`).

    한계: net 데이터라 일중 회전은 안 보이고, 구간 시작 이전 포지션을
    베이스라인 0 으로 둔 *구간 상대* 손익 — 절대 손익이 아니다.

    ``{s}_amount`` 미병합 시 빈 리스트 반환(holdings 단독 df).
    """
    if df.empty or "close" not in df.columns:
        return []
    if any(c not in df.columns for c in PNL_AMOUNT_COLS):
        return []
    closes = df["close"].astype(float)
    closes = closes[closes > 0]
    if closes.empty:
        return []
    last_close = float(closes.iloc[-1])

    rows: list[dict] = []
    for s in KEY_SUBJECTS:
        nq = pd.to_numeric(df[f"{s}_net_qty"], errors="coerce").fillna(0)
        na = pd.to_numeric(df[f"{s}_amount"], errors="coerce").fillna(0)
        buy, sell = nq > 0, nq < 0           # 순매수일 / 순매도일
        buy_qty, buy_amt = float(nq[buy].sum()), float(na[buy].sum())
        sell_qty, sell_amt = float(nq[sell].sum()), float(na[sell].sum())  # 둘 다 −
        sell_mag = -sell_qty                 # 순매도 수량 절댓값
        net_qty, net_amt = buy_qty + sell_qty, buy_amt + sell_amt
        buy_avg = (buy_amt / buy_qty) if buy_qty != 0 else None
        sell_avg = (sell_amt / sell_qty) if sell_qty != 0 else None  # −/− = +

        # 실현: 왕복 매칭분(매도평단 − 매수평단). 미실현: 남은 순포지션 현재가 평가.
        matched = min(buy_qty, sell_mag)
        realized = matched * (sell_avg - buy_avg) if matched > 0 else 0.0
        if net_qty > 0:                      # 순매수 잔여 → 매수평단 기준, 보유 평가손익
            unrealized = net_qty * (last_close - buy_avg)
            unrealized_kind = "평가"
        elif net_qty < 0:                    # 순매도 초과분 → 매도평단 기준, 기회손익
            unrealized = net_qty * (last_close - sell_avg)
            unrealized_kind = "기회"
        else:
            unrealized = 0.0
            unrealized_kind = None
        pnl = realized + unrealized          # = last_close*net_qty − net_amt (항등식)
        rows.append({
            "subject": s,
            "label": SUBJECT_LABELS[s],
            "buy_qty": int(buy_qty),
            "buy_avg": round(buy_avg, 1) if buy_avg is not None else None,
            "sell_qty": int(sell_mag),       # 양수 표기
            "sell_avg": round(sell_avg, 1) if sell_avg is not None else None,
            "net_qty": int(net_qty),
            "net_amount": net_amt,
            "last_close": round(last_close, 1),
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "unrealized_kind": unrealized_kind,  # "평가"(순매수 보유) / "기회"(순매도) / None
            "pnl": pnl,
        })
    rows.sort(key=lambda r: r["pnl"], reverse=True)
    return rows


# === 통합 리포트 =============================================================

def compute_advanced_report(df: pd.DataFrame) -> dict:
    """holdings DataFrame → 6 종 정교한 분석 결과 dict.

    무거운 연산(VAR fit, Granger 5 lag × 5 주체 × 2방향 등) 포함이라
    streamlit 에서는 expander 열 때만 호출되도록 캐시 권장.
    """
    out: dict = {
        "adf": [], "granger": [], "var_irf": None,
        "cointegration": [], "mutual_info": [], "rolling_r": [], "pnl": [],
        "config": {
            "key_subjects": list(KEY_SUBJECTS),
            "granger_max_lag": GRANGER_MAX_LAG,
            "var_horizon": VAR_HORIZON,
            "rolling_window": ROLLING_WINDOW,
        },
    }
    if df.empty or "price_change_pct" not in df.columns:
        return out

    df = df.sort_values("date").reset_index(drop=True)
    # 상장 전(close=0) 또는 결측 close 행 제거 — pct_change 가 inf/NaN 발생해
    # statsmodels (Granger/VAR/coint) 가 MissingDataError 던지는 원인.
    # volume=0 유령행(비유동·거래정지 구간 기준가 노이즈)도 제거 — close>0 만으로는
    # 안 걸러져 returns 시계열을 오염시킴.
    df = df[(df["close"] > 0) & (df["volume"] > 0)].reset_index(drop=True)
    # 상장 첫날 price_change_pct 가 inf — 제거.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["price_change_pct"]
    ).reset_index(drop=True)
    if len(df) < 50:
        return out
    ret = df["price_change_pct"].astype(float)
    close = df["close"].astype(float)

    # [1] ADF — return, close, 그리고 핵심 주체의 cum_qty / net_qty
    adf_rows: list[dict] = []
    adf_rows.append({"series": "return (price_change_pct)",
                     "label": "수익률 (정상성 기대)", **adf_test(ret)})
    adf_rows.append({"series": "close",
                     "label": "종가 (비정상 기대)", **adf_test(close)})
    for s in KEY_SUBJECTS:
        adf_rows.append({"series": f"{s}_cum_qty",
                         "label": f"{SUBJECT_LABELS[s]} 누적",
                         **adf_test(df[f"{s}_cum_qty"].astype(float))})
        adf_rows.append({"series": f"{s}_net_qty",
                         "label": f"{SUBJECT_LABELS[s]} 일별",
                         **adf_test(df[f"{s}_net_qty"].astype(float))})
    out["adf"] = adf_rows

    # [2] Granger — 양방향. stationary 변수만 안전 → net_qty + return.
    granger_rows: list[dict] = []
    for s in KEY_SUBJECTS:
        net = df[f"{s}_net_qty"].astype(float)
        # net → return
        fwd = granger_test(ret, net)
        # return → net (chasing 방향)
        bwd = granger_test(net, ret)
        granger_rows.append({
            "subject": s, "label": SUBJECT_LABELS[s],
            "net_causes_return": fwd,
            "return_causes_net": bwd,
        })
    out["granger"] = granger_rows

    # [3] VAR — return + 핵심 주체 net_qty (모두 stationary)
    var_cols = ["return"] + [f"{s}_net" for s in KEY_SUBJECTS]
    var_df = pd.DataFrame({"return": ret})
    for s in KEY_SUBJECTS:
        var_df[f"{s}_net"] = df[f"{s}_net_qty"].astype(float)
    out["var_irf"] = var_irf(var_df, var_cols, horizon=VAR_HORIZON)

    # [4] Cointegration — cum_qty 와 close. 핵심 주체 별.
    coint_rows: list[dict] = []
    for s in KEY_SUBJECTS:
        cum = df[f"{s}_cum_qty"].astype(float)
        res = cointegration_test(cum, close)
        coint_rows.append({"subject": s, "label": SUBJECT_LABELS[s], **res})
    out["cointegration"] = coint_rows

    # [5] Mutual Information — net_qty vs return (Pearson 보완)
    mi_rows: list[dict] = []
    for s in KEY_SUBJECTS:
        net = df[f"{s}_net_qty"].astype(float)
        mi = mutual_info(net, ret)
        mi_rows.append({"subject": s, "label": SUBJECT_LABELS[s], "mi": mi})
    # |MI| 큰 순 정렬
    mi_rows.sort(key=lambda r: r["mi"], reverse=True)
    out["mutual_info"] = mi_rows

    # [6] Rolling correlation (60일) — DCC 단순판
    rolling_rows: list[dict] = []
    for s in KEY_SUBJECTS:
        net = df[f"{s}_net_qty"].astype(float)
        summary = rolling_correlation_summary(net, ret)
        rolling_rows.append({"subject": s, "label": SUBJECT_LABELS[s],
                             **summary})
    out["rolling_r"] = rolling_rows

    # [7] 주체별 손익 근사 — {s}_amount 가 병합돼 있을 때만 채워짐
    out["pnl"] = pnl_attribution(df)

    return out


# === 마크다운 렌더 ==========================================================

def render_advanced_markdown(adv: dict) -> str:
    """advanced report dict → 마크다운 (CLI / 옵시디언 공용)."""
    from dashboard import interpretation as itp  # noqa: PLC0415

    lines: list[str] = ["## 7. 정교한 분석 (Granger·VAR·MI 외)"]
    if not adv["adf"]:
        lines.append("")
        lines.append("*데이터 부족 — 분석 미수행.*")
        return "\n".join(lines)

    cfg = adv["config"]
    lines.append("")
    lines.append(f"핵심 주체: {', '.join(SUBJECT_LABELS[s] for s in cfg['key_subjects'])}")
    lines.append(f"Granger max lag={cfg['granger_max_lag']} · "
                 f"VAR horizon={cfg['var_horizon']} · "
                 f"Rolling window={cfg['rolling_window']}일")
    lines.append("")

    # ADF
    lines.append("### 7-1. ADF 정상성 검정")
    lines.append("")
    lines.append("| 시리즈 | ADF | p | 정상성 | n |")
    lines.append("|---|---:|---:|:---:|---:|")
    for r in adv["adf"]:
        ok = "✓" if r["is_stationary"] else "✗ 비정상"
        adf_s = "n/a" if r["adf_stat"] is None else f"{r['adf_stat']:+.3f}"
        p_s = "n/a" if r["p_value"] is None else f"{r['p_value']:.4f}"
        lines.append(f"| {r['label']} | {adf_s} | {p_s} | {ok} | {r['n_obs']:,} |")
    lines.append("")
    lines.append("*해석*: 종가·cum_qty 같은 누적 시계열이 비정상으로 나오는 게 보통. "
                 "level r 은 spurious 위험 — cointegration 결과로 보강.")
    lines.append("")
    lines.append(f"> **자동 해석**: {itp.interpret_adf(adv)}")
    lines.append("")

    # Granger
    lines.append("### 7-2. Granger Causality (양방향)")
    lines.append("")
    lines.append("**net_qty → return** — 매매가 미래 가격을 *예측*하는가?")
    lines.append("")
    lines.append("| 주체 | " + " | ".join(f"lag {k}" for k in range(1, 6)) + " |")
    lines.append("|---|" + "---:|" * 5)
    for r in adv["granger"]:
        cells = []
        by_lag = {x["lag"]: x for x in r["net_causes_return"]}
        for k in range(1, 6):
            x = by_lag.get(k)
            if x is None:
                cells.append("—")
            else:
                mark = "✓" if x["significant"] else ""
                cells.append(f"p={x['p_value']:.3f} {mark}".strip())
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**return → net_qty (chasing)** — 가격 변동이 다음 매매를 유발?")
    lines.append("")
    lines.append("| 주체 | " + " | ".join(f"lag {k}" for k in range(1, 6)) + " |")
    lines.append("|---|" + "---:|" * 5)
    for r in adv["granger"]:
        cells = []
        by_lag = {x["lag"]: x for x in r["return_causes_net"]}
        for k in range(1, 6):
            x = by_lag.get(k)
            if x is None:
                cells.append("—")
            else:
                mark = "✓" if x["significant"] else ""
                cells.append(f"p={x['p_value']:.3f} {mark}".strip())
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"> **자동 해석**: {itp.interpret_granger(adv)}")
    lines.append("")

    # VAR
    irf = adv["var_irf"] or {}
    lines.append("### 7-3. VAR + 누적 충격반응(IRF)")
    lines.append("")
    if not irf or irf.get("order") is None:
        lines.append("*VAR fit 실패 — 데이터 부족 또는 공선성*")
    else:
        lines.append(f"VAR 차수(AIC) = **{irf['order']}**, horizon = {irf['horizon']}일")
        lines.append("")
        lines.append("**1-σ 매매 충격이 누적 수익률에 미치는 효과** "
                     "(첫째·5일·10일 후 누적값)")
        lines.append("")
        lines.append("| 충격원 → return | t=0 | t=5 | t=10 |")
        lines.append("|---|---:|---:|---:|")
        for s in KEY_SUBJECTS:
            key = f"{s}_net->return"
            vals = irf["irf_cum"].get(key, [])
            if not vals:
                continue
            v0 = vals[0] if len(vals) > 0 else 0.0
            v5 = vals[5] if len(vals) > 5 else vals[-1]
            v10 = vals[10] if len(vals) > 10 else vals[-1]
            lines.append(f"| {SUBJECT_LABELS[s]} | {v0:+.4f} | {v5:+.4f} | {v10:+.4f} |")
    lines.append("")
    lines.append(f"> **자동 해석**: {itp.interpret_var_irf(adv)}")
    lines.append("")

    # Cointegration
    lines.append("### 7-4. Cointegration — cum_qty ↔ close")
    lines.append("")
    lines.append("| 주체 | score | p | 공적분 |")
    lines.append("|---|---:|---:|:---:|")
    for r in adv["cointegration"]:
        ok = "✓ 장기 균형" if r["is_cointegrated"] else "✗"
        sc = "n/a" if r["score"] is None else f"{r['score']:+.3f}"
        p_s = "n/a" if r["p_value"] is None else f"{r['p_value']:.4f}"
        lines.append(f"| {r['label']} | {sc} | {p_s} | {ok} |")
    lines.append("")
    lines.append("*공적분 ✓ 면 두 시계열이 장기적으로 함께 움직임 → level r 신뢰 가능. "
                 "✗ 면 추세 동조성으로 spurious 가능.*")
    lines.append("")
    lines.append(f"> **자동 해석**: {itp.interpret_cointegration(adv)}")
    lines.append("")

    # MI
    lines.append("### 7-5. Mutual Information — net_qty ↔ return (비선형 포함)")
    lines.append("")
    lines.append("| 주체 | MI (큰 순) |")
    lines.append("|---|---:|")
    for r in adv["mutual_info"]:
        lines.append(f"| {r['label']} | {r['mi']:.4f} |")
    lines.append("")
    lines.append("*Pearson r 이 작은데 MI 가 크면 비선형 의존성 신호.*")
    lines.append("")
    lines.append(f"> **자동 해석**: {itp.interpret_mi(adv)}")
    lines.append("")

    # Rolling
    rw = adv["config"]["rolling_window"]
    lines.append(f"### 7-6. Rolling correlation ({rw}일 윈도우) — DCC 단순판")
    lines.append("")
    lines.append("| 주체 | mean | std | min | p10 | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in adv["rolling_r"]:
        if r["mean"] is None:
            continue
        lines.append(f"| {r['label']} | {r['mean']:+.3f} | {r['std']:.3f} | "
                     f"{r['min']:+.3f} | {r['p10']:+.3f} | "
                     f"{r['p90']:+.3f} | {r['max']:+.3f} |")
    lines.append("")
    lines.append("*std 가 크면 상관 강도가 시기마다 크게 변동(regime 변화).*")
    lines.append("")
    lines.append(f"> **자동 해석**: {itp.interpret_rolling(adv)}")
    lines.append("")

    # 7-7. 주체별 손익 근사
    lines.append("### 7-7. 주체별 손익 근사 (구간 mark-to-market)")
    lines.append("")
    if not adv.get("pnl"):
        lines.append("*거래대금(amount) 미병합 — 손익 추정 생략.*")
    else:
        lc = adv["pnl"][0]["last_close"]
        lines.append(f"구간 net 흐름을 순매수일/순매도일로 분리(현재가 "
                     f"{lc:,.0f}원). **실현** = 왕복 매칭분(매도평단−매수평단). "
                     "**미실현 평가** = 순매수 잔여 보유의 현재가 평가, "
                     "**미실현 기회** = 순매도 초과분의 매도 타이밍 평가(보유 아님).")
        lines.append("")
        lines.append("| 주체 | 순매수량 | 순매수평단 | 순매도량 | 순매도평단 | 실현(억) | 미실현 평가(억) | 미실현 기회(억) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in adv["pnl"]:
            ba = "—" if r["buy_avg"] is None else f"{r['buy_avg']:,.0f}"
            sa = "—" if r["sell_avg"] is None else f"{r['sell_avg']:,.0f}"
            u = r["unrealized_pnl"] / 1e8
            ev = f"{u:+,.1f}" if r["unrealized_kind"] == "평가" else "—"
            op = f"{u:+,.1f}" if r["unrealized_kind"] == "기회" else "—"
            lines.append(
                f"| {r['label']} | {r['buy_qty']:,} | {ba} | "
                f"{r['sell_qty']:,} | {sa} | {r['realized_pnl'] / 1e8:+,.1f} | "
                f"{ev} | {op} |"
            )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**구현하지 않은 4 종 — 사유**")
    lines.append("")
    lines.append("| 기법 | 사유 |")
    lines.append("|---|---|")
    lines.append("| **DCC-GARCH** | `arch` 라이브러리(C 확장) 추가 필요, fit 시간 길고 종목·주체별로 모델링이 무거움. Rolling correlation 으로 1차 근사. |")
    lines.append("| **Transfer Entropy** | 안정적 PyPI 패키지 부재(직접 구현 필요), 표본 크기에 매우 민감해 1,200일로는 추정 분산이 큼. Granger + MI 로 선·비선형 정보 흐름은 부분 커버. |")
    lines.append("| **Hawkes process** | `tick` 라이브러리는 macOS arm64 빌드 이슈가 잦고, 이벤트 자기·교차 유발 모델은 일별 데이터(저빈도)보다 분/틱 데이터에서 가치가 높음. |")
    lines.append("| **Wavelet coherence** | `pywt` + 자체 구현 부담. 결과 해석에 푸리에/웨이블릿 도메인 친숙도 필요. 본 분석 목적(주체별 상관 구조)에 비해 가독성 낮음. |")
    return "\n".join(lines)
