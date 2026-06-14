"""Stage 1 게이트 — 시장 도움 제거 채점 (새 루프 1차 관문).

배경 (2026-06-11 실측): 백테스트 게이트·차감-바스켓 알파는 모두 베타 틸트에
속는다 — 출렁임 큰 종목을 고르면 상승장에서 그 초과분이 실력처럼 보이고,
레짐이 꺾이면 무너진다(SELECTED 14: 회귀 절편 전원 0, 모멘텀 β 1.2~3.3,
하락창 차감알파 유의 음수).

채점: 거래 단위 회귀  net = α + β·basket
  - α(절편)  = 같은 보유기간 바스켓이 0일 때의 기대 초과수익
               = **시장 도움을 뺀 종목선택 실력. 합격 근거는 오직 이것.**
  - β(기울기) = 시장 민감도 배율 — 보고만, 게이트하지 않음.
  - 하락창   = 바스켓이 음(-)이던 거래들의 차감알파 — 베타 위장의 직접 증상.

게이트 금지 지표(과거 오판정 봉인): 승률, 완전투자 벤치마크 대비 총수익,
차감-바스켓 알파 단독.

사용 (홀드아웃 채점 + 원장 기록):
    uv run python -m tradingagents.newloop.stage1_gate \\
        --family momfo-ride --hypothesis "외인추세+장기보유 모멘텀" \\
        --ids 3009,3010,3011 --out artifacts/newloop/stage1_xxx.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from tradingagents.newloop import ledger

# ── 게이트 사양 (동결 — 변경 시 버전 상향 + 사유 기록) ──────────────────
# s1-v1 (2026-06-11): 최초. t(α)≥3.0, 하락창 t>-2.0, 30거래.
# s1-v2 (2026-06-11): 적대적 검증 반영 —
#   C2: 같은 시기 중첩 거래는 독립 표본이 아님(상관 복제로 t 2.04→6.78 부풀림
#       실증) → 진입월 군집-로버스트 표준오차로 교체 + 최소 군집 수 요구.
#   W1: 하락창 표본 3건으로 ③검사 무력화 실증 → 최소 10건 미만이면 보류.
#   W2: 미세 알파(α 0.05%)×거대 표본이 t만으로 통과 실증 → 경제성 하한 추가
#       (슬리피지·모형오차 버퍼보다 커야 의미).
#   W3: 반올림된 t로 판정 시 경계 뒤집힘 실증 → 판정은 raw, 표시만 반올림.
# s1-v3 (2026-06-12): 다중검정 보정 —
#   고정 t≥3.0 은 단일검정 하한(HLZ 2016)일 뿐, 홀드아웃을 N 번 들여다보면
#   운만으로 통과가 누적된다(N=770·t≥3 → 기대 거짓통과 ≈1.07 실측 계산).
#   합격선을 누적 채점 수 N 에 맞춰 동적 상향(Šidák FWER 보정) + 원장에 총 N
#   상한(ledger.HOLDOUT_TRIAL_BUDGET). N 은 게이트가 ledger 에서 읽어온다.
GATE_VERSION = "s1-v3"
GATE_MIN_TRADES = 30      # 미만이면 운/실력 구분 불가 → insufficient(판정 보류)
GATE_MIN_CLUSTERS = 8     # 진입월 군집 최소 수 — 시기 다양성 없이는 t 신뢰 불가
GATE_MIN_T_ALPHA = 3.0    # α 의 t 하한(단일검정, HLZ 2016) — 동적 합격선의 절대 floor
GATE_MIN_ALPHA_PCT = 0.3  # α 경제성 하한(%) — 슬리피지·모형오차 이하의 알파는 무의미
GATE_MIN_DOWN_TRADES = 10  # 하락창 최소 표본 — 미만이면 위장 검사 불가 → 보류
GATE_DOWN_T_FLOOR = -2.0  # 하락창 차감알파가 유의하게 음수면 베타 위장 → 탈락
GATE_FWER_ALPHA = 0.05    # 홀드아웃 수명 전체의 family-wise 거짓통과 목표
BOOTSTRAP_B = 2000        # wild bootstrap 재표본 수 (임계 보정)
_Z99 = 2.3263478740408408  # Φ⁻¹(0.99) 단측 — 부트스트랩 꼬리 팽창계수 기준점

_EULER_GAMMA = 0.5772156649015329
_NORM = statistics.NormalDist()  # 표준정규 (stdlib — scipy 비의존)


def _quantile(xs: list[float], q: float) -> float:
    """정렬·선형보간 분위 (numpy 비의존)."""
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _resampled_t_samples(xs, ys, groups, t_alpha_fn, method, block_len, B, seed):
    """귀무(α=0) 하 재표본 t(α) 경험분포. method='wild'(군집 부호반전) | 'mbb'
    (이동블록 잔차 복원추출). 평균 0 전체모형 잔차를 써 알파 누출을 막는다."""
    n = len(xs)
    mx = sum(xs) / n
    sxx_c = sum((x - mx) ** 2 for x in xs)
    if n < 10 or sxx_c <= 0:
        return None
    my = sum(ys) / n
    b_full = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx_c
    a_full = my - b_full * mx
    resid = [ys[i] - a_full - b_full * xs[i] for i in range(n)]   # mean 0
    rng = random.Random(seed)
    ts: list[float] = []
    if method == "wild":
        uniq = list(dict.fromkeys(groups))
        for _ in range(B):
            sign = {g: (1.0 if rng.random() < 0.5 else -1.0) for g in uniq}
            ys_star = [b_full * xs[i] + resid[i] * sign[groups[i]] for i in range(n)]
            t = t_alpha_fn(xs, ys_star, groups)
            if t is not None and math.isfinite(t):
                ts.append(t)
    else:  # mbb — 이동블록 잔차 복원추출(시계열 자기상관 보존 + HAC 소표본 재현)
        L = max(2, block_len)
        nblocks = -(-n // L)
        for _ in range(B):
            rr: list[float] = []
            for _b in range(nblocks):
                s = rng.randint(0, n - L)
                rr.extend(resid[s:s + L])
            rr = rr[:n]
            ys_star = [b_full * xs[i] + rr[i] for i in range(n)]
            t = t_alpha_fn(xs, ys_star, groups)
            if t is not None and math.isfinite(t):
                ts.append(t)
    return ts if len(ts) >= 100 else None


def wild_bootstrap_t_crit(xs, ys, groups, t_alpha_fn, n_trials: int,
                          *, method: str = "wild", block_len: int = 5,
                          B: int | None = None, seed: int = 20260612) -> float | None:
    """귀무(α=0) 부트스트랩으로 동적 t(α) 합격선을 데이터에서 보정.

    정규분위(deflated_t_threshold)는 소표본 군집/HAC t 의 두꺼운 꼬리를 못 막아
    실현 FWER 이 목표(0.05)를 크게 초과한다(재감사 실증: 소표본 HAC-t 에서
    P(t≥3)=1.9% = 명목 14배, 400회 채점 FWER≈0.96). 이를 막으려 이 표본의 귀무
    t 분포를 직접 만들어 임계를 잡는다.

    method='wild' (Stage1 군집-로버스트): 군집(groups) 단위 Rademacher 부호반전.
    method='mbb'  (Stage2 HAC 시계열): 길이 block_len 이동블록 잔차 복원추출 —
      부호반전은 HAC 소표본 과대산포를 못 재현하므로(실측) 블록 복원추출로 교체.

    극단 분위는 B 가 천문학적이어야 직접 추정되므로, 99% 분위에서 잰 '정규 대비
    꼬리 팽창계수'(≥1)를 목표 꼬리확률의 정규-z 에 곱해 외삽한다(floor 의도 포함).

    t_alpha_fn(xs, ys_star, groups) -> t(α) 또는 None. degenerate 면 None
    (호출부가 정규 임계로 폴백).
    """
    ts = _resampled_t_samples(xs, ys, groups, t_alpha_fn, method, block_len,
                              B or BOOTSTRAP_B, seed)
    if not ts:
        return None
    q99 = _quantile([abs(t) for t in ts], 0.99)   # 양측 대칭 귀무 → |t| 상위
    if q99 <= 0:
        return None
    inflation = max(1.0, q99 / _Z99)              # 정규 대비 꼬리 팽창 (≥1)
    # 목표 꼬리확률 = FWER 시도당(α')과 HLZ 단일검정 강도(t≥3.0 ⇒ ≈0.00135) 중
    # 더 엄격한 쪽. 그 정규-z 를 꼬리 팽창계수로 부풀려 임계로(floor 의도까지 보정).
    alpha_prime = -math.expm1(math.log1p(-GATE_FWER_ALPHA) / max(n_trials, 1))
    p_target = min(alpha_prime, _NORM.cdf(-GATE_MIN_T_ALPHA))
    z_target = -_NORM.inv_cdf(p_target)           # ≥ GATE_MIN_T_ALPHA
    return inflation * z_target                   # inflation≥1·z_target≥3.0 ⇒ ≥3.0

DEFAULT_WINDOW_START = "2025-07-01"
STRAT_DB = os.path.expanduser("~/.tradingagents/hermes/strategies_v2.db")  # 읽기전용 입력


def deflated_t_threshold(n_trials: int) -> float:
    """누적 시도 수 N 에 맞춰 동적 상향된 t(α) 합격선 (Šidák FWER 보정).

    홀드아웃을 N 번 들여다봤을 때 '운으로 적어도 하나 통과'할 확률을
    GATE_FWER_ALPHA 이하로 묶으려면 시도당 유의수준을 α'=1-(1-α)^(1/N) 로
    조여야 한다 → 임계 t = Φ⁻¹(1-α'). 단 단일검정 하한 GATE_MIN_T_ALPHA(3.0,
    HLZ 2016) 밑으로는 절대 내리지 않는다. N≈38 부터 3.0 을 넘어 상승
    (N=100→3.28, N=400→3.66).
    """
    if n_trials <= 1:
        return GATE_MIN_T_ALPHA
    # α' = 1-(1-α)^(1/N) 를 expm1 로 안정 계산하고, 상측분위는 대칭형
    # -Φ⁻¹(α') 로 구한다(1-α' 가 거대 N 에서 1.0 으로 반올림돼 inv_cdf 가
    # 터지는 것을 회피).
    alpha_prime = -math.expm1(math.log1p(-GATE_FWER_ALPHA) / n_trials)
    return max(GATE_MIN_T_ALPHA, -_NORM.inv_cdf(alpha_prime))


def dsr_benchmark_t(n_trials: int) -> float | None:
    """N 번 시도 시 순수 운이 낼 '기대 최대 t' (Bailey-López de Prado deflated
    benchmark). 관측 t 가 이보다 못하면 운의 기대치 이하 — 보고용 진단."""
    if n_trials < 2:
        return None
    # Φ⁻¹(1-q) = -Φ⁻¹(q) — 거대 N 에서 1-q 가 1.0 으로 반올림되는 것 회피
    z1 = -_NORM.inv_cdf(1.0 / n_trials)
    z2 = -_NORM.inv_cdf(1.0 / (n_trials * math.e))
    return (1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2


def _fwer_p(t_alpha: float, n_trials: int) -> float:
    """N 번 시도에서 운만으로 이 t 이상이 적어도 하나 나올 확률 = 1-(1-p)^N."""
    p = _NORM.cdf(-t_alpha)
    return 1.0 - (1.0 - p) ** max(n_trials, 1)


def ols_alpha_beta_clustered(xs: list[float], ys: list[float], clusters: list[str]):
    """단순회귀 y = a + b·x, 군집-로버스트(CR1) 표준오차.

    같은 군집(진입월)의 거래들은 같은 시장 구간을 공유해 잔차가 상관됨 —
    iid 가정의 t 는 유효 표본을 과대평가한다(상관 복제 공격 실증). 군집 합산
    스코어로 분산을 추정해 이를 흡수한다. 반환 (a, t_a, b, t_b_vs1, n_clusters).
    """
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    a = my - b * mx
    resid = [ys[i] - (a + b * xs[i]) for i in range(n)]

    # Bread = (X'X)^-1, X=[1,x] — 2x2 역행렬 직접 계산
    sx = sum(xs)
    sxx_raw = sum(x * x for x in xs)
    det = n * sxx_raw - sx * sx
    if det <= 0:
        return None
    inv = [[sxx_raw / det, -sx / det], [-sx / det, n / det]]

    # Meat = Σ_g (X_g'e_g)(X_g'e_g)'
    g_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for i, c in enumerate(clusters):
        g_sums[c][0] += resid[i]
        g_sums[c][1] += xs[i] * resid[i]
    G = len(g_sums)
    if G < 2:
        return None
    meat = [[0.0, 0.0], [0.0, 0.0]]
    for s0, s1 in g_sums.values():
        meat[0][0] += s0 * s0
        meat[0][1] += s0 * s1
        meat[1][0] += s1 * s0
        meat[1][1] += s1 * s1

    # V = c · Bread · Meat · Bread, CR1 보정 c = G/(G-1) · (n-1)/(n-2)
    c1 = (G / (G - 1)) * ((n - 1) / (n - 2))
    bm = [[sum(inv[i][k] * meat[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    v = [[c1 * sum(bm[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    se_a = math.sqrt(max(v[0][0], 0.0))
    se_b = math.sqrt(max(v[1][1], 0.0))
    t_a = a / se_a if se_a > 0 else 0.0
    t_b = (b - 1.0) / se_b if se_b > 0 else 0.0
    return a, t_a, b, t_b, G


def _mean_t(xs: list[float]):
    """(n, 평균, t) — 1표본 t (표본표준편차 ddof=1). 표본 부족/무분산이면 t=None."""
    if not xs:
        return 0, None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return 1, m, None
    sd = statistics.stdev(xs)
    return len(xs), m, (m / (sd / math.sqrt(len(xs))) if sd > 0 else None)


def score_pairs(pairs: list[tuple[float, float, str]], n_trials: int = 1) -> dict:
    """(net%, basket%, 진입월) 목록 → 기계 판정. 순수 함수(엔진 비의존).

    n_trials = 홀드아웃 누적 채점 수(이 채점 포함). 합격선을 다중검정 보정해
    동적 상향한다 (n_trials=1 이면 단일검정 = 기본 t≥3.0).

    판정 = 전 조건 AND. 판정은 전부 raw 값으로 하고 출력만 반올림한다.
      ① n ≥ GATE_MIN_TRADES, 군집(진입월) ≥ GATE_MIN_CLUSTERS   (미달: insufficient)
      ② t(α) ≥ deflated_t_threshold(n_trials) AND α ≥ GATE_MIN_ALPHA_PCT
      ③ 하락창 n ≥ GATE_MIN_DOWN_TRADES (미달: insufficient)
         AND 하락창 차감알파 t > GATE_DOWN_T_FLOOR
    """
    t_crit_norm = deflated_t_threshold(n_trials)
    t_crit = t_crit_norm  # 회귀 성공 후 부트스트랩 임계로 교체
    dsr_bench = dsr_benchmark_t(n_trials)
    out = {
        "gate_version": GATE_VERSION,
        "n_trades": len(pairs), "n_clusters": 0,
        "alpha_pct": None, "t_alpha": None, "beta": None, "t_beta_vs1": None,
        "down": {"n": 0, "mean_pct": None, "t": None},
        "up": {"n": 0, "mean_pct": None, "t": None},
        "multiplicity": {"n_trials": n_trials, "t_crit": round(t_crit, 3),
                         "t_crit_normal": round(t_crit_norm, 3), "t_crit_method": "normal",
                         "fwer_alpha": GATE_FWER_ALPHA, "fwer_p": None,
                         "dsr_benchmark_t": round(dsr_bench, 3) if dsr_bench is not None else None,
                         "dsr": None},
        "checks": {"enough_trades": None, "enough_clusters": None,
                   "alpha_significant": None, "alpha_material": None,
                   "enough_down": None, "down_not_broken": None},
        "status": "insufficient", "gate_passed": False,
    }
    if len(pairs) < GATE_MIN_TRADES:
        out["checks"]["enough_trades"] = False
        return out
    out["checks"]["enough_trades"] = True

    nets = [p[0] for p in pairs]
    baskets = [p[1] for p in pairs]
    months = [p[2] for p in pairs]
    reg = ols_alpha_beta_clustered(baskets, nets, months)
    if reg is None:
        return out
    a, t_a, b, t_b, n_clusters = reg
    out["n_clusters"] = n_clusters
    out.update(alpha_pct=round(a, 4), t_alpha=round(t_a, 3),
               beta=round(b, 4), t_beta_vs1=round(t_b, 3))
    # 소표본 군집 t 의 두꺼운 꼬리를 데이터에서 보정 (정규임계 폴백)
    def _t_clustered(xx, yy, gg):
        r = ols_alpha_beta_clustered(xx, yy, gg)
        return r[1] if r is not None else None
    t_boot = wild_bootstrap_t_crit(baskets, nets, months, _t_clustered, n_trials)
    if t_boot is not None:
        t_crit = t_boot
        out["multiplicity"]["t_crit"] = round(t_crit, 3)
        out["multiplicity"]["t_crit_method"] = "wild_bootstrap"
    out["multiplicity"]["fwer_p"] = round(_fwer_p(t_a, n_trials), 6)
    if dsr_bench is not None:
        out["multiplicity"]["dsr"] = round(_NORM.cdf(t_a - dsr_bench), 4)
    if n_clusters < GATE_MIN_CLUSTERS:
        out["checks"]["enough_clusters"] = False
        return out
    out["checks"]["enough_clusters"] = True

    down = [n_ - x for n_, x in zip(nets, baskets) if x < 0]
    up = [n_ - x for n_, x in zip(nets, baskets) if x > 0]
    dn_n, dn_m, dn_t = _mean_t(down)
    up_n, up_m, up_t = _mean_t(up)
    out["down"] = {"n": dn_n, "mean_pct": round(dn_m, 4) if dn_m is not None else None,
                   "t": round(dn_t, 3) if dn_t is not None else None}
    out["up"] = {"n": up_n, "mean_pct": round(up_m, 4) if up_m is not None else None,
                 "t": round(up_t, 3) if up_t is not None else None}

    # 판정은 raw 값으로 (반올림 경계 뒤집힘 방지 — s1-v2 W3)
    out["checks"]["alpha_significant"] = bool(t_a >= t_crit)  # 동적 합격선 (s1-v3)
    out["checks"]["alpha_material"] = bool(a >= GATE_MIN_ALPHA_PCT)
    if dn_n < GATE_MIN_DOWN_TRADES or dn_t is None:
        # 위장 여부를 못 본 채 합격시키는 것이 가장 위험 — 보류.
        out["checks"]["enough_down"] = False
        return out
    out["checks"]["enough_down"] = True
    out["checks"]["down_not_broken"] = bool(dn_t > GATE_DOWN_T_FLOOR)

    out["gate_passed"] = bool(
        out["checks"]["alpha_significant"] and out["checks"]["alpha_material"]
        and out["checks"]["down_not_broken"]
    )
    out["status"] = "pass" if out["gate_passed"] else "fail"
    return out


def collect_pairs(spec: dict, tickers: list[str], loader, basket_ret,
                  w0: str, w1: str | None) -> list[tuple[float, float, str]]:
    """홀드아웃에서 실현 거래의 (net%, 같은 구간 바스켓%, 진입월) 수집. forced_eod 제외."""
    from tradingagents.hermes.backtest_engine_v2 import _simulate_v2

    pairs: list[tuple[float, float, str]] = []
    for tk in tickers:
        df, _ = loader(tk)
        if df is None or df.empty:
            continue
        try:
            trades, _daily, _active, _cdf = _simulate_v2(spec, df)
        except Exception:
            continue
        for t in trades:
            ed = t["entry_date"]
            if ed < w0 or (w1 is not None and ed > w1):
                continue
            if t.get("exit_reason") == "forced_eod":
                continue
            br = basket_ret(ed, t["exit_date"])
            if br is None:
                continue
            pairs.append((float(t["net_ret_pct"]), float(br), str(ed)[:7]))
    return pairs


def score(spec: dict, tickers: list[str], loader, basket_ret,
          w0: str = DEFAULT_WINDOW_START, w1: str | None = None,
          n_trials: int = 1) -> dict:
    out = score_pairs(collect_pairs(spec, tickers, loader, basket_ret, w0, w1), n_trials)
    out["window"] = [w0, w1]
    return out


def _load_spec_from_db(sid: int) -> dict:
    con = sqlite3.connect(f"file:{STRAT_DB}?mode=ro", uri=True)
    row = con.execute("SELECT spec_json FROM strategies WHERE id=?", (sid,)).fetchone()
    if row is None:
        raise SystemExit(f"전략 id {sid} 없음: {STRAT_DB}")
    return json.loads(row[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage1 게이트 — 시장 도움 제거 채점")
    ap.add_argument("--family", required=True, help="가설 계열 키 (원장 사전등록 단위)")
    ap.add_argument("--hypothesis", default=None, help="가설 한 줄 (family 최초 등록 시 필수)")
    ap.add_argument("--ids", default=None, help="기존 strategies_v2.db 의 전략 id 콤마 목록")
    ap.add_argument("--spec-file", default=None, help="spec JSON 파일(단일 또는 배열)")
    ap.add_argument("--window", default=f"{DEFAULT_WINDOW_START}:",
                    help="진입일 창 'YYYY-MM-DD:YYYY-MM-DD' (끝 생략 가능)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    if not args.ids and not args.spec_file:
        ap.error("--ids 또는 --spec-file 필요")

    # 사전등록 + 재응시 한도 확인 (채점보다 먼저 — 시험지 마모 방지)
    if args.hypothesis:
        ledger.register_family(args.family, args.hypothesis)
    ok, why = ledger.can_submit(args.family)
    if not ok:
        print(f"제출 거부 [{args.family}]: {why}")
        return 2

    specs: list[tuple[str, dict]] = []
    if args.ids:
        for s in args.ids.split(","):
            sid = int(s)
            specs.append((f"db:{sid}", _load_spec_from_db(sid)))
    if args.spec_file:
        with open(args.spec_file, encoding="utf-8") as f:
            loaded = json.load(f)
        # 채점 전 spec 전수 검증 — 깨진 spec이 0거래→insufficient로 제출 슬롯을
        # 낭비하지 않도록, 하나라도 무효면 원장 기록 없이 중단한다.
        from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
        for i, sp in enumerate(loaded if isinstance(loaded, list) else [loaded]):
            try:
                validate_spec_v2(sp)
            except Exception as e:
                print(f"spec 무효(제출 미기록): #{i} {sp.get('name', '?')} — {e}")
                return 2
            specs.append((f"file:{sp.get('name', i)}", sp))

    # 다중검정 보정: 홀드아웃 누적 채점 수 N 에 맞춰 합격선을 동적 상향한다.
    # 예산 초과는 채점(시험지 소모) 전에 거른다.
    cum_before = ledger.cumulative_trials()
    n_eff = cum_before + len(specs)
    budget = ledger.HOLDOUT_TRIAL_BUDGET
    if n_eff > budget:
        print(f"제출 거부 [{args.family}]: 홀드아웃 예산 초과 "
              f"(누적 {cum_before}+{len(specs)} > {budget}) — 새 홀드아웃 수집 필요")
        return 2
    t_crit = deflated_t_threshold(n_eff)

    w0, _, w1 = args.window.partition(":")
    w1 = w1 or None

    from tradingagents.newloop.holdout import holdout_universe
    tickers, loader, basket_ret = holdout_universe()

    print(f"Stage1 {GATE_VERSION} | family={args.family} | 홀드아웃 {len(tickers)}종목 | 창 {w0}~{w1 or '끝'}")
    print(f"  다중검정 N={n_eff} (예산 {budget}, 잔여 {budget - n_eff}) → 동적 합격선 t(α)≥{t_crit:.2f}")
    print(f"{'ref':>10} {'n':>5} {'군집':>4} {'α%':>7} {'t(α)':>6} {'β':>6} {'하락n':>5} {'하락t':>6} {'FWERp':>8}  판정")
    results = []
    for ref, sp in specs:
        r = score(sp, tickers, loader, basket_ret, w0, w1, n_trials=n_eff)
        results.append({"strategy_ref": ref, "gate_version": GATE_VERSION,
                        "status": r["status"], "result": r})
        dn = r["down"]
        fp = r["multiplicity"]["fwer_p"]
        print(f"{ref:>10} {r['n_trades']:>5} {r['n_clusters']:>4} "
              f"{r['alpha_pct'] if r['alpha_pct'] is not None else '-':>7} "
              f"{r['t_alpha'] if r['t_alpha'] is not None else '-':>6} "
              f"{r['beta'] if r['beta'] is not None else '-':>6} "
              f"{dn['n']:>5} {dn['t'] if dn['t'] is not None else '-':>6} "
              f"{fp if fp is not None else '-':>8}  {r['status']}")

    sub = ledger.record_submission(args.family, results)
    st = ledger.family_state(args.family)
    print(f"\n원장 기록: 제출 #{sub}/{st['max_submissions']} → family 상태: {st['status']}")

    if args.out:
        out_dir = os.path.dirname(args.out) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "gate_version": GATE_VERSION,
                "thresholds": {"min_trades": GATE_MIN_TRADES, "min_clusters": GATE_MIN_CLUSTERS,
                               "min_t_alpha": GATE_MIN_T_ALPHA, "min_alpha_pct": GATE_MIN_ALPHA_PCT,
                               "min_down_trades": GATE_MIN_DOWN_TRADES, "down_t_floor": GATE_DOWN_T_FLOOR},
                "multiplicity": {"n_trials": n_eff, "cumulative_before": cum_before,
                                 "fwer_alpha": GATE_FWER_ALPHA, "t_crit": round(t_crit, 3),
                                 "budget": budget, "budget_remaining": budget - n_eff},
                "family": args.family, "submission": sub,
                "window": [w0, w1], "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
