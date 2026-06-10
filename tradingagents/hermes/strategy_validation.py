"""전략 검증 v3 — 시간 분할(walk-forward) 게이트 + 누수 방지 + 다중검정 가드.

배경: 기존 ``run_universe_backtest_v2`` 의 in/out 분할은 종목 md5 해시
(cross-sectional)라 in·out 이 **동일 기간**을 본다. 이는 "다른 종목으로 이식
가능한가"만 검증할 뿐 "학습에 안 쓰인 **미래**에도 통하는가"(out-of-time
generalization)를 측정하지 못한다. 진단(``scripts/v3_timesplit_diagnostic``)에서
종목분할 in/out 격차 중앙값 0.38 vs 시간분할 IS/OOS 격차 1.57 로, 종목분할이
시간 불안정성을 ~4배 과소평가함을 확인했다.

이 모듈은 **엔진을 수정/복사하지 않는다.** 검증된 ``_simulate_v2`` 를 그대로
재사용하되 분할 축만 시간으로 바꾼다.

설계 — 윈도우 격리(window isolation):
  IS 와 OOS 를 **서로 겹치지 않는 날짜 구간**으로 잘라 각각 독립 시뮬레이션한다.
  이로써 (1) IS 포지션은 IS 경계에서 강제청산되어 OOS 로 새지 않고(purging 자동),
  (2) OOS 는 무포지션에서 시작한다. 비용은 OOS 시작부의 신호 워밍업 손실과 IS
  경계 거래의 조기청산 — 둘 다 **보수적**(과대평가가 아니라 과소평가) 방향이다.
  embargo 는 IS 종료와 OOS 시작 사이에 거래일 갭을 둬 잔존 직렬상관을 끊는다.

게이트는 ``passes_full_gate_v2`` 를 그대로 재사용한다(두 split 모두 Sharpe/MDD/거래수
통과, 승률 조건 없음 — 의미 동일, 축만 시간). 즉 IS·OOS **양쪽**에서 강해야 통과한다.

참고: 본 모듈은 우선순위 1(시간분할)·2(purge/embargo)를 구현한다. PBO/Deflated
Sharpe/CPCV(우선순위 4~6)는 통과 후보가 복수로 나온 뒤 별도 추가한다.
"""
from __future__ import annotations

import math
import sys
from typing import Callable, Optional

import pandas as pd
from scipy.stats import norm

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.market_history import fetch_kospi, fetch_usdkrw
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import (
    GATE_V2_MIN_SHARPE,
    _market_overlay_multiplier,
    _simulate_v2,
    passes_full_gate_v2,
)
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2

IN_SAMPLE_PCT = 70           # 시간축 IS 비율(종목분할과 동일 — 축만 다름)
DEFAULT_EMBARGO_DAYS = 12    # IS↔OOS 갭(거래일). López de Prado h≈1%·T 권장에 부합.
_HI_SENTINEL = "9999-12-31"  # 마지막 날짜까지 포함하기 위한 상한
GATE_EXCESS_MIN_IR = 0.5     # (구) KOSPI 대비 초과 IR 하한 — 생존편향 포함이라 위양성多. 진단용 보존.
GATE_POOL_MIN_IR = 0.3       # (신) 생존풀(유니버스 동일가중) 대비 초과 IR 하한 — 생존편향 중립 '실력'.
# 유니버스가 전부 현재 생존자라 KOSPI 초과는 생존 프리미엄(클린기간 ~3.3배)을 실력으로 오인한다.
# 벤치마크를 생존풀로 바꾸면 그 프리미엄이 전략·벤치마크 양쪽에서 상쇄돼 순수 종목선택력만 남는다.
# 신호 계산은 실제 KOSPI 유지(market_filter 등 불변), 초과수익 벤치마크만 생존풀로 교체한다.
# OOS/IS 채점 상한 — 2025-06 이후 한국 증시 비정상 급등(생존자 풀 동일가중 ~18.8배 vs
# KOSPI ~4.3배)이 검증에 섞이면 성과가 구조적으로 과대평가된다(CLAUDE.md 규칙). 데이터
# 그리드를 이 날짜에서 잘라 채점하면 IS·OOS 모두 상한이 적용된다. 포워드 관찰은 date_hi=None.
SCORE_DATE_HI = "2025-06-30"


def _universe_dates(holdings: dict[str, pd.DataFrame]) -> list[str]:
    """전 종목 거래일 union — 정렬된 ISO 문자열(lexicographic=chronological)."""
    dates: set[str] = set()
    for df in holdings.values():
        dates.update(df["date"].astype(str).tolist())
    return sorted(dates)


def _survivor_pool_index(holdings: dict) -> Optional[pd.DataFrame]:
    """유니버스(생존 종목군) 동일가중 누적 인덱스 — 생존편향 중립 벤치마크.

    종목이 전부 현재 생존자라 KOSPI 초과는 생존 프리미엄을 실력으로 오인한다. 이 풀을
    벤치마크로 쓰면 프리미엄이 전략·벤치마크 양쪽에서 상쇄돼 순수 종목선택력만 남는다.
    반환 형식은 KOSPI df 와 동일(date/close)이라 bt._market_daily_returns 가 그대로 쓴다.
    """
    rets = []
    for _tk, df in holdings.items():
        d = df[bt._num(df["close"]) > 0]
        if d.empty:
            continue
        idx = d["date"].astype(str).to_numpy()
        rets.append(pd.Series(bt._num(d["close"]).pct_change().to_numpy(), index=idx, name=_tk))
    if not rets:
        return None
    ew = pd.concat(rets, axis=1).sort_index().mean(axis=1).fillna(0.0)
    close = (1.0 + ew).cumprod()
    return pd.DataFrame({"date": close.index, "close": close.to_numpy()})


def _window_metrics(spec: dict, holdings: dict, kospi, lo: str, hi: str, usdkrw=None,
                    benchmark=None) -> dict:
    """``[lo, hi)`` 날짜 구간만 잘라 종목군 시뮬 → 메트릭(bt._metrics).

    각 종목 df 를 구간으로 필터해 ``_simulate_v2`` 에 넘긴다. 구간 밖 데이터는
    신호 계산에서도 제외되므로(워밍업 손실) 누수가 원천 차단된다.
    """
    ret_frames, act_frames, all_trades = [], [], []
    n_tk = 0
    for tk in sorted(holdings):
        df = holdings[tk]
        d = df["date"].astype(str)
        sub = df[(d >= lo) & (d < hi)]
        if len(sub) < 3:
            continue
        sub = sub.reset_index(drop=True)
        trades, daily, active, cdf = _simulate_v2(spec, sub, kospi=kospi)
        bt._attach_kospi(trades, kospi)
        idx = cdf["date"].astype(str).to_numpy()
        ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
        act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        all_trades.extend(trades)
        n_tk += 1
    port = bt._combine_korea_stock_portfolio(ret_frames, act_frames)
    bench = kospi if benchmark is None else benchmark   # 초과수익 벤치마크(신호는 kospi 고정)
    overlay_mult = _market_overlay_multiplier(port.index, kospi, spec.get("market_overlay"), usdkrw=usdkrw)
    if spec.get("market_overlay"):
        port = port * overlay_mult
        market = bt._market_daily_returns(bench, port.index)
        invested = bt._invested_weight_korea(act_frames).reindex(port.index).fillna(0.0) * overlay_mult
        excess = port - invested * market
    else:
        excess = bt._excess_daily_korea(port, act_frames, bench)
    m = bt._metrics(all_trades, port, excess_daily=excess)
    m["_window"] = [lo, hi if hi != _HI_SENTINEL else "end"]
    m["_n_tickers"] = n_tk
    return m


def run_time_split_validation(
    spec: dict,
    tickers: list[str],
    *,
    loader: Callable[[str], tuple] = load_holdings,
    kospi_fetcher: Optional[Callable[[str, str], pd.DataFrame]] = fetch_kospi,
    usdkrw_fetcher: Optional[Callable[[str, str], pd.DataFrame]] = fetch_usdkrw,
    in_sample_pct: int = IN_SAMPLE_PCT,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    date_hi: Optional[str] = SCORE_DATE_HI,
) -> dict:
    """단일 시간 분할(IS=과거 / OOS=미래) 검증 + embargo 갭 + 시간 게이트.

    반환은 ``run_universe_backtest_v2`` 와 호환되는 형태:
    ``{in_sample, out_sample, gate_passed, universe_size, split}``.
    여기서 in_sample=IS(과거), out_sample=OOS(미래)다.
    """
    validate_spec_v2(spec)

    holdings: dict[str, pd.DataFrame] = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader(tk)
        if df is None or df.empty:
            continue
        if date_hi:                                  # OOS/IS 채점 상한 (CLAUDE.md 2025-06 룰)
            df = df[df["date"].astype(str) <= date_hi]
            if df.empty:
                continue
        holdings[tk] = df
        d0, d1 = str(df["date"].iloc[0]), str(df["date"].iloc[-1])
        min_d = d0 if (min_d is None or d0 < min_d) else min_d
        max_d = d1 if (max_d is None or d1 > max_d) else max_d

    kospi = None
    if holdings and kospi_fetcher is not None and min_d and max_d:
        try:
            kospi = kospi_fetcher(min_d, max_d)
        except Exception:        # noqa: BLE001
            kospi = None
    usdkrw = None
    needs_fx = spec.get("market_overlay", {}).get("type") == "usdkrw_trailing_return_ma_scale"
    if needs_fx and holdings and min_d and max_d and usdkrw_fetcher is not None:
        try:
            usdkrw = usdkrw_fetcher(min_d, max_d)
        except Exception:        # noqa: BLE001
            usdkrw = None

    dates = _universe_dates(holdings)
    if len(dates) < 10:
        raise ValueError("시간 분할에 거래일이 부족합니다.")
    cut_idx = int(len(dates) * in_sample_pct / 100)
    cutoff = dates[cut_idx]
    oos_idx = min(cut_idx + embargo_days, len(dates) - 1)
    oos_start = dates[oos_idx]

    is_m = _window_metrics(spec, holdings, kospi, dates[0], cutoff, usdkrw=usdkrw)
    oos_m = _window_metrics(spec, holdings, kospi, oos_start, _HI_SENTINEL, usdkrw=usdkrw)

    return {
        "in_sample": is_m,
        "out_sample": oos_m,
        "gate_passed": passes_full_gate_v2(is_m, oos_m),
        "universe_size": len(holdings),
        "split": {
            "axis": "time",
            "in_sample_pct": in_sample_pct,
            "cutoff": cutoff,
            "oos_start": oos_start,
            "embargo_days": embargo_days,
            "data_span": [min_d, max_d],
        },
    }


def run_walk_forward_validation(
    spec: dict,
    tickers: list[str],
    *,
    loader: Callable[[str], tuple] = load_holdings,
    kospi_fetcher: Optional[Callable[[str, str], pd.DataFrame]] = fetch_kospi,
    usdkrw_fetcher: Optional[Callable[[str, str], pd.DataFrame]] = fetch_usdkrw,
    is_years: float = 3.0,
    oos_years: float = 1.0,
    mode: str = "rolling",          # "rolling" | "anchored"
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    date_hi: Optional[str] = SCORE_DATE_HI,
    benchmark: str = "pool",        # "pool"=생존풀(신 게이트, 생존편향 중립) | "kospi"=구 게이트
) -> dict:
    """Walk-forward 검증 — 윈도우를 전진시키며 (IS→OOS) 사이클 반복.

    각 윈도우의 OOS 메트릭 분포를 집계한다. 게이트는 점추정이 아니라 **분포 기반**:
    모든 OOS 윈도우 sharpe>0(손실 윈도우 없음) AND OOS sharpe 중앙값 > 임계.
    ``mode='anchored'`` 면 IS 시작점을 고정(누적 확장), ``'rolling'`` 이면 고정 길이
    슬라이드. 두 모드 비교로 레짐 이동을 진단할 수 있다(rolling≫anchored → 구데이터 해악).
    """
    validate_spec_v2(spec)
    if mode not in ("rolling", "anchored"):
        raise ValueError(f"mode must be rolling|anchored, got {mode!r}")

    holdings: dict[str, pd.DataFrame] = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader(tk)
        if df is None or df.empty:
            continue
        if date_hi:                                  # OOS/IS 채점 상한 (CLAUDE.md 2025-06 룰)
            df = df[df["date"].astype(str) <= date_hi]
            if df.empty:
                continue
        holdings[tk] = df
        d0, d1 = str(df["date"].iloc[0]), str(df["date"].iloc[-1])
        min_d = d0 if (min_d is None or d0 < min_d) else min_d
        max_d = d1 if (max_d is None or d1 > max_d) else max_d
    kospi = None
    if holdings and kospi_fetcher is not None and min_d and max_d:
        try:
            kospi = kospi_fetcher(min_d, max_d)
        except Exception:        # noqa: BLE001
            kospi = None
    usdkrw = None
    needs_fx = spec.get("market_overlay", {}).get("type") == "usdkrw_trailing_return_ma_scale"
    if needs_fx and holdings and min_d and max_d and usdkrw_fetcher is not None:
        try:
            usdkrw = usdkrw_fetcher(min_d, max_d)
        except Exception:        # noqa: BLE001
            usdkrw = None

    if benchmark not in ("pool", "kospi"):
        raise ValueError(f"benchmark must be pool|kospi, got {benchmark!r}")
    bench_df = _survivor_pool_index(holdings) if benchmark == "pool" else None
    gate_min_ir = GATE_POOL_MIN_IR if benchmark == "pool" else GATE_EXCESS_MIN_IR

    dates = _universe_dates(holdings)
    is_len = int(is_years * bt.TRADING_DAYS_PER_YEAR)
    oos_len = int(oos_years * bt.TRADING_DAYS_PER_YEAR)
    step = oos_len
    windows = []
    is_start_idx = 0
    is_end_idx = is_len
    while is_end_idx + embargo_days + oos_len <= len(dates):
        oos_lo = is_end_idx + embargo_days
        oos_hi = oos_lo + oos_len
        windows.append((
            dates[is_start_idx], dates[is_end_idx],
            dates[oos_lo], dates[min(oos_hi, len(dates) - 1)],
        ))
        if mode == "rolling":
            is_start_idx += step
        is_end_idx += step

    results = []
    for (is_lo, is_hi, oos_lo, oos_hi) in windows:
        is_m = _window_metrics(spec, holdings, kospi, is_lo, is_hi, usdkrw=usdkrw, benchmark=bench_df)
        oos_m = _window_metrics(spec, holdings, kospi, oos_lo, oos_hi, usdkrw=usdkrw, benchmark=bench_df)
        results.append({
            "is_window": [is_lo, is_hi], "oos_window": [oos_lo, oos_hi],
            "in_sample": is_m, "out_sample": oos_m,
            "window_gate": passes_full_gate_v2(is_m, oos_m),
        })

    import statistics as _st

    # 채택 기준은 *시장 대비 초과수익 IR*(베타 제거 알파). raw sharpe 는 진단용으로만 함께 보고.
    oos_excess = [r["out_sample"].get("excess_sharpe") for r in results]
    valid_excess = [s for s in oos_excess if s is not None]
    oos_ex_median = round(_st.median(valid_excess), 4) if valid_excess else None
    oos_ex_min = round(min(valid_excess), 4) if valid_excess else None
    # 모든 OOS 윈도우가 시장을 이기고(IR>0) 중앙값이 임계 초과해야 통과.
    passed = bool(
        len(results) >= 2
        and len(valid_excess) == len(results)
        and oos_ex_min is not None and oos_ex_min > 0
        and oos_ex_median is not None and oos_ex_median > gate_min_ir
    )

    oos_sharpes = [r["out_sample"]["sharpe"] for r in results
                   if r["out_sample"]["sharpe"] is not None]
    oos_median = round(_st.median(oos_sharpes), 4) if oos_sharpes else None
    oos_min = round(min(oos_sharpes), 4) if oos_sharpes else None

    return {
        "mode": mode,
        "n_windows": len(results),
        "windows": results,
        "oos_excess_ir_median": oos_ex_median,
        "oos_excess_ir_min": oos_ex_min,
        "oos_sharpe_median": oos_median,    # 진단용(raw)
        "oos_sharpe_min": oos_min,          # 진단용(raw)
        "gate_passed": passed,
        "gate_metric": "excess_ir",
        "benchmark": benchmark,            # "pool"(생존편향 중립) | "kospi"(구)
        "gate_min_ir": gate_min_ir,
        "data_span": [min_d, max_d],
    }


# === 다중검정 가드 (False Strategy Theorem / MinBTL) ===
# Bailey, Borwein, López de Prado, Zhu (2014), "Pseudo-Mathematics and Financial
# Charlatanism", Notices of the AMS 61(5), Theorem 3.1. (워크플로우 검증: confirmed)
_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """참 skill=0 인 ``n_trials`` 개 독립 전략의 기대 최대 Sharpe(연율).

    이 값을 못 넘는 후보 Sharpe 는 순수 다중검정 운(noise)과 구별 불가하다.
    ``sr_std`` = 시행 Sharpe 들의 표준편차(연율). 기본 1.0 은 표준화 가정.
    """
    if n_trials < 2:
        return 0.0
    g = _EULER_GAMMA
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - g) * z1 + g * z2)


def max_justifiable_trials(data_years: float, target_sharpe: float = GATE_V2_MIN_SHARPE) -> int:
    """주어진 데이터 길이에서 ``target_sharpe`` 를 우연 아닌 것으로 볼 수 있는 시행 상한.

    MinBTL 상한식 ``MinBTL < 2·ln(N)/SR²`` 의 역산: ``N_max ≈ exp(T·SR²/2)``.
    예) 5년·SR=1.0 → ~12, 10년·SR=1.0 → ~148 (시행이 상관되면 유효 N 으로 더 엄격).
    """
    if target_sharpe <= 0 or data_years <= 0:
        return 0
    return int(math.exp(data_years * target_sharpe * target_sharpe / 2.0))


# === 엔진 버전(provenance) — 저장 메트릭 재현성 추적 ===
# 진단에서 확인된 문제: 엔진 in-place 수정 후 strategies_v2.db 저장값이 현재
# 코드로 재현 불가(예 #400 저장 in_sharpe 1.44 vs 재계산 0.87). 백테스트 결과를
# 결정하는 소스 파일들의 md5 단축본을 row 에 함께 저장해, 어떤 엔진이 만든
# 메트릭인지 식별 가능하게 한다.
def engine_version() -> str:
    """백테스트/검증 결과를 결정하는 소스의 단축 md5 (8자)."""
    import hashlib
    from tradingagents.hermes import backtest_engine_v2, strategy_spec_v2
    h = hashlib.md5()
    for mod in (bt, backtest_engine_v2, strategy_spec_v2, sys.modules[__name__]):
        try:
            with open(mod.__file__, "rb") as f:
                h.update(f.read())
        except (OSError, AttributeError):
            continue
    return h.hexdigest()[:8]
