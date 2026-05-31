"""트레일링 스톱(적응형 청산) 실험 — 기존 엔진 무수정, 신규 파일.

목적: 상위 near-miss 전략의 단일 병목인 MDD 를, 보유 중 *고점(peak) 대비 일정%
하락* 시 청산하는 트레일링 스톱으로 실제로 줄일 수 있는지 백테스트로 검증한다.
고정 일수(max_hold)·고정 손절(stop_loss)은 상황을 무시하는 "달력/절대" 청산인
반면, 트레일링 스톱은 수익을 따라 올라간 고점을 기준으로 후퇴분만 잘라 MDD 를
직접 겨냥한다.

기존 검증 자산 재사용 (수정 0):
  backtest_engine     : _num / _metrics / _combine / _attach_kospi / split_of /
                        TX_COST_ONE_WAY
  backtest_engine_v2  : _and_v2 / _attach_market_columns / passes_full_gate_v2 /
                        passes_single_gate_v2
  strategy_spec_v2    : validate_spec_v2

look-ahead 불변식 유지 (기존 _simulate_v2 와 동일):
  - 고점 peak 은 close[e..j] 로만 갱신 (미래 미참조).
  - 트레일링 트리거는 row j 종가로 판정하되 *체결은 close[j+1]*.
  - trail_pct=None 이면 트레일링 분기는 죽고 결과가 _simulate_v2 와 **완전 동일**
    (baseline 등가 — 본 모듈의 회귀 기준).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes import backtest_engine_v2 as v2
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2


def _simulate_trailing(
    spec: dict,
    df: pd.DataFrame,
    *,
    kospi: Optional[pd.DataFrame] = None,
    trail_pct: Optional[float],
    keep_fixed_sl: bool = True,
):
    """spec + 단일 종목 holdings → (trades, daily_ret, active, clean_df).

    ``_simulate_v2`` 와 동일하되 청산 우선순위에 트레일링 스톱을 끼운다:
      stop_loss > **trailing_stop** > take_profit > signal > max_hold.
    ``keep_fixed_sl=False`` 면 고정 손절을 끄고 트레일링만 하방 보호로 쓴다.
    """
    # close>0 + volume>0 — _simulate_v2 와 필터 정렬(거래량 0 유령행 제거).
    # trail_pct=None 등가(회귀 불변식) 유지: 입력 필터가 엔진과 동일해야 함.
    df = df[(bt._num(df["close"]) > 0) & (bt._num(df["volume"]) > 0)].reset_index(drop=True)
    df = v2._attach_market_columns(df, kospi)
    n = len(df)
    daily_ret = pd.Series(0.0, index=range(n))
    active = pd.Series(False, index=range(n))
    trades: list[dict] = []
    if n < 3:
        return trades, daily_ret, active, df

    entry_sig = v2._and_v2(df, spec["entry"]["all_of"]).to_numpy()
    exit_sig = v2._and_v2(df, spec["exit"].get("signal_all_of", [])).to_numpy()
    close = bt._num(df["close"]).to_numpy(dtype=float)
    dates = df["date"].astype(str).to_numpy()
    sl = spec["exit"].get("stop_loss_pct") if keep_fixed_sl else None
    tp = spec["exit"].get("take_profit_pct")
    mh = spec["exit"]["max_hold_days"]
    tx = bt.TX_COST_ONE_WAY
    trail = None if trail_pct is None else float(trail_pct)

    i = 0
    while i < n - 2:
        if not entry_sig[i]:
            i += 1
            continue
        e = i + 1
        entry_price = close[e]
        peak = entry_price                       # 보유 중 최고 종가 (trailing 기준)

        exit_idx: Optional[int] = None
        reason = None
        j = e
        while j < n:
            held = j - e
            cj = close[j]
            if cj > peak:                        # 고점 갱신 — close[e..j] 만 참조
                peak = cj
            ret = cj / entry_price - 1
            trig = None
            if sl is not None and ret <= -sl / 100.0:
                trig = "stop_loss"
            elif trail is not None and cj <= peak * (1 - trail / 100.0):
                trig = "trailing_stop"
            elif tp is not None and ret >= tp / 100.0:
                trig = "take_profit"
            elif exit_sig[j]:
                trig = "signal"
            elif held >= mh:
                trig = "max_hold"
            if trig is not None:
                if j + 1 < n:
                    exit_idx, reason = j + 1, trig
                else:
                    exit_idx, reason = j, "forced_eod"
                break
            j += 1
        if exit_idx is None:
            exit_idx, reason = n - 1, "forced_eod"

        exit_price = close[exit_idx]
        gross = exit_price / entry_price - 1
        net = (exit_price * (1 - tx)) / (entry_price * (1 + tx)) - 1
        trades.append({
            "entry_date": str(dates[e]),
            "exit_date": str(dates[exit_idx]),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_ret_pct": round(float(gross) * 100, 4),
            "net_ret_pct": round(float(net) * 100, 4),
            "hold_days": int(exit_idx - e),
            "exit_reason": reason,
        })

        for t in range(e + 1, exit_idx + 1):
            daily_ret.iloc[t] = close[t] / close[t - 1] - 1
            active.iloc[t] = True
        first = e + 1
        daily_ret.iloc[first] = (1 + daily_ret.iloc[first]) / (1 + tx) - 1
        daily_ret.iloc[exit_idx] = (1 + daily_ret.iloc[exit_idx]) * (1 - tx) - 1

        i = exit_idx + 1

    return trades, daily_ret, active, df


def run_universe_trailing(
    spec: dict,
    *,
    loaded: dict[str, pd.DataFrame],
    kospi: Optional[pd.DataFrame],
    trail_pct: Optional[float],
    keep_fixed_sl: bool = True,
) -> dict:
    """사전 캐시된 holdings/KOSPI 로 종목군 트레일링 백테스트.

    ``run_universe_backtest_v2`` 와 동일 구조지만 (1) 시뮬을 ``_simulate_trailing``
    으로 교체, (2) parquet/지수 I/O 를 호출자가 1회 preload 해 넘긴다(변형 반복시
    중복 로드 제거).
    """
    validate_spec_v2(spec)
    results: dict[str, dict] = {}
    for split in ("in", "out"):
        all_trades: list[dict] = []
        ret_frames: list[pd.Series] = []
        act_frames: list[pd.Series] = []
        for tk, df in loaded.items():
            if bt.split_of(tk) != split:
                continue
            trades, daily, active, cdf = _simulate_trailing(
                spec, df, kospi=kospi, trail_pct=trail_pct, keep_fixed_sl=keep_fixed_sl)
            bt._attach_kospi(trades, kospi)
            all_trades.extend(trades)
            idx = cdf["date"].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        results[split] = bt._metrics(all_trades, bt._combine(ret_frames, act_frames))

    in_m, out_m = results["in"], results["out"]
    return {
        "in_sample": in_m,
        "out_sample": out_m,
        "gate_passed": v2.passes_full_gate_v2(in_m, out_m),
    }
