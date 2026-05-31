"""레짐 게이트(시장 추세 진입 필터) 실험 — 기존 엔진 무수정, 신규 파일.

가설: 상위 near-miss 전략의 MDD 는 *폭락장에 진입한* 거래들이 동시에 물려
생긴다. KOSPI 가 추세 하락(N일 이평 아래)인 동안 **신규 진입을 전면 차단**하면
포트폴리오 MDD 가 줄어드는가? (이미 보유 중인 포지션은 그대로 — 진입 필터다.)

구현: 전략의 ``entry.all_of`` 에 ``market_filter(above_ma, window)`` 신호를 AND
로 덧붙인 뒤 기존 v2 시뮬(``_simulate_v2``)을 그대로 호출한다. market_filter
평가·KOSPI 결합(``_attach_market_columns``, merge_asof backward)은 검증·테스트
끝난 기존 자산이라 look-ahead 0 이 보장된다. ``validate_spec_v2`` 의 entry≤3 캡
과 MA 그리드를 우회하기 위해 ``_simulate_v2`` 를 직접 호출한다(엔진 자체는
spec 을 검증하지 않음).
"""
from __future__ import annotations

import copy
from typing import Optional

import pandas as pd

from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes import backtest_engine_v2 as v2


def augment_with_regime(spec: dict, *, window: int, mode: str = "above_ma") -> dict:
    """entry.all_of 에 market_filter 레짐 신호를 AND 로 덧붙인 새 spec(deepcopy)."""
    out = copy.deepcopy(spec)
    out["entry"]["all_of"].append(
        {"signal": "market_filter", "mode": mode, "window": window})
    return out


def run_universe_regime(
    spec: dict,
    *,
    loaded: dict[str, pd.DataFrame],
    kospi: Optional[pd.DataFrame],
    regime_window: Optional[int],
    mode: str = "above_ma",
) -> dict:
    """사전 캐시된 holdings/KOSPI 로 레짐 게이트 종목군 백테스트.

    ``regime_window=None`` 이면 게이트 없이 원 spec 그대로(baseline). 그 외에는
    ``market_filter(mode, regime_window)`` 를 entry 에 AND 로 추가해 평가한다.
    """
    aug = (augment_with_regime(spec, window=regime_window, mode=mode)
           if regime_window else spec)
    results: dict[str, dict] = {}
    for split in ("in", "out"):
        all_trades: list[dict] = []
        ret_frames: list[pd.Series] = []
        act_frames: list[pd.Series] = []
        for tk, df in loaded.items():
            if bt.split_of(tk) != split:
                continue
            trades, daily, active, cdf = v2._simulate_v2(aug, df, kospi=kospi)
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
