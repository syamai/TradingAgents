"""파라미터화 수급 룰 전략의 결정론적 백테스트 엔진 (LLM-free).

순수 함수 모듈. ``run_backtest`` 는 단일 종목 holdings DataFrame 에 spec 을
적용해 trades + 메트릭을 산출하고, ``run_universe_backtest`` 는 종목군을 해시로
in/out-sample 분할해 각각 집계 + 채택 게이트를 판정한다.

look-ahead 차단 (타협 불가, 코드 구조로 보장):
  **모든 신호는 row i 까지의 데이터로만 평가하고, 체결은 row i+1 의 ``close``.**
  - 진입: 신호 row i (종가 확정) → entry_price = close[i+1].
  - 청산: 보유 중 매 봉 close[j] 로 손절/익절/신호/보유일 평가 → exit = close[j+1].
    (intrabar high/low 미사용 — OHLC 낙관 회피.)
  - rolling 은 전부 과거방향(trailing). ``shift(-k)`` 금지.
  - 단일 포지션(청산 전 재진입 금지). 데이터 끝 미청산분은 마지막 close 강제청산.

수익률·위험 메트릭:
  - per-trade net 수익률은 거래비용을 곱셈으로 반영:
    ``net = exit*(1-tx) / (entry*(1+tx)) - 1`` (tx=편도).
  - Sharpe/MDD/누적수익률은 종목별 일별 net 수익률을 *active 종목 동일가중*
    포트폴리오로 결합한 equity curve 에서 산출 (per-trade net 과 정합).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes.labeling import _close_at_or_after
from tradingagents.hermes.strategy_spec import validate_spec

# 거래비용 — 편도(수수료+세금 근사). entry/exit 각 레그에 곱셈 반영.
TX_COST_ONE_WAY = 0.0023
TRADING_DAYS_PER_YEAR = 252

# 종목 분할 — md5(code6) % 100 < IN_SAMPLE_PCT → in-sample.
IN_SAMPLE_PCT = 70

# 채택 게이트 (in/out 양쪽 충족 + 격차).
GATE_MIN_WIN_RATE = 0.60
GATE_MIN_SHARPE = 1.2
GATE_MAX_DRAWDOWN_PCT = -20.0      # mdd_pct >= 이 값 (드로다운이 더 얕아야)
GATE_MIN_TRADES = 50
GATE_MAX_WIN_RATE_GAP = 0.10       # |in.win_rate - out.win_rate|
SHARPE_MIN_TRADES = 2              # 미만이면 sharpe None (게이트 fail)

# 비정상 일중 점프 가드 — 분할/감자 미조정·극저유동 등으로 close 가 직전 거래일
# 대비 비현실적으로 튄 봉. 한국 일일 가격제한(±30%) 상 정상 연속 거래일엔 불가능한
# 배율이라, KIS 수정주가가 분할/감자를 조정해주는 정상 종목은 전부 마스크 False.
_SPLIT_JUMP_HI = 2.5
_SPLIT_JUMP_LO = 0.4


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _bad_bar_mask(close: np.ndarray) -> np.ndarray:
    """직전 거래일 대비 close 점프가 비정상(>HI 또는 <LO)인 봉 마스크.

    분할/감자 미조정·극저유동 데이터 아티팩트를 거래·수익 계상에서 제외하기 위함.
    index 0 은 기준봉이 없어 항상 False.
    """
    n = len(close)
    if n < 2:
        return np.zeros(n, dtype=bool)
    jump = np.ones(n)
    jump[1:] = close[1:] / close[:-1]
    return (jump > _SPLIT_JUMP_HI) | (jump < _SPLIT_JUMP_LO)


def _code6(ticker: str) -> str:
    return str(ticker).split(".")[0]


def split_of(ticker: str) -> str:
    """종목 → 'in' / 'out'. 6자리 코드 md5 해시 기반, 결정론적·재현 가능.

    ``005930`` 과 ``005930.KS`` 는 같은 split (suffix 무시).
    """
    import hashlib
    h = int(hashlib.md5(_code6(ticker).encode("utf-8")).hexdigest(), 16) % 100
    return "in" if h < IN_SAMPLE_PCT else "out"


# === 신호 평가 (벡터화, look-ahead 0) ===


def _eval_signal(df: pd.DataFrame, sig: dict) -> pd.Series:
    """단일 신호 → row 별 boolean Series (row i 까지 데이터로만 평가).

    NaN(초기 window 부족 등) 비교는 pandas 에서 False 로 떨어진다.
    """
    t = sig["signal"]

    if t == "net_streak":
        col = _num(df[f"{sig['subject']}_net_qty"]).fillna(0)
        pos = (col > 0) if sig["sign"] == "buy" else (col < 0)
        # 연속 run 길이 — cumcount 는 과거→현재만 누적 (미래 미참조).
        run = (pos != pos.shift()).cumsum()
        length = pos.groupby(run).cumcount() + 1
        streak = length.where(pos, 0)
        return streak >= sig["min_days"]

    if t == "pct_threshold":
        col = _num(df[f"{sig['subject']}_pct"])
        return (col >= sig["value"]) if sig["op"] == ">=" else (col <= sig["value"])

    if t == "pct_delta":
        col = _num(df[f"{sig['subject']}_pct"])
        delta = col - col.shift(sig["window"])   # shift>0 = 과거 참조
        return (delta >= sig["value"]) if sig["op"] == ">=" else (delta <= sig["value"])

    if t == "net_vol_ratio":
        nq = _num(df[f"{sig['subject']}_net_qty"]).fillna(0)
        vol = _num(df["volume"])
        num = nq.rolling(sig["window"]).sum()       # trailing
        den = vol.rolling(sig["window"]).sum()
        ratio = num / den.where(den > 0)            # den<=0 → NaN → False
        return (ratio >= sig["value"]) if sig["op"] == ">=" else (ratio <= sig["value"])

    if t == "price_filter":
        close = _num(df["close"])
        ma = close.rolling(sig["window"]).mean()    # trailing
        return (close > ma) if sig["mode"] == "above_ma" else (close < ma)

    raise ValueError(f"unknown signal type: {t!r}")


def _and(df: pd.DataFrame, signals: list) -> pd.Series:
    """신호 AND 결합. 빈 리스트 → 전부 False."""
    if not signals:
        return pd.Series(False, index=df.index)
    out = pd.Series(True, index=df.index)
    for s in signals:
        out = out & _eval_signal(df, s).fillna(False)
    return out


# === 시뮬레이션 ===


def _simulate(spec: dict, df: pd.DataFrame):
    """spec + 단일 종목 holdings → (trades, daily_ret, active, clean_df).

    daily_ret/active 는 clean_df(close/volume 0·결측 제거 후) index 기준 일별
    시계열 — 종목별 net 수익률(보유일만 비0)과 보유 여부. 포트폴리오 결합에 사용.
    """
    # close>0 만으로는 거래량 0 유령행(비유동·거래정지 구간 기준가 노이즈)이
    # 안 걸러져 진입/청산가를 오염시킴 — volume>0 도 함께 요구.
    df = df[(_num(df["close"]) > 0) & (_num(df["volume"]) > 0)].reset_index(drop=True)
    n = len(df)
    daily_ret = pd.Series(0.0, index=range(n))
    active = pd.Series(False, index=range(n))
    trades: list[dict] = []
    if n < 3:   # 진입(i+1) 후 최소 1 거래일 보유 불가
        return trades, daily_ret, active, df

    entry_sig = _and(df, spec["entry"]["all_of"]).to_numpy()
    exit_sig = _and(df, spec["exit"].get("signal_all_of", [])).to_numpy()
    close = _num(df["close"]).to_numpy(dtype=float)
    bad_bar = _bad_bar_mask(close)       # 분할/감자 미조정 등 비정상 점프 봉 — 거래 제외
    dates = df["date"].astype(str).to_numpy()
    sl = spec["exit"].get("stop_loss_pct")
    tp = spec["exit"].get("take_profit_pct")
    mh = spec["exit"]["max_hold_days"]
    tx = TX_COST_ONE_WAY

    i = 0
    while i < n - 2:                      # e=i+1 <= n-2 → 청산 봉(>=e+1) 보장
        if not entry_sig[i] or bad_bar[i + 1]:   # 진입 봉(i+1)이 비정상 점프면 보류
            i += 1
            continue
        e = i + 1
        entry_price = close[e]

        exit_idx: Optional[int] = None
        reason = None
        j = e
        while j < n:
            if bad_bar[j]:               # 비정상 점프 봉 — 트리거 평가·청산 보류
                j += 1
                continue
            held = j - e
            ret = close[j] / entry_price - 1
            trig = None
            if sl is not None and ret <= -sl / 100.0:
                trig = "stop_loss"
            elif tp is not None and ret >= tp / 100.0:
                trig = "take_profit"
            elif exit_sig[j]:
                trig = "signal"
            elif held >= mh:
                trig = "max_hold"
            if trig is not None:
                k = j + 1
                while k < n and bad_bar[k]:   # 청산 봉도 비정상 점프 회피
                    k += 1
                if k < n:
                    exit_idx, reason = k, trig
                else:
                    exit_idx, reason = j, "forced_eod"   # 정상 청산 봉 없음
                break
            j += 1
        if exit_idx is None:
            last = n - 1
            while last > e and bad_bar[last]:    # 강제청산도 정상 봉에서
                last -= 1
            exit_idx, reason = last, "forced_eod"

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

        # 보유 구간 (e+1 .. exit_idx) 일별 net 수익률 + active.
        for t in range(e + 1, exit_idx + 1):
            # 비정상 점프 봉(및 그 직후 복귀 봉)은 가짜 수익이라 미계상.
            daily_ret.iloc[t] = 0.0 if bad_bar[t] else close[t] / close[t - 1] - 1
            active.iloc[t] = True
        # 거래비용 곱셈 반영 — 진입 레그는 첫 보유일에, 청산 레그는 청산일에.
        first = e + 1
        daily_ret.iloc[first] = (1 + daily_ret.iloc[first]) / (1 + tx) - 1
        daily_ret.iloc[exit_idx] = (1 + daily_ret.iloc[exit_idx]) * (1 - tx) - 1

        i = exit_idx + 1                 # 단일 포지션 — 청산 봉 이후 재진입

    return trades, daily_ret, active, df


def _attach_kospi(trades: list, kospi: Optional[pd.DataFrame]) -> None:
    """각 trade 에 보유구간 KOSPI 수익·초과수익(net - kospi) 부여. in-place.

    kospi 결손 시 None. 휴장일은 ``_close_at_or_after`` 로 직후 거래일 보정.
    """
    for tr in trades:
        if kospi is None or len(kospi) == 0:
            tr["kospi_ret_pct"] = None
            tr["excess_ret_pct"] = None
            continue
        sk = _close_at_or_after(kospi, tr["entry_date"])
        ek = _close_at_or_after(kospi, tr["exit_date"])
        if not sk or not ek or sk["close"] <= 0:
            tr["kospi_ret_pct"] = None
            tr["excess_ret_pct"] = None
            continue
        kr = (ek["close"] / sk["close"] - 1) * 100
        tr["kospi_ret_pct"] = round(kr, 4)
        tr["excess_ret_pct"] = round(tr["net_ret_pct"] - kr, 4)


# === 메트릭 ===


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())


def _metrics(trades: list, port_daily: pd.Series) -> dict:
    """trades(pooled) + 포트폴리오 일별 수익률 → 메트릭 dict (JSON 직렬화 가능)."""
    n = len(trades)
    empty = {
        "n_trades": 0, "win_rate": None, "avg_net_ret_pct": None,
        "avg_hold_days": None, "cum_return_pct": 0.0, "sharpe": None,
        "mdd_pct": 0.0, "avg_excess_ret_pct": None,
    }
    if n == 0:
        return empty

    nets = [t["net_ret_pct"] for t in trades]
    wins = sum(1 for x in nets if x > 0)
    excesses = [t["excess_ret_pct"] for t in trades if t.get("excess_ret_pct") is not None]

    r = port_daily.to_numpy(dtype=float)
    equity = np.cumprod(1.0 + r) if len(r) else np.array([])
    cum = float(equity[-1] - 1) if len(equity) else 0.0

    sharpe = None
    if n >= SHARPE_MIN_TRADES and len(r) and r.std(ddof=0) > 0:
        sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))

    return {
        "n_trades": n,
        "win_rate": round(wins / n, 4),
        "avg_net_ret_pct": round(sum(nets) / n, 4),
        "avg_hold_days": round(sum(t["hold_days"] for t in trades) / n, 2),
        "cum_return_pct": round(cum * 100, 4),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "mdd_pct": round(_max_drawdown(equity) * 100, 4),
        "avg_excess_ret_pct": round(sum(excesses) / len(excesses), 4) if excesses else None,
    }


def passes_single_gate(m: dict) -> bool:
    """한 split 의 메트릭이 게이트 통과인가."""
    return (
        m["n_trades"] >= GATE_MIN_TRADES
        and m["win_rate"] is not None and m["win_rate"] >= GATE_MIN_WIN_RATE
        and m["sharpe"] is not None and m["sharpe"] >= GATE_MIN_SHARPE
        and m["mdd_pct"] >= GATE_MAX_DRAWDOWN_PCT
    )


def passes_full_gate(in_m: dict, out_m: dict) -> bool:
    """in/out 양쪽 게이트 + 승률 격차 조건 (과적합 탐지)."""
    if not (passes_single_gate(in_m) and passes_single_gate(out_m)):
        return False
    if in_m["win_rate"] is None or out_m["win_rate"] is None:
        return False
    return abs(in_m["win_rate"] - out_m["win_rate"]) <= GATE_MAX_WIN_RATE_GAP


# === 공개 진입점 ===


def run_backtest(
    spec: dict, df: pd.DataFrame, *, kospi: Optional[pd.DataFrame] = None,
) -> dict:
    """단일 종목 백테스트. 반환 ``{"metrics": {...}, "trades": [...]}``."""
    validate_spec(spec)
    trades, daily, _active, _cdf = _simulate(spec, df)
    _attach_kospi(trades, kospi)
    return {"metrics": _metrics(trades, daily), "trades": trades}


def _combine(ret_frames: list, act_frames: list) -> pd.Series:
    """종목별 일별 net 수익률 → active 종목 동일가중 포트폴리오 일별 수익률.

    날짜 union 으로 정렬. 그날 active 한 종목들의 net 수익률 평균(없으면 0).
    """
    if not ret_frames:
        return pd.Series(dtype=float)
    R = pd.concat(ret_frames, axis=1).sort_index()
    # union 정렬 시 결측은 NaN — ``== True`` 로 bool 변환(NaN→False). fillna
    # 다운캐스팅 deprecation 회피.
    A = (pd.concat(act_frames, axis=1).sort_index() == True)  # noqa: E712
    R = R.where(A, 0.0)
    count = A.sum(axis=1)
    port = R.sum(axis=1) / count.where(count > 0)
    return port.fillna(0.0)


def run_universe_backtest(
    spec: dict,
    tickers: list[str],
    *,
    loader: Callable[[str], tuple] = load_holdings,
    kospi_fetcher: Optional[Callable[[str, str], pd.DataFrame]] = fetch_kospi,
) -> dict:
    """종목군 백테스트 — in/out-sample 분할 집계 + 채택 게이트.

    반환: ``{in_sample, out_sample, gate_passed, universe_size, n_in, n_out}``.
    ``loader(ticker) -> (df, meta)``, ``kospi_fetcher(start, end) -> df`` 는
    테스트에서 fake 주입 가능 (LabelingScheduler 의 fetcher 주입 패턴).
    """
    validate_spec(spec)

    loaded: dict[str, pd.DataFrame] = {}
    min_d = max_d = None
    for tk in tickers:
        df, _meta = loader(tk)
        if df is None or df.empty:
            continue
        loaded[tk] = df
        d0, d1 = str(df["date"].iloc[0]), str(df["date"].iloc[-1])
        min_d = d0 if (min_d is None or d0 < min_d) else min_d
        max_d = d1 if (max_d is None or d1 > max_d) else max_d

    kospi = None
    if loaded and kospi_fetcher is not None and min_d and max_d:
        try:
            kospi = kospi_fetcher(min_d, max_d)
        except Exception:        # noqa: BLE001 — 지수 fetch 실패 시 초과수익만 결손
            kospi = None

    results: dict[str, dict] = {}
    for split in ("in", "out"):
        all_trades: list[dict] = []
        ret_frames: list[pd.Series] = []
        act_frames: list[pd.Series] = []
        for tk, df in loaded.items():
            if split_of(tk) != split:
                continue
            trades, daily, active, cdf = _simulate(spec, df)
            _attach_kospi(trades, kospi)
            all_trades.extend(trades)
            idx = cdf["date"].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        results[split] = _metrics(all_trades, _combine(ret_frames, act_frames))

    in_m, out_m = results["in"], results["out"]
    return {
        "in_sample": in_m,
        "out_sample": out_m,
        "gate_passed": passes_full_gate(in_m, out_m),
        "universe_size": len(loaded),
        "n_in": sum(1 for tk in loaded if split_of(tk) == "in"),
        "n_out": sum(1 for tk in loaded if split_of(tk) == "out"),
    }
