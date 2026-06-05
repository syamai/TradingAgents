"""단타·단기보유(1~수일) 횡단면 신호 백테스트 — swing V2(xsec/spec) 와 별개 트랙.

진짜 스캘핑(초~분)은 분봉·틱이 없어 불가. 여기서 다루는 건 **일봉으로 검증 가능한
단타급** 전략 — 보유 1~수일, 우리 차별데이터(투자자별 일별 순매수·일별 공매도비중)를
직접 입력으로 쓰는 횡단면 랭킹. 학술근거·매핑은 vault ``단타·스캘핑 전략 논문`` 참조.

look-ahead 0 규약: 신호는 종가 확정 후 알려지는 값(수급/공매도/overnight)이므로 row r
(=날짜 t)에서 신호 → **진입가 open[r+1]**, 청산가 open[r+1+rb]. 종가(MOC) 진입은 종가
확정 후 체결 불가라 비현실적이므로 시가-시가 보유로 검증한다(Tier A 노트의 체결편의 경고).

신호(전부 trailing, 높을수록 long 선호):
    short_low      저공매도압력 컨트래리언 = −trailing W 평균 short_volume_ratio.
                   공매도 전면금지 구간(SHORT_BAN_WINDOWS)은 신호 단절 → 해당 리밸런스 제외.
    flow_foreign   외국인 순매수의 거래대금정규화 (Kang 2025: 외국인은 dollar-volume 정규화).
    flow_smart_dumb 외국인−개인 순매수(smart−dumb)의 거래대금정규화 (Ryu 2012 / NAT 2026).
    overnight_tow  과거 W일 overnight(open/전일close−1) 누적 = Tug-of-War (Lou-Polk-Skouras 2019).
    reversal       단기 가격반전 = −과거 W일 수익(롱=패자). skip 으로 bid-ask bounce 완화.

공통 통제(노트 §5): net 왕복비용(2×TX_COST_ONE_WAY, 기본 편도 0.015%) 차감, ETF 제외,
유동성 하위 제외(저유동·저가 bounce 완화), 채점 상한 2025-06-30, 연도별 시장초과 IR 게이트.
"""
from __future__ import annotations

import statistics as _st

import numpy as np
import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.xsec_backtest import SCORE_DATE_HI, GATE_EXCESS_MIN_IR
from tradingagents.hermes.forward_test import ETF_CODES

# 공매도 전면금지 구간(한국) — short_low 신호는 이 구간에 구조적으로 단절되므로 제외.
SHORT_BAN_WINDOWS = (("2020-03-16", "2021-05-02"), ("2023-11-06", "2025-03-30"))

SIGNAL_KINDS = ("short_low", "flow_foreign", "flow_smart_dumb", "overnight_tow", "reversal")


def _signal_series(df: pd.DataFrame, kind: str, window: int, skip: int) -> pd.Series:
    """종목 1개의 trailing 신호 (높을수록 long 선호, look-ahead 0).

    skip>0 은 신호를 추가로 ``skip`` 일 지연시킨다 — (a) 단기반전·overnight 의
    bid-ask bounce 완화(직전 봉 제외), (b) **공매도/수급 공시지연 보정**(KRX 일별
    공매도는 통상 T+1 공시 → skip=1 로 진입 전 확정 데이터만 사용).
    """
    close = bt._num(df["close"])
    if kind == "short_low":
        if "short_volume_ratio" not in df.columns:
            return pd.Series(np.nan, index=df.index)
        sig = -bt._num(df["short_volume_ratio"]).rolling(window).mean()
    elif kind in ("flow_foreign", "flow_smart_dumb"):
        dv = (close * bt._num(df["volume"])).rolling(window).sum()
        if kind == "flow_foreign":
            net = bt._num(df["foreign_net_qty"]) * close
        else:
            net = (bt._num(df["foreign_net_qty"]) - bt._num(df["retail_net_qty"])) * close
        sig = net.rolling(window).sum() / dv.replace(0, np.nan)
    elif kind == "overnight_tow":
        sig = (bt._num(df["open"]) / close.shift(1) - 1.0).rolling(window).sum()
    elif kind == "reversal":
        sig = -(close / close.shift(window) - 1.0)
    else:
        raise ValueError(f"unknown signal kind: {kind!r}")
    return sig.shift(skip) if skip else sig


def _wide(holdings: dict, kind: str, window: int, skip: int, liq_trailing: int = 0):
    """종목별 open/close/유동성/signal 을 공통 날짜 캘린더(union) wide 행렬로.

    liq_trailing>0 이면 거래대금을 trailing W일 평균으로(xsec 와 동일), 0 이면 당일값.
    """
    opens, closes, vols, sigs = {}, {}, {}, {}
    for tk, df in holdings.items():
        idx = df["date"].astype(str).to_numpy()
        def _ser(vals):
            s = pd.Series(np.asarray(vals), index=idx)
            return s[~s.index.duplicated()]
        dv = bt._num(df["close"]) * bt._num(df["volume"])
        if liq_trailing > 0:
            dv = dv.rolling(liq_trailing).mean()
        opens[tk] = _ser(bt._num(df["open"]).to_numpy())
        closes[tk] = _ser(bt._num(df["close"]).to_numpy())
        vols[tk] = _ser(dv.to_numpy())
        sigs[tk] = _ser(_signal_series(df, kind, window, skip).to_numpy())
    open_m = pd.DataFrame(opens).sort_index()
    close_m = pd.DataFrame(closes).reindex(open_m.index)
    liq_m = pd.DataFrame(vols).reindex(open_m.index)
    sig_m = pd.DataFrame(sigs).reindex(open_m.index)
    return open_m, close_m, liq_m, sig_m


def _in_ban(date: str) -> bool:
    return any(lo <= date <= hi for lo, hi in SHORT_BAN_WINDOWS)


def run_daytrade_backtest(holdings: dict, kospi, *, kind: str, window: int = 5,
                          rebalance: int = 1, quantile: float = 0.2, skip: int = 0,
                          min_liq_pct: float = 0.3, cost_one_way: float = bt.TX_COST_ONE_WAY,
                          price: str = "open", exclude_ban: bool = True,
                          liq_trailing: int = 20,
                          date_lo: str | None = None,
                          date_hi: str | None = SCORE_DATE_HI) -> dict:
    """단타 횡단면 랭킹 백테스트 (시가-시가 보유 기본, net 비용). 반환: 메트릭 dict.

    진입 price[r+1], 청산 price[r+1+rebalance]. 신호 상위 quantile 동일가중 long.
    유동성 하위 min_liq_pct 제외(저유동 bid-ask bounce 완화).

    reconcile 토글: price('open'|'close'), exclude_ban(공매도 금지구간 제외),
    liq_trailing(거래대금 측정창; 기본 20=trailing 평균으로 xsec 와 정합. 0=당일값은
    worst-year 를 인위적으로 좋게 보이는 노이즈라 진단용으로만).
    """
    if kind not in SIGNAL_KINDS:
        raise ValueError(f"unknown kind: {kind}")
    if price not in ("open", "close"):
        raise ValueError("price must be open|close")
    open_m, close_m, liq_m, sig_m = _wide(holdings, kind, window, skip, liq_trailing)
    if date_lo is not None:
        m = open_m.index >= date_lo
        open_m, close_m, liq_m, sig_m = open_m[m], close_m[m], liq_m[m], sig_m[m]
    if date_hi is not None:
        m = open_m.index <= date_hi
        open_m, close_m, liq_m, sig_m = open_m[m], close_m[m], liq_m[m], sig_m[m]
    px_m = open_m if price == "open" else close_m
    dates = list(open_m.index)
    kseries = None
    if kospi is not None:
        kk = kospi.copy()
        ks = pd.Series(bt._num(kk["close"]).to_numpy(), index=kk["date"].astype(str).to_numpy())
        kseries = ks[~ks.index.duplicated()].reindex(dates)

    per_ret, per_ex, per_year, n_pick = [], [], [], []
    rt_cost = 2 * cost_one_way
    start = window + skip + 2
    for r in range(start, len(dates) - rebalance - 1, rebalance):
        if exclude_ban and kind == "short_low" and _in_ban(dates[r]):
            continue
        sig = sig_m.iloc[r]
        entry, exit_ = px_m.iloc[r + 1], px_m.iloc[r + 1 + rebalance]
        elig = sig.notna() & entry.notna() & exit_.notna() & (entry > 0)
        names = sig[elig]
        if min_liq_pct > 0 and len(names) >= 10:
            liq_row = liq_m.iloc[r][names.index]
            keep = (liq_row.rank(pct=True) >= min_liq_pct).reindex(names.index).fillna(False)
            names = names[keep]
        if len(names) < 10:
            continue
        k = max(1, int(round(len(names) * quantile)))
        picks = names.nlargest(k).index
        rets = (exit_[picks] / entry[picks] - 1.0) - rt_cost
        pr = float(rets.mean())
        per_ret.append(pr)
        n_pick.append(len(picks))
        per_year.append(dates[r + 1][:4])
        if kseries is not None and pd.notna(kseries.iloc[r + 1]) and pd.notna(kseries.iloc[r + 1 + rebalance]):
            mkt = float(kseries.iloc[r + 1 + rebalance]) / float(kseries.iloc[r + 1]) - 1.0
            per_ex.append(pr - mkt)
        else:
            per_ex.append(float("nan"))

    n = len(per_ret)
    if n < 4:
        return {"kind": kind, "gate_passed": False, "n_periods": n, "reason": "too few periods"}

    ppy = 252.0 / rebalance
    arr = np.array(per_ret, float)
    eq = np.cumprod(1 + arr)
    sharpe = float(arr.mean() / arr.std() * (ppy ** 0.5)) if arr.std() > 0 else None
    cum = float((eq[-1] - 1) * 100)
    ex = np.array([e for e in per_ex if not np.isnan(e)], float)
    excess_ir = float(ex.mean() / ex.std() * (ppy ** 0.5)) if len(ex) > 2 and ex.std() > 0 else None

    by_year = {}
    for y in sorted(set(per_year)):
        ys = np.array([per_ex[i] for i in range(n) if per_year[i] == y and not np.isnan(per_ex[i])], float)
        if len(ys) >= 4 and ys.std() > 0:
            by_year[y] = round(float(ys.mean() / ys.std() * (ppy ** 0.5)), 3)
    yr_irs = list(by_year.values())
    gate = bool(len(yr_irs) >= 3 and min(yr_irs) > 0 and _st.median(yr_irs) > GATE_EXCESS_MIN_IR)

    return {
        "kind": kind, "window": window, "rebalance": rebalance, "quantile": quantile, "skip": skip,
        "n_periods": n, "avg_picks": round(float(np.mean(n_pick)), 1),
        "sharpe": None if sharpe is None else round(sharpe, 3),
        "cum_return_pct": round(cum, 1),
        "excess_ir": None if excess_ir is None else round(excess_ir, 3),
        "by_year_excess_ir": by_year,
        "yr_ir_median": round(_st.median(yr_irs), 3) if yr_irs else None,
        "yr_ir_min": round(min(yr_irs), 3) if yr_irs else None,
        "gate_passed": gate,
    }


def _load():
    """수급 유니버스 로드 + ETF 제외(stock 단타 전략 기준)."""
    tickers, loader, kf, uf = _preload(KisHistoryStore().list_tickers())
    H = {tk: loader(tk)[0] for tk in tickers if str(tk)[:6] not in ETF_CODES}
    return H, kf(None, None)


# 스캔 그리드 — Tier A 전략(우리 차별데이터) 위주 + reversal 대조.
_GRID = [
    ("short_low", dict(window=5, rebalance=5, skip=0)),
    ("short_low", dict(window=5, rebalance=1, skip=0)),
    ("short_low", dict(window=10, rebalance=5, skip=0)),
    ("flow_foreign", dict(window=5, rebalance=1, skip=0)),
    ("flow_foreign", dict(window=5, rebalance=5, skip=0)),
    ("flow_foreign", dict(window=10, rebalance=5, skip=0)),
    ("flow_smart_dumb", dict(window=5, rebalance=1, skip=0)),
    ("flow_smart_dumb", dict(window=5, rebalance=5, skip=0)),
    ("flow_smart_dumb", dict(window=10, rebalance=5, skip=0)),
    ("overnight_tow", dict(window=20, rebalance=5, skip=0)),
    ("overnight_tow", dict(window=60, rebalance=20, skip=0)),
    ("reversal", dict(window=5, rebalance=5, skip=1)),
    ("reversal", dict(window=20, rebalance=20, skip=1)),
]


def main() -> int:
    print(f"[daytrade] preloading universe (cutoff ≤ {SCORE_DATE_HI}, ETF 제외, net 비용) ...", flush=True)
    H, K = _load()
    print(f"[daytrade] universe={len(H)} 종목\n", flush=True)
    rows = []
    for kind, kw in _GRID:
        r = run_daytrade_backtest(H, K, kind=kind, **kw)
        rows.append(r)
        tag = f"{kind} W{r.get('window')} rb{r.get('rebalance')} skip{r.get('skip')}"
        print(f"[{'PASS' if r.get('gate_passed') else 'fail'}] {tag:34s} "
              f"exIR={r.get('excess_ir')} ir_med={r.get('yr_ir_median')} ir_min={r.get('yr_ir_min')} "
              f"sharpe={r.get('sharpe')} cum={r.get('cum_return_pct')}% n={r.get('n_periods')} "
              f"picks={r.get('avg_picks')}", flush=True)
    print()
    passed = [r for r in rows if r.get("gate_passed")]
    print(f"=== 게이트 통과 {len(passed)}/{len(rows)} (fair gate: 모든 연도 시장초과 IR>0 AND 중앙>0.5) ===")
    for r in sorted(rows, key=lambda x: (x.get("excess_ir") or -9), reverse=True)[:5]:
        print(f"  {r['kind']} W{r.get('window')} rb{r.get('rebalance')}: exIR={r.get('excess_ir')} "
              f"by_year={r.get('by_year_excess_ir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
