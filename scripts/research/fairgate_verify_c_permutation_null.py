"""테스트 C — 순열/랜덤-진입 null.

신호에 정보가 없을 때(결정론적 의사난수 진입) 공정게이트가 얼마나 통과시키는지의
경험 분포를 만들어, 관측(strict 18/1005, conc median IR~1.0)과 비교한다.

설계:
- _preload(1회)로 199종목 holdings + KOSPI 캐시.
- conc 11개의 실제 entry 진입 발생률(p_conc)을 측정 → 거래 빈도 정렬 기준.
- 랜덤-진입 null: entry 를 ticker6+date+seed 결정론적 md5 해시 < threshold 마스크로
  대체(정의상 가격/수급과 무상관). exit/보유는 conc 와 동일(TP/SL/MH 그리드).
- _and_v2 를 monkeypatch: entry 신호 리스트(비어있지 않음)면 해시 마스크 반환,
  빈 리스트(conc exit signal_all_of 없음)면 원본대로 전부 False.
- ticker 식별: holdings df 에 '_nullc_tk6' 컬럼을 미리 주입 → 패치가 그 값으로 해시.
- 각 null 을 run_walk_forward_validation(동일 KOSPI 벤치·동일 정책)으로 평가.
- strict 통과 비율 + oos_excess_ir_median 분포(p50/p90/p99/max) 산출.

난수 불가 → 종목코드·날짜·seed 기반 결정론적 md5 해시로 의사난수 마스크 생성.
"""
from __future__ import annotations

import hashlib
import statistics as st
import time

import numpy as np
import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes import backtest_engine_v2 as btv2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_validation import (
    engine_version,
    run_walk_forward_validation,
)

TIME_BUDGET_S = 12 * 60   # ~12분 예산
TK_COL = "_nullc_tk6"     # 패치된 _and_v2 가 종목코드를 읽을 임시 컬럼

# conc 가족 exit 그리드 (tp8/10, sl8, h20). 진입 빈도는 entry 마스크 p 로 맞추고,
# exit/보유는 conc 변형들을 순환 사용해 보유기간 분포까지 정렬한다.
CONC_EXITS = [
    {"take_profit_pct": 8.0, "stop_loss_pct": 8.0, "max_hold_days": 20},
    {"take_profit_pct": 10.0, "stop_loss_pct": 8.0, "max_hold_days": 20},
]

# conc 11개 합격 entry — 실제 진입 발생률 측정용 (subject, min_days, lookback, min_r).
CONC_ENTRIES = [
    ("foreign", 5, 60, 0.3), ("foreign_registered", 3, 60, 0.2),
    ("foreign_registered", 3, 60, 0.2), ("foreign_registered", 3, 60, 0.3),
    ("foreign_registered", 3, 60, 0.3), ("foreign_registered", 5, 60, 0.2),
    ("foreign_registered", 5, 60, 0.3), ("foreign", 3, 60, 0.2),
    ("foreign", 3, 60, 0.2), ("foreign", 3, 60, 0.3), ("foreign", 3, 60, 0.3),
]


def _conc_entry_signals(subj, md, lb, mr):
    return [
        {"signal": "net_streak", "subject": subj, "min_days": md, "sign": "buy"},
        {"signal": "rolling_corr", "subject": subj, "lookback": lb, "min_r": mr},
        {"signal": "price_filter", "mode": "above_ma", "window": 20},
    ]


_ORIG_AND_V2 = btv2._and_v2


def _hash_mask(df: pd.DataFrame, ticker6: str, seed: str, threshold: int) -> pd.Series:
    """ticker6+date+seed 결정론적 md5 → [0,1000) 정수 < threshold 인 row True.

    가격/수급과 무상관(정의상 정보 0). threshold/1000 ≈ 진입 발생 확률.
    """
    dates = df["date"].astype(str).to_numpy()
    vals = np.empty(len(dates), dtype=np.int64)
    for i, d in enumerate(dates):
        h = hashlib.md5(f"{seed}|{ticker6}|{d}".encode()).digest()
        vals[i] = int.from_bytes(h[:4], "big") % 1000
    return pd.Series(vals < threshold, index=df.index)


def measure_conc_entry_rate(holdings: dict[str, pd.DataFrame]) -> float:
    """conc 11개 entry 의 평균 진입 발생률(신호=True row 비율) 측정.

    _simulate_v2 와 동일하게 close>0 & volume>0 필터 후 _and_v2(entry) True 비율을
    종목·전략별로 집계해 전체 평균을 낸다.
    """
    rates = []
    for subj, md, lb, mr in CONC_ENTRIES:
        sigs = _conc_entry_signals(subj, md, lb, mr)
        for df in holdings.values():
            d = df[(bt._num(df["close"]) > 0) & (bt._num(df["volume"]) > 0)]
            if len(d) < 3:
                continue
            d = d.reset_index(drop=True)
            mask = _ORIG_AND_V2(d, sigs)
            rates.append(float(mask.mean()))
    return float(np.mean(rates)) if rates else 0.0


def main() -> int:
    t0 = time.time()
    raw = KisHistoryStore().list_tickers()
    print(f"[testC] preloading {len(raw)} tickers ...", flush=True)
    tickers, loader, kospi_fetcher, usdkrw_fetcher = _preload(raw)

    # holdings 에 종목코드 컬럼을 주입해 패치된 _and_v2 가 ticker 를 알 수 있게 한다.
    holdings_raw = {tk: loader(tk)[0] for tk in tickers}
    holdings_tagged: dict[str, pd.DataFrame] = {}
    for tk, df in holdings_raw.items():
        d = df.copy()
        d[TK_COL] = bt._code6(tk)
        holdings_tagged[tk] = d

    def tagged_loader(tk):
        return holdings_tagged[tk], {}

    print(f"[testC] universe={len(tickers)} engine={engine_version()}", flush=True)

    p_conc = measure_conc_entry_rate(holdings_raw)
    threshold = max(1, round(p_conc * 1000))
    print(f"[testC] conc 평균 진입 발생률 p_conc={p_conc:.5f} "
          f"→ 해시 threshold={threshold}/1000", flush=True)

    null_irs = []
    null_pass = 0
    null_traded = []
    null_oos_trades = []
    evaluated = 0
    seed_i = 0

    while time.time() - t0 < TIME_BUDGET_S:
        seed = f"nullC-{seed_i}"
        seed_i += 1
        exit_cfg = CONC_EXITS[seed_i % len(CONC_EXITS)]
        spec = {
            "spec_version": 2,
            "name": f"randnull-{seed}",
            "direction": "long",
            # entry placeholder(검증 통과용). 실제 평가는 패치가 해시 마스크로 대체.
            "entry": {"all_of": [
                {"signal": "price_filter", "mode": "above_ma", "window": 20},
            ]},
            "exit": dict(exit_cfg),
        }

        def patched_and(df, signals, _seed=seed, _thr=threshold):
            if not signals:                       # exit 의 빈 signal_all_of → 원본(False)
                return _ORIG_AND_V2(df, signals)
            if TK_COL in df.columns and len(df):  # entry → 해시 마스크
                tk6 = str(df[TK_COL].iloc[0])
                return _hash_mask(df, tk6, _seed, _thr)
            return pd.Series(False, index=df.index)

        btv2._and_v2 = patched_and
        try:
            wf = run_walk_forward_validation(
                spec, tickers, loader=tagged_loader, kospi_fetcher=kospi_fetcher)
        finally:
            btv2._and_v2 = _ORIG_AND_V2

        evaluated += 1
        med = wf["oos_excess_ir_median"]
        if med is not None:
            null_irs.append(med)
        if wf["gate_passed"]:
            null_pass += 1
        traded_windows = sum(
            1 for w in wf["windows"] if w["out_sample"]["n_trades"] > 0)
        null_traded.append(traded_windows)
        # OOS 전체 거래수(첫 윈도우) — 빈도 정렬 sanity
        if wf["windows"]:
            null_oos_trades.append(
                sum(w["out_sample"]["n_trades"] for w in wf["windows"]) / len(wf["windows"]))

        if evaluated % 10 == 0:
            print(f"[testC] evaluated={evaluated} pass={null_pass} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    def pct(xs, q):
        if not xs:
            return None
        s = sorted(xs)
        idx = min(len(s) - 1, int(q / 100 * len(s)))
        return round(s[idx], 4)

    n = evaluated
    valid = [x for x in null_irs if x is not None]
    print("\n" + "=" * 60, flush=True)
    print("[testC] RESULT — random-entry null", flush=True)
    print(f"  engine_version           = {engine_version()}", flush=True)
    print(f"  evaluated samples        = {n}", flush=True)
    print(f"  p_conc (entry rate)      = {p_conc:.5f} (threshold={threshold}/1000)", flush=True)
    print(f"  median OOS trades/window = "
          f"{round(st.median(null_oos_trades),1) if null_oos_trades else 0}", flush=True)
    print(f"  median windows traded    = {st.median(null_traded) if null_traded else 0}", flush=True)
    print("  --- (a) strict gate pass ---", flush=True)
    print(f"  null strict pass         = {null_pass}/{n} = {null_pass/n*100:.2f}%", flush=True)
    print(f"  --- (b) oos_excess_ir_median dist ({len(valid)} valid) ---", flush=True)
    print(f"  p50  = {pct(valid,50)}", flush=True)
    print(f"  p90  = {pct(valid,90)}", flush=True)
    print(f"  p99  = {pct(valid,99)}", flush=True)
    print(f"  max  = {round(max(valid),4) if valid else None}", flush=True)
    print(f"  mean = {round(st.mean(valid),4) if valid else None}", flush=True)
    print(f"  std  = {round(st.pstdev(valid),4) if len(valid)>1 else None}", flush=True)

    obs_conc_median_ir = 1.0
    if valid:
        lt = sum(1 for x in valid if x < obs_conc_median_ir)
        conc_pctile = lt / len(valid) * 100
        print("  --- 관측 위치 ---", flush=True)
        print(f"  conc median IR≈{obs_conc_median_ir} → null 분포 percentile ≈ {conc_pctile:.2f}%", flush=True)
        print(f"  관측 strict 1.8% vs null strict {null_pass/n*100:.2f}%", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
