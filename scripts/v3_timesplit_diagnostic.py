"""v3 검증 진단 — 기존 전략을 '시간 분할'로 재평가해 '종목 분할'과 비교.

목적(읽기 전용 진단): 현행 in/out 게이트는 종목 md5 해시(cross-sectional)로
분할하므로 in·out 이 동일한 2021–2026 구간을 본다. 이 스크립트는 엔진
(``_simulate_v2``)을 그대로 재사용하되, 종목 split 대신 **거래/일별수익률을
날짜로 70/30 분할**(chronological)하여 시간 일반화(out-of-time) 성과를 재계산한다.

핵심 질문: "종목 분할에서 좋아 보이던 전략이 시간 분할에선 무너지는가?"

엔진·DB·라이브 루프는 일절 수정하지 않는다. ``strategies_v2.db`` 는 읽기만 한다.

실행:
    uv run python -m scripts.v3_timesplit_diagnostic --limit 30 --order in_sharpe
    uv run python -m scripts.v3_timesplit_diagnostic --limit 3   # 빠른 검증
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _attach_market_columns, _simulate_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_validation import engine_version

IN_SAMPLE_PCT = bt.IN_SAMPLE_PCT  # 70 — 종목 분할과 동일 비율로 시간 분할(축만 다름)
ROBUST_BAR = 0.5  # 견고성 기준: 두 split 양쪽 모두 sharpe 가 이 값보다 커야 함


def _time_cutoff(holdings: dict[str, pd.DataFrame], pct: int) -> str:
    """전 종목 거래일 union 의 pct 분위수 날짜(ISO 문자열, lexicographic=chronological)."""
    dates: set[str] = set()
    for df in holdings.values():
        dates.update(df["date"].astype(str).tolist())
    ordered = sorted(dates)
    k = int(len(ordered) * pct / 100)
    return ordered[min(k, len(ordered) - 1)]


def _simulate_all(spec: dict, holdings: dict, kospi):
    """전 종목 _simulate_v2 → (all_trades, port_daily(date-index)).

    run_universe_backtest_v2 와 동일하게 종목별 일별수익률을 date-index Series 로
    쌓아 _combine. 단 종목 split 으로 안 거르고 전부 합친다(시간으로 나눌 것이므로).
    """
    ret_frames, act_frames, all_trades = [], [], []
    for tk, df in holdings.items():
        trades, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
        bt._attach_kospi(trades, kospi)
        idx = cdf["date"].astype(str).to_numpy()
        ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
        act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
        all_trades.extend(trades)
    port = bt._combine(ret_frames, act_frames)
    return all_trades, port


def _split_by_time(all_trades: list, port: pd.Series, cutoff: str):
    """거래는 entry_date, 일별수익률은 날짜 인덱스로 IS/OOS(시간) 분할."""
    is_tr = [t for t in all_trades if t["entry_date"] < cutoff]
    oos_tr = [t for t in all_trades if t["entry_date"] >= cutoff]
    is_port = port[port.index < cutoff]
    oos_port = port[port.index >= cutoff]
    return bt._metrics(is_tr, is_port), bt._metrics(oos_tr, oos_port)


def _split_by_ticker(spec: dict, holdings: dict, kospi):
    """파이프라인 정합성 자체검증용 — 종목 split 으로 재계산(DB 값과 대조)."""
    results = {}
    for split in ("in", "out"):
        ret_frames, act_frames, trades_all = [], [], []
        for tk, df in holdings.items():
            if bt.split_of(tk) != split:
                continue
            trades, daily, active, cdf = _simulate_v2(spec, df, kospi=kospi)
            bt._attach_kospi(trades, kospi)
            idx = cdf["date"].astype(str).to_numpy()
            ret_frames.append(pd.Series(daily.to_numpy(), index=idx, name=tk))
            act_frames.append(pd.Series(active.to_numpy(), index=idx, name=tk))
            trades_all.extend(trades)
        results[split] = bt._metrics(trades_all, bt._combine(ret_frames, act_frames))
    return results["in"], results["out"]


def _sharpe(m: dict):
    return m["sharpe"]


def _fmt(m: dict) -> str:
    def g(k, w=5, p=2):
        v = m.get(k)
        return f"{'n/a':>{w}}" if v is None else f"{v:{w}.{p}f}"
    return f"shp{g('sharpe')} win{g('win_rate',5,2)} mdd{g('mdd_pct',7,1)} n{m.get('n_trades',0):>4}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30, help="평가할 전략 수(상위 N)")
    ap.add_argument("--order", default="in_sharpe",
                    choices=["in_sharpe", "out_sharpe", "id"],
                    help="정렬 기준(상위 N 선정)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    store = StrategyStoreV2()
    strategies = store.list()
    print(f"[diag] 저장된 전략 {len(strategies)}개", flush=True)

    # 상위 N 선정 — '종목 분할에서 좋아 보이는' 후보를 집중 검증
    def keyfn(s):
        v = s.get(args.order if args.order != "id" else "id")
        if args.order == "id":
            return v
        return -(v if v is not None else -999)  # None/낮은 sharpe 후순위
    strategies.sort(key=keyfn)
    targets = strategies[:args.limit]

    raw_tickers = KisHistoryStore().list_tickers()
    print(f"[diag] preloading {len(raw_tickers)} tickers ...", flush=True)
    tickers, loader, kospi_fetcher = _preload(raw_tickers)
    holdings = {tk: loader(tk)[0] for tk in tickers}
    kospi = kospi_fetcher(None, None)
    cutoff = _time_cutoff(holdings, IN_SAMPLE_PCT)
    span_lo = min(str(df["date"].iloc[0]) for df in holdings.values())
    span_hi = max(str(df["date"].iloc[-1]) for df in holdings.values())
    ev = engine_version()
    print(f"[diag] universe={len(holdings)} kospi={'OK' if kospi is not None else 'NONE'} "
          f"data={span_lo}~{span_hi} time-cutoff(@{IN_SAMPLE_PCT}%)={cutoff} engine={ev}", flush=True)
    print(f"[diag] 비교축: 현재엔진 종목분할(xsec, 같은기간·다른종목) vs "
          f"현재엔진 시간분할(time, IS<{cutoff}<=OOS=미래)", flush=True)
    print(f"[diag] 견고성 기준: 두 split 양쪽 모두 sharpe>{ROBUST_BAR} 여야 'robust'\n", flush=True)

    rows = []
    verify_done = False
    for i, s in enumerate(targets, 1):
        spec = s["spec"]
        all_trades, port = _simulate_all(spec, holdings, kospi)
        is_t, oos_t = _split_by_time(all_trades, port, cutoff)
        x_in, x_out = _split_by_ticker(spec, holdings, kospi)  # 현재엔진 종목분할

        # 첫 전략: DB drift 점검(현재엔진 종목분할 vs DB 저장값 — 재현성 확인)
        if not verify_done:
            print(f"[drift] #{s['id']} 종목분할 현재엔진 vs DB저장 — "
                  f"in_sharpe {x_in['sharpe']} vs {s.get('in_sharpe')} | "
                  f"out_sharpe {x_out['sharpe']} vs {s.get('out_sharpe')} "
                  f"(차이 크면 저장 후 엔진 수정됨)", flush=True)
            verify_done = True

        rows.append({
            "id": s["id"], "name": s["name"],
            "db_in_sharpe": s.get("in_sharpe"), "db_out_sharpe": s.get("out_sharpe"),
            "xsec_in": x_in, "xsec_out": x_out,
            "time_is": is_t, "time_oos": oos_t,
        })
        print(f"[{i:>3}/{len(targets)}] #{s['id']:>4} {s['name'][:30]:30} | "
              f"xsec IN[{_fmt(x_in)}] OUT[{_fmt(x_out)}] "
              f"|| time IS[{_fmt(is_t)}] OOS[{_fmt(oos_t)}]", flush=True)

    # === 집계 ===
    def shp(m):
        return m["sharpe"]

    def both_pos(a, b, bar):
        return (shp(a) is not None and shp(a) > bar
                and shp(b) is not None and shp(b) > bar)

    xsec_robust = [r for r in rows if both_pos(r["xsec_in"], r["xsec_out"], ROBUST_BAR)]
    time_robust = [r for r in rows if both_pos(r["time_is"], r["time_oos"], ROBUST_BAR)]

    def gap(a, b):
        if shp(a) is None or shp(b) is None:
            return None
        return round(abs(shp(a) - shp(b)), 3)
    xsec_gaps = [g for r in rows if (g := gap(r["xsec_in"], r["xsec_out"])) is not None]
    time_gaps = [g for r in rows if (g := gap(r["time_is"], r["time_oos"])) is not None]

    import statistics as st
    med = lambda v: round(st.median(v), 3) if v else None

    print("\n" + "=" * 74)
    print(f"집계 (상위 {len(rows)}개, 정렬={args.order})")
    print(f"  종목분할 in·out 양쪽 sharpe>{ROBUST_BAR} (robust) : {len(xsec_robust)}개")
    print(f"  시간분할 IS·OOS 양쪽 sharpe>{ROBUST_BAR} (robust) : {len(time_robust)}개  ← 진짜 견고성")
    print(f"  in/out sharpe 격차 중앙값 — 종목분할 {med(xsec_gaps)} vs 시간분할 {med(time_gaps)}")
    print(f"    (종목분할 격차 작음=안정적으로 보임 / 시간분할 격차 큼=실제론 시간에 불안정)")
    print("=" * 74)

    out_path = Path(args.out) if args.out else (
        Path.home() / ".tradingagents" / "hermes" / "reports"
        / f"v3-timesplit-diagnostic-top{args.limit}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "cutoff": cutoff, "in_sample_pct": IN_SAMPLE_PCT, "robust_bar": ROBUST_BAR,
        "engine_version": ev, "data_span": [span_lo, span_hi],
        "n_strategies": len(rows), "order": args.order,
        "summary": {
            "xsec_robust": len(xsec_robust), "time_robust": len(time_robust),
            "xsec_gap_median": med(xsec_gaps), "time_gap_median": med(time_gaps),
        },
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[diag] 결과 저장: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
