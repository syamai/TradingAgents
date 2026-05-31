"""레짐 게이트 실험 러너 — 상위 near-miss 전략의 MDD 감소 여부 검증.

실행: ``uv run python -m tradingagents.hermes.run_regime_experiment``

각 spec 에 대해 baseline + KOSPI 레짐 게이트(60/120/200일선 위에서만 진입)를
백테스트하고 in/out 승률·샤프·MDD·거래수 + 게이트 통과 여부를 표로 출력한다.
preload 는 run_trailing_experiment 와 동일(holdings/KOSPI 1회 로딩).
"""
from __future__ import annotations

import json
import os
import sqlite3

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.regime_gate_experiment import run_universe_regime

DB = os.path.expanduser("~/.tradingagents/hermes/strategies_v2.db")
TOP_N = 6

# (라벨, regime_window) — None=baseline
VARIANTS = [
    ("baseline       ", None),
    ("regime>MA60    ", 60),
    ("regime>MA120   ", 120),
    ("regime>MA200   ", 200),
]


def _load_top_specs(n: int) -> list[dict]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, name, spec_json, in_mdd, out_mdd
           FROM strategies
           WHERE in_win_rate > 0.5 AND out_win_rate > 0.5 AND out_sharpe > 1.0
           ORDER BY out_sharpe DESC LIMIT ?""",
        (n,),
    ).fetchall()
    con.close()
    return [{"id": r["id"], "name": r["name"], "spec": json.loads(r["spec_json"]),
             "db_in_mdd": r["in_mdd"], "db_out_mdd": r["out_mdd"]} for r in rows]


def _gate_reason(in_m: dict, out_m: dict) -> str:
    def fail(m):
        bad = []
        if not (m["win_rate"] and m["win_rate"] > 0.50):
            bad.append("win")
        if not (m["sharpe"] and m["sharpe"] > 1.0):
            bad.append("shp")
        if not (m["mdd_pct"] >= -20.0):
            bad.append("mdd")
        if not (m["n_trades"] >= 50):
            bad.append("n")
        return bad
    bi, bo = fail(in_m), fail(out_m)
    if not bi and not bo:
        if abs(in_m["win_rate"] - out_m["win_rate"]) > 0.10:
            return "FAIL:gap"
        return "PASS"
    return "FAIL:" + ",".join(sorted(set(bi) | set(bo)))


def _fmt(m: dict) -> str:
    w = f"{m['win_rate']:.3f}" if m["win_rate"] is not None else "  -  "
    s = f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "  - "
    return f"win {w}  shp {s:>5}  mdd {m['mdd_pct']:7.2f}  n {m['n_trades']:>4}"


def main() -> None:
    print(f"[1/3] 상위 {TOP_N} near-miss spec 로드 …")
    specs = _load_top_specs(TOP_N)
    for s in specs:
        print(f"   #{s['id']:>3} {s['name']}")

    print("[2/3] 유니버스 holdings + KOSPI preload …")
    loaded: dict = {}
    min_d = max_d = None
    for tk in KisHistoryStore().list_tickers():
        df, _meta = load_holdings(tk)
        if df is None or df.empty:
            continue
        loaded[tk] = df
        d0, d1 = str(df["date"].iloc[0]), str(df["date"].iloc[-1])
        min_d = d0 if (min_d is None or d0 < min_d) else min_d
        max_d = d1 if (max_d is None or d1 > max_d) else max_d
    n_in = sum(1 for tk in loaded if bt.split_of(tk) == "in")
    n_out = sum(1 for tk in loaded if bt.split_of(tk) == "out")
    print(f"   종목 {len(loaded)}개 (in {n_in} / out {n_out}), 기간 {min_d}~{max_d}")
    kospi = fetch_kospi(min_d, max_d)
    print(f"   KOSPI {len(kospi)} rows")

    print("[3/3] baseline + 레짐 게이트 백테스트 …\n")
    for s in specs:
        print(f"=== #{s['id']} {s['name']}  (DB MDD in {s['db_in_mdd']:.1f} / "
              f"out {s['db_out_mdd']:.1f}) ===")
        for label, win in VARIANTS:
            res = run_universe_regime(
                s["spec"], loaded=loaded, kospi=kospi, regime_window=win)
            im, om = res["in_sample"], res["out_sample"]
            gate = _gate_reason(im, om)
            mark = "  <<< GATE PASS" if gate == "PASS" else ""
            print(f"    {label} | in: {_fmt(im)} | out: {_fmt(om)} | {gate}{mark}")
        print()


if __name__ == "__main__":
    main()
