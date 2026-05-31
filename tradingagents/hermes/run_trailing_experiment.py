"""트레일링 스톱 실험 러너 — 상위 near-miss 전략의 MDD 감소 여부 검증.

실행: ``uv run python -m tradingagents.hermes.run_trailing_experiment``

흐름:
  1. strategies_v2.db 에서 in/out 승률>0.5 · out 샤프>1.0 상위 N개 spec 로드.
  2. 유니버스 holdings + KOSPI 를 1회 preload (모든 변형이 재사용).
  3. 각 spec 에 대해 baseline(저장된 v2 그대로) + 트레일링 변형들 백테스트.
  4. in/out 승률·샤프·MDD·거래수 + 게이트 통과 여부를 표로 출력.
     baseline 의 MDD 가 DB 저장값과 일치하는지(재현성)도 함께 표시.
"""
from __future__ import annotations

import json
import os
import sqlite3

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.trailing_stop_experiment import run_universe_trailing

DB = os.path.expanduser("~/.tradingagents/hermes/strategies_v2.db")
TOP_N = 6

# (라벨, trail_pct, keep_fixed_sl)
VARIANTS = [
    ("baseline      ", None, True),    # = 저장된 v2 (트레일링 없음)
    ("trail15 + SL  ", 15.0, True),
    ("trail12 + SL  ", 12.0, True),
    ("trail10 + SL  ", 10.0, True),
    ("trail8  + SL  ", 8.0, True),
    ("trail10 (only)", 10.0, False),   # 고정 손절 제거, 트레일링만 하방보호
    ("trail8  (only)", 8.0, False),
]


def _load_top_specs(n: int) -> list[dict]:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT id, name, spec_json, in_mdd, out_mdd, in_sharpe, out_sharpe
           FROM strategies
           WHERE in_win_rate > 0.5 AND out_win_rate > 0.5 AND out_sharpe > 1.0
           ORDER BY out_sharpe DESC LIMIT ?""",
        (n,),
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "name": r["name"], "spec": json.loads(r["spec_json"]),
            "db_in_mdd": r["in_mdd"], "db_out_mdd": r["out_mdd"],
        })
    return out


def _gate_reason(in_m: dict, out_m: dict) -> str:
    """게이트 실패 사유 약식 (통과면 'PASS')."""
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
        iw, ow = in_m["win_rate"], out_m["win_rate"]
        if abs(iw - ow) > 0.10:
            return "FAIL:gap"
        return "PASS"
    tags = set(bi) | set(bo)
    return "FAIL:" + ",".join(sorted(tags))


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
    tickers = KisHistoryStore().list_tickers()
    loaded: dict = {}
    min_d = max_d = None
    for tk in tickers:
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
    kospi = None
    try:
        kospi = fetch_kospi(min_d, max_d)
        print(f"   KOSPI {len(kospi)} rows")
    except Exception as e:  # noqa: BLE001
        print(f"   KOSPI fetch 실패: {e!r} (market_filter/초과수익 결손)")

    print("[3/3] baseline + 트레일링 변형 백테스트 …\n")
    for s in specs:
        print(f"=== #{s['id']} {s['name']}  (DB MDD in {s['db_in_mdd']:.1f} / "
              f"out {s['db_out_mdd']:.1f}) ===")
        print(f"    {'variant':<15} | in : {'':<38} | out: {'':<38} | gate")
        for label, trail, keep_sl in VARIANTS:
            res = run_universe_trailing(
                s["spec"], loaded=loaded, kospi=kospi,
                trail_pct=trail, keep_fixed_sl=keep_sl)
            im, om = res["in_sample"], res["out_sample"]
            gate = _gate_reason(im, om)
            mark = "  <<< GATE PASS" if gate == "PASS" else ""
            print(f"    {label} | in: {_fmt(im)} | out: {_fmt(om)} | {gate}{mark}")
        print()


if __name__ == "__main__":
    main()
