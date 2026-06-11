#!/usr/bin/env python3
"""3-arm forward 실험 평가 — 동결(2026-06-10) 이후 페이퍼 실현거래를 arm별로 집계하고
KOSPI buy&hold(Arm C 벤치마크)와 비교한다. 매일 실행해 누적 추적.

Arm A (틱손절 견고성 바스켓):  1087 1091 1081 2746 1013
Arm B (ride 변형, 넓은손절/긴보유): 3009 3010 3011  (각각 1087/1091/1081 entry + ride exit)
Arm C (벤치마크): KOSPI — 동결~최신 buy&hold 수익률.

실현손익은 포지션 비중(스냅샷 per_name_weight) 가중: 슬리브자본 대비 % = Σ(weight×net).
손절 실현손실을 별도 분리해 추적(빈도가 아닌 비중가중 손실이 핵심). 미청산 보유의 평가손익
(unrealized)은 미포함 — 청산된 거래만. run마다 중복기록되므로 (sid,ticker,entry,exit) dedup.
forward = entry_date >= FREEZE.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date

FREEZE="2026-06-10"
ARMS={"A_tight":[1087,1091,1081,2746,1013], "B_ride":[3009,3010,3011],
      "D_maB":[2930,2939,2942], "E_flow":[2884,2866,634]}  # D/E=생존풀 게이트 통과 진짜 실력
# 승격 픽 — D·E 각 1. D=생존풀 med 1위. E=활동·견고 균형(2866, med0.74/min0.51/Sharpe0.72/379거래).
# (634는 med 1.29로 더 높으나 183거래·Sharpe0.37 저활동 → forward 관찰엔 2866이 유리.)
PROMOTED=[(2939,"D-best maB med1.64"), (2866,"E-best absorption med0.74")]
PAPER="/Users/selab/.tradingagents/hermes/paper_trades.db"
STRAT="/Users/selab/.tradingagents/hermes/strategies_v2.db"
# 기계판독 환류용 JSON — 일자별 파일이 누적 이력(FREEZE 변경 포함)을 보존한다.
OUT_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "forward_track")


def weighted_realized(con, sids):
    """슬리브별 비중가중 실현손익(청산 거래만). 반환 {sid:{all,sl,n,sl_n,wins}} (단위 %)."""
    ph=','.join('?'*len(sids))
    rows=con.execute(
        f"""SELECT e.strategy_id, e.ticker, e.entry_date, e.exit_date, e.net_ret_pct, e.exit_reason,
                   s.per_name_weight_pct
            FROM paper_trade_events e
            LEFT JOIN paper_strategy_snapshots s ON s.run_id=e.run_id AND s.strategy_id=e.strategy_id
            WHERE e.strategy_id IN ({ph}) AND e.entry_date>=?""", (*sids, FREEZE)).fetchall()
    seen={}
    for sid,tk,ed,xd,net,reason,wt in rows:
        seen.setdefault((sid,tk,ed,xd), (sid, net, reason, wt if wt is not None else 3.0))
    agg={sid:{"all":0.0,"sl":0.0,"n":0,"sl_n":0,"wins":0} for sid in sids}
    for (sid,_,_,_),(_,net,reason,wt) in seen.items():
        a=agg[sid]; w=wt/100.0
        a["all"]+=w*net; a["n"]+=1
        if net>0: a["wins"]+=1
        if reason=="stop_loss":
            a["sl"]+=w*net; a["sl_n"]+=1
    return agg


def _arm_stats(agg, n_sleeves):
    """arm 집계: (청산, 손절, 승, 가중총실현%, 가중손절손실%). arm 자본 대비 = 슬리브합 ÷ 슬리브수(동일자본)."""
    n=sum(a["n"] for a in agg.values()); sln=sum(a["sl_n"] for a in agg.values())
    wins=sum(a["wins"] for a in agg.values())
    return n, sln, wins, sum(a["all"] for a in agg.values())/n_sleeves, sum(a["sl"] for a in agg.values())/n_sleeves


def _verdicts(n, all_drag, sl_drag, sl_n, kospi_pct):
    """판정 3종 기계화 — ①가중총실현>0 ②KOSPI 초과 ③손절손실 미잠식. 데이터 없으면 None(판정 불가)."""
    if not n:
        return {"realized_positive": None, "beats_kospi": None, "stoploss_not_eroding": None}
    return {
        "realized_positive": all_drag > 0,
        "beats_kospi": (all_drag > kospi_pct) if kospi_pct is not None else None,
        # 손절 가중손실 절대값 < 손절 외 실현손익 → 손절이 총실현을 잠식하지 않음. 손절 0건이면 잠식 없음.
        "stoploss_not_eroding": True if not sl_n else abs(sl_drag) < (all_drag - sl_drag),
    }


def build_payload(as_of, latest, arm_aggs, promo, kospi_pct, kospi_window):
    """탐색 루프 환류용 기계판독 페이로드. arm_aggs={arm:{sid:agg}}, promo=승격픽 dict 리스트."""
    arms=[]; strategies=[]
    for arm,sids in ARMS.items():
        agg=arm_aggs[arm]
        n,sln,wins,all_drag,sl_drag=_arm_stats(agg, len(sids))
        arms.append({
            "arm_key": arm, "strategy_ids": sids,
            "n_closed_trades": n, "n_stop_loss": sln,
            "weighted_realized_pct": round(all_drag,4),
            "stop_loss_erosion_pct": round(sl_drag,4),
            "win_rate": round(wins/n,4) if n else None,
            "kospi_return_pct": round(kospi_pct,4) if kospi_pct is not None else None,
            "verdicts": _verdicts(n, all_drag, sl_drag, sln, kospi_pct),
        })
        for sid in sids:  # 전략별 상세 — 청산 거래가 있는 슬리브만
            a=agg[sid]
            if a["n"]:
                strategies.append({
                    "strategy_id": sid, "arm_key": arm,
                    "n_closed_trades": a["n"], "n_stop_loss": a["sl_n"],
                    "win_rate": round(a["wins"]/a["n"],4),
                    "weighted_realized_pct": round(a["all"],4),
                    "stop_loss_erosion_pct": round(a["sl"],4),
                })
    return {
        "as_of": as_of, "freeze_date": FREEZE, "latest_data_date": latest,
        "selected_strategy_ids": sorted({sid for sids in ARMS.values() for sid in sids}),
        "kospi": {"return_pct": round(kospi_pct,4) if kospi_pct is not None else None,
                  "window": list(kospi_window)},
        "arms": arms, "promotion_picks": promo, "strategies": strategies,
    }


def write_json(payload, out_dir=OUT_DIR):
    """arms_eval_{as_of}.json(일자별 누적 보존) + arms_eval_latest.json(최신 복사본) 기록."""
    os.makedirs(out_dir, exist_ok=True)
    daily=os.path.join(out_dir, f"arms_eval_{payload['as_of']}.json")
    text=json.dumps(payload, ensure_ascii=False, indent=2)
    for p in (daily, os.path.join(out_dir, "arms_eval_latest.json")):
        with open(p, "w", encoding="utf-8") as f:
            f.write(text+"\n")
    return daily


def kospi_return(d0, d1):
    if d1 < d0:
        return None, (None, None)
    from tradingagents.dataflows.market_history import fetch_kospi
    k=fetch_kospi(d0, d1); k["date"]=k["date"].astype(str)
    k=k[(k["date"]>=d0)&(k["date"]<=d1)]
    c=k["close"].to_numpy()
    if len(c)<2:
        return None, (None, None)
    return (c[-1]/c[0]-1)*100, (str(k["date"].iloc[0]), str(k["date"].iloc[-1]))


def main():
    pc=sqlite3.connect(PAPER)
    sc=sqlite3.connect(STRAT); sc.row_factory=sqlite3.Row
    latest=pc.execute("SELECT max(latest_data_date) FROM paper_runs").fetchone()[0] or FREEZE
    print(f"=== 3-arm forward 평가 (동결 {FREEZE} ~ {latest}) ===\n")
    print(f"{'arm':>8} {'청산':>4} {'손절':>4} {'승률':>5} {'가중총실현%':>11} {'가중손절손실%':>13}  (백테스트 medIR/Shp)")
    arm_aggs={}
    for arm,sids in ARMS.items():
        agg=weighted_realized(pc, sids); arm_aggs[arm]=agg
        n,sln,wins,all_drag,sl_drag=_arm_stats(agg, len(sids))
        wr=f"{wins/n:.0%}" if n else "-"
        bt=sc.execute(f"SELECT avg(wf_excess_ir_median) m, avg(out_sharpe) s FROM strategies WHERE id IN ({','.join('?'*len(sids))})", sids).fetchone()
        print(f"{arm:>8} {n:>4} {sln:>4} {wr:>5} {all_drag:>+11.3f} {sl_drag:>+13.3f}  (medIR {bt['m']:.2f}, Shp {bt['s']:.2f})")
        if n:  # 슬리브별 손절손실 분해
            for sid in sids:
                a=agg[sid]
                if a["sl_n"]: print(f"         └ #{sid}: 손절 {a['sl_n']}건 → 가중손실 {a['sl']:+.3f}%")
    print("\n=== 승격 픽 (가장 우수 D/E 각 1) forward 추적 ===")
    promo=[]
    for sid,tag in PROMOTED:
        a=weighted_realized(pc,[sid])[sid]
        nm=sc.execute("SELECT name FROM strategies WHERE id=?",(sid,)).fetchone()
        nm=nm["name"] if nm else "?"
        wr=f"{a['wins']/a['n']:.0%}" if a["n"] else "-"
        print(f"  [{tag}] #{sid} {nm[:30]}: 청산 {a['n']} 손절 {a['sl_n']} 승률 {wr} | 가중실현 {a['all']:+.3f}% (손절손실 {a['sl']:+.3f}%)")
        promo.append({
            "strategy_id": sid, "tag": tag, "name": nm,
            "n_closed_trades": a["n"], "n_stop_loss": a["sl_n"],
            "win_rate": round(a["wins"]/a["n"],4) if a["n"] else None,
            "weighted_realized_pct": round(a["all"],4),
            "stop_loss_erosion_pct": round(a["sl"],4),
        })
    kr,(d0,d1)=kospi_return(FREEZE, latest)
    if kr is not None:
        print(f"\n[C_KOSPI 벤치마크] {d0}~{d1} buy&hold: {kr:+.2f}%")
    else:
        print(f"\n[C_KOSPI 벤치마크] forward 창 아직 없음(최신 {latest} < 동결 {FREEZE}). 거래일 누적 후 산출.")
    print("\n판정: ① arm 가중총실현 > 0  ② KOSPI 초과  ③ 손절손실이 총실현을 잠식하지 않을 것.")
    print("주: 미청산 보유의 평가손익(unrealized)은 미포함 — 청산 거래만. 매일 실행해 누적 추적.")
    # JSON 환류 — stdout(Telegram 보고)에는 섞지 않고 stderr로만 경로 안내.
    path=write_json(build_payload(date.today().isoformat(), latest, arm_aggs, promo, kr, (d0,d1)))
    print(f"[JSON] {path}", file=sys.stderr)
    pc.close(); sc.close()


if __name__=="__main__":
    raise SystemExit(main())
