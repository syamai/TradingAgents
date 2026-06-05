"""trading-ai → Hermes MCP 서버 v2 (신규 신호 + 완화 게이트).

기존 ``mcp_server.py`` 를 무수정 유지하고, v2 전략 도구만 노출하는 **별도 stdio
서버**. Hermes 가 ``strategy-researcher-v2`` 스킬로 이 서버의 도구를 호출해
신규 신호(price_drop/trend_slope/rolling_corr) 기반 전략을 자율 연구한다.

캐시 로더·유니버스는 기존 ``mcp_server`` 헬퍼를 그대로 재사용(중복 0).

도구 (3):
  - ``backtest_strategy_v2``  : v2 spec 종목군 백테스트 (탐색용, 저장 X)
  - ``save_strategy_v2``      : 내부 재백테스트 후 ``strategies_v2.db`` 영구 저장
  - ``list_strategies_v2``    : 누적 시도/통과 조회 (기본 brief — 토큰 절약)

게이트(v2): 승률>0.50 · 샤프>1.0 · MDD≥-20% · 거래≥50 · in/out 격차≤10%p,
in/out 양쪽 충족 시에만 ``gate_passed=True``.

실행: ``uv run python -m tradingagents.hermes.mcp_server_v2``
저장: ``~/.tradingagents/hermes/strategies_v2.db`` (기존 strategies.db 와 분리).
"""
from __future__ import annotations

from typing import Optional, Union

from mcp.server.fastmcp import FastMCP

from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.mcp_server import _engine_loader, _universe
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_validation import (
    engine_version,
    run_walk_forward_validation,
)


def _v2_store() -> StrategyStoreV2:
    """StrategyStoreV2 — ``~/.tradingagents/hermes/strategies_v2.db``."""
    return StrategyStoreV2()


mcp = FastMCP("trading-ai-hermes-v2")


def _fair_gate(wf: dict) -> bool:
    """공정 게이트 — walk-forward × 시장대비 초과수익 IR. 모든 OOS 창에서 시장을
    이기고(IR>0) 중앙 IR>임계 여야 True. 단일 분할 레짐편향·시장베타 오인 제거."""
    return bool(wf["gate_passed"])


@mcp.tool()
def backtest_strategy_v2(spec: dict, universe: Optional[list[str]] = None) -> dict:
    """v2 수급 룰 전략을 종목군에 백테스트 (탐색용 — 저장 안 함).

    ★ v3: 종목분할(xsec)과 **시간분할(time, IS=과거/OOS=미래)** 을 모두 평가한다.
    최종 ``gate_passed`` 는 **두 게이트 모두 통과**해야 True. 시간분할이 진짜
    관문이다 — 종목분할만 좋고 시간분할이 무너지는 전략(레짐 베팅)은 통과 못 한다.
    전략을 개선할 때 ``time_out_sample.sharpe`` (미래 구간)와 ``time_in_sample.sharpe``
    (과거 구간)가 **둘 다 1.0 초과**하도록 맞춰라 — 한쪽만 높으면 시간 게이트 실패.

    v2 신호 어휘 (기존 5종 + 신규 3종):
      - ``price_drop``   : 최근 W일 종가 수익률 <= -X% (눌림목/역추세 진입)
        params: ``window∈{3,5,10,20}``, ``value∈{3,5,8,10,15}``
      - ``trend_slope``  : ``{subject}_net_qty`` 누적합의 W일 변화 방향
        params: ``subject``, ``window∈{10,20,60}``, ``direction∈{up,down}``
      - ``rolling_corr`` : ``{subject}_net_qty`` vs ``price_change_pct`` 의 W일
        trailing Pearson r >= min_r (수급-가격 동조 필터)
        params: ``subject``, ``lookback∈{20,60,120}``, ``min_r∈{0.1,0.2,0.3,0.5}``
      - 기존 5종: net_streak / pct_threshold / pct_delta / net_vol_ratio / price_filter
    spec_version 은 2 (또는 1). entry 1~3 신호 AND, exit signal 0~2 + stop_loss_pct
    /take_profit_pct (옵션) + max_hold_days (필수). 수치는 그리드 값만(위반 ValueError).

    look-ahead 0 (신호 row i → 체결 close[i+1]), 거래비용 편도 0.015% 반영.

    args:
        spec: v2 전략 spec dict. 스키마 위반 시 ValueError.
        universe: 종목 코드 리스트. None 이면 KIS 히스토리 전체.

    returns:
        ``{in_sample:{win_rate,sharpe,mdd_pct,cum_return_pct,n_trades,...},
           out_sample:{...}, gate_passed:bool, universe_size, n_in, n_out}``.
        게이트(승률>0.50·샤프>1.0·MDD≥-20%·거래≥50·in/out격차≤10%p)는 in/out
        양쪽 충족 시에만 True.
    """
    validate_spec_v2(spec)
    tickers = universe if universe else _universe()
    xsec = run_universe_backtest_v2(spec, tickers, loader=_engine_loader)
    wf = run_walk_forward_validation(spec, tickers, loader=_engine_loader)
    return {
        "in_sample": xsec["in_sample"],          # 종목분할(진단)
        "out_sample": xsec["out_sample"],
        "xsec_gate_passed": xsec["gate_passed"],
        "wf_n_windows": wf["n_windows"],         # 공정 게이트(1차 관문)
        "wf_excess_ir_median": wf["oos_excess_ir_median"],
        "wf_excess_ir_min": wf["oos_excess_ir_min"],
        "wf_oos_sharpe_median": wf["oos_sharpe_median"],   # raw(진단)
        "gate_passed": _fair_gate(wf),
        "gate_min_ir": wf["gate_min_ir"],
        "portfolio_policy": xsec.get("portfolio_policy"),
        "universe_size": xsec["universe_size"],
    }


@mcp.tool()
def save_strategy_v2(spec: dict, name: Optional[str] = None) -> dict:
    """v2 전략을 *내부 재백테스트* 후 ``strategies_v2.db`` 영구 저장.

    전달 메트릭을 신뢰하지 않고 spec 으로 직접 재백테스트(LLM 위조 방지). 동일
    로직(spec_hash) 전략이 이미 있으면 재실행 없이 ``duplicate`` 반환. 게이트
    미통과도 저장(연구 로그 + 시도 카운트 + 재시도 차단).

    args:
        spec: v2 전략 spec dict. 스키마 위반 시 ValueError.
        name: 사람이 읽는 식별자 (옵션 — spec.name 사용).

    returns:
        신규: ``{strategy_id, duplicate:False, gate_passed, in_sample,
        out_sample, universe_size}``. 중복: ``{strategy_id, duplicate:True,
        gate_passed}``.
    """
    validate_spec_v2(spec)
    store = _v2_store()
    existing = store.get_by_hash(spec_hash(spec))
    if existing is not None:
        return {
            "strategy_id": existing["id"],
            "duplicate": True,
            "gate_passed": existing["gate_passed"],
        }
    tickers = _universe()
    result = run_universe_backtest_v2(spec, tickers, loader=_engine_loader)
    wf_result = run_walk_forward_validation(spec, tickers, loader=_engine_loader)
    sid, _is_new = store.save(
        spec, result, name=name,
        wf_result=wf_result, engine_version=engine_version(),
    )
    return {
        "strategy_id": sid,
        "duplicate": False,
        "gate_passed": _fair_gate(wf_result),
        "in_sample": result["in_sample"],
        "out_sample": result["out_sample"],
        "wf_n_windows": wf_result["n_windows"],
        "wf_excess_ir_median": wf_result["oos_excess_ir_median"],
        "wf_excess_ir_min": wf_result["oos_excess_ir_min"],
        "universe_size": result["universe_size"],
    }


@mcp.tool()
def list_strategies_v2(
    gate_passed: Optional[bool] = None, brief: bool = True,
) -> Union[dict, list[dict]]:
    """``strategies_v2.db`` 전략 조회 — 연구 진행/통과 카운트 single source.

    기본 ``brief=True`` 는 토큰 절약형 요약을 반환(spec_json·전체 행 제외) — 매
    tick 호출해도 O(n) 폭증 없음. 전략 *중복*은 ``save_strategy_v2`` 가 spec_hash
    로 서버단에서 멱등 차단하므로, 진행 추적엔 brief 로 충분.

    args:
        gate_passed: True 면 통과만, False 면 미통과만, None 전체.
        brief: True(기본)=경량 요약, False=전체 행(spec 포함).

    returns:
        brief=True: ``{total, passed, top_by_in_sharpe:[{id,name,in_win,in_sharpe,
        in_mdd,in_n,out_win,out_sharpe,out_mdd,gate}, ...최대 5]}``.
        brief=False: 전체 전략 dict 리스트.
    """
    rows = _v2_store().list(gate_passed=gate_passed)
    if not brief:
        return rows
    passed = sum(1 for r in rows if r["gate_passed"])  # 공정 게이트(walk-forward 초과수익 IR)
    # walk-forward OOS 초과수익 IR 최소값(최악 구간 알파) 기준 정렬 — '모든 구간에서
    # 시장을 이긴' 후보가 위로.
    def _wf_min(r):
        v = r.get("wf_excess_ir_min")
        return v if v is not None else -999.0
    top = sorted(rows, key=_wf_min, reverse=True)[:5]
    top_brief = [{
        "id": r["id"], "name": r["name"],
        "in_sharpe": r["in_sharpe"], "out_sharpe": r["out_sharpe"],  # xsec raw(진단)
        "wf_excess_ir_median": r.get("wf_excess_ir_median"),
        "wf_excess_ir_min": r.get("wf_excess_ir_min"),
        "wf_n_windows": r.get("wf_n_windows"),
        "gate": r["gate_passed"],
    } for r in top]
    return {"total": len(rows), "passed": passed,
            "top_by_wf_excess_ir_min": top_brief}


def main():
    """stdio transport 진입점."""
    mcp.run()


if __name__ == "__main__":
    main()
