"""Forward 테스트 레지스트리 — 동결 전략을 2025-06-30 이전(SCORE_DATE_HI)으로 선정·동결,
2025-06-30 이후(급등구간+미래)를 편향0 holdout 으로 관찰한다.

생존편향은 과거 백테스트를 영구 오염시키므로 유일한 클린 검증은 forward 관찰. 채점 컷오프
이후 데이터는 비정상 급등 레짐이라 절대수익이 아닌 **시장초과 IR**로만 본다(그조차 신중).

register(): 동결 정의 + IS(컷오프 이전) 메트릭을 JSON 기록. observe(): forward(컷오프 이후)
메트릭을 재실행 가능하게 계산. 데이터가 쌓일수록 observe() 를 다시 돌리면 된다.
"""
from __future__ import annotations

import json
from pathlib import Path

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.xsec_backtest import run_xsec_rank_backtest, SCORE_DATE_HI

REGISTRY = Path.home() / ".tradingagents/hermes/forward_registry.json"

# 수급 유니버스에 섞인 ETF — stock 팩터 전략서 제외(ETF 는 구조적으로 공매도 비중 낮아
# 저공매도 픽을 오염). 2026-06 시점 top-cap 유니버스에서 식별한 12개.
ETF_CODES = frozenset({
    "069500", "091160", "102110", "122630", "133690", "278530",
    "360750", "379800", "379810", "396500", "459580", "488770",
})

# 동결 대표 전략 — 상관 제거(서로 다른 아이디어), 전부 forward 관찰 가능(rb20).
# 점검(2026-06-02): ETF 제외 + 유동성 필터 적용 시 저공매도 엣지 붕괴 → illiquidity 아티팩트.
# 무필터(진단)와 유동성필터(tradable) 버전을 함께 동결해 forward 추적.
FROZEN = [
    {"id": "xsec-short-w5", "kind": "short", "direction": "low", "window": 5,
     "quantile": 0.2, "rebalance": 20, "min_liq_pct": 0.0,
     "thesis": "저공매도압력 (무필터, 진단용)",
     "verdict": "illiquidity 아티팩트 — 유동성필터(min_liq 0.3) 시 exIR 1.94→1.15·게이트 탈락. 거래 불가, 관찰만"},
    {"id": "xsec-short-w5-liq30", "kind": "short", "direction": "low", "window": 5,
     "quantile": 0.2, "rebalance": 20, "min_liq_pct": 0.3,
     "thesis": "저공매도압력 (tradable, 하위30% 유동성 제외)",
     "verdict": "tradable 기준 IS 게이트 탈락(exIR 1.15, 최악연도 IR −2.0). 엣지 아님 — forward 추적"},
    {"id": "xsec-mom-w120", "kind": "mom", "direction": "high", "window": 120,
     "quantile": 0.2, "rebalance": 20, "min_liq_pct": 0.0,
     "thesis": "6개월 횡단면 모멘텀",
     "verdict": "IS 경계 탈락(2018 음수)+forward 음수. 유동성 강화 시 더 악화(exIR 1.17→0.52)"},
    {"id": "xsec-lowvol-w60", "kind": "vol", "direction": "low", "window": 60,
     "quantile": 0.2, "rebalance": 20, "min_liq_pct": 0.0,
     "thesis": "저변동성 (음성 대조군)",
     "verdict": "부재 — IS/forward/횡단면 모두 음수"},
]

_KEEP = ("excess_ir", "sharpe", "cum_return_pct", "yr_ir_min", "gate_passed",
         "n_periods", "avg_pick_liq_pct", "avg_picks")


def _load():
    """수급 유니버스 로드 + ETF 제외(stock 팩터 전략 기준)."""
    tickers, loader, kf = _preload(KisHistoryStore().list_tickers())
    H = {tk: loader(tk)[0] for tk in tickers if str(tk)[:6] not in ETF_CODES}
    return H, kf(None, None)


def _metrics(H, K, e, **kw):
    r = run_xsec_rank_backtest(H, K, kind=e["kind"], window=e["window"], quantile=e["quantile"],
                               rebalance=e["rebalance"], direction=e["direction"],
                               min_liq_pct=e.get("min_liq_pct", 0.0), **kw)
    return {k: r.get(k) for k in _KEEP}


def register(registered_at: str, H=None, K=None):
    """동결 전략 + IS(≤SCORE_DATE_HI) 메트릭을 레지스트리에 기록."""
    if H is None:
        H, K = _load()
    out = [{**e, "freeze_date": SCORE_DATE_HI, "registered_at": registered_at,
            "is_metrics": _metrics(H, K, e)} for e in FROZEN]
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def observe(H=None, K=None):
    """forward(>SCORE_DATE_HI) 메트릭 계산. 데이터가 쌓이면 재실행."""
    if H is None:
        H, K = _load()
    reg = json.loads(REGISTRY.read_text())
    return [{"id": e["id"], "thesis": e["thesis"], "verdict": e.get("verdict", ""),
             "is": e["is_metrics"],
             "fwd": _metrics(H, K, e, date_lo=SCORE_DATE_HI, date_hi=None)} for e in reg]


if __name__ == "__main__":
    import sys
    when = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    H, K = _load()
    register(when, H, K)
    print(f"registered {len(FROZEN)} strategies → {REGISTRY}\n")
    for r in observe(H, K):
        i, f = r["is"], r["fwd"]
        print(f"[{r['id']}] {r['thesis']}")
        print(f"   판정: {r['verdict']}")
        print(f"   IS(≤{SCORE_DATE_HI}):  exIR={i['excess_ir']} gate={i['gate_passed']} "
              f"yr_ir_min={i['yr_ir_min']} sharpe={i['sharpe']} pick유동성%={i.get('avg_pick_liq_pct')}")
        print(f"   FWD(>{SCORE_DATE_HI}): exIR={f['excess_ir']} sharpe={f['sharpe']} "
              f"cum={f['cum_return_pct']}% nper={f['n_periods']}")
