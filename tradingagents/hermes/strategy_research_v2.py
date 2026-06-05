"""수급 룰 전략 자율 연구 루프 v2 (LLM-free, 결정론적).

승률(>0.5)·샤프(>1.0) 완화 게이트를 목표로 신규 신호(price_drop/trend_slope/
rolling_corr)를 활용한 후보 전략을 생성→백테스트→게이트 판정→영속화한다.

종료: gate_passed=1 발견 OR ``MAX_EVAL``(=100) 개 평가 완료.
실행: ``uv run python -m tradingagents.hermes.strategy_research_v2``
결과: ``~/.tradingagents/hermes/strategies_v2.db`` 에 매 평가 즉시 저장.

전략 생성은 12개 아키타입의 결정론적 열거다. 앞 8개는 기존 루프 호환용,
뒤 4개는 KOSPI 시장 레짐 필터를 쓰는 추가 루프용 신규 아키타입이다.
"""
from __future__ import annotations

import os
import sys
from itertools import zip_longest

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.market_history import fetch_kospi, fetch_usdkrw
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2
from tradingagents.hermes.strategy_validation import (
    engine_version,
    run_walk_forward_validation,
)

# 한 실행에서 평가할 최대 후보 수. cron 단발 tick(120s 제한)에서는 환경변수
# V2_LLMFREE_MAX 로 소량(예: 8)만 돌려 timeout 을 피한다. 미설정 시 100.
MAX_EVAL = int(os.environ.get("V2_LLMFREE_MAX", "100"))

# 1차 100-run 에서 주로 탐색한 주체군.
SUBJECTS = ["foreign_registered", "foreign", "pension", "other_corp", "securities"]

# 추가 루프에서 넓힐 주체군 (기존 100개에서 상대적으로 저탐색).
# 400개+ 평가 이후에는 FR contrarian 재반복보다 비FR/단기·중기 주체 위주 탐색을 우선한다.
NEXT_LOOP_SUBJECTS = [
    "investment_trust",
    "insurance",
    "bank",
    "private_equity",
    "foreign_unregistered",
]

_ABBR = {
    "foreign_registered": "fr",
    "foreign": "fg",
    "foreign_unregistered": "fu",
    "pension": "pn",
    "other_corp": "oc",
    "securities": "sc",
    "investment_trust": "it",
    "private_equity": "pe",
    "insurance": "in",
    "bank": "bk",
}


def _candidates() -> list[dict]:
    """13개 아키타입 × 그리드 → 후보 spec 리스트 (라운드로빈 인터리브)."""
    a1, a2, a3, a4 = [], [], [], []
    a5, a6, a7, a8 = [], [], [], []
    a9, a10, a11, a12 = [], [], [], []
    a13: list[dict] = []  # 신규: 저공매도압력 (short_ratio 신호)
    a14: list[dict] = []  # 신규: 장기 가격 모멘텀 (price_return 신호)
    a15: list[dict] = []  # 신규: 저변동성 방어 (realized_vol 신호)
    a16: list[dict] = []  # 신규: smart-money 합의 + 개인 흡수 (flow_consensus)
    a17: list[dict] = []  # 신규: 분산 매집 + 유동성 필터 (flow_dispersion/liquidity)
    a18: list[dict] = []  # 신규: 수급 가속 + 단기자금 이탈 청산 (flow_accel/fast_money_unwind)

    # A1 DIP_SUPPORT: 눌림목 + 수급 매수 지지 (기존 100-run 재현용)
    for subj in SUBJECTS:
        for W, X in [(5, 5), (5, 8), (10, 8), (10, 10), (20, 10)]:
            for md in (2, 3):
                for TP, SL, MH in [(5.0, 5.0, 10), (8.0, 8.0, 20), (5.0, 8.0, 10)]:
                    a1.append({
                        "spec_version": 2,
                        "name": f"dipsup-{_ABBR[subj]}-d{W}_{int(X)}-ns{md}-tp{int(TP)}sl{int(SL)}h{MH}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "price_drop", "window": W, "value": X},
                            {"signal": "net_streak", "subject": subj, "min_days": md, "sign": "buy"},
                        ]},
                        "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
                    })

    # A2 DIP_TREND: 상승 레짐 내 눌림목 (기존 100-run 재현용)
    for subj in SUBJECTS:
        for W, X in [(5, 5), (10, 8), (10, 10)]:
            for tw in (20, 60):
                for TP, SL, MH in [(5.0, 5.0, 10), (8.0, 8.0, 20)]:
                    a2.append({
                        "spec_version": 2,
                        "name": f"diptr-{_ABBR[subj]}-d{W}_{int(X)}-ts{tw}-tp{int(TP)}sl{int(SL)}h{MH}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "price_drop", "window": W, "value": X},
                            {"signal": "trend_slope", "subject": subj, "window": tw, "direction": "up"},
                        ]},
                        "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
                    })

    # A3 CONCORDANCE: 수급 매수 + 가격 동조 확인 + 추세 필터 (기존 100-run 재현용)
    for subj in SUBJECTS:
        for md in (3, 5):
            for lb, mr in [(60, 0.2), (60, 0.3)]:
                for TP, SL, MH in [(8.0, 8.0, 20), (10.0, 8.0, 20)]:
                    a3.append({
                        "spec_version": 2,
                        "name": f"conc-{_ABBR[subj]}-ns{md}-cr{lb}_{int(mr * 100)}-tp{int(TP)}sl{int(SL)}h{MH}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "net_streak", "subject": subj, "min_days": md, "sign": "buy"},
                            {"signal": "rolling_corr", "subject": subj, "lookback": lb, "min_r": mr},
                            {"signal": "price_filter", "mode": "above_ma", "window": 20},
                        ]},
                        "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
                    })

    # A4 DIP_BELOW_MA: 순수 역추세 (기존 100-run 재현용)
    for W, X in [(5, 5), (5, 8), (10, 8), (10, 10), (20, 10), (20, 15)]:
        for maw in (20, 60):
            for TP, MH in [(5.0, 5), (5.0, 10), (8.0, 10)]:
                a4.append({
                    "spec_version": 2,
                    "name": f"diponly-d{W}_{int(X)}-bma{maw}-tp{int(TP)}h{MH}",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "price_drop", "window": W, "value": X},
                        {"signal": "price_filter", "mode": "below_ma", "window": maw},
                    ]},
                    "exit": {"take_profit_pct": TP, "max_hold_days": MH},
                })

    # A5 REGIME_EXIT_PULLBACK: 상승 레짐 내 얕은 눌림목 + 레짐 붕괴 시 청산.
    for subj in NEXT_LOOP_SUBJECTS:
        for W, X in [(3, 3), (5, 3), (5, 5)]:
            for maw in (5, 20):
                for SL, TP, MH in [(3.0, 5.0, 5), (3.0, 8.0, 10), (5.0, 5.0, 10)]:
                    a5.append({
                        "spec_version": 2,
                        "name": f"rgx-{_ABBR[subj]}-d{W}_{int(X)}-ma{maw}-sl{int(SL)}tp{int(TP)}h{MH}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "price_drop", "window": W, "value": X},
                            {"signal": "trend_slope", "subject": subj, "window": 20, "direction": "up"},
                            {"signal": "price_filter", "mode": "above_ma", "window": maw},
                        ]},
                        "exit": {
                            "signal_all_of": [
                                {"signal": "trend_slope", "subject": subj, "window": 10, "direction": "down"},
                            ],
                            "take_profit_pct": TP,
                            "stop_loss_pct": SL,
                            "max_hold_days": MH,
                        },
                    })

    # A6 FLOW_REVERSAL_EXIT: 매수 streak 진입 + 매도 streak 청산.
    for subj in NEXT_LOOP_SUBJECTS:
        for md_in, md_out in [(2, 2), (3, 2), (3, 3)]:
            for mr in (0.1, 0.2):
                for TP, SL, MH in [(5.0, 3.0, 5), (8.0, 5.0, 10)]:
                    a6.append({
                        "spec_version": 2,
                        "name": f"frev-{_ABBR[subj]}-in{md_in}-out{md_out}-rc{int(mr * 100)}-sl{int(SL)}tp{int(TP)}h{MH}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "net_streak", "subject": subj, "min_days": md_in, "sign": "buy"},
                            {"signal": "rolling_corr", "subject": subj, "lookback": 20, "min_r": mr},
                        ]},
                        "exit": {
                            "signal_all_of": [
                                {"signal": "net_streak", "subject": subj, "min_days": md_out, "sign": "sell"},
                            ],
                            "take_profit_pct": TP,
                            "stop_loss_pct": SL,
                            "max_hold_days": MH,
                        },
                    })

    # A7 FLOW_SCALE: 거래량 대비 유의미한 순매수 + 추세 동조.
    for subj in NEXT_LOOP_SUBJECTS:
        for vw, vr in [(3, 0.05), (5, 0.1), (10, 0.1)]:
            for maw in (5, 20):
                for TP, SL, MH in [(5.0, 3.0, 5), (8.0, 5.0, 10)]:
                    a7.append({
                        "spec_version": 2,
                        "name": f"fscale-{_ABBR[subj]}-vw{vw}r{int(vr * 100)}-ma{maw}-sl{int(SL)}tp{int(TP)}h{MH}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "net_vol_ratio", "subject": subj, "window": vw, "op": ">=", "value": vr},
                            {"signal": "trend_slope", "subject": subj, "window": 10, "direction": "up"},
                            {"signal": "price_filter", "mode": "above_ma", "window": maw},
                        ]},
                        "exit": {
                            "signal_all_of": [
                                {"signal": "price_filter", "mode": "below_ma", "window": 5},
                            ],
                            "take_profit_pct": TP,
                            "stop_loss_pct": SL,
                            "max_hold_days": MH,
                        },
                    })

    # A8 OWNERSHIP_ROTATION: 비중 상승 + 추세 상향을 함께 본 ownership 전환형.
    for subj in ("investment_trust", "insurance", "foreign_registered", "bank"):
        for delta_w, delta_v, thr_v in [(3, 2, 10), (5, 2, 10), (10, 5, 20)]:
            for SL, TP, MH in [(3.0, 5.0, 5), (5.0, 8.0, 10)]:
                a8.append({
                    "spec_version": 2,
                    "name": f"orot-{_ABBR[subj]}-dw{delta_w}dv{delta_v}tv{thr_v}-sl{int(SL)}tp{int(TP)}h{MH}",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "pct_delta", "subject": subj, "window": delta_w, "op": ">=", "value": delta_v},
                        {"signal": "pct_threshold", "subject": subj, "op": ">=", "value": thr_v},
                        {"signal": "trend_slope", "subject": subj, "window": 10, "direction": "up"},
                    ]},
                    "exit": {
                        "signal_all_of": [
                            {"signal": "pct_delta", "subject": subj, "window": 3, "op": "<=", "value": 2},
                        ],
                        "take_profit_pct": TP,
                        "stop_loss_pct": SL,
                        "max_hold_days": MH,
                    },
                })

    # A9 MARKET_REGIME_PULLBACK: KOSPI 상승 레짐에서만 얕은 눌림목 진입.
    for subj in ("investment_trust", "insurance", "private_equity", "foreign_unregistered"):
        for W, X in [(3, 3), (5, 3), (5, 5)]:
            for mw in (20, 60):
                for SL, TP, MH in [(3.0, 5.0, 5), (3.0, 8.0, 10)]:
                    a9.append({
                        "spec_version": 2,
                        "name": f"mrp-{_ABBR[subj]}-d{W}_{int(X)}-km{mw}-sl{int(SL)}tp{int(TP)}h{MH}",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "price_drop", "window": W, "value": X},
                            {"signal": "trend_slope", "subject": subj, "window": 20, "direction": "up"},
                            {"signal": "market_filter", "mode": "above_ma", "window": mw},
                        ]},
                        "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
                    })

    # A10 MARKET_CONCORDANCE: 수급-가격 동조 + KOSPI 레짐으로 시장 급락 구간 제외.
    for subj in ("investment_trust", "insurance", "bank", "foreign_unregistered"):
        for md in (2, 3):
            for mr in (0.1, 0.2):
                for mw in (20, 60):
                    a10.append({
                        "spec_version": 2,
                        "name": f"mconc-{_ABBR[subj]}-ns{md}-rc{int(mr * 100)}-km{mw}-sl3tp5h5",
                        "direction": "long",
                        "entry": {"all_of": [
                            {"signal": "net_streak", "subject": subj, "min_days": md, "sign": "buy"},
                            {"signal": "rolling_corr", "subject": subj, "lookback": 20, "min_r": mr},
                            {"signal": "market_filter", "mode": "above_ma", "window": mw},
                        ]},
                        "exit": {"take_profit_pct": 5.0, "stop_loss_pct": 3.0, "max_hold_days": 5},
                    })

    # A11 MARKET_FLOW_SCALE: 거래량 대비 순매수 + 시장 레짐 + 짧은 리스크 절단.
    for subj in ("investment_trust", "private_equity", "insurance", "foreign_unregistered"):
        for vw, vr in [(3, 0.05), (5, 0.05), (5, 0.1)]:
            for mw in (20, 60):
                a11.append({
                    "spec_version": 2,
                    "name": f"mfscale-{_ABBR[subj]}-vw{vw}r{int(vr * 100)}-km{mw}-sl3tp5h5",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "net_vol_ratio", "subject": subj, "window": vw, "op": ">=", "value": vr},
                        {"signal": "trend_slope", "subject": subj, "window": 10, "direction": "up"},
                        {"signal": "market_filter", "mode": "above_ma", "window": mw},
                    ]},
                    "exit": {"take_profit_pct": 5.0, "stop_loss_pct": 3.0, "max_hold_days": 5},
                })

    # A12 DEFENSIVE_ROTATION: 장기 주체 비중 상승 + KOSPI 60/120일 레짐.
    for subj in ("foreign_registered", "investment_trust", "insurance", "pension"):
        for delta_w, delta_v in [(3, 2), (5, 2), (10, 5)]:
            for mw in (60, 120):
                a12.append({
                    "spec_version": 2,
                    "name": f"defrot-{_ABBR[subj]}-dw{delta_w}dv{delta_v}-km{mw}-sl3tp5h5",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "pct_delta", "subject": subj, "window": delta_w, "op": ">=", "value": delta_v},
                        {"signal": "trend_slope", "subject": subj, "window": 20, "direction": "up"},
                        {"signal": "market_filter", "mode": "above_ma", "window": mw},
                    ]},
                    "exit": {"take_profit_pct": 5.0, "stop_loss_pct": 3.0, "max_hold_days": 5},
                })

    # A13 SHORT_PRESSURE: 저공매도압력(공매도 비중 낮음) + 추세/매집 — 신규 short_ratio 신호.
    # 학술 근거: 고공매도 종목은 저수익(Boehmer-Huszar-Jordan) → long-only 는 저공매도 선호.
    # 시드 백테스트에서 공정 게이트 통과한 템플릿(short_ratio<=v AND 추세) 의 그리드 변형.
    for subj in ("foreign", "foreign_registered", "pension"):
        for sw, sv in [(10, 10.0), (20, 15.0), (10, 15.0)]:
            for TP, SL, MH in [(15.0, 8.0, 40), (10.0, 5.0, 20)]:
                a13.append({
                    "spec_version": 2,
                    "name": f"shortp-{_ABBR[subj]}-sr{sw}_{int(sv)}-tr60-tp{int(TP)}sl{int(SL)}h{MH}",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "short_ratio", "window": sw, "op": "<=", "value": sv},
                        {"signal": "trend_slope", "subject": subj, "window": 60, "direction": "up"},
                    ]},
                    "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
                })
    # 주체 무관 가격추세 버전 (저공매도 + 60일선 위).
    for sw, sv in [(10, 10.0), (20, 15.0), (5, 10.0)]:
        for TP, SL, MH in [(15.0, 8.0, 40), (10.0, 5.0, 20)]:
            a13.append({
                "spec_version": 2,
                "name": f"shortp-px-sr{sw}_{int(sv)}-ma60-tp{int(TP)}sl{int(SL)}h{MH}",
                "direction": "long",
                "entry": {"all_of": [
                    {"signal": "short_ratio", "window": sw, "op": "<=", "value": sv},
                    {"signal": "price_filter", "mode": "above_ma", "window": 60},
                ]},
                "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
            })

    # A14 MOMENTUM: 장기 가격 모멘텀(price_return) + 추세/시장 레짐 — 신규 price_return 신호.
    # 학술 근거: 횡단면 모멘텀(Jegadeesh-Titman) — 검증 최강 이상현상. long-only leg + 레짐필터로
    # 모멘텀 폭락(Daniel-Moskowitz) 완화. rw=120/250 으로 6~12개월 모멘텀 표현.
    for rw, rv in [(120, 10.0), (250, 20.0), (120, 20.0), (250, 30.0)]:
        for maw in (20, 60):
            for TP, SL, MH in [(15.0, 8.0, 40), (20.0, 10.0, 60)]:
                a14.append({
                    "spec_version": 2,
                    "name": f"mom-pr{rw}_{int(rv)}-ma{maw}-tp{int(TP)}sl{int(SL)}h{MH}",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "price_return", "window": rw, "op": ">=", "value": rv},
                        {"signal": "price_filter", "mode": "above_ma", "window": maw},
                    ]},
                    "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
                })

    # A15 LOW_VOL: 저변동성 방어(realized_vol 낮음) + 추세 — 신규 realized_vol 신호.
    # 학술 근거: 저변동성 이상현상(Baker-Bradley-Wurgler, Frazzini-Pedersen). long-only leg 가 더 견고.
    for vw, vv in [(60, 30.0), (120, 30.0), (60, 40.0), (120, 40.0)]:
        for TP, SL, MH in [(15.0, 8.0, 40), (10.0, 5.0, 20)]:
            a15.append({
                "spec_version": 2,
                "name": f"lowvol-rv{vw}_{int(vv)}-ma60-tp{int(TP)}sl{int(SL)}h{MH}",
                "direction": "long",
                "entry": {"all_of": [
                    {"signal": "realized_vol", "window": vw, "op": "<=", "value": vv},
                    {"signal": "price_filter", "mode": "above_ma", "window": 60},
                ]},
                "exit": {"take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
            })

    # A16 SMART_CONSENSUS: 여러 smart-money 주체가 동시에 순매수하고 개인이 파는 흡수자 패턴.
    # price_drop 없이 구조적 합의/분산 우선순위 1을 직접 검증한다.
    for group, min_buyers in [("fast_money", 2), ("foreign_pair", 2), ("institution_defensive", 2), ("broad_smart", 3)]:
        for w in (3, 5, 10):
            for lw, lv in [(20, 5_000_000_000.0), (60, 5_000_000_000.0), (20, 10_000_000_000.0)]:
                a16.append({
                    "spec_version": 2,
                    "name": f"cons-{group}-w{w}mb{min_buyers}-liq{lw}_{int(lv/1e9)}b-sl5tp8h10",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "flow_consensus", "group": group, "window": w, "min_buyers": min_buyers, "require_retail_sell": True},
                        {"signal": "liquidity_filter", "window": lw, "op": ">=", "value": lv},
                    ]},
                    "exit": {"signal_all_of": [{"signal": "fast_money_unwind", "window": 3}], "take_profit_pct": 8.0, "stop_loss_pct": 5.0, "max_hold_days": 10},
                })

    # A17 DISPERSION_ACCUM: 한 주체 쏠림이 아닌 분산 매집 + 유동성 통제.
    for group, max_share in [("broad_smart", 0.5), ("broad_smart", 0.6), ("institution_defensive", 0.6), ("fast_money", 0.7)]:
        for w in (5, 10, 20):
            for TP, SL, MH in [(8.0, 5.0, 10), (10.0, 5.0, 20)]:
                a17.append({
                    "spec_version": 2,
                    "name": f"disp-{group}-w{w}s{int(max_share*100)}-liq60_5b-tp{int(TP)}sl{int(SL)}h{MH}",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "flow_dispersion", "group": group, "window": w, "max_share": max_share},
                        {"signal": "liquidity_filter", "window": 60, "op": ">=", "value": 5_000_000_000.0},
                    ]},
                    "exit": {"signal_all_of": [{"signal": "fast_money_unwind", "window": 5}], "take_profit_pct": TP, "stop_loss_pct": SL, "max_hold_days": MH},
                })

    # A18 FLOW_ACCEL_UNWIND: 수급 2차 미분으로 진입하고 단기자금 동시 이탈 시 청산.
    for subj in ("foreign_unregistered", "private_equity", "investment_trust", "pension"):
        for short, long in [(3, 20), (5, 20), (10, 60)]:
            for zwin, zlb, zmin in [(5, 60, 1.0), (10, 120, 1.5)]:
                a18.append({
                    "spec_version": 2,
                    "name": f"accel-{_ABBR[subj]}-s{short}l{long}-z{zwin}_{zlb}_{int(zmin*10)}-sl5tp8h10",
                    "direction": "long",
                    "entry": {"all_of": [
                        {"signal": "flow_accel", "subject": subj, "short": short, "long": long},
                        {"signal": "flow_zscore", "subject": subj, "window": zwin, "lookback": zlb, "min_z": zmin},
                        {"signal": "liquidity_filter", "window": 20, "op": ">=", "value": 5_000_000_000.0},
                    ]},
                    "exit": {"signal_all_of": [{"signal": "fast_money_unwind", "window": 3}], "take_profit_pct": 8.0, "stop_loss_pct": 5.0, "max_hold_days": 10},
                })

    # 라운드로빈 인터리브 — 신규 아키타입(A16~A18 수급/유동성, A9~A12 KOSPI 레짐)을 최우선 배치.
    out: list[dict] = []
    for tup in zip_longest(a16, a17, a18, a9, a10, a11, a12, a5, a6, a7, a8, a13, a14, a15, a1, a2, a3, a4):
        for s in tup:
            if s is not None:
                out.append(s)
    return out


def _dedup(specs: list[dict]) -> list[dict]:
    """spec_hash 기준 중복 제거 (인터리브 순서 보존)."""
    seen, out = set(), []
    for s in specs:
        validate_spec_v2(s)
        h = spec_hash(s)
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out


def _phase_filter(specs: list[dict], existing_count: int) -> list[dict]:
    """평가 수가 충분히 쌓인 뒤엔 2차 루프용 아키타입에 집중한다.

    400개+ 평가 시 legacy 역추세 반복(A1~A4)을 뒤로 미뤄서,
    시장 레짐/rotation/flow 계열(A5~A12)만 먼저 소진한다.
    """
    if existing_count < 400:
        return specs
    advanced_prefixes = (
        "cons-", "disp-", "accel-",
        "shortp-", "mom-", "lowvol-",
        "mrp-", "mconc-", "mfscale-", "defrot-",
        "rgx-", "frev-", "fscale-", "orot-",
    )
    advanced = [s for s in specs if s["name"].startswith(advanced_prefixes)]
    legacy = [s for s in specs if not s["name"].startswith(advanced_prefixes)]
    return advanced + legacy


def _fmt(m: dict) -> str:
    def g(k):
        v = m.get(k)
        return "  n/a" if v is None else f"{v:5.2f}"
    return (f"win{g('win_rate')} shp{g('sharpe')} exIR{g('excess_sharpe')} "
            f"mdd{g('mdd_pct')} n{m.get('n_trades', 0):>4}")


def _existing_hashes(store: StrategyStoreV2) -> set[str]:
    """이미 저장된 spec_hash 집합 — 추가 루프에서 중복 재백테스트 방지."""
    with store._conn() as c:
        rows = c.execute("SELECT spec_hash FROM strategies").fetchall()
    return {r[0] for r in rows}


def _merge_short(df, ticker: str, store) -> object:
    """holdings df 에 short 엔드포인트의 공매도 비중 컬럼을 date 기준 left-merge.

    same-day 값(close 시점 확정)이라 look-ahead 0. short 데이터 부재/실패 시 df 원본
    반환 → ``short_ratio`` 신호는 컬럼 부재로 False(하위호환).
    """
    try:
        sdf = store.read(ticker, "short")
    except Exception:        # noqa: BLE001
        return df
    if sdf is None or getattr(sdf, "empty", True) or "short_volume_ratio" not in sdf.columns:
        return df
    cols = ["date", "short_volume_ratio"]
    if "short_amount_ratio" in sdf.columns:
        cols.append("short_amount_ratio")
    sdf = sdf[cols].copy()
    sdf["date"] = sdf["date"].astype(str)
    out = df.copy()
    out["date"] = out["date"].astype(str)
    return out.merge(sdf, on="date", how="left")


def _preload(tickers: list[str]):
    """holdings 전부 1회 로딩(메모리 캐시) + short 비중 머지 + 시장데이터 1회.

    반환 ``(loaded_tickers, cached_loader, cached_kospi_fetcher, cached_usdkrw_fetcher)``.
    시장/FX fetch 를 spec 별로 반복하면 캐시 TTL·네트워크 결손 때문에 같은 전략의
    지표가 run 마다 달라질 수 있어, 연구 루프 시작 시점의 스냅샷으로 고정한다.
    """
    holdings: dict[str, object] = {}
    min_d = max_d = None
    _short_store = KisHistoryStore()
    for tk in tickers:
        df, _meta = load_holdings(tk)
        if df is None or df.empty:
            continue
        holdings[tk] = _merge_short(df, tk, _short_store)
        d0, d1 = str(df["date"].iloc[0]), str(df["date"].iloc[-1])
        min_d = d0 if (min_d is None or d0 < min_d) else min_d
        max_d = d1 if (max_d is None or d1 > max_d) else max_d

    kospi = None
    if min_d and max_d:
        try:
            kospi = fetch_kospi(min_d, max_d)
        except Exception:        # noqa: BLE001
            kospi = None
    usdkrw = None
    if min_d and max_d:
        try:
            usdkrw = fetch_usdkrw(min_d, max_d)
        except Exception:        # noqa: BLE001
            usdkrw = None

    def loader(tk):
        return holdings[tk], {}

    def kospi_fetcher(_s, _e):
        return kospi

    def usdkrw_fetcher(_s, _e):
        return usdkrw

    return sorted(holdings.keys()), loader, kospi_fetcher, usdkrw_fetcher


def main() -> int:
    raw_tickers = KisHistoryStore().list_tickers()
    store = StrategyStoreV2()
    existing_hashes = _existing_hashes(store)
    cands_all = _phase_filter(_dedup(_candidates()), len(existing_hashes))
    cands = [s for s in cands_all if spec_hash(s) not in existing_hashes]
    print(f"[research-v2] preloading {len(raw_tickers)} tickers ...", flush=True)
    tickers, loader, kospi_fetcher, usdkrw_fetcher = _preload(raw_tickers)
    print(f"[research-v2] universe={len(tickers)} candidates={len(cands)} "
          f"(all={len(cands_all)} existing={len(existing_hashes)}) "
          f"MAX_EVAL={MAX_EVAL} gate: walk-forward 초과수익 IR>0.5 (모든 OOS창 시장초과)",
          flush=True)

    evaluated = 0
    passed = []
    for spec in cands:
        if evaluated >= MAX_EVAL:
            break
        result = run_universe_backtest_v2(
            spec, tickers, loader=loader, kospi_fetcher=kospi_fetcher,
            usdkrw_fetcher=usdkrw_fetcher)
        # 공정 게이트 — walk-forward × 시장대비 초과수익 IR. 모든 OOS 창에서 시장을
        # 이기고(IR>0) 중앙 IR>임계 여야 통과. 단일 분할 레짐편향·시장베타 오인을 제거.
        wf_result = run_walk_forward_validation(
            spec, tickers, loader=loader, kospi_fetcher=kospi_fetcher,
            usdkrw_fetcher=usdkrw_fetcher)
        sid, is_new = store.save(spec, result, name=spec["name"],
                                 wf_result=wf_result,
                                 engine_version=engine_version())
        evaluated += 1
        gate = wf_result["gate_passed"]
        inm, outm = result["in_sample"], result["out_sample"]
        flag = "PASS" if gate else "    "
        print(f"[{evaluated:>3}/{MAX_EVAL}] {flag} #{sid} {spec['name'][:34]:34} "
              f"| xsec in {_fmt(inm)} out {_fmt(outm)} "
              f"| wf nwin={wf_result['n_windows']} exIRmed={wf_result['oos_excess_ir_median']} "
              f"exIRmin={wf_result['oos_excess_ir_min']}", flush=True)
        if gate:
            passed.append((sid, spec, result))
            print(f"\n🎯 GATE PASSED (walk-forward 초과수익 IR) — #{sid} {spec['name']}", flush=True)
            print(f"   wf: nwin={wf_result['n_windows']} "
                  f"exIR med={wf_result['oos_excess_ir_median']} "
                  f"min={wf_result['oos_excess_ir_min']}", flush=True)
            print(f"   xsec in:{_fmt(inm)} out:{_fmt(outm)}", flush=True)
            break

    print(f"\n[research-v2] done. evaluated={evaluated} passed={len(passed)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
