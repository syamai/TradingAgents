"""종목별 주도세력(per-stock leader) 추종 전략 발굴·검증.

배경: newloop spec 은 scale-free(한 신호규칙을 유니버스 전체 적용)라 "종목 A=외국인,
B=사모" 같은 종목별 주도세력 지정을 표현 못 한다. 이 모듈은 검증된 엔진 부품을
재사용하되(``_simulate_v2``·``ols_alpha_beta_clustered``·시간분할 상수), 종목마다 leader
subject 를 주입한 spec 을 인스턴스화해 per-stock 으로 발굴·검증하는 신규 레이어다.

가설(사용자): 종목마다 고유한 주도세력이 있고, 그 주도세력의 매수(누적 z급증)·매도
(누적 되돌림=flow_unwind) 타이밍을 따르면 성공확률이 높다. 모든 종목이 해당되진 않으니
**명확히 해당되는 종목만** 발굴한다.

선택편향 차단(설계 핵심):
  - 시간 3분할: SELECT_IS(leader 선정·튜닝 전부) / embargo / SCORE_OOS(≤2025-06-30 채점)
    / FORWARD(>2025-07-01 관찰전용). per-stock 은 종목 고유라 시간축이 유일한 OOS.
  - Stage A(IS): 종목별 9주체 × 진입그리드 백테스트 → IS 게이트 통과 중 t(α) 최대 1개를
    leader 로. Stage B(OOS): 동결 leader 를 OOS 에서 단 1회 채점.
  - IS-OOS 부호일치(둘 다 α>0) 요구가 Stage A 선택편향을 흡수(우연 leader 는 OOS 부호반전).
  - 횡단 BH-FDR(α=0.10)로 종목 간 다중검정 통제. deflated_t_threshold(N_eff)는 진단 동봉.
  - placebo: leader 를 무작위 다른 주체로 셔플 → 합격률이 기대 위양성에 수렴해야 정상.

벤치마크: 거래별 alpha 는 KOSPI 같은구간 수익 대비 초과(시장초과 = 급등 레짐 베타 중립).
비용·룩어헤드0 는 ``_simulate_v2``(BUY/SELL 0.099%×2, i→i+1 익일체결)가 그대로 보장.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.market_history import fetch_kospi
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _simulate_v2
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.hermes.strategy_validation import (
    DEFAULT_EMBARGO_DAYS,
    IN_SAMPLE_PCT,
    SCORE_DATE_HI,
    _HI_SENTINEL,
    engine_version,
)
from tradingagents.newloop.stage1_gate import (
    GATE_DOWN_T_FLOOR,
    _mean_t,
    deflated_t_threshold,
    ols_alpha_beta_clustered,
)

# 후보 주도세력 (개인=numeraire 제외, 통합 foreign 제외 — 세분 9주체).
LEADER_SUBJECTS: tuple[str, ...] = (
    "foreign_registered", "foreign_unregistered", "pension", "private_equity",
    "investment_trust", "securities", "bank", "insurance", "other_corp",
)
# 매수 신호 그리드 (lookback=120 고정 — expanding-z min_periods 규율 근사 + 시행수 억제).
ENTRY_WINDOW_GRID: tuple[int, ...] = (5, 10)
ENTRY_LOOKBACK = 120
ENTRY_MIN_Z_GRID: tuple[float, ...] = (1.5, 2.0)
# 매도 = cum_qty leader-exit(flow_unwind) 고정 1셋 + 안전장치(사용자 결정: 처음부터 leader-exit).
EXIT_UNWIND_WINDOW = 20
EXIT_UNWIND_DRAWDOWN = 50.0
EXIT_STOP_LOSS = 8.0
EXIT_MAX_HOLD = 20

# per-stock floor — 단일종목 z-진입은 거래가 희소(6년 IS에 ~7~30건)해 stage1 우주
# 게이트(n≥30·t≥3.0)는 도달 불가. 설계 의도대로 IS 는 *관대한 선정기*(Stage A 자유
# 선정), 통계 rigor 는 OOS 부호일치 + 횡단 BH-FDR + forward(Stage B/3)에 둔다.
# IS floor 는 추정 타당성(군집-t 유효 표본) + 양의 알파 + 잡음 배제 수준의 최소치다.
IS_MIN_TRADES = 15                       # 군집-로버스트 추정에 필요한 최소 거래수
IS_MIN_CLUSTERS = 6                      # 진입월 군집 — 시기 다양성
IS_MIN_T_SELECT = 1.5                    # 선정용 약한 t (유의 바 아님 — 잡음 leader 배제)
OOS_MIN_TRADES = 8                       # OOS 창은 짧아 더 완화(추정 가능 최소)
OOS_MIN_CLUSTERS = 4
FDR_ALPHA = 0.10                         # 탐색적(사용자 결정) — forward 가 최종 검증대

_NORM = statistics.NormalDist()
_OOS_HI = "2025-07-01"                   # SCORE_OOS 상한(배타) = SCORE_DATE_HI 다음날 = FWD_START
FWD_START = "2025-07-01"                 # forward 관찰 시작(채점 금지 구간)
MIN_FORWARD_WEEKS = 4
REGISTRY = Path.home() / ".tradingagents/newloop/leader_forward_registry.json"

# 파일럿 종목(plan): 일진전기·DB손해보험 + 외국인 생존 13.
PILOT_TICKERS: tuple[str, ...] = (
    "103590", "005830", "000660", "042660", "006400", "086280", "086520",
    "290650", "056080", "348370", "214150", "183300", "080220",
)


# === spec 빌더 ===

def make_leader_spec(subject: str, *, entry_window: int, min_z: float,
                     entry_lookback: int = ENTRY_LOOKBACK,
                     unwind_window: int = EXIT_UNWIND_WINDOW,
                     unwind_drawdown: float = EXIT_UNWIND_DRAWDOWN,
                     stop_loss: float = EXIT_STOP_LOSS,
                     max_hold: int = EXIT_MAX_HOLD) -> dict:
    """leader subject 를 entry(flow_zscore)·exit(flow_unwind)에 주입한 정규 v2 spec."""
    spec = {
        "spec_version": 2,
        "name": f"leader-{subject}-w{entry_window}-z{min_z}",
        "direction": "long",
        "entry": {"all_of": [
            {"signal": "flow_zscore", "subject": subject,
             "window": entry_window, "lookback": entry_lookback, "min_z": min_z}]},
        "exit": {"signal_all_of": [
            {"signal": "flow_unwind", "subject": subject,
             "window": unwind_window, "drawdown_pct": unwind_drawdown}],
            "stop_loss_pct": stop_loss, "max_hold_days": max_hold},
    }
    validate_spec_v2(spec)
    return spec


# === 시장(KOSPI) 같은구간 수익 basket — 거래별 시장초과 알파용 ===

def _kospi_ret_fn(kospi: Optional[pd.DataFrame]) -> Callable[[str, str], Optional[float]]:
    if kospi is None or kospi.empty:
        return lambda _a, _b: None
    s = pd.Series(bt._num(kospi["close"]).to_numpy(dtype=float),
                  index=kospi["date"].astype(str).to_numpy())
    s = s[~s.index.duplicated(keep="last")]

    def f(entry_date: str, exit_date: str) -> Optional[float]:
        if entry_date in s.index and exit_date in s.index:
            e, x = float(s[entry_date]), float(s[exit_date])
            if e > 0:
                return (x / e - 1.0) * 100.0
        return None

    return f


# === 윈도우 슬라이스 백테스트 → (net%, KOSPI%, 진입월) pairs (누수0: 경계 강제청산) ===

def _window_pairs(spec: dict, df: pd.DataFrame, kospi_fn, lo: str, hi: str) -> list[tuple]:
    d = df["date"].astype(str)
    sub = df[(d >= lo) & (d < hi)].reset_index(drop=True)
    if len(sub) < 30:
        return []
    trades, _daily, _active, _cdf = _simulate_v2(spec, sub)
    pairs: list[tuple] = []
    for t in trades:
        if t.get("exit_reason") == "forced_eod":     # 경계/말단 강제청산 제외(보수적)
            continue
        br = kospi_fn(t["entry_date"], t["exit_date"])
        if br is None:
            continue
        pairs.append((float(t["net_ret_pct"]), float(br), str(t["entry_date"])[:7]))
    return pairs


def _score(pairs: list[tuple]) -> Optional[dict]:
    """pairs → 군집-로버스트 alpha/t (stage1 계산기 재사용). 게이트 판정은 호출자."""
    if len(pairs) < 3:
        return None
    nets = [p[0] for p in pairs]
    baskets = [p[1] for p in pairs]
    months = [p[2] for p in pairs]
    reg = ols_alpha_beta_clustered(baskets, nets, months)
    if reg is None:
        return None
    a, t_a, b, _t_b, n_clusters = reg
    down = [nn - x for nn, x in zip(nets, baskets) if x < 0]
    dn_n, _dn_m, dn_t = _mean_t(down)
    return {
        "n_trades": len(pairs), "n_clusters": int(n_clusters),
        "alpha_pct": round(a, 4), "t_alpha": round(t_a, 3), "beta": round(b, 4),
        "down_n": dn_n, "down_t": round(dn_t, 3) if dn_t is not None else None,
    }


def _is_select(m: dict) -> bool:
    """IS 선정기(Stage A) — 추정 타당성 + 양의 시장초과 알파 + 하락위장 아님.

    유의(t≥3.0) 바가 아니라 *선정* 바다. 진짜 leader 증명은 OOS 부호일치 + 횡단
    FDR + forward 가 맡는다(단일종목 t 는 거래 희소로 3.0 도달 불가).
    """
    return bool(
        m["n_trades"] >= IS_MIN_TRADES
        and m["n_clusters"] >= IS_MIN_CLUSTERS
        and m["alpha_pct"] > 0                      # 매수→상승(롱 전략 방향)
        and m["t_alpha"] >= IS_MIN_T_SELECT         # 약한 신호 바(잡음 배제)
        and (m["down_t"] is None or m["down_t"] > GATE_DOWN_T_FLOOR)
    )


def _split_dates(df: pd.DataFrame) -> Optional[dict]:
    """종목 거래일을 SELECT_IS / embargo / SCORE_OOS(≤2025-06-30) 로 분할."""
    d = df["date"].astype(str)
    dates = sorted(d[d <= SCORE_DATE_HI].unique().tolist())
    if len(dates) < 300:                  # IS+OOS 채점에 충분한 표본 없으면 제외
        return None
    cut = int(len(dates) * IN_SAMPLE_PCT / 100)
    oos_idx = min(cut + DEFAULT_EMBARGO_DAYS, len(dates) - 1)
    return {"is_lo": dates[0], "is_hi": dates[cut],
            "oos_lo": dates[oos_idx], "oos_hi": _OOS_HI}


# === 종목 1개 발굴 (Stage A 선정 + Stage B OOS 채점) ===

def discover_stock(ticker: str, *, loader=load_holdings, kospi=None,
                   subject_pool: tuple[str, ...] = LEADER_SUBJECTS) -> dict:
    df, meta = loader(ticker)
    market = (meta or {}).get("market") if isinstance(meta, dict) else None
    name = (meta or {}).get("name") if isinstance(meta, dict) else None
    base = {"ticker": ticker, "market": market, "name": name}
    if df is None or df.empty:
        return {**base, "skipped": "no_data", "n_trials": 0, "qualified": False}
    sp = _split_dates(df)
    if sp is None:
        return {**base, "skipped": "insufficient_history", "n_trials": 0, "qualified": False}
    kfn = _kospi_ret_fn(kospi)

    # Stage A — IS: 9주체 × 진입그리드 → IS 게이트 통과 후보. leader = t(α) 최대 1개.
    candidates: list[tuple] = []
    n_trials = 0
    for subj in subject_pool:
        if f"{subj}_net_qty" not in df.columns:
            continue
        for w in ENTRY_WINDOW_GRID:
            for z in ENTRY_MIN_Z_GRID:
                spec = make_leader_spec(subj, entry_window=w, min_z=z)
                n_trials += 1
                m = _score(_window_pairs(spec, df, kfn, sp["is_lo"], sp["is_hi"]))
                if m is not None and _is_select(m):
                    candidates.append((subj, spec, m))

    if not candidates:
        return {**base, "leader_subject": None, "qualified": False,
                "n_trials": n_trials, "split": sp, "is_metrics": None, "oos_metrics": None}

    subj, spec, is_m = max(candidates, key=lambda c: c[2]["t_alpha"])
    # Stage B — OOS: 동결 leader spec 단 1회 채점.
    oos_m = _score(_window_pairs(spec, df, kfn, sp["oos_lo"], sp["oos_hi"]))
    return {
        **base, "leader_subject": subj,
        "entry_spec": spec["entry"], "exit_spec": spec["exit"],
        "n_trials": n_trials, "n_candidates": len(candidates), "split": sp,
        "is_metrics": is_m, "oos_metrics": oos_m,
    }


# === 횡단 BH-FDR ===

def _bh_survivors(pvals: list[float], alpha: float) -> set[int]:
    if not pvals:
        return set()
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    keep: set[int] = set()
    for rank, idx in enumerate(order, 1):
        if pvals[idx] <= alpha * rank / m:
            keep = set(order[:rank])
    return keep


def _finalize(results: list[dict], *, n_eff: int, fdr_alpha: float = FDR_ALPHA) -> None:
    """OOS floor + 부호일치 통과분에 횡단 BH-FDR 적용 → qualified 마킹 (in-place)."""
    t_crit_strict = deflated_t_threshold(n_eff)
    eligible: list[dict] = []
    for r in results:
        r["qualified"] = False
        is_m, oos_m = r.get("is_metrics"), r.get("oos_metrics")
        if not is_m or not oos_m:
            continue
        floor = (oos_m["n_trades"] >= OOS_MIN_TRADES and oos_m["n_clusters"] >= OOS_MIN_CLUSTERS)
        sign_match = bool(is_m["alpha_pct"] > 0 and oos_m["alpha_pct"] > 0)
        r["oos_floor_pass"] = floor
        r["sign_match"] = sign_match
        if floor and sign_match and oos_m["t_alpha"] is not None:
            r["p_value"] = round(_NORM.cdf(-oos_m["t_alpha"]), 6)   # 단측(α>0)
            r["strict_pass"] = bool(oos_m["t_alpha"] >= t_crit_strict)  # 진단(FWER-strict)
            eligible.append(r)
    keep = _bh_survivors([r["p_value"] for r in eligible], fdr_alpha)
    for i, r in enumerate(eligible):
        r["fdr_survivor"] = i in keep
        r["qualified"] = bool(i in keep)


# === 발굴 오케스트레이션 ===

def discover(tickers: list[str], *, loader=load_holdings,
             kospi_fetcher: Callable = fetch_kospi, fdr_alpha: float = FDR_ALPHA) -> dict:
    kospi = None
    try:
        kospi = kospi_fetcher("2015-01-01", SCORE_DATE_HI)
    except Exception:        # noqa: BLE001
        kospi = None
    results = [discover_stock(tk, loader=loader, kospi=kospi) for tk in tickers]
    n_eff = sum(int(r.get("n_trials", 0)) for r in results)
    _finalize(results, n_eff=n_eff, fdr_alpha=fdr_alpha)
    qualified = [r for r in results if r.get("qualified")]
    return {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "engine_version": engine_version(),
        "score_date_hi": SCORE_DATE_HI,
        "fdr_alpha": fdr_alpha,
        "n_eff": n_eff,
        "t_crit_strict": round(deflated_t_threshold(n_eff), 3),
        "n_universe": len(tickers),
        "n_qualified": len(qualified),
        "stocks": results,
    }


def placebo(tickers: list[str], *, loader=load_holdings,
            kospi_fetcher: Callable = fetch_kospi, seed: int = 7,
            fdr_alpha: float = FDR_ALPHA) -> dict:
    """음성 대조군 — 종목별 leader 후보를 무작위 1주체로 강제 후 동일 파이프라인.

    선택편향이 제거됐다면 placebo 합격률은 기대 위양성(≈fdr_alpha)에 수렴해야 한다.
    """
    kospi = None
    try:
        kospi = kospi_fetcher("2015-01-01", SCORE_DATE_HI)
    except Exception:        # noqa: BLE001
        kospi = None
    rng = random.Random(seed)
    results = []
    for tk in tickers:
        forced = (rng.choice(LEADER_SUBJECTS),)     # 무작위 1주체만 후보로
        results.append(discover_stock(tk, loader=loader, kospi=kospi, subject_pool=forced))
    n_eff = sum(int(r.get("n_trials", 0)) for r in results)
    _finalize(results, n_eff=n_eff, fdr_alpha=fdr_alpha)
    return {"seed": seed, "n_qualified": sum(1 for r in results if r.get("qualified")),
            "n_universe": len(tickers), "stocks": results}


# === per-stock forward 트래커 (forward.py 원칙 차용 — ledger 비접촉, 클램프 우회) ===

def _read_registry() -> list[dict]:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return []


def _write_registry(reg: list[dict]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def register_leaders(leader_map: dict, registered_at: Optional[str] = None) -> int:
    """qualified 종목을 forward_registry 에 동결 등록. 신규 ticker 만 추가(재등록 무시)."""
    registered_at = registered_at or datetime.now(timezone.utc).astimezone().isoformat()
    reg = _read_registry()
    seen = {e["ticker"] for e in reg}
    added = 0
    for r in leader_map.get("stocks", []):
        if not r.get("qualified") or r["ticker"] in seen:
            continue
        reg.append({
            "ticker": r["ticker"], "market": r.get("market"), "name": r.get("name"),
            "leader_subject": r["leader_subject"],
            "entry_spec": r["entry_spec"], "exit_spec": r["exit_spec"],
            "freeze_date": SCORE_DATE_HI, "registered_at": registered_at,
            "engine_version": leader_map.get("engine_version"),
            "is_metrics": r.get("is_metrics"), "oos_metrics": r.get("oos_metrics"),
        })
        added += 1
    _write_registry(reg)
    return added


def observe_leaders(now_iso: Optional[str] = None, *, loader=load_holdings,
                    kospi_fetcher: Callable = fetch_kospi) -> list[dict]:
    """등록 leader 의 forward(>2025-07-01) 관찰 메트릭 + 경과주수·판정자격. 채점 아님."""
    reg = _read_registry()
    if not reg:
        return []
    kospi = None
    try:
        kospi = kospi_fetcher("2015-01-01", "2027-01-01")     # forward 까지 덮음
    except Exception:        # noqa: BLE001
        kospi = None
    kfn = _kospi_ret_fn(kospi)
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now(timezone.utc).astimezone()
    rows = []
    for e in reg:
        spec = {"spec_version": 2, "name": f"fwd-{e['ticker']}", "direction": "long",
                "entry": e["entry_spec"], "exit": e["exit_spec"]}
        df, _meta = loader(e["ticker"])
        fwd = None
        if df is not None and not df.empty:
            fwd = _score(_window_pairs(spec, df, kfn, FWD_START, _HI_SENTINEL))
        weeks = (now - datetime.fromisoformat(e["registered_at"])).days / 7.0
        rows.append({"ticker": e["ticker"], "leader_subject": e["leader_subject"],
                     "registered_at": e["registered_at"], "weeks_elapsed": round(weeks, 1),
                     "eligible_for_decision": weeks >= MIN_FORWARD_WEEKS,
                     "is": e.get("is_metrics"), "oos": e.get("oos_metrics"), "fwd": fwd})
    return rows


# === CLI ===

def main() -> int:
    ap = argparse.ArgumentParser(description="종목별 주도세력 추종 전략 발굴·검증")
    ap.add_argument("--discover", action="store_true", help="유니버스 스캔 → leader_map JSON")
    ap.add_argument("--placebo", action="store_true", help="음성 대조군(주체 셔플) 합격률")
    ap.add_argument("--register", action="store_true", help="--map-file 의 qualified 를 forward 등록")
    ap.add_argument("--observe", action="store_true", help="등록 leader forward 관찰")
    ap.add_argument("--tickers", default=None, help="쉼표구분 종목코드 (기본: 파일럿 13)")
    ap.add_argument("--map-file", default=None, help="--register 입력 leader_map JSON")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    tickers = (args.tickers.split(",") if args.tickers else list(PILOT_TICKERS))

    if args.discover:
        res = discover(tickers)
        if args.placebo:
            res["placebo"] = placebo(tickers)
        _emit(res, args.out)
        q = [r for r in res["stocks"] if r.get("qualified")]
        print(f"발굴 {res['n_universe']}종목 | N_eff={res['n_eff']} (t_crit_strict={res['t_crit_strict']}) "
              f"| qualified {res['n_qualified']}: "
              + ", ".join(f"{r['ticker']}({r['leader_subject']},OOSα={r['oos_metrics']['alpha_pct']}%"
                          f",t={r['oos_metrics']['t_alpha']})" for r in q))
        if "placebo" in res:
            print(f"placebo 합격 {res['placebo']['n_qualified']}/{res['placebo']['n_universe']} "
                  f"(기대 위양성 ≈ {FDR_ALPHA*max(res['n_qualified'],1):.1f})")
        return 0

    if args.placebo:
        res = placebo(tickers)
        _emit(res, args.out)
        print(f"placebo 합격 {res['n_qualified']}/{res['n_universe']}")
        return 0

    if args.register:
        if not args.map_file:
            print("--register 에는 --map-file 필요")
            return 2
        leader_map = json.loads(Path(args.map_file).read_text())
        added = register_leaders(leader_map)
        print(f"forward 등록 {added}건 → {REGISTRY}")
        return 0

    if args.observe:
        rows = observe_leaders()
        _emit(rows, args.out)
        for r in rows:
            print(f"{r['ticker']}({r['leader_subject']}) {r['weeks_elapsed']}주 "
                  f"eligible={r['eligible_for_decision']} fwd={r['fwd']}")
        return 0

    ap.print_help()
    return 1


def _emit(obj, out: Optional[str]) -> None:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
