"""Hermes 가설 파이프라인 과거 데이터 백테스트.

운영 진입 시 신규 종목 가설은 호라이즌(2~4주) 만기까지 결과 확인이 불가하다.
대신 *과거 시점* as_of_date 로 가설을 생성하고, 이미 실현된 KOSPI 상대 수익으로
지금 즉시 라벨링해 적중률을 뽑는다.

데이터 누수(look-ahead) 차단:
  - 분석가 입력은 라이브 KIS 가 아니라 ``~/.tradingagents/kis_history/kis.db``
    (point-in-time 5y 스토어) 에서 as_of_date 이하만 슬라이스해 주입.
  - news/sentiment/fundamentals 분석가는 검색 API 가 최신 결과를 끌어와
    미래 정보 오염 위험 → 백테스트에서 *제외*. 수급(KIS)·기술적(가격/지표)만.
  - 라벨링은 기존 LabelingScheduler 그대로 — as_of~target 실현 가격(yfinance).

두 가지 가설 생성 방식 비교:
  - rule : 수급 raw 수치(외국인+기관 5일 순매수 부호)에서 방향 도출. LLM 0.
  - llm  : 수급+기술적 분석가 리포트(gemma)를 LLM 이 읽고 방향/스탠스 추출.

저장소는 운영 ``hypotheses.db`` 를 오염시키지 않도록 별도 root 사용:
  - rule → ~/.tradingagents/hermes/backtest/rule/hypotheses.db
  - llm  → ~/.tradingagents/hermes/backtest/llm/hypotheses.db

usage:
  uv run python scripts/hermes_backtest.py rule          # 규칙 기반 생성+라벨
  uv run python scripts/hermes_backtest.py llm            # LLM 합성 생성+라벨 (subset)
  uv run python scripts/hermes_backtest.py report         # 두 저장소 결과 집계만
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# === 백테스트 설정 ===

BACKTEST_TODAY = "2026-05-30"  # 라벨링 기준 '오늘' (모든 호라이즌 만기 확보)

# 대형주 20 (전부 kis.db 보유 확인됨)
TICKERS = [
    "005930", "000660", "005380", "005490", "035420",
    "051910", "006400", "035720", "105560", "055550",
    "000270", "012330", "068270", "207940", "028260",
    "066570", "003550", "015760", "017670", "034730",
]
# LLM 합성은 비용/시간이 커 subset 만 (앞 2 종목 × 6 as-of = 12 cell)
LLM_TICKERS = TICKERS[:2]

AS_OF_DATES = [
    "2025-09-15", "2025-10-15", "2025-11-17",
    "2025-12-15", "2026-01-15", "2026-02-16",
]
HORIZONS = [2, 4]

RULE_LOOKBACK = 5      # 수급 신호 집계 거래일 수
RULE_BAND = 0.15       # |순매수비율| 이 band 미만이면 neutral

KIS_DB = Path.home() / ".tradingagents" / "kis_history" / "kis.db"
BACKTEST_ROOT = Path.home() / ".tradingagents" / "hermes" / "backtest"
ANALYST_MODEL = "gemma4:26b-a4b"

# === kis.db point-in-time 리더 ===

_INVESTOR_COLS = [
    "date", "close",
    "foreign_amount", "foreign_registered_amount", "foreign_unregistered_amount",
    "institution_amount", "pension_amount", "private_equity_amount",
    "investment_trust_amount", "securities_amount", "bank_amount",
    "insurance_amount", "retail_amount", "other_corp_amount",
]


def _kis_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{KIS_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_window(table: str, cols: list[str], code6: str, end_date: str, n: int) -> list[dict]:
    """``end_date`` 이하 최신 n 거래일 (newest first) 을 dict 리스트로."""
    with _kis_conn() as c:
        rows = c.execute(
            f"SELECT {', '.join(cols)} FROM {table} "
            f"WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT ?",
            (code6, end_date, n),
        ).fetchall()
    return [dict(r) for r in rows]


# === 분석가 KIS fetcher monkeypatch (라이브 → kis.db) ===

def install_kis_history_patch() -> None:
    """분석가가 호출하는 3개 kis_api fetcher 를 kis.db 리더로 교체.

    분석가 코드는 *수정하지 않는다* (Hermes 규약: wrap, don't edit).
    """
    from tradingagents.dataflows import kis_api

    def fetch_investor_trend(code6, end_date, lookback_days=7):
        return _read_window("investor", _INVESTOR_COLS, code6, end_date, lookback_days)

    def fetch_program_trading(code6, end_date, lookback_days=7):
        return _read_window(
            "program", ["date", "close", "net_qty", "net_amount"],
            code6, end_date, lookback_days,
        )

    def fetch_short_interest(code6, start_date, end_date, lookback_days=7):
        return _read_window(
            "short",
            ["date", "close", "short_qty", "short_volume_ratio",
             "short_amount", "short_amount_ratio"],
            code6, end_date, lookback_days,
        )

    kis_api.fetch_investor_trend = fetch_investor_trend
    kis_api.fetch_program_trading = fetch_program_trading
    kis_api.fetch_short_interest = fetch_short_interest


# === 규칙 기반 신호 ===

def rule_signal(code6: str, as_of: str) -> dict:
    """수급 5일 외국인+기관 순매수 부호 → 방향.

    return: {direction, ratio, net, gross, n}
    """
    rows = _read_window("investor", _INVESTOR_COLS, code6, as_of, RULE_LOOKBACK)
    if not rows:
        return {"direction": "neutral", "ratio": 0.0, "net": 0, "gross": 0, "n": 0}
    net = sum(r["foreign_amount"] + r["institution_amount"] for r in rows)
    gross = sum(abs(r["foreign_amount"]) + abs(r["institution_amount"]) for r in rows)
    ratio = net / gross if gross else 0.0
    if ratio >= RULE_BAND:
        direction = "bullish"
    elif ratio <= -RULE_BAND:
        direction = "bearish"
    else:
        direction = "neutral"
    return {"direction": direction, "ratio": ratio, "net": net, "gross": gross, "n": len(rows)}


def _horizon_pred(direction: str, horizon: int) -> float:
    base = 2.0 + (horizon - 2) * 0.5  # labeling.horizon_threshold 와 동일 스케일
    if direction == "bullish":
        return base
    if direction == "bearish":
        return -base
    return 0.0


def build_rule_record(code6: str, as_of: str, sig: dict) -> dict:
    direction = sig["direction"]
    conf = round(min(0.9, 0.5 + 0.4 * min(1.0, abs(sig["ratio"]))), 3)
    stance = {"bullish": "bullish", "bearish": "bearish", "neutral": "neutral"}[direction]
    excerpt = (
        f"{sig['n']}거래일 외국인+기관 순매수 {sig['net']:+,}원 "
        f"(방향비율 {sig['ratio']:+.2f}, band ±{RULE_BAND})"
    )
    return {
        "ticker": f"{code6}.KS",
        "as_of_date": as_of,
        "overall_stance": stance,
        "overall_confidence": conf,
        "hypotheses": [
            {
                "id": f"h{h}",
                "claim": f"수급 {direction} 신호 — {h}주 호라이즌 KOSPI 상대 {direction}",
                "direction": direction,
                "confidence": conf,
                "evidence_tools": ["rule:supply_demand_net5"],
                "evidence_excerpts": [excerpt],
                "horizon_weeks": h,
                "predicted_relative_return_pct": _horizon_pred(direction, h),
            }
            for h in HORIZONS
        ],
    }


# === LLM 합성 (수급 + 기술적 분석가 → gemma 추출) ===

_EXTRACT_PROMPT = """다음은 한국 종목 {ticker} 의 {as_of} 기준 분석가 리포트다.
수급(KIS)·기술적(가격/지표) 신호만 근거로, 향후 2~4주 스윙 전망을 판단하라.

<수급 리포트>
{supply}
</수급 리포트>

<기술적 리포트>
{market}
</기술적 리포트>

아래 JSON 만 출력하라 (설명·마크다운 금지):
{{"overall_stance": "bullish|moderately_bullish|neutral|moderately_bearish|bearish",
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0~1.0,
  "claim": "한 줄 핵심 주장",
  "evidence_excerpts": ["리포트에서 인용한 구체 수치 1~2개"]}}"""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def build_llm_record(code6: str, as_of: str, llm) -> dict | None:
    from tradingagents.hermes.analyst_runner import run_analyst

    ticker = f"{code6}.KS"
    supply = run_analyst(ticker, as_of, "supply_demand", llm=llm)
    try:
        market = run_analyst(ticker, as_of, "market", llm=llm)
    except Exception as e:  # noqa: BLE001 — 기술적 실패 시 수급만으로 진행
        market = f"<unavailable: {e}>"

    prompt = _EXTRACT_PROMPT.format(
        ticker=ticker, as_of=as_of, supply=supply[:6000], market=market[:6000]
    )
    resp = llm.invoke(prompt)
    parsed = _extract_json(getattr(resp, "content", str(resp)))
    if not parsed:
        return None

    direction = parsed.get("direction")
    if direction not in {"bullish", "bearish", "neutral"}:
        return None
    stance = parsed.get("overall_stance")
    valid_stances = {
        "bullish", "moderately_bullish", "neutral",
        "moderately_bearish", "bearish",
    }
    if stance not in valid_stances:
        stance = direction
    conf = parsed.get("confidence", 0.5)
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        conf = 0.5
    claim = parsed.get("claim") or f"수급+기술 종합 {direction}"
    excerpts = parsed.get("evidence_excerpts") or [claim]
    if not isinstance(excerpts, list) or not excerpts:
        excerpts = [claim]

    return {
        "ticker": ticker,
        "as_of_date": as_of,
        "overall_stance": stance,
        "overall_confidence": round(float(conf), 3),
        "hypotheses": [
            {
                "id": f"h{h}",
                "claim": f"[{h}주] {claim}",
                "direction": direction,
                "confidence": round(float(conf), 3),
                "evidence_tools": ["analyst_supply_demand", "analyst_market"],
                "evidence_excerpts": [str(x) for x in excerpts][:4],
                "horizon_weeks": h,
                "predicted_relative_return_pct": _horizon_pred(direction, h),
            }
            for h in HORIZONS
        ],
    }


# === 생성 + 라벨링 ===

def _store(method: str):
    from tradingagents.hermes.hypothesis_store import HypothesisStore
    return HypothesisStore(root=BACKTEST_ROOT / method)


def _reset_store(method: str) -> None:
    db = BACKTEST_ROOT / method / "hypotheses.db"
    if db.exists():
        db.unlink()


def generate_rule() -> int:
    _reset_store("rule")
    store = _store("rule")
    n = 0
    for code6 in TICKERS:
        for as_of in AS_OF_DATES:
            sig = rule_signal(code6, as_of)
            if sig["n"] == 0:
                continue
            store.save(build_rule_record(code6, as_of, sig))
            n += len(HORIZONS)
    print(f"[rule] saved {n} hypotheses ({len(TICKERS)} tickers × {len(AS_OF_DATES)} dates × {len(HORIZONS)} horizons)")
    return n


def generate_llm() -> int:
    from tradingagents.dataflows.config import get_config, set_config
    from tradingagents.llm_clients.factory import create_llm_client

    cfg = get_config()
    cfg["analyst_mode"] = "prefetch"
    cfg["output_language"] = "Korean"
    cfg["quick_think_llm"] = ANALYST_MODEL
    set_config(cfg)

    install_kis_history_patch()
    llm = create_llm_client(
        "ollama", ANALYST_MODEL, base_url="http://localhost:11434/v1"
    ).get_llm()

    _reset_store("llm")
    store = _store("llm")
    n, skipped = 0, 0
    total = len(LLM_TICKERS) * len(AS_OF_DATES)
    i = 0
    for code6 in LLM_TICKERS:
        for as_of in AS_OF_DATES:
            i += 1
            print(f"[llm] ({i}/{total}) {code6} @ {as_of} ...", flush=True)
            try:
                rec = build_llm_record(code6, as_of, llm)
            except Exception as e:  # noqa: BLE001
                print(f"      error: {e}")
                rec = None
            if rec is None:
                skipped += 1
                continue
            store.save(rec)
            n += len(HORIZONS)
    print(f"[llm] saved {n} hypotheses, skipped {skipped} cells")
    return n


def label(method: str) -> dict:
    from tradingagents.hermes.labeling import LabelingScheduler
    sched = LabelingScheduler(_store(method))
    counts = sched.run_pending(today=BACKTEST_TODAY)
    print(f"[{method}] labeling: {counts}")
    return counts


# === 결과 집계 ===

def aggregate(method: str) -> dict | None:
    store = _store(method)
    hyps = store.list()
    rows = []
    for h in hyps:
        labels = store.get_labels(h["id"])
        auto = next((l for l in labels if l["label_kind"] == "auto_relative"), None)
        if auto is None:
            continue
        rows.append({
            "ticker": h["ticker"], "as_of": h["as_of_date"],
            "direction": h["direction"], "horizon": h["horizon_weeks"],
            "confidence": h["confidence"], "verdict": auto["verdict"],
            "rel": auto["actual_relative_return_pct"],
            "abs": auto["actual_return_pct"],
        })
    if not rows:
        return None

    def hit(subset):
        if not subset:
            return (0, 0, 0.0)
        right = sum(1 for r in subset if r["verdict"] == "right")
        return (right, len(subset), 100.0 * right / len(subset))

    total = hit(rows)
    by_dir = {d: hit([r for r in rows if r["direction"] == d])
              for d in ("bullish", "bearish", "neutral")}
    by_hz = {h: hit([r for r in rows if r["horizon"] == h]) for h in HORIZONS}
    nonneutral = [r for r in rows if r["direction"] != "neutral"]
    directional = hit(nonneutral)

    # 베이스라인: 같은 표본에 "항상 X" 를 적용했을 때 적중률.
    def baseline(direction):
        from tradingagents.hermes.labeling import horizon_threshold, judge_verdict
        right = 0
        for r in rows:
            thr = horizon_threshold(r["horizon"])
            if judge_verdict(direction, r["rel"], thr) == "right":
                right += 1
        return 100.0 * right / len(rows)

    mean_rel = sum(r["rel"] for r in rows) / len(rows)
    return {
        "method": method, "rows": rows, "total": total,
        "by_dir": by_dir, "by_hz": by_hz, "directional": directional,
        "mean_rel": mean_rel,
        "baseline": {d: baseline(d) for d in ("bullish", "bearish", "neutral")},
    }


def _fmt(t):
    return f"{t[0]}/{t[1]} ({t[2]:.0f}%)"


def report() -> None:
    lines = ["# Hermes 가설 백테스트 결과", "",
             f"- 라벨링 기준일: {BACKTEST_TODAY}",
             f"- as_of 시점: {', '.join(AS_OF_DATES)}",
             f"- 호라이즌: {HORIZONS}주 / 임계값 2w±2% 4w±3% (KOSPI 상대)", ""]
    any_data = False
    for method in ("rule", "llm"):
        agg = aggregate(method)
        if agg is None:
            lines += [f"## {method}: (라벨된 가설 없음)", ""]
            continue
        any_data = True
        lines += [
            f"## {method}",
            "",
            f"- **전체 적중률: {_fmt(agg['total'])}**  | 평균 KOSPI 상대수익 {agg['mean_rel']:+.2f}%",
            f"- 방향성 가설만(neutral 제외): {_fmt(agg['directional'])}",
            "",
            "| 구분 | 적중률 |",
            "|---|---|",
            f"| bullish | {_fmt(agg['by_dir']['bullish'])} |",
            f"| bearish | {_fmt(agg['by_dir']['bearish'])} |",
            f"| neutral | {_fmt(agg['by_dir']['neutral'])} |",
            f"| 2주 | {_fmt(agg['by_hz'][2])} |",
            f"| 4주 | {_fmt(agg['by_hz'][4])} |",
            "",
            "베이스라인(같은 표본에 항상 동일 방향 적용 시 적중률):",
            f"- 항상 bullish {agg['baseline']['bullish']:.0f}% / "
            f"항상 bearish {agg['baseline']['bearish']:.0f}% / "
            f"항상 neutral {agg['baseline']['neutral']:.0f}%",
            "",
        ]
    out = "\n".join(lines)
    print("\n" + out)
    if any_data:
        report_path = BACKTEST_ROOT / "report.md"
        report_path.write_text(out, encoding="utf-8")
        print(f"\n[report] written to {report_path}")


# === entrypoint ===

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "rule":
        generate_rule()
        label("rule")
        report()
    elif cmd == "llm":
        generate_llm()
        label("llm")
        report()
    elif cmd == "report":
        report()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
