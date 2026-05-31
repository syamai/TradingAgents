"""실제 Hermes 파이프라인 과거 백테스트 드라이버.

각 (ticker, as_of) 셀마다 백테스트 env 를 세팅하고 *실제* ``hermes -z`` 를
호출한다. MCP 서버가 HERMES_BACKTEST_AS_OF 를 보고 모든 데이터 도구를 그 시점
이하로 강제 → 누수 0. 가설은 격리 저장소에 쌓이고, 호라이즌 만기는 이미
지났으므로 즉시 라벨링·집계한다.

운영 db(~/.tradingagents/hermes/hypotheses.db)는 건드리지 않는다 —
HERMES_BACKTEST_STORE_ROOT 로 격리.

usage:
  uv run python scripts/hermes_backtest_run.py run --cells 1   # 1셀 e2e 검증
  uv run python scripts/hermes_backtest_run.py run             # 12셀 전체
  uv run python scripts/hermes_backtest_run.py report          # 라벨링+집계만
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKTEST_TODAY = "2026-05-30"
STORE_ROOT = Path.home() / ".tradingagents" / "hermes" / "backtest_hermes"

TICKERS = ["005930", "000660"]   # 삼성전자, SK하이닉스
AS_OF_DATES = [
    "2025-09-15", "2025-10-15", "2025-11-17",
    "2025-12-15", "2026-01-15", "2026-02-16",
]
PER_CELL_TIMEOUT = 1200  # 초 (gemma 분석가 콜드로딩 여유)


def cells() -> list[tuple[str, str]]:
    return [(f"{t}.KS", d) for t in TICKERS for d in AS_OF_DATES]


ANALYST_MODEL = "gemma4:e4b"  # 프로덕션 26b-a4b(~10분/콜) 대신 경량 동계열 모델


def run_one(ticker: str, as_of: str, log_path: Path) -> dict:
    # env 가 아니라 상태 파일로 전달 — Hermes 가 MCP 서브프로세스 env 를 필터링하므로.
    from tradingagents.hermes import backtest as bt
    bt.write_state(as_of=as_of, store_root=str(STORE_ROOT), analyst_model=ANALYST_MODEL)
    env = dict(os.environ)
    prompt = (
        f"{ticker} 종목을 기준일 {as_of} 기준으로 스윙 분석해줘. "
        f"분석 기준일(as_of_date)은 반드시 {as_of} 로 설정하고, "
        f"오늘 날짜를 쓰지 마라."
    )
    with open(log_path, "w") as f:
        proc = subprocess.run(
            ["hermes", "-z", prompt, "-s", "stock-analyst", "--yolo"],
            env=env, stdout=f, stderr=subprocess.STDOUT,
            timeout=PER_CELL_TIMEOUT,
        )
    # 저장 결과 확인 (이 as_of 로 레코드가 들어왔는가)
    from tradingagents.hermes.hypothesis_store import HypothesisStore
    store = HypothesisStore(root=STORE_ROOT)
    saved = store.list(ticker=ticker, as_of_date=as_of)
    return {"exit": proc.returncode, "saved": len(saved)}


def run(n_cells: int | None) -> None:
    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = STORE_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    todo = cells()[: n_cells] if n_cells else cells()
    print(f"[run] {len(todo)} cells → {STORE_ROOT}")
    for i, (ticker, as_of) in enumerate(todo, 1):
        log_path = log_dir / f"{ticker}_{as_of}.log"
        print(f"[run] ({i}/{len(todo)}) {ticker} @ {as_of} ...", flush=True)
        try:
            r = run_one(ticker, as_of, log_path)
            print(f"      exit={r['exit']} saved={r['saved']} → {log_path}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"      TIMEOUT (>{PER_CELL_TIMEOUT}s) → {log_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"      ERROR: {e}", flush=True)
    from tradingagents.hermes import backtest as bt
    bt.clear_state()  # 운영 모드 복귀
    report()


def report() -> None:
    from tradingagents.hermes.hypothesis_store import HypothesisStore
    from tradingagents.hermes.labeling import (
        LabelingScheduler, horizon_threshold, judge_verdict,
    )

    store = HypothesisStore(root=STORE_ROOT)
    counts = LabelingScheduler(store).run_pending(today=BACKTEST_TODAY)
    print(f"[label] {counts}")

    rows = []
    for h in store.list():
        auto = next(
            (l for l in store.get_labels(h["id"])
             if l["label_kind"] == "auto_relative"), None
        )
        if auto is None:
            continue
        rows.append({
            "ticker": h["ticker"], "as_of": h["as_of_date"],
            "direction": h["direction"], "horizon": h["horizon_weeks"],
            "confidence": h["confidence"], "verdict": auto["verdict"],
            "rel": auto["actual_relative_return_pct"],
        })
    if not rows:
        print("[report] 라벨된 가설 없음")
        return

    def hit(sub):
        if not sub:
            return "0/0 (–)"
        r = sum(1 for x in sub if x["verdict"] == "right")
        return f"{r}/{len(sub)} ({100*r/len(sub):.0f}%)"

    mean_rel = sum(r["rel"] for r in rows) / len(rows)
    by_dir = {d: [r for r in rows if r["direction"] == d]
              for d in ("bullish", "bearish", "neutral")}
    nonneutral = [r for r in rows if r["direction"] != "neutral"]

    def baseline(direction):
        right = sum(
            1 for r in rows
            if judge_verdict(direction, r["rel"], horizon_threshold(r["horizon"])) == "right"
        )
        return f"{100*right/len(rows):.0f}%"

    lines = [
        "# 실제 Hermes 과거 백테스트 결과", "",
        f"- 라벨링 기준일: {BACKTEST_TODAY}",
        f"- 모델: Hermes orchestrator(Codex gpt-5.4) + 분석가(gemma4:26b-a4b)",
        f"- 셀: {', '.join(TICKERS)} × {len(AS_OF_DATES)} 시점",
        f"- 라벨된 가설 수: {len(rows)}", "",
        f"- **전체 적중률: {hit(rows)}**  | 평균 KOSPI 상대수익 {mean_rel:+.2f}%",
        f"- 방향성 가설만(neutral 제외): {hit(nonneutral)}", "",
        "| 구분 | 적중률 |", "|---|---|",
        f"| bullish | {hit(by_dir['bullish'])} |",
        f"| bearish | {hit(by_dir['bearish'])} |",
        f"| neutral | {hit(by_dir['neutral'])} |",
        f"| 2주 | {hit([r for r in rows if r['horizon']==2])} |",
        f"| 4주 | {hit([r for r in rows if r['horizon']==4])} |",
        f"| 3주 | {hit([r for r in rows if r['horizon']==3])} |", "",
        "베이스라인(같은 표본에 항상 동일 방향): "
        f"bullish {baseline('bullish')} / bearish {baseline('bearish')} / neutral {baseline('neutral')}",
    ]
    out = "\n".join(lines)
    print("\n" + out)
    (STORE_ROOT / "report.md").write_text(out, encoding="utf-8")
    print(f"\n[report] → {STORE_ROOT / 'report.md'}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "run":
        n = None
        if "--cells" in sys.argv:
            n = int(sys.argv[sys.argv.index("--cells") + 1])
        run(n)
    elif cmd == "report":
        report()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
