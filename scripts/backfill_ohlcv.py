"""KIS history store 의 OHLCV 5컬럼(open/high/low/close/volume) backfill.

KIS API 응답 파싱이 OHLCV 모두 추출하도록 확장된 후(``kis_api.py`` 변경),
기존 종목 parquet 의 investor 테이블은 close만 보유한 상태. 이 스크립트는
*OHLCV 누락 종목만* 골라 KIS API 에서 전체 기간을 재수집한다.

재시작 가능: 이미 ``volume`` 컬럼이 존재하는 종목은 skip.

사용:
    uv run python scripts/backfill_ohlcv.py            # 모든 비OHLCV 종목
    uv run python scripts/backfill_ohlcv.py 005830 000810  # 명시 종목만
    uv run python scripts/backfill_ohlcv.py --years 3  # 기본 5년
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collect_kis_history import collect_one  # noqa: E402
from tradingagents.dataflows.kis_history_store import KisHistoryStore  # noqa: E402


def _needs_backfill(store: KisHistoryStore, ticker: str) -> bool:
    """investor parquet에 ``volume`` 컬럼 없으면 backfill 필요."""
    df = store.read(ticker, "investor")
    if df.empty:
        return False  # 데이터 자체가 없으면 skip (별도 collect 필요)
    return "volume" not in df.columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="명시 종목. 비우면 store 전체.")
    parser.add_argument("--years", type=int, default=5, help="수집 기간(년). 기본 5.")
    parser.add_argument(
        "--force", action="store_true",
        help="OHLCV 이미 있어도 강제 재수집.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="대상 종목만 출력하고 실제 수집은 skip.",
    )
    args = parser.parse_args(argv)

    store = KisHistoryStore()
    candidates = args.tickers or store.list_tickers()
    targets = [t for t in candidates if args.force or _needs_backfill(store, t)]

    print(f"전체 후보 {len(candidates)} / backfill 대상 {len(targets)}")
    if args.dry_run:
        for t in targets:
            print(f"  - {t}")
        return 0

    if not targets:
        print("backfill 할 종목 없음 — 모두 OHLCV 보유.")
        return 0

    t_start = time.time()
    results: dict[str, dict] = {}
    for i, ticker in enumerate(targets, start=1):
        print(f"\n[{i}/{len(targets)}] {ticker}")
        summary = collect_one(
            store, ticker, years=args.years, full=True, endpoints=["investor"],
        )
        results[ticker] = summary

    elapsed = time.time() - t_start
    n_ok = sum(
        1 for s in results.values()
        if s.get("endpoints", {}).get("investor", {}).get("status") == "ok"
    )
    n_err = sum(
        1 for s in results.values()
        if s.get("endpoints", {}).get("investor", {}).get("status") == "error"
    )
    print(f"\n=== 완료: {n_ok} 성공 / {n_err} 실패 / {elapsed:.0f}초 ===")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
