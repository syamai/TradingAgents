"""기존 raw investor 데이터에서 holdings derived 테이블을 일괄 재계산.

일반 흐름에서는 ``KisHistoryStore.write(endpoint='investor')`` 가 자동으로
``_materialize_holdings`` 를 호출하므로 별도 명령 불필요. 이 CLI는 다음 경우
사용:
  - holdings 정책 변경(분모·컬럼 구조 등) 후 일괄 재계산
  - parquet 손상·SQLite drop 후 derived 복구
  - 신규 종목을 raw 만 미리 import한 뒤 derived 만들기

사용:
    # 한 종목
    uv run python compute_holdings.py --ticker 005930

    # 전체 (store.list_tickers)
    uv run python compute_holdings.py --all

    # 상태만 확인
    uv run python compute_holdings.py --list
"""
from __future__ import annotations

import argparse
import sys
import time

import tradingagents  # noqa: F401 — dotenv 로딩
from tradingagents.dataflows.kis_history_store import KisHistoryStore


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="holdings derived 테이블 일괄 재계산")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", help="단일 종목 코드 (예: 005930 또는 005930.KS)")
    g.add_argument("--all", action="store_true",
                   help="store.list_tickers() 모든 종목 재계산")
    g.add_argument("--list", action="store_true",
                   help="현재 holdings·tickers 메타 상태 출력 (read-only)")
    return p.parse_args(argv)


def _do_one(store: KisHistoryStore, ticker: str) -> tuple[str, int, float]:
    t0 = time.time()
    n = store.materialize_holdings(ticker)
    elapsed = time.time() - t0
    return ticker, n, elapsed


def _list_status(store: KisHistoryStore) -> int:
    tickers = store.list_tickers()
    print(f"store root: {store.root}")
    print(f"종목 수: {len(tickers)}")
    print()
    print(f"{'ticker':<8} | {'종목명':<20} | {'시장':<6} | {'investor':>10} | {'holdings':>10}")
    print("-" * 75)
    for t in tickers[:30]:
        meta = store.get_ticker_metadata(t) or {}
        name = meta.get("company_name", "-")[:20]
        market = meta.get("market", "-")
        inv_last = store.last_date(t, "investor") or "-"
        hold_last = store.last_date(t, "holdings") or "-"
        print(f"{t:<8} | {name:<20} | {market:<6} | {inv_last:>10} | {hold_last:>10}")
    if len(tickers) > 30:
        print(f"... ({len(tickers) - 30} more)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    store = KisHistoryStore()

    if args.list:
        return _list_status(store)

    if args.ticker:
        tickers = [args.ticker]
    else:  # --all
        tickers = store.list_tickers()
        if not tickers:
            print("ERROR: store에 종목이 없습니다.", file=sys.stderr)
            return 2

    print(f"holdings 재계산 — {len(tickers)} 종목")
    print()
    overall_t0 = time.time()
    ok = skip = err = 0
    for t in tickers:
        try:
            ticker, n, elapsed = _do_one(store, t)
            if n == 0:
                print(f"  ⏭  {t}: raw investor 없음 — skip")
                skip += 1
            else:
                print(f"  ✅ {ticker}: {n} rows holdings in {elapsed*1000:.0f}ms")
                ok += 1
        except Exception as exc:
            print(f"  ❌ {t}: {type(exc).__name__}: {exc}")
            err += 1

    print()
    print(f"=== 완료 ({time.time()-overall_t0:.1f}s): "
          f"ok={ok}, skip={skip}, error={err} ===")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
