"""KIS 수급 데이터 장기 히스토리 수집 CLI.

분석가 노드의 응답 캐시(7일 lookback)와 무관한 영구 저장소
(``~/.tradingagents/kis_history/``)에 5년+ 시계열을 sliding-window로 수집.

사용 예:
    # 005930.KS 5년치 수집 (parquet + sqlite 둘 다)
    uv run python collect_kis_history.py 005930.KS --years 5

    # 증분 업데이트만 (마지막 저장일 이후만)
    uv run python collect_kis_history.py 005930.KS --years 5

    # 전체 재수집 (기존 무시)
    uv run python collect_kis_history.py 005930.KS --years 5 --full

    # parquet만
    uv run python collect_kis_history.py 005930.KS --years 5 --store parquet

    # 다종목
    uv run python collect_kis_history.py 005930.KS 000660.KS 035720.KQ --years 5

호출 횟수 (5년 1종목):
    investor 42 + program 42 + short 13 ≈ 97회, ~10초 (100ms 간격 + backoff)

자격증명: ``KIS_APP_KEY`` / ``KIS_APP_SECRET`` / ``KIS_ENV=real|mock``
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta

import tradingagents  # noqa: F401 — dotenv 자동 로딩 트리거
from tradingagents.dataflows import kis_api
from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.dataflows.korean_utils import is_korean_ticker, to_naver_code

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _top_by_marketcap(n: int, market: str = "kospi") -> list[str]:
    """네이버 금융에서 시가총액 상위 N 종목 코드(6자리) 추출.

    KIS의 ``market-cap`` ranking endpoint가 0행을 반환하는 시점에 폴백.
    한 페이지 50종목 → ``ceil(n/50)`` 페이지 스크래핑. EUC-KR 디코드.

    ``market``: ``"kospi"`` (sosok=0) 또는 ``"kosdaq"`` (sosok=1).
    """
    import re
    import urllib.request

    sosok = {"kospi": 0, "kosdaq": 1}.get(market.lower())
    if sosok is None:
        raise ValueError(f"market must be 'kospi' or 'kosdaq', got {market!r}")

    codes: list[str] = []
    seen: set[str] = set()
    pages_needed = (n + 49) // 50
    for page in range(1, pages_needed + 1):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("euc-kr", errors="ignore")
        for code in re.findall(r'/item/main\.naver\?code=(\d{6})', html):
            if code not in seen:
                codes.append(code)
                seen.add(code)
                if len(codes) >= n:
                    return codes
    return codes


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIS 수급 5년치 등 장기 히스토리 수집")
    p.add_argument("tickers", nargs="*", help="종목 코드 (005930.KS 또는 005930). --top-kospi 사용 시 생략 가능.")
    p.add_argument(
        "--top-kospi",
        type=int,
        metavar="N",
        help="KOSPI 시가총액 상위 N종목 자동 선정 (네이버 스크래핑)",
    )
    p.add_argument(
        "--top-kosdaq",
        type=int,
        metavar="N",
        help="KOSDAQ 시가총액 상위 N종목 자동 선정",
    )
    p.add_argument("--years", type=int, default=5, help="수집 기간 (년, 기본 5)")
    p.add_argument(
        "--store",
        choices=["parquet", "sqlite", "both"],
        default="both",
        help="저장 백엔드 (기본 both)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="기존 데이터 무시하고 전체 재수집. 기본은 증분 업데이트.",
    )
    p.add_argument(
        "--endpoints",
        nargs="+",
        choices=["investor", "program", "short"],
        default=["investor", "program", "short"],
        help="수집할 endpoint (기본 셋 모두)",
    )
    return p.parse_args(argv)


def _resolve_window(
    store: KisHistoryStore, ticker: str, endpoint: str, years: int, full: bool
) -> tuple[str, str] | None:
    """수집할 (start_date, end_date) 결정. None이면 skip (이미 최신)."""
    today = date.today()
    end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")  # 어제
    if full:
        start = today - timedelta(days=365 * years)
        return start.strftime("%Y-%m-%d"), end_date
    last = store.last_date(ticker, endpoint)
    if last is None:
        start = today - timedelta(days=365 * years)
        return start.strftime("%Y-%m-%d"), end_date
    # 증분: 마지막+1일 부터 어제까지
    from datetime import datetime as _dt
    last_dt = _dt.strptime(last, "%Y-%m-%d").date()
    start_dt = last_dt + timedelta(days=1)
    if start_dt > today - timedelta(days=1):
        return None  # already up to date
    return start_dt.strftime("%Y-%m-%d"), end_date


FETCHERS = {
    "investor": kis_api.fetch_investor_trend_range,
    "program": kis_api.fetch_program_trading_range,
    "short": kis_api.fetch_short_interest_range,
}


def collect_one(
    store: KisHistoryStore, ticker: str, *, years: int, full: bool, endpoints: list[str]
) -> dict:
    """한 종목 수집. 결과 요약 dict 반환."""
    if not is_korean_ticker(ticker):
        print(f"  ❌ {ticker}: not a Korean ticker — skipping")
        return {"ticker": ticker, "skipped": "non_korean"}
    code6 = to_naver_code(ticker)
    summary = {"ticker": ticker, "endpoints": {}}
    for endpoint in endpoints:
        window = _resolve_window(store, ticker, endpoint, years, full)
        if window is None:
            print(f"  ⏭  {ticker} / {endpoint}: up to date")
            summary["endpoints"][endpoint] = {"status": "up_to_date"}
            continue
        start, end = window
        t0 = time.time()
        last_progress = [0]

        def _progress(n_rows):
            if n_rows - last_progress[0] >= 60:
                print(f"     ↻ {n_rows} rows...", flush=True)
                last_progress[0] = n_rows

        try:
            rows = FETCHERS[endpoint](code6, start, end, progress_cb=_progress)
        except Exception as exc:
            print(f"  ❌ {ticker} / {endpoint}: {type(exc).__name__}: {exc}")
            summary["endpoints"][endpoint] = {"status": "error", "error": str(exc)[:120]}
            continue
        elapsed = time.time() - t0
        n_total = store.write(ticker, endpoint, rows)
        print(
            f"  ✅ {ticker} / {endpoint}: collected {len(rows)} rows in "
            f"{elapsed:.1f}s · store total {n_total} rows (range {start} → {end})"
        )
        summary["endpoints"][endpoint] = {
            "status": "ok", "collected": len(rows), "total": n_total,
            "elapsed_sec": round(elapsed, 1),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # --top-kospi / --top-kosdaq 옵션 처리: tickers 인자에 자동 추가
    tickers = list(args.tickers)
    if args.top_kospi:
        print(f"네이버 시가총액 페이지 스크래핑 — KOSPI 상위 {args.top_kospi}종목...")
        top = _top_by_marketcap(args.top_kospi, market="kospi")
        print(f"  → {len(top)} codes: {top[:5]} ... {top[-3:]}")
        tickers.extend(top)
    if args.top_kosdaq:
        print(f"네이버 시가총액 페이지 스크래핑 — KOSDAQ 상위 {args.top_kosdaq}종목...")
        top = _top_by_marketcap(args.top_kosdaq, market="kosdaq")
        print(f"  → {len(top)} codes: {top[:5]} ... {top[-3:]}")
        tickers.extend(top)
    if not tickers:
        print("ERROR: 종목 코드 인자 또는 --top-kospi/--top-kosdaq N 중 하나는 필요합니다.", file=sys.stderr)
        return 2

    if args.store == "both":
        backends = ("parquet", "sqlite")
    else:
        backends = (args.store,)
    store = KisHistoryStore(backends=backends)

    print(
        f"KIS history collection — {len(tickers)} tickers, years={args.years}, "
        f"endpoints={args.endpoints}, backends={backends}, "
        f"mode={'FULL' if args.full else 'INCREMENTAL'}"
    )
    print(f"store root: {store.root}")
    print()

    overall_t0 = time.time()
    summaries = []
    for ticker in tickers:
        print(f"=== {ticker} ===")
        s = collect_one(
            store, ticker,
            years=args.years, full=args.full, endpoints=args.endpoints,
        )
        summaries.append(s)
        print()

    print(f"=== 전체 완료 ({time.time()-overall_t0:.1f}s) ===")
    for s in summaries:
        if s.get("skipped"):
            print(f"  {s['ticker']}: skipped ({s['skipped']})")
            continue
        eps = s.get("endpoints", {})
        statuses = [f"{ep}:{info['status']}" for ep, info in eps.items()]
        print(f"  {s['ticker']}: " + " | ".join(statuses))
    return 0


if __name__ == "__main__":
    sys.exit(main())
