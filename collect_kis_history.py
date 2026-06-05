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

    # 1분봉 OHLCV (장후 매일 누적용). 증분 모드면 마지막 저장일 다음날부터,
    # 신규면 보관 한계 ~1년 이내만 수집. KIS는 분봉을 ~1년만 보관하므로
    # 장기 코퍼스는 매일 수집·누적이 유일한 경로.
    uv run python collect_kis_history.py 005930.KS --endpoints minute

호출 횟수 (5년 1종목):
    investor 42 + program 42 + short 13 ≈ 97회, ~10초 (100ms 간격 + backoff)
    minute: 거래일당 ~4회 → 1년 누적 백필 시 ~1,000회/종목 (~2분), 일일 증분은 ~4회/종목

자격증명: ``KIS_APP_KEY`` / ``KIS_APP_SECRET`` / ``KIS_ENV=real|mock``
"""
from __future__ import annotations

import argparse
import fcntl
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


def _top_by_marketcap(n: int, market: str = "kospi") -> list[tuple[str, str]]:
    """네이버 금융에서 시가총액 상위 N 종목 (코드, 종목명) 튜플 추출.

    KIS의 ``market-cap`` ranking endpoint가 0행을 반환하는 시점에 폴백.
    한 페이지 50종목 → ``ceil(n/50)`` 페이지 스크래핑. EUC-KR 디코드.

    ``market``: ``"kospi"`` (sosok=0) 또는 ``"kosdaq"`` (sosok=1).

    HTML 패턴: ``/item/main.naver?code=NNNNNN" class="tltle">종목명<``
    """
    import re
    import urllib.request

    sosok = {"kospi": 0, "kosdaq": 1}.get(market.lower())
    if sosok is None:
        raise ValueError(f"market must be 'kospi' or 'kosdaq', got {market!r}")

    pattern = re.compile(r'/item/main\.naver\?code=(\d{6})" class="tltle">([^<]+)')
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    pages_needed = (n + 49) // 50
    for page in range(1, pages_needed + 1):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("euc-kr", errors="ignore")
        for code, name in pattern.findall(html):
            if code not in seen:
                out.append((code, name.strip()))
                seen.add(code)
                if len(out) >= n:
                    return out
    return out


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
    p.add_argument(
        "--all-stored",
        action="store_true",
        help="이미 store에 저장된 모든 종목을 대상에 추가 (일일 cron 증분 수집용)",
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
        choices=["investor", "program", "short", "minute"],
        default=["investor", "program", "short"],
        help=(
            "수집할 endpoint (기본 일별 수급 3종). 'minute'=1분봉 OHLCV — "
            "거래일당 ~380봉·~4회 호출로 무거우니 별도 지정하고 보관 한계 "
            "~1년 이내만 수집된다 (예: --endpoints minute)."
        ),
    )
    return p.parse_args(argv)


def _resolve_window(
    store: KisHistoryStore, ticker: str, endpoint: str, years: int, full: bool
) -> tuple[str, str] | None:
    """수집할 (start_date, end_date) 결정. None이면 skip (이미 최신)."""
    today = date.today()
    # 분봉(가격 OHLCV)은 장 마감·체결확정(~15:40) 후 당일치가 KIS에 올라온다 → end=오늘.
    # 수급(investor/program/short)은 KIS가 당일분을 익일 확정 → end=어제(T-1).
    last_day = today if endpoint == "minute" else today - timedelta(days=1)
    end_date = last_day.strftime("%Y-%m-%d")
    if full:
        start = today - timedelta(days=365 * years)
        return start.strftime("%Y-%m-%d"), end_date
    last = store.last_date(ticker, endpoint)
    if last is None:
        start = today - timedelta(days=365 * years)
        return start.strftime("%Y-%m-%d"), end_date
    # 증분: 마지막+1일 부터 last_day 까지.
    # minute endpoint는 last가 "YYYY-MM-DD HH:MM:SS" 타임스탬프라 앞 10자만 사용.
    from datetime import datetime as _dt
    last_dt = _dt.strptime(last[:10], "%Y-%m-%d").date()
    start_dt = last_dt + timedelta(days=1)
    if start_dt > last_day:
        return None  # already up to date
    return start_dt.strftime("%Y-%m-%d"), end_date


FETCHERS = {
    "investor": kis_api.fetch_investor_trend_range,
    "program": kis_api.fetch_program_trading_range,
    "short": kis_api.fetch_short_interest_range,
    "minute": kis_api.fetch_minute_bars_range,
}

# KIS 분봉 서버 보관 한계 ~1년(+여유). 그 이전 거래일은 빈 응답이라
# 백필 윈도우를 잘라 호출 낭비를 막는다.
_MINUTE_RETENTION_DAYS = 370


def _clamp_minute_window(window: tuple[str, str]) -> tuple[str, str] | None:
    """분봉 수집 윈도우를 보관 한계(~1년) 이내로 클램프. 전체가 한계 밖이면 None."""
    start, end = window
    earliest = (date.today() - timedelta(days=_MINUTE_RETENTION_DAYS)).strftime("%Y-%m-%d")
    if start < earliest:
        start = earliest
    if start > end:
        return None
    return start, end


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
        if window is not None and endpoint == "minute":
            window = _clamp_minute_window(window)
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
    # (code, name, market) 트리플 — store.set_ticker_metadata 호출용. tickers 인자로
    # 직접 받은 코드는 종목명 모름 → 빈 문자열로 두고 메타 호출 skip.
    metadata: list[tuple[str, str, str]] = []
    if args.top_kospi:
        print(f"네이버 시가총액 페이지 스크래핑 — KOSPI 상위 {args.top_kospi}종목...")
        top = _top_by_marketcap(args.top_kospi, market="kospi")
        print(f"  → {len(top)} codes: {[c for c,_ in top[:5]]} ... {[c for c,_ in top[-3:]]}")
        for code, name in top:
            tickers.append(code)
            metadata.append((code, name, "KOSPI"))
    if args.top_kosdaq:
        print(f"네이버 시가총액 페이지 스크래핑 — KOSDAQ 상위 {args.top_kosdaq}종목...")
        top = _top_by_marketcap(args.top_kosdaq, market="kosdaq")
        print(f"  → {len(top)} codes: {[c for c,_ in top[:5]]} ... {[c for c,_ in top[-3:]]}")
        for code, name in top:
            tickers.append(code)
            metadata.append((code, name, "KOSDAQ"))
    if args.store == "both":
        backends = ("parquet", "sqlite")
    else:
        backends = (args.store,)
    store = KisHistoryStore(backends=backends)

    if args.all_stored:
        stored = store.list_tickers()
        print(f"--all-stored: store에 저장된 {len(stored)}종목 대상에 추가")
        tickers.extend(stored)
    # 중복 제거 (순서 보존) — --all-stored 와 명시 인자가 겹칠 수 있음
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        print("ERROR: 종목 코드 인자 또는 --top-kospi/--top-kosdaq/--all-stored 중 하나는 필요합니다.", file=sys.stderr)
        return 2

    # 백필(수 시간)과 일일 cron 이 동시에 같은 종목을 write 하면 parquet
    # read-merge-write 가 서로의 갱신을 덮어쓸 수 있다. 단일 실행 보장용 파일 락
    # — 잡지 못하면 이번 실행은 skip(다른 수집이 진행 중). 프로세스 종료 시 해제.
    _lock_f = open(store.root / ".collect.lock", "w")
    try:
        fcntl.flock(_lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("다른 수집 프로세스가 실행 중 — 이번 실행 skip", file=sys.stderr)
        return 0

    # 종목명·시장 메타 우선 저장. 다중 호출 시 collect 진행 중 사용자가 SQL로 종목명 조회 가능.
    if metadata:
        print(f"tickers 메타 저장 ({len(metadata)} 종목)...")
        for code, name, market in metadata:
            store.set_ticker_metadata(code, name, market)

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
