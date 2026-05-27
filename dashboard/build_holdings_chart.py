"""CLI: holdings derived 테이블 → 인터랙티브 HTML 차트 1장.

사용 예:
    uv run python dashboard/build_holdings_chart.py 005930
    uv run python dashboard/build_holdings_chart.py 005930 --start 2024-01-01
    uv run python dashboard/build_holdings_chart.py 000660 --output /tmp/x.html
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import tradingagents  # noqa: F401 — dotenv 로딩
from dashboard.holdings_chart import load_holdings, make_figure, save_html

_NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _default_output(ticker: str, company_name: str | None) -> Path:
    today = date.today().isoformat()
    safe_name = (company_name or "unknown").replace(" ", "_").replace("/", "_")
    return _NOTEBOOKS_DIR / f"holdings_{ticker}_{safe_name}_{today}.html"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="holdings → plotly HTML")
    p.add_argument("ticker", help="종목 코드 (6자리 또는 .KS/.KQ)")
    p.add_argument("--start", help="시작일 YYYY-MM-DD (옵션)")
    p.add_argument("--end", help="종료일 YYYY-MM-DD (옵션)")
    p.add_argument("--output", help="출력 HTML 경로 (옵션)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    df, meta = load_holdings(args.ticker, start=args.start, end=args.end)
    if df.empty:
        print(f"ERROR: holdings 데이터 없음 — {args.ticker}", file=sys.stderr)
        print("  → 먼저 collect_kis_history.py 로 raw investor 수집 필요.",
              file=sys.stderr)
        return 2

    company_name = meta.get("company_name", "unknown")
    ticker_norm = meta.get("ticker", args.ticker)
    market = meta.get("market", "-")
    title = (
        f"{company_name} ({ticker_norm}) · {market} · "
        f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]} · {len(df)}일"
    )

    fig = make_figure(df, title=title)

    out_path = Path(args.output) if args.output else _default_output(ticker_norm, company_name)
    save_html(fig, out_path)
    print(f"✅ {out_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
