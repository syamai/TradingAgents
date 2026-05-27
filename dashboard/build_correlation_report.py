"""CLI: 종목 holdings 데이터 → 주가-수급 상관 분석 보고서.

산출 위치:
  - SQLite ``analysis_reports`` 테이블 (--save-db, 기본 활성)
  - 옵시디언 vault 마크다운 (--obsidian, vault 자동 탐지)
  - 파일시스템 마크다운 (--md PATH)

사용 예:
    uv run python dashboard/build_correlation_report.py 267270
    uv run python dashboard/build_correlation_report.py 267270 --no-db --md /tmp/x.md
    uv run python dashboard/build_correlation_report.py 005930 --obsidian
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import tradingagents  # noqa: F401 — dotenv
from dashboard.correlation_analysis import (
    compute_correlation_report, render_markdown,
)
from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.kis_history_store import KisHistoryStore

_OBSIDIAN_REL = Path("Projects/trading-ai/reports")


def _find_obsidian_vault() -> Path | None:
    """``~/Documents/*/.obsidian`` 첫 매치의 vault root."""
    docs = Path.home() / "Documents"
    if not docs.exists():
        return None
    for entry in docs.iterdir():
        if (entry / ".obsidian").is_dir():
            return entry
    return None


def _obsidian_report_path(vault: Path, ticker: str, company_name: str | None) -> Path:
    safe = (company_name or "unknown").replace(" ", "_").replace("/", "_")
    return vault / _OBSIDIAN_REL / f"correlation_{ticker}_{safe}.md"


def _frontmatter(report: dict) -> str:
    w = report.get("window") or {}
    return "\n".join([
        "---",
        f"ticker: {report['ticker']}",
        f"company_name: {report.get('company_name') or ''}",
        f"market: {report.get('market') or ''}",
        f"type: analysis-report",
        f"analysis_kind: correlation",
        f"window_start: {w.get('start', '')}",
        f"window_end: {w.get('end', '')}",
        f"n_days: {report.get('n_days', 0)}",
        f"generated_at: {report['generated_at']}",
        f"tags: [analysis, correlation, supply-demand]",
        f"related: [[수급 보유 시각화 계획]]",
        "---",
        "",
    ])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="주가-수급 상관 분석 보고서")
    p.add_argument("ticker", help="종목 코드 (6자리 또는 .KS/.KQ)")
    p.add_argument("--start", help="시작일 YYYY-MM-DD (옵션, 디폴트: 전체)")
    p.add_argument("--end", help="종료일 YYYY-MM-DD (옵션, 디폴트: 전체)")
    p.add_argument("--no-db", action="store_true",
                   help="SQLite analysis_reports 저장 안 함")
    p.add_argument("--obsidian", action="store_true",
                   help="옵시디언 vault에 마크다운 저장")
    p.add_argument("--md", metavar="PATH",
                   help="마크다운을 임의 경로에 추가 저장")
    p.add_argument("--print", action="store_true",
                   help="마크다운을 stdout에도 출력")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    store = KisHistoryStore()

    df, meta = load_holdings(args.ticker, start=args.start, end=args.end,
                              store=store)
    if df.empty:
        print(f"ERROR: holdings 데이터 없음 — {args.ticker}", file=sys.stderr)
        return 2

    ticker = meta.get("ticker") or args.ticker
    report = compute_correlation_report(
        df, ticker=ticker,
        company_name=meta.get("company_name"),
        market=meta.get("market"),
    )
    md_body = render_markdown(report)
    md_full = _frontmatter(report) + md_body + "\n"

    saved_paths: list[str] = []

    # 1) SQLite
    if not args.no_db:
        as_of = report["window"]["end"]
        store.write_analysis_report(ticker, "correlation", as_of, report)
        saved_paths.append(f"sqlite: analysis_reports[{ticker}, correlation, {as_of}]")

    # 2) Obsidian
    if args.obsidian:
        vault = _find_obsidian_vault()
        if vault is None:
            print("WARN: 옵시디언 vault 못 찾음 — ~/Documents/*/.obsidian 없음",
                  file=sys.stderr)
        else:
            path = _obsidian_report_path(vault, ticker, meta.get("company_name"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(md_full, encoding="utf-8")
            saved_paths.append(f"obsidian: {path}")

    # 3) 임의 경로
    if args.md:
        path = Path(args.md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md_full, encoding="utf-8")
        saved_paths.append(f"file: {path}")

    if args.print or not saved_paths:
        print(md_full)

    for s in saved_paths:
        print(f"✅ {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
