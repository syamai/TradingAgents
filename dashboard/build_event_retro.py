"""CLI: 이벤트 리뷰 보고서 (이벤트·뉴스·거시·DART·동종·LLM 합성).

산출 위치 (옵션):
  - SQLite ``analysis_reports`` (--save-db, 기본 활성)
  - 옵시디언 vault 마크다운 (--obsidian)
  - 파일시스템 (--md PATH)

사용 예:
    uv run python dashboard/build_event_retro.py 005830 2025-02-14 2025-04-14
    uv run python dashboard/build_event_retro.py 005830 2025-02-14 2025-04-14 \\
        --industry 손해보험 --llm-synthesis --obsidian
    uv run python dashboard/build_event_retro.py 005830 2025-02-14 2025-04-14 \\
        --peers 000810,001450,000060,000540 --no-search
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import tradingagents  # noqa: F401 — dotenv
from dashboard.event_retro import (
    compute_event_retro, render_event_retro_markdown,
)
from dashboard.event_retro_synthesis import (
    DEFAULT_MODEL as RETRO_SYN_MODEL,
    DEFAULT_PROVIDER as RETRO_SYN_PROVIDER,
    synthesize_event_retro, render_synthesis_markdown,
)
from dashboard.holdings_chart import load_holdings
from tradingagents.dataflows.kis_history_store import KisHistoryStore

_OBSIDIAN_REL = Path("Projects/trading-ai/reports")


def _find_obsidian_vault() -> Path | None:
    docs = Path.home() / "Documents"
    if not docs.exists():
        return None
    for entry in docs.iterdir():
        if (entry / ".obsidian").is_dir():
            return entry
    return None


def _obsidian_report_path(
    vault: Path, ticker: str, company_name: str | None,
    start: str, end: str,
) -> Path:
    safe = (company_name or "unknown").replace(" ", "_").replace("/", "_")
    return vault / _OBSIDIAN_REL / f"event_{ticker}_{safe}_{start}_to_{end}.md"


def _frontmatter(report: dict) -> str:
    meta = report.get("meta", {})
    return "\n".join([
        "---",
        f"ticker: {meta.get('ticker', '')}",
        f"company_name: {meta.get('company_name', '')}",
        f"market: {meta.get('market', '')}",
        f"report_kind: event_retro",
        f"period_start: {meta.get('start', '')}",
        f"period_end: {meta.get('end', '')}",
        f"generated_at: {meta.get('generated_at', '')}",
        f"tags: [analysis, event-retro]",
        "---",
        "",
    ])


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="이벤트 리뷰 보고서 CLI")
    p.add_argument("ticker", help="6자리 종목 코드")
    p.add_argument("start", help="회고 시작일 YYYY-MM-DD")
    p.add_argument("end", help="회고 종료일 YYYY-MM-DD")
    p.add_argument("--industry", help="산업 키워드 (검색 보강)", default=None)
    p.add_argument("--peers", help="동종 수동 (콤마 구분 ticker)", default=None)
    p.add_argument("--no-search", action="store_true", help="뉴스 검색 비활성")
    p.add_argument("--no-macro", action="store_true", help="거시 비활성")
    p.add_argument("--no-dart", action="store_true", help="DART 공시 비활성")
    p.add_argument("--no-peers", action="store_true", help="동종 비교 비활성")
    p.add_argument("--llm-synthesis", action="store_true",
                   help=f"LLM 통합 호출 (분류+자연어 단락). "
                        f"디폴트 {RETRO_SYN_PROVIDER}/{RETRO_SYN_MODEL}")
    p.add_argument("--llm-provider", default=RETRO_SYN_PROVIDER)
    p.add_argument("--llm-model", default=RETRO_SYN_MODEL)
    p.add_argument("--no-db", action="store_true",
                   help="SQLite analysis_reports 저장 안 함")
    p.add_argument("--obsidian", action="store_true",
                   help="옵시디언 vault 에 마크다운 저장")
    p.add_argument("--md", metavar="PATH",
                   help="마크다운을 임의 경로에 저장")
    p.add_argument("--print", action="store_true",
                   help="마크다운을 stdout 에도 출력")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    store = KisHistoryStore()

    peers_override = (
        [p.strip() for p in args.peers.split(",") if p.strip()]
        if args.peers else None
    )

    print(f"→ 회고 생성 {args.ticker} {args.start} ~ {args.end}", file=sys.stderr)
    report = compute_event_retro(
        args.ticker, args.start, args.end,
        store=store,
        enable_search=not args.no_search,
        enable_macro=not args.no_macro,
        enable_dart=not args.no_dart,
        enable_peers=not args.no_peers,
        industry_keyword=args.industry,
        peers_override=peers_override,
    )
    if report.get("period_summary") is None:
        print(f"ERROR: 데이터 부족 — {args.ticker}", file=sys.stderr)
        for w in report.get("warnings") or []:
            print(f"  - {w}", file=sys.stderr)
        return 2

    if args.llm_synthesis:
        print(f"→ LLM 통합 호출 ({args.llm_provider}/{args.llm_model})...",
              file=sys.stderr)
        report = synthesize_event_retro(
            report, provider=args.llm_provider, model=args.llm_model,
        )

    md_body = render_event_retro_markdown(report)
    syn_md = render_synthesis_markdown(report)
    if syn_md:
        md_body += "\n\n" + syn_md
    md_full = _frontmatter(report) + md_body + "\n"

    saved_paths: list[str] = []

    # 1) SQLite
    if not args.no_db:
        try:
            store.write_analysis_report(
                args.ticker, "event_retro", args.end, report,
            )
            saved_paths.append(
                f"sqlite: analysis_reports[{args.ticker}, event_retro, {args.end}]"
            )
        except Exception as exc:
            print(f"WARN: SQLite 저장 실패: {exc}", file=sys.stderr)

    # 2) 옵시디언
    if args.obsidian:
        vault = _find_obsidian_vault()
        if vault is None:
            print("WARN: 옵시디언 vault 못 찾음", file=sys.stderr)
        else:
            path = _obsidian_report_path(
                vault, args.ticker,
                report.get("meta", {}).get("company_name"),
                args.start, args.end,
            )
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

    # warnings 항상 stderr 에
    for w in report.get("warnings") or []:
        print(f"⚠ {w}", file=sys.stderr)
    for s in saved_paths:
        print(f"✅ {s}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
