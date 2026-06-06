"""트렌드 신호 수집 cron 진입점 (LLM-free).

    uv run python -m tradingagents.hermes.trend_collect --source apewisdom
    uv run python -m tradingagents.hermes.trend_collect --source finviz_unusual --dry-run

``SOURCE_REGISTRY`` 에서 ``--source`` 의 fetcher 를 찾아 수집하고 ``TrendStore`` 에
적재한다. ``labeling_cron`` 과 동일하게 dict 결과를 JSON 으로 출력하고 **항상
exit 0** (엔드포인트 실패는 status 로 분류, 예외 미전파 → cron tick 이 죽지 않고
다음 tick 에서 자가복구).
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from ..dataflows.trend_store import TrendStore
from ..dataflows.trends import SOURCE_REGISTRY, UNIVERSE_SOURCES


def run(
    source: str,
    *,
    market: Optional[str] = None,
    asof_date: Optional[str] = None,
    dry_run: bool = False,
    store: Optional[TrendStore] = None,
) -> dict:
    """단일 소스 수집 → 적재. 반환=상태 dict(절대 예외 안 올림)."""
    if source not in SOURCE_REGISTRY:
        return {
            "status": "error",
            "source": source,
            "reason": f"unknown source; known={sorted(SOURCE_REGISTRY)}",
        }
    spec = SOURCE_REGISTRY[source]
    mkt = market or spec["market"]
    kwargs: dict = {"market": mkt}
    if asof_date is not None:
        kwargs["asof_date"] = asof_date
    # per-ticker 소스는 universe 를 주입. us=그날 뜬 hot_candidates,
    # kr=미국 fade 와치리스트의 유사 한국 종목(kr_peer_bridge).
    if source in UNIVERSE_SOURCES:
        st = store or TrendStore()
        if mkt == "kr":
            from .kr_peer_bridge import kr_universe
            kwargs["universe"] = kr_universe(asof_date=asof_date, store=st)
        else:
            kwargs["universe"] = st.hot_candidates(mkt, asof_date=asof_date, limit=10)
    try:
        rows, fetch_ok = spec["fn"](**kwargs)
    except Exception as exc:  # fetcher 는 graceful 이지만 방어적으로 캐치
        return {
            "status": "error",
            "source": source,
            "market": mkt,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not fetch_ok:
        return {
            "status": "fetch_failed",
            "source": source,
            "market": mkt,
            "collected": len(rows),
            "written": 0,
        }
    written = 0
    if not dry_run:
        store = store or TrendStore()
        written = store.write(rows)
    return {
        "status": "ok",
        "source": source,
        "market": mkt,
        "collected": len(rows),
        "written": written,
        "dry_run": dry_run,
        "top": [
            {"entity": r.entity, "abnormal": r.abnormal_value, "rank": r.rank}
            for r in rows[:5]
        ],
    }


def run_market(
    market: str = "us",
    *,
    asof_date: Optional[str] = None,
    dry_run: bool = False,
    store: Optional[TrendStore] = None,
) -> list:
    """해당 market 의 모든 등록 소스를 순차 수집(store 공유)."""
    if store is None and not dry_run:
        store = TrendStore()
    results = []
    for name, spec in SOURCE_REGISTRY.items():
        if spec["market"] != market:
            continue
        results.append(
            run(name, market=market, asof_date=asof_date, dry_run=dry_run, store=store)
        )
    return results


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="트렌드 신호 수집기 (LLM-free)")
    ap.add_argument(
        "--source", default=None,
        help=f"one of {sorted(SOURCE_REGISTRY)} (생략 시 --market 전체 소스)",
    )
    ap.add_argument("--market", default="us", help="us | kr (기본=us)")
    ap.add_argument("--asof-date", default=None, help="YYYY-MM-DD (기본=오늘)")
    ap.add_argument("--dry-run", action="store_true", help="적재 없이 결과만 출력")
    ap.add_argument(
        "--digest", action="store_true",
        help="수집 후 fade 와치리스트 다이제스트 텍스트를 출력(Telegram 전송용)",
    )
    args = ap.parse_args(argv)

    if args.source:
        results = [
            run(args.source, market=args.market, asof_date=args.asof_date,
                dry_run=args.dry_run)
        ]
    else:
        results = run_market(
            args.market, asof_date=args.asof_date, dry_run=args.dry_run
        )

    if args.digest:
        from .trend_rank import format_digest

        print(format_digest(args.market, asof_date=args.asof_date))
    else:
        print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
