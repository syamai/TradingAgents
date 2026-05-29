"""일별 사후 라벨링 크론 진입점.

PRD G3 — LLM 호출 0. 미라벨 가설 중 호라이즌 경과한 것을 KOSPI 상대 수익
기준으로 자동 라벨링한다.

실행:
    uv run python -m tradingagents.hermes.labeling_cron
    uv run python -m tradingagents.hermes.labeling_cron --today 2026-06-12
    uv run python -m tradingagents.hermes.labeling_cron --dry-run

등록 (예시 — macOS launchd 일별):
    매일 9 시 KST 실행. ``com.tradingai.hermes.labeling.plist`` 작성 →
    ``launchctl load -w ~/Library/LaunchAgents/<...>.plist``.

또는 cron:
    0 9 * * * cd /Users/selab/Source/trading-ai && \
        uv run python -m tradingagents.hermes.labeling_cron \
        >> ~/.tradingagents/hermes/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from tradingagents.hermes.hypothesis_store import HypothesisStore
from tradingagents.hermes.labeling import LabelingScheduler


def run(today: Optional[str] = None, *, dry_run: bool = False) -> dict:
    """라벨링 1 회 실행. 반환 = 상태별 카운트 dict.

    dry_run 시 store 는 같은 미라벨 가설 목록을 미리 보여주되 ``add_label``
    호출하지 않도록 LabelingScheduler 의 store 를 fake 로 wrap. 단순화 위해
    ``dry_run`` 은 store.list() 만 호출 + 보여주기.
    """
    store = HypothesisStore()
    if dry_run:
        pending = store.list()
        return {
            "dry_run": True,
            "pending_count": len(pending),
            "by_horizon": _group_by_horizon(pending),
        }
    sched = LabelingScheduler(store)
    return sched.run_pending(today=today)


def _group_by_horizon(records: list[dict]) -> dict:
    """가설 리스트를 horizon_weeks 별 카운트 dict 로."""
    counts: dict[int, int] = {}
    for r in records:
        h = int(r["horizon_weeks"])
        counts[h] = counts.get(h, 0) + 1
    return {str(k): v for k, v in sorted(counts.items())}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="trading-ai / Hermes 사후 가격 라벨링 일별 크론",
    )
    parser.add_argument(
        "--today",
        help="라벨링 기준일 (YYYY-MM-DD). 미지정 시 오늘 (UTC).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 라벨 추가 없이 대기 중 가설만 카운트.",
    )
    args = parser.parse_args(argv)

    result = run(today=args.today, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 라벨링 자체 실패는 없음 — fetch 실패는 status 로 분류. 항상 0 종료.
    return 0


if __name__ == "__main__":
    sys.exit(main())
