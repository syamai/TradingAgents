"""네이버 종목토론방 글수·감성 트렌드 신호(한국).

기존 ``naver_discussion.collect_naver_discussion(코드)`` 를 래핑해 ``asof_date``
당일 게시글 수를 ``SignalRow`` 로 만든다. 개인 투자자 관심이 한 종목에 몰리면
토론방 글이 폭증한다는 가정. Phase1 은 raw 글수(``abnormal_value=글수``)이며,
forward 로 baseline 을 쌓아 Phase2 에서 z-score(SV-Delta)로 전환한다.

Phase 2: 당일 글들의 공감/비공감으로 **감성 틸트**(board_sentiment)도 함께 낸다.
이는 글수(crowding 강도)와 달리 *방향*(낙관/비관)이라 fade_score(과열 강도)에
섞으면 안 되므로 ``rank=None`` 으로 적재 → ``_rank_score`` 가 0 → fade_score 무영향.
같은 source('naver_board')라 2소스 동의(독립성) 카운트에도 더해지지 않는다 —
토론방 글수·감성은 같은 게시판에서 나오므로 독립 소스가 아니다(독립 2소스는
naver_datalab 검색량). 디지스트에는 분위기 한 줄로만 표시.

per-ticker 소스라 universe(브리지가 만든 한국 후보)를 주입받는다. universe 가
없으면 첫 tick 으로 보고 빈 결과(실패 아님)를 반환한다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from ..naver_discussion import _parse_date, collect_naver_discussion
from .base import COINCIDENT, SignalRow

_SOURCE = "naver_board"
_METRIC = "board_volume"
_SENTIMENT_METRIC = "board_sentiment"
_MIN_VOTES = 5  # 공감+비공감 표본이 이보다 적으면 감성 노이즈 → 미산출
_KST = timezone(timedelta(hours=9))  # 토론방 글 일자는 KST 기준


def _today() -> str:
    return datetime.now(_KST).date().isoformat()


def collect_naver_board(
    market: str = "kr",
    *,
    universe: Optional[list] = None,
    asof_date: Optional[str] = None,
    lookback_days: int = 2,
    limit: int = 200,
    timeout: float = 5.0,  # 시그니처 일관성용(naver_discussion 자체 timeout 사용)
) -> tuple[list[SignalRow], bool]:
    """한국 후보별 종목토론방 ``asof_date`` 당일 글수 → SignalRow 리스트.

    반환 ``(rows, fetch_ok)``. universe 없으면 ``([], True)``(첫 tick). 전 종목
    수집 실패면 ``([], False)``(다음 tick 자가복구).
    """
    if not universe:
        return [], True

    asof = asof_date or _today()
    rows: list[SignalRow] = []
    any_ok = False
    for code in universe:
        err, posts = collect_naver_discussion(
            code, limit=limit, lookback_days=lookback_days
        )
        if err is not None:
            continue
        any_ok = True
        count = 0
        up_sum = 0
        down_sum = 0
        for p in posts:
            d = _parse_date(p.get("date", ""))
            if d is not None and d.date().isoformat() == asof:
                count += 1
                up_sum += int(p.get("up", 0) or 0)
                down_sum += int(p.get("down", 0) or 0)
        if count <= 0:
            continue
        rows.append(
            SignalRow(
                release_date=asof,
                asof_date=asof,
                market=market,
                entity=code,
                source=_SOURCE,
                metric=_METRIC,
                leadingness=COINCIDENT,
                raw_value=float(count),
                abnormal_value=float(count),
            )
        )
        # 감성 틸트 — 당일 글들의 공감/비공감 순도. rank 미부여(fade_score 무영향).
        votes = up_sum + down_sum
        if votes >= _MIN_VOTES:
            tilt = (up_sum - down_sum) / votes  # [-1, 1]
            rows.append(
                SignalRow(
                    release_date=asof,
                    asof_date=asof,
                    market=market,
                    entity=code,
                    source=_SOURCE,
                    metric=_SENTIMENT_METRIC,
                    leadingness=COINCIDENT,
                    raw_value=float(votes),
                    abnormal_value=round(tilt, 4),
                )
            )

    if not rows and not any_ok:
        return [], False
    # 글수(board_volume) 행만 순위 부여 — 감성 행은 rank=None 유지.
    vol = [r for r in rows if r.metric == _METRIC]
    vol.sort(key=lambda r: r.abnormal_value or 0.0, reverse=True)
    for i, r in enumerate(vol):
        r.rank = i + 1
    return rows, True
