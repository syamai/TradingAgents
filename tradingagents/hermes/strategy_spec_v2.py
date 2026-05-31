"""전략 spec v2 — 신규 신호 3종 추가 (기존 strategy_spec.py 무수정 확장).

기존 5종(net_streak/pct_threshold/pct_delta/net_vol_ratio/price_filter)은
``strategy_spec._validate_signal`` 에 그대로 위임하고, 승률 향상을 노린 신규
신호 3종만 여기서 정의·검증한다.

신규 신호 (전부 trailing — look-ahead 0):
  - ``price_drop``   : 최근 W일 종가 수익률 <= -X% (역추세/눌림목 진입 트리거)
  - ``trend_slope``  : ``{subject}_net_qty`` 누적합의 W일 변화 방향 (수급 추세 레짐)
  - ``rolling_corr`` : ``{subject}_net_qty`` vs ``price_change_pct`` 의 W일 trailing
                       Pearson r >= min_r (수급-가격 동조 필터)
  - ``market_filter``: KOSPI 종가가 N일 이평 위/아래인지 확인하는 시장 레짐 필터

``rolling_corr`` 는 무방향(|r|)이 아니라 양의 동조(r>=min_r)만 — "이 주체가
사면 이 종목이 오른다"는 확인 필터. 방향은 entry 의 net_streak buy 와 AND 로
결합해 부여한다(검증에서 지적된 무방향 결함 회피).
"""
from __future__ import annotations

from tradingagents.hermes import strategy_spec as base

# 기존 5종은 base 에 위임.
_BASE_SIGNALS: frozenset[str] = frozenset(
    {"net_streak", "pct_threshold", "pct_delta", "net_vol_ratio", "price_filter"}
)

# 신규 신호 그리드 (이산 — 과최적화 방어).
PRICE_DROP_WINDOW_GRID: tuple[int, ...] = (3, 5, 10, 20)
PRICE_DROP_PCT_GRID: tuple[float, ...] = (3.0, 5.0, 8.0, 10.0, 15.0)
TREND_SLOPE_WINDOW_GRID: tuple[int, ...] = (10, 20, 60)
CORR_LOOKBACK_GRID: tuple[int, ...] = (20, 60, 120)
CORR_MIN_R_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
MARKET_FILTER_WINDOW_GRID: tuple[int, ...] = (20, 60, 120)
MARKET_FILTER_MODE_GRID: frozenset[str] = frozenset({"above_ma", "below_ma"})

_DIRECTIONS: frozenset[str] = frozenset({"up", "down"})

SUPPORTED_VERSIONS: frozenset[int] = frozenset({1, 2})


def _validate_signal_v2(sig: dict, where: str) -> None:
    """신호 검증 — 기존 5종은 base 위임, 신규 3종은 여기서."""
    if not isinstance(sig, dict):
        raise ValueError(f"{where}: signal must be a dict, got {type(sig).__name__}")
    stype = base._need(sig, "signal", where)

    if stype in _BASE_SIGNALS:
        base._validate_signal(sig, where)
        return

    if stype == "price_drop":
        base._in_grid(
            base._need(sig, "window", where), PRICE_DROP_WINDOW_GRID, where, "window"
        )
        base._in_grid(
            base._need(sig, "value", where), PRICE_DROP_PCT_GRID, where, "value"
        )

    elif stype == "trend_slope":
        base._check_subject(sig, where)
        base._in_grid(
            base._need(sig, "window", where), TREND_SLOPE_WINDOW_GRID, where, "window"
        )
        if base._need(sig, "direction", where) not in _DIRECTIONS:
            raise ValueError(f"{where}: direction must be one of {sorted(_DIRECTIONS)}")

    elif stype == "rolling_corr":
        base._check_subject(sig, where)
        base._in_grid(
            base._need(sig, "lookback", where), CORR_LOOKBACK_GRID, where, "lookback"
        )
        base._in_grid(
            base._need(sig, "min_r", where), CORR_MIN_R_GRID, where, "min_r"
        )

    elif stype == "market_filter":
        mode = base._need(sig, "mode", where)
        if mode not in MARKET_FILTER_MODE_GRID:
            raise ValueError(
                f"{where}: mode must be one of {sorted(MARKET_FILTER_MODE_GRID)}"
            )
        base._in_grid(
            base._need(sig, "window", where), MARKET_FILTER_WINDOW_GRID, where, "window"
        )

    else:
        raise ValueError(f"{where}: unknown signal type {stype!r}")


def validate_spec_v2(spec: dict) -> None:
    """전략 spec v2 검증. 실패 시 ``ValueError``.

    ``strategy_spec.validate_spec`` 와 동일 구조이되 spec_version∈{1,2} 허용,
    신호 검증만 ``_validate_signal_v2`` 사용 (신규 3종 + 기존 5종 superset).
    """
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a dict, got {type(spec).__name__}")
    if spec.get("spec_version") not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"spec_version must be in {sorted(SUPPORTED_VERSIONS)}, "
            f"got {spec.get('spec_version')!r}"
        )
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if spec.get("direction") != "long":
        raise ValueError(
            f"only direction='long' supported (1차), got {spec.get('direction')!r}"
        )

    entry = spec.get("entry")
    if not isinstance(entry, dict):
        raise ValueError("entry must be a dict with 'all_of'")
    all_of = entry.get("all_of")
    if not isinstance(all_of, list) or not (
        1 <= len(all_of) <= base.MAX_ENTRY_SIGNALS
    ):
        raise ValueError(
            f"entry.all_of must have 1..{base.MAX_ENTRY_SIGNALS} signals, "
            f"got {len(all_of) if isinstance(all_of, list) else all_of!r}"
        )
    for i, s in enumerate(all_of):
        _validate_signal_v2(s, f"entry.all_of[{i}]")

    exit_ = spec.get("exit")
    if not isinstance(exit_, dict):
        raise ValueError("exit must be a dict")
    sig_all = exit_.get("signal_all_of", [])
    if not isinstance(sig_all, list) or len(sig_all) > base.MAX_EXIT_SIGNALS:
        raise ValueError(
            f"exit.signal_all_of must have 0..{base.MAX_EXIT_SIGNALS} signals"
        )
    for i, s in enumerate(sig_all):
        _validate_signal_v2(s, f"exit.signal_all_of[{i}]")

    sl = exit_.get("stop_loss_pct")
    if sl is not None:
        base._in_grid(sl, base.STOP_LOSS_GRID, "exit", "stop_loss_pct")
    tp = exit_.get("take_profit_pct")
    if tp is not None:
        base._in_grid(tp, base.TAKE_PROFIT_GRID, "exit", "take_profit_pct")
    mh = exit_.get("max_hold_days")
    if mh is None:
        raise ValueError("exit.max_hold_days is required (무한보유 방지)")
    base._in_grid(mh, base.MAX_HOLD_GRID, "exit", "max_hold_days")
