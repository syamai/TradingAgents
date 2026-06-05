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
  - ``flow_zscore``   : ``{subject}_net_qty`` W일 누적의 자기 과거(N일) 분포 대비
                        z-score >= min_z (수급 서프라이즈 = 이례적 순매수)
  - ``flow_accel``    : ``{subject}_net_qty`` 단기(A일)·장기(B일) 평균 교차 — 단기가
                        장기를 상회하면 매집 가속(A<B 강제)
  - ``flow_divergence``: 지정 주체 W일 순매수 누적 > 0 AND 개인(retail) W일 누적 < 0
                        (주체 간 로테이션 — smart-money 매집 + 개인 매도)
  - ``flow_consensus`` : 지정 주체군에서 W일 순매수 주체 수 >= min_buyers. 옵션으로
                        개인 순매도 동반을 요구해 smart-money 합의/개인 흡수를 표현.
  - ``flow_dispersion``: 지정 주체군 W일 순매수 총합 > 0 이면서 최대 단일 주체 비중 <= max_share
                        (한 주체 쏠림보다 분산 매집을 선호).
  - ``fast_money_unwind``: 외국인비등록 + 사모가 W일 동시 순매도. 주로 exit 에서
                        단기자금 이탈/과열 해소를 감지.
  - ``liquidity_filter``: W일 평균 거래대금(close*volume) 임계. 저유동 아티팩트 차단.
  - ``short_ratio``   : 공매도 거래량 비중(short_volume_ratio %)의 W일 trailing 평균
                        임계 — 낮은 공매도 압력(<=) 선호. short 데이터 부재 시 False.
  - ``price_return``  : 종가의 W일 trailing 수익률(%) 임계 — 장기 모멘텀(>= 양수) /
                        역추세(<=) 신호. 긴 윈도우(120/250)로 12-1 류 모멘텀 표현.
  - ``realized_vol``  : 일별수익률의 W일 trailing 실현변동성(연율화 %) 임계 — 저변동성
                        팩터(<=) / 고변동성(>=). close 만 사용.
  - ``volume_surge``  : 단기 평균 거래량 / 장기 평균 거래량 비율. 거래 참여 급증.
  - ``range_compression``: W일 고저폭 / 종가가 낮은 변동성 수축 상태.
  - ``breakout_high`` : 전일 기준 W일 고점에 근접/돌파. 당일 이후 정보 미사용.
  - ``close_location``: 당일 종가가 일중 고저 범위의 상단에 위치하는 강한 종가.

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
# 수급 흐름-변화 신호 3종 그리드 (이산 — 과최적화 방어).
FLOW_ZSCORE_WINDOW_GRID: tuple[int, ...] = (3, 5, 10, 20)
FLOW_ZSCORE_LOOKBACK_GRID: tuple[int, ...] = (60, 120, 250)
FLOW_ZSCORE_MIN_Z_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
FLOW_ACCEL_SHORT_GRID: tuple[int, ...] = (3, 5, 10)
FLOW_ACCEL_LONG_GRID: tuple[int, ...] = (20, 60)
FLOW_DIVERGENCE_WINDOW_GRID: tuple[int, ...] = (3, 5, 10, 20)
FLOW_GROUPS: dict[str, tuple[str, ...]] = {
    "fast_money": ("foreign_unregistered", "private_equity"),
    "foreign_pair": ("foreign_registered", "foreign_unregistered"),
    "institution_defensive": ("pension", "insurance", "investment_trust"),
    "broad_smart": ("foreign_registered", "foreign_unregistered", "private_equity", "investment_trust", "pension"),
}
FLOW_CONSENSUS_MIN_BUYERS_GRID: tuple[int, ...] = (2, 3, 4)
FLOW_DISPERSION_MAX_SHARE_GRID: tuple[float, ...] = (0.5, 0.6, 0.7)
FAST_MONEY_UNWIND_WINDOW_GRID: tuple[int, ...] = (3, 5, 10)
LIQUIDITY_WINDOW_GRID: tuple[int, ...] = (20, 60)
LIQUIDITY_VALUE_GRID: tuple[float, ...] = (1_000_000_000.0, 5_000_000_000.0, 10_000_000_000.0, 20_000_000_000.0)
# 공매도 압력 신호 그리드 (short_volume_ratio = 공매도 거래량 비중 %, 이산).
SHORT_RATIO_VALUE_GRID: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0)
# 장기 가격 모멘텀 신호 그리드 (긴 윈도우 — 12-1 류 모멘텀, 이산).
PRICE_RETURN_WINDOW_GRID: tuple[int, ...] = (20, 60, 120, 250)
PRICE_RETURN_VALUE_GRID: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0)
# 실현변동성(저변동성 팩터) 신호 그리드 (연율화 %, 이산).
REALIZED_VOL_WINDOW_GRID: tuple[int, ...] = (20, 60, 120)
REALIZED_VOL_VALUE_GRID: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0)
VOLUME_SURGE_SHORT_GRID: tuple[int, ...] = (3, 5, 10)
VOLUME_SURGE_LONG_GRID: tuple[int, ...] = (20, 60)
VOLUME_SURGE_MIN_RATIO_GRID: tuple[float, ...] = (1.5, 2.0, 3.0)
RANGE_COMPRESSION_WINDOW_GRID: tuple[int, ...] = (10, 20, 60)
RANGE_COMPRESSION_MAX_PCT_GRID: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)
BREAKOUT_HIGH_WINDOW_GRID: tuple[int, ...] = (20, 60, 120)
BREAKOUT_HIGH_PROXIMITY_GRID: tuple[float, ...] = (0.0, 2.0, 5.0)
CLOSE_LOCATION_MIN_GRID: tuple[float, ...] = (0.6, 0.75, 0.9)

# 시장 하락 회피 overlay (포트폴리오 결합 후 KOSPI trailing return 으로 exposure 축소).
# 값은 이산 그리드로 제한해 사후 과최적화/임의 소수 저장을 막는다.
MARKET_OVERLAY_TYPE_GRID: frozenset[str] = frozenset({"kospi_trailing_return_scale", "usdkrw_trailing_return_ma_scale"})
MARKET_OVERLAY_WINDOW_GRID: tuple[int, ...] = (20, 40, 60, 80, 120, 200)
MARKET_OVERLAY_MA_WINDOW_GRID: tuple[int, ...] = (20, 60, 120, 200)
MARKET_OVERLAY_THRESHOLD_GRID: tuple[float, ...] = (-10.0, -8.0, -6.0, -5.0, -3.0, 0.0, 2.0, 3.0, 5.0)
MARKET_OVERLAY_STOCK_WEIGHT_GRID: tuple[float, ...] = (0.0, 20.0, 30.0, 45.0, 60.0, 70.0, 90.0)

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

    elif stype == "flow_zscore":
        base._check_subject(sig, where)
        base._in_grid(
            base._need(sig, "window", where), FLOW_ZSCORE_WINDOW_GRID, where, "window"
        )
        base._in_grid(
            base._need(sig, "lookback", where), FLOW_ZSCORE_LOOKBACK_GRID, where, "lookback"
        )
        base._in_grid(
            base._need(sig, "min_z", where), FLOW_ZSCORE_MIN_Z_GRID, where, "min_z"
        )

    elif stype == "flow_accel":
        base._check_subject(sig, where)
        short = base._need(sig, "short", where)
        long = base._need(sig, "long", where)
        base._in_grid(short, FLOW_ACCEL_SHORT_GRID, where, "short")
        base._in_grid(long, FLOW_ACCEL_LONG_GRID, where, "long")
        if not short < long:
            raise ValueError(f"{where}: short({short}) must be < long({long})")

    elif stype == "flow_divergence":
        base._check_subject(sig, where)
        base._in_grid(
            base._need(sig, "window", where), FLOW_DIVERGENCE_WINDOW_GRID, where, "window"
        )

    elif stype == "flow_consensus":
        group = base._need(sig, "group", where)
        if group not in FLOW_GROUPS:
            raise ValueError(f"{where}: group must be one of {sorted(FLOW_GROUPS)}")
        base._in_grid(
            base._need(sig, "window", where), FLOW_DIVERGENCE_WINDOW_GRID, where, "window"
        )
        base._in_grid(
            base._need(sig, "min_buyers", where), FLOW_CONSENSUS_MIN_BUYERS_GRID, where, "min_buyers"
        )
        if "require_retail_sell" in sig and not isinstance(sig["require_retail_sell"], bool):
            raise ValueError(f"{where}: require_retail_sell must be bool when provided")

    elif stype == "flow_dispersion":
        group = base._need(sig, "group", where)
        if group not in FLOW_GROUPS:
            raise ValueError(f"{where}: group must be one of {sorted(FLOW_GROUPS)}")
        base._in_grid(
            base._need(sig, "window", where), FLOW_DIVERGENCE_WINDOW_GRID, where, "window"
        )
        base._in_grid(
            base._need(sig, "max_share", where), FLOW_DISPERSION_MAX_SHARE_GRID, where, "max_share"
        )

    elif stype == "fast_money_unwind":
        base._in_grid(
            base._need(sig, "window", where), FAST_MONEY_UNWIND_WINDOW_GRID, where, "window"
        )

    elif stype == "liquidity_filter":
        base._in_grid(
            base._need(sig, "window", where), LIQUIDITY_WINDOW_GRID, where, "window"
        )
        if base._need(sig, "op", where) not in base._OPS:
            raise ValueError(f"{where}: op must be one of {sorted(base._OPS)}")
        base._in_grid(
            base._need(sig, "value", where), LIQUIDITY_VALUE_GRID, where, "value"
        )

    elif stype == "short_ratio":
        base._in_grid(
            base._need(sig, "window", where), base.WINDOW_GRID, where, "window"
        )
        if base._need(sig, "op", where) not in base._OPS:
            raise ValueError(f"{where}: op must be one of {sorted(base._OPS)}")
        base._in_grid(
            base._need(sig, "value", where), SHORT_RATIO_VALUE_GRID, where, "value"
        )

    elif stype == "price_return":
        base._in_grid(
            base._need(sig, "window", where), PRICE_RETURN_WINDOW_GRID, where, "window"
        )
        if base._need(sig, "op", where) not in base._OPS:
            raise ValueError(f"{where}: op must be one of {sorted(base._OPS)}")
        base._in_grid(
            base._need(sig, "value", where), PRICE_RETURN_VALUE_GRID, where, "value"
        )

    elif stype == "realized_vol":
        base._in_grid(
            base._need(sig, "window", where), REALIZED_VOL_WINDOW_GRID, where, "window"
        )
        if base._need(sig, "op", where) not in base._OPS:
            raise ValueError(f"{where}: op must be one of {sorted(base._OPS)}")
        base._in_grid(
            base._need(sig, "value", where), REALIZED_VOL_VALUE_GRID, where, "value"
        )

    elif stype == "volume_surge":
        short = base._need(sig, "short", where)
        long = base._need(sig, "long", where)
        base._in_grid(short, VOLUME_SURGE_SHORT_GRID, where, "short")
        base._in_grid(long, VOLUME_SURGE_LONG_GRID, where, "long")
        if not short < long:
            raise ValueError(f"{where}: short({short}) must be < long({long})")
        base._in_grid(
            base._need(sig, "min_ratio", where), VOLUME_SURGE_MIN_RATIO_GRID, where, "min_ratio"
        )

    elif stype == "range_compression":
        base._in_grid(
            base._need(sig, "window", where), RANGE_COMPRESSION_WINDOW_GRID, where, "window"
        )
        base._in_grid(
            base._need(sig, "max_pct", where), RANGE_COMPRESSION_MAX_PCT_GRID, where, "max_pct"
        )

    elif stype == "breakout_high":
        base._in_grid(
            base._need(sig, "window", where), BREAKOUT_HIGH_WINDOW_GRID, where, "window"
        )
        base._in_grid(
            base._need(sig, "proximity_pct", where), BREAKOUT_HIGH_PROXIMITY_GRID, where, "proximity_pct"
        )

    elif stype == "close_location":
        base._in_grid(
            base._need(sig, "min_pos", where), CLOSE_LOCATION_MIN_GRID, where, "min_pos"
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

    overlay = spec.get("market_overlay")
    if overlay is not None:
        if not isinstance(overlay, dict):
            raise ValueError("market_overlay must be a dict when provided")
        otype = base._need(overlay, "type", "market_overlay")
        if otype not in MARKET_OVERLAY_TYPE_GRID:
            raise ValueError(
                f"market_overlay.type must be one of {sorted(MARKET_OVERLAY_TYPE_GRID)}"
            )
        base._in_grid(
            base._need(overlay, "window", "market_overlay"),
            MARKET_OVERLAY_WINDOW_GRID,
            "market_overlay",
            "window",
        )
        if base._need(overlay, "op", "market_overlay") not in base._OPS:
            raise ValueError(f"market_overlay.op must be one of {sorted(base._OPS)}")
        base._in_grid(
            base._need(overlay, "threshold_pct", "market_overlay"),
            MARKET_OVERLAY_THRESHOLD_GRID,
            "market_overlay",
            "threshold_pct",
        )
        base._in_grid(
            base._need(overlay, "risk_stock_weight_pct", "market_overlay"),
            MARKET_OVERLAY_STOCK_WEIGHT_GRID,
            "market_overlay",
            "risk_stock_weight_pct",
        )
        if otype == "usdkrw_trailing_return_ma_scale":
            base._in_grid(
                base._need(overlay, "ma_window", "market_overlay"),
                MARKET_OVERLAY_MA_WINDOW_GRID,
                "market_overlay",
                "ma_window",
            )
        shock = overlay.get("shock_cap")
        if shock is not None:
            if not isinstance(shock, dict):
                raise ValueError("market_overlay.shock_cap must be a dict when provided")
            base._in_grid(
                base._need(shock, "window", "market_overlay.shock_cap"),
                MARKET_OVERLAY_WINDOW_GRID,
                "market_overlay.shock_cap",
                "window",
            )
            if base._need(shock, "op", "market_overlay.shock_cap") not in base._OPS:
                raise ValueError(
                    f"market_overlay.shock_cap.op must be one of {sorted(base._OPS)}"
                )
            base._in_grid(
                base._need(shock, "threshold_pct", "market_overlay.shock_cap"),
                MARKET_OVERLAY_THRESHOLD_GRID,
                "market_overlay.shock_cap",
                "threshold_pct",
            )
            base._in_grid(
                base._need(shock, "cap_stock_weight_pct", "market_overlay.shock_cap"),
                MARKET_OVERLAY_STOCK_WEIGHT_GRID,
                "market_overlay.shock_cap",
                "cap_stock_weight_pct",
            )
