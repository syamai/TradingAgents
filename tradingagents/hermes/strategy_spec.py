"""파라미터화된 수급 룰 전략 spec — 스키마·신호 어휘·검증.

전략은 진입(``entry``)·청산(``exit``) 규칙을 담은 dict. 모든 신호는 holdings
컬럼(``{subject}_net_qty`` / ``{subject}_pct`` / ``close`` / ``volume``)만
사용하며 *scale-free* — 종목군 일반화(시총·가격대가 다른 199 종목에 동일 룰
적용)를 위해 절대 주수 임계 대신 부호·비중·거래량 대비 비율로 표현한다.

과최적화 방어: 수치 파라미터는 이산 그리드 값만 허용(연속 튜닝 금지), entry
신호 ≤3·exit 신호 ≤2. ``validate_spec`` 가 위반을 ``ValueError`` 로 거부한다
(``hypothesis_store._validate_record`` 와 동일한 수동 검증 패턴).

신호 어휘 (5종, 전부 long-only 진입/청산에 사용):
  - ``net_streak``    : ``{subject}_net_qty`` 가 연속 N거래일 동일 부호
  - ``pct_threshold`` : ``{subject}_pct`` (누적 영향력 비중%) 임계
  - ``pct_delta``     : ``{subject}_pct`` 의 N일 변화량 (비중 추세)
  - ``net_vol_ratio`` : N일 누적 net_qty / N일 누적 volume (거래량 대비 강도)
  - ``price_filter``  : 종가 vs N일 이동평균 위치 (추세 동조 필터)
"""
from __future__ import annotations

import hashlib
import json

from tradingagents.dataflows.kis_holdings import SUBS_10

# 신호 subject: 10 sub-주체 + 외국인 통합. MEMORY 시드의 "기관 통합 분석 금지,
# sub-주체 분리" 의무와 정합 — "institution" 같은 통합 카테고리는 노출하지 않음.
VALID_SUBJECTS: frozenset[str] = frozenset(SUBS_10) | {"foreign"}

# 이산 파라미터 그리드 (과최적화 방어 — 이 값만 허용).
MIN_DAYS_GRID: tuple[int, ...] = (2, 3, 4, 5, 7, 10)
WINDOW_GRID: tuple[int, ...] = (1, 3, 5, 10, 20)
PCT_GRID: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0, 50.0)
PCT_DELTA_GRID: tuple[float, ...] = (2.0, 5.0, 10.0)
VOL_RATIO_GRID: tuple[float, ...] = (0.05, 0.1, 0.2)
MA_GRID: tuple[int, ...] = (5, 20, 60)
STOP_LOSS_GRID: tuple[float, ...] = (3.0, 5.0, 8.0, 10.0)
TAKE_PROFIT_GRID: tuple[float, ...] = (5.0, 8.0, 10.0, 15.0, 20.0)
MAX_HOLD_GRID: tuple[int, ...] = (5, 10, 20, 40, 60)

_OPS: frozenset[str] = frozenset({">=", "<="})
_SIGNS: frozenset[str] = frozenset({"buy", "sell"})
_MA_MODES: frozenset[str] = frozenset({"above_ma", "below_ma"})

MAX_ENTRY_SIGNALS = 3
MAX_EXIT_SIGNALS = 2


def _need(sig: dict, key: str, where: str):
    if key not in sig:
        raise ValueError(f"{where}: signal missing field {key!r}")
    return sig[key]


def _in_grid(val, grid: tuple, where: str, name: str) -> None:
    # int/float 혼용 허용 (예: 5 와 5.0 동일) — 그리드 멤버십은 == 비교.
    if not any(val == g for g in grid):
        raise ValueError(f"{where}: {name}={val!r} not in grid {list(grid)}")


def _check_subject(sig: dict, where: str) -> str:
    subj = _need(sig, "subject", where)
    if subj not in VALID_SUBJECTS:
        raise ValueError(
            f"{where}: subject={subj!r} not in {sorted(VALID_SUBJECTS)}"
        )
    return subj


def _validate_signal(sig: dict, where: str) -> None:
    """단일 신호 검증. 실패 시 ``ValueError``."""
    if not isinstance(sig, dict):
        raise ValueError(f"{where}: signal must be a dict, got {type(sig).__name__}")
    stype = _need(sig, "signal", where)

    if stype == "net_streak":
        _check_subject(sig, where)
        _in_grid(_need(sig, "min_days", where), MIN_DAYS_GRID, where, "min_days")
        if _need(sig, "sign", where) not in _SIGNS:
            raise ValueError(f"{where}: sign must be one of {sorted(_SIGNS)}")

    elif stype == "pct_threshold":
        _check_subject(sig, where)
        if _need(sig, "op", where) not in _OPS:
            raise ValueError(f"{where}: op must be one of {sorted(_OPS)}")
        _in_grid(_need(sig, "value", where), PCT_GRID, where, "value")

    elif stype == "pct_delta":
        _check_subject(sig, where)
        _in_grid(_need(sig, "window", where), WINDOW_GRID, where, "window")
        if _need(sig, "op", where) not in _OPS:
            raise ValueError(f"{where}: op must be one of {sorted(_OPS)}")
        _in_grid(_need(sig, "value", where), PCT_DELTA_GRID, where, "value")

    elif stype == "net_vol_ratio":
        _check_subject(sig, where)
        _in_grid(_need(sig, "window", where), WINDOW_GRID, where, "window")
        if _need(sig, "op", where) not in _OPS:
            raise ValueError(f"{where}: op must be one of {sorted(_OPS)}")
        _in_grid(_need(sig, "value", where), VOL_RATIO_GRID, where, "value")

    elif stype == "price_filter":
        if _need(sig, "mode", where) not in _MA_MODES:
            raise ValueError(f"{where}: mode must be one of {sorted(_MA_MODES)}")
        _in_grid(_need(sig, "window", where), MA_GRID, where, "window")

    else:
        raise ValueError(f"{where}: unknown signal type {stype!r}")


def validate_spec(spec: dict) -> None:
    """전략 spec 전체 검증. 실패 시 ``ValueError``.

    검증 항목: spec_version==1, name 비어있지 않음, direction=='long',
    entry.all_of (1..3 신호), exit.signal_all_of (0..2 신호),
    stop_loss_pct/take_profit_pct (옵션, 그리드), max_hold_days (필수, 그리드).
    """
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a dict, got {type(spec).__name__}")
    if spec.get("spec_version") != 1:
        raise ValueError(f"spec_version must be 1, got {spec.get('spec_version')!r}")
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
    if not isinstance(all_of, list) or not (1 <= len(all_of) <= MAX_ENTRY_SIGNALS):
        raise ValueError(
            f"entry.all_of must have 1..{MAX_ENTRY_SIGNALS} signals, "
            f"got {len(all_of) if isinstance(all_of, list) else all_of!r}"
        )
    for i, s in enumerate(all_of):
        _validate_signal(s, f"entry.all_of[{i}]")

    exit_ = spec.get("exit")
    if not isinstance(exit_, dict):
        raise ValueError("exit must be a dict")
    sig_all = exit_.get("signal_all_of", [])
    if not isinstance(sig_all, list) or len(sig_all) > MAX_EXIT_SIGNALS:
        raise ValueError(
            f"exit.signal_all_of must have 0..{MAX_EXIT_SIGNALS} signals"
        )
    for i, s in enumerate(sig_all):
        _validate_signal(s, f"exit.signal_all_of[{i}]")

    sl = exit_.get("stop_loss_pct")
    if sl is not None:
        _in_grid(sl, STOP_LOSS_GRID, "exit", "stop_loss_pct")
    tp = exit_.get("take_profit_pct")
    if tp is not None:
        _in_grid(tp, TAKE_PROFIT_GRID, "exit", "take_profit_pct")
    mh = exit_.get("max_hold_days")
    if mh is None:
        raise ValueError("exit.max_hold_days is required (무한보유 방지)")
    _in_grid(mh, MAX_HOLD_GRID, "exit", "max_hold_days")


def _canonical(spec: dict) -> dict:
    """해시용 정규형 — name 제외(식별자는 로직 동일성과 무관), all_of 정렬
    (AND 는 교환법칙 — [A,B] 와 [B,A] 동일 전략)."""
    def _sig_key(s: dict) -> str:
        return json.dumps(s, sort_keys=True, ensure_ascii=False)

    exit_ = spec.get("exit", {})
    return {
        "spec_version": spec["spec_version"],
        "direction": spec["direction"],
        "entry": {"all_of": sorted(spec["entry"]["all_of"], key=_sig_key)},
        "exit": {
            "signal_all_of": sorted(exit_.get("signal_all_of", []), key=_sig_key),
            "stop_loss_pct": exit_.get("stop_loss_pct"),
            "take_profit_pct": exit_.get("take_profit_pct"),
            "max_hold_days": exit_["max_hold_days"],
        },
    }


def spec_hash(spec: dict) -> str:
    """전략 로직 정규형의 md5 — 동일 변이 중복 저장 멱등 차단용.

    ``name`` 은 무시(같은 로직, 다른 이름 = 같은 전략). 호출 전 ``validate_spec``
    통과를 가정.
    """
    canon = json.dumps(_canonical(spec), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(canon.encode("utf-8")).hexdigest()
