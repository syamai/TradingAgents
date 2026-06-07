"""strategy_research_v2 신규 기술신호 아키타입(A19~A21) 회귀 테스트.

상한(2025-06-30) 적용 측정에서 공정게이트 통과율이 입증된 신호만 결정적 탐색
후보 생성기에 편입했다: breakout_high(단독 43%)·breakout_high+volume_surge(29%)·
close_location(좁은 z5 영역). range_compression 은 단독 엣지가 없어 제외.
"""
from __future__ import annotations

import pytest

from tradingagents.hermes.strategy_research_v2 import _candidates, _dedup, _phase_filter
from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2

NEW_PREFIXES = ("tbreak-", "tdvb-", "tclose-")


def _entry_signals(spec: dict) -> set[str]:
    out = set()
    for c in spec.get("entry", {}).get("all_of", []):
        if "signal" in c:
            out.add(c["signal"])
    return out


@pytest.mark.unit
class TestNewTechnicalArchetypes:
    def test_candidates_include_new_archetypes_with_expected_counts(self):
        names = [s["name"] for s in _candidates()]
        assert sum(n.startswith("tbreak-") for n in names) == 24   # A19 breakout
        assert sum(n.startswith("tdvb-") for n in names) == 24     # A20 dispersion+vol+breakout
        assert sum(n.startswith("tclose-") for n in names) == 12   # A21 close_location(좁은 z5)

    def test_all_new_specs_validate(self):
        for s in _candidates():
            if s["name"].startswith(NEW_PREFIXES):
                validate_spec_v2(s)  # 무효 시 raise

    def test_new_specs_dedup_clean(self):
        ded = _dedup(_candidates())
        assert sum(s["name"].startswith(NEW_PREFIXES) for s in ded) == 60

    def test_archetypes_use_expected_target_signals(self):
        sigs = {p: set() for p in NEW_PREFIXES}
        for s in _candidates():
            for p in NEW_PREFIXES:
                if s["name"].startswith(p):
                    sigs[p] |= _entry_signals(s)
        assert "breakout_high" in sigs["tbreak-"]
        assert {"breakout_high", "volume_surge"} <= sigs["tdvb-"]
        assert "close_location" in sigs["tclose-"]
        # range_compression 은 측정상 엣지가 없어 어느 신규 아키타입에도 들어가면 안 됨.
        all_new = sigs["tbreak-"] | sigs["tdvb-"] | sigs["tclose-"]
        assert "range_compression" not in all_new

    def test_tclose_only_narrow_z5_region(self):
        # z3_60·z10_250 은 상한 적용 측정에서 전멸 → z5_120 만 편입.
        for s in _candidates():
            if s["name"].startswith("tclose-"):
                assert "z5_120_15" in s["name"]

    def test_phase_filter_keeps_supply_demand_before_technical(self):
        # 신규 기술 아키타입은 검증된 수급 아키타입(A16~A18)의 평가를 굶기지 않도록 그 뒤에 와야 한다.
        adv = _phase_filter(_dedup(_candidates()), 500)
        first_tech = next(i for i, s in enumerate(adv) if s["name"].startswith(NEW_PREFIXES))
        first_sd = next(i for i, s in enumerate(adv) if s["name"].startswith(("cons-", "disp-", "accel-")))
        assert first_sd < first_tech
