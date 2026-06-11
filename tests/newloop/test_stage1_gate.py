"""Stage1 게이트 순수 로직 테스트 — score_pairs / ols_alpha_beta."""

from __future__ import annotations

import pytest

from tradingagents.newloop.stage1_gate import (
    GATE_MIN_TRADES,
    ols_alpha_beta,
    score_pairs,
)


def _wiggle(i: int) -> float:
    """결정적 미세 노이즈 — 무분산 퇴화 방지용 (±0.1 교대)."""
    return 0.1 if i % 2 == 0 else -0.1


def _baskets(n: int) -> list[float]:
    """상승·하락 섞인 바스켓 수익 — -3 ~ +6% 순환."""
    cycle = [-3.0, -1.5, 1.0, 2.5, 4.0, 6.0]
    return [cycle[i % len(cycle)] for i in range(n)]


@pytest.mark.unit
class TestOls:
    def test_exact_line_recovers_alpha_beta(self):
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        ys = [2.0 + 1.5 * x + _wiggle(i) for i, x in enumerate(xs)]
        a, t_a, b, _t_b = ols_alpha_beta(xs, ys)
        assert a == pytest.approx(2.0, abs=0.15)
        assert b == pytest.approx(1.5, abs=0.1)
        assert t_a > 3  # 노이즈 0.1 대비 절편 2.0 — 강하게 유의

    def test_degenerate_inputs(self):
        assert ols_alpha_beta([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # x 무변동
        assert ols_alpha_beta([1.0, 2.0], [1.0, 2.0]) is None  # n<3


@pytest.mark.unit
class TestScorePairs:
    def test_pure_beta_disguise_fails(self):
        """순수 베타 위장(net=2×basket): 절편 0 + 하락창 유의 음수 → fail."""
        xs = _baskets(60)
        pairs = [(2.0 * x + _wiggle(i), x) for i, x in enumerate(xs)]
        r = score_pairs(pairs)
        assert r["status"] == "fail"
        assert r["gate_passed"] is False
        assert abs(r["alpha_pct"]) < 0.5
        assert r["beta"] == pytest.approx(2.0, abs=0.1)
        assert r["checks"]["alpha_significant"] is False
        assert r["checks"]["down_not_broken"] is False  # 하락창에서 바스켓보다 더 깨짐

    def test_genuine_alpha_passes(self):
        """진짜 실력(net=basket+2%): 절편 유의 + 하락창 무손상 → pass."""
        xs = _baskets(60)
        pairs = [(x + 2.0 + _wiggle(i), x) for i, x in enumerate(xs)]
        r = score_pairs(pairs)
        assert r["status"] == "pass"
        assert r["gate_passed"] is True
        assert r["alpha_pct"] == pytest.approx(2.0, abs=0.2)
        assert r["t_alpha"] >= 3.0
        assert r["down"]["mean_pct"] == pytest.approx(2.0, abs=0.2)

    def test_insufficient_trades(self):
        pairs = [(1.0 + _wiggle(i), 1.0) for i in range(GATE_MIN_TRADES - 1)]
        r = score_pairs(pairs)
        assert r["status"] == "insufficient"
        assert r["checks"]["enough_trades"] is False
        assert r["gate_passed"] is False

    def test_no_down_window_is_held_not_passed(self):
        """하락 거래 0건(일방향 장)이면 강한 알파라도 합격 아닌 보류."""
        xs = [1.0, 2.0, 3.0, 4.0] * 15  # 전부 양수 바스켓
        pairs = [(x + 2.0 + _wiggle(i), x) for i, x in enumerate(xs)]
        r = score_pairs(pairs)
        assert r["checks"]["alpha_significant"] is True
        assert r["down"]["n"] == 0
        assert r["status"] == "insufficient"
        assert r["gate_passed"] is False
