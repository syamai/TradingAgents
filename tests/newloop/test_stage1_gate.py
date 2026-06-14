"""Stage1 게이트(s1-v2) 순수 로직 테스트 — score_pairs / ols_alpha_beta_clustered."""

from __future__ import annotations

import pytest

from tradingagents.newloop.stage1_gate import (
    GATE_MIN_CLUSTERS,
    GATE_MIN_DOWN_TRADES,
    GATE_MIN_T_ALPHA,
    GATE_MIN_TRADES,
    SCORE_DATE_HI,
    deflated_t_threshold,
    dsr_benchmark_t,
    ols_alpha_beta_clustered,
    score,
    score_pairs,
)


def _wiggle(i: int) -> float:
    """결정적 미세 노이즈 — 무분산 퇴화 방지용 (±0.1 교대)."""
    return 0.1 if i % 2 == 0 else -0.1


MONTHS = [f"2025-{m:02d}" for m in range(1, 11)]  # 군집 10개


def _pairs(alpha: float, beta: float, n_per_month: int = 6,
           month_noise: float = 0.0) -> list[tuple[float, float, str]]:
    """월별 바스켓 수익(-3~+6% 순환) 위에 y = α + β·x (+월 공통 노이즈) 생성.

    month_noise 는 같은 달 거래들이 공유하는 잔차 — 군집 상관의 원천.
    """
    cycle = [-3.0, -1.5, 1.0, 2.5, 4.0, 6.0]
    pairs = []
    i = 0
    for mi, month in enumerate(MONTHS):
        e_m = month_noise * (1 if mi % 2 == 0 else -1)
        for j in range(n_per_month):
            x = cycle[(mi + j) % len(cycle)]
            pairs.append((alpha + beta * x + e_m + _wiggle(i), x, month))
            i += 1
    return pairs


@pytest.mark.unit
class TestOls:
    def test_exact_line_recovers_alpha_beta(self):
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 3
        cl = [f"m{i % 6}" for i in range(len(xs))]
        ys = [2.0 + 1.5 * x + _wiggle(i) for i, x in enumerate(xs)]
        a, t_a, b, _t_b, g = ols_alpha_beta_clustered(xs, ys, cl)
        assert a == pytest.approx(2.0, abs=0.15)
        assert b == pytest.approx(1.5, abs=0.1)
        assert g == 6
        assert t_a > 3

    def test_degenerate_inputs(self):
        assert ols_alpha_beta_clustered([1.0] * 3, [1.0, 2.0, 3.0], ["a", "b", "c"]) is None
        assert ols_alpha_beta_clustered([1.0, 2.0], [1.0, 2.0], ["a", "b"]) is None
        # 군집 1개 — 분산 추정 불가
        assert ols_alpha_beta_clustered([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ["a", "a", "a"]) is None


@pytest.mark.unit
class TestScorePairs:
    def test_pure_beta_disguise_fails(self):
        """순수 베타 위장(net=2×basket): 절편 0 + 하락창 유의 음수 → fail."""
        r = score_pairs(_pairs(alpha=0.0, beta=2.0))
        assert r["status"] == "fail"
        assert abs(r["alpha_pct"]) < 0.5
        assert r["beta"] == pytest.approx(2.0, abs=0.1)
        assert r["checks"]["alpha_significant"] is False
        assert r["checks"]["down_not_broken"] is False  # 하락창에서 바스켓보다 더 깨짐

    def test_genuine_alpha_passes(self):
        """진짜 실력(net=basket+2%): 절편 유의+경제적 + 하락창 무손상 → pass."""
        r = score_pairs(_pairs(alpha=2.0, beta=1.0))
        assert r["status"] == "pass"
        assert r["alpha_pct"] == pytest.approx(2.0, abs=0.2)
        assert r["t_alpha"] >= 3.0
        assert r["checks"]["alpha_material"] is True
        assert r["down"]["n"] >= GATE_MIN_DOWN_TRADES

    def test_replication_attack_blocked(self):
        """C2 회귀: 월 공통 잔차를 공유하는 대량 복제가 t를 부풀리지 못한다.

        월 노이즈 ±1.2 위 α=0.5 — 군집 무시 t라면 n=120으로 유의해 보이지만,
        군집-로버스트 t는 월 10개 표본의 분산을 반영해 3 미만이어야 한다.
        """
        r = score_pairs(_pairs(alpha=0.5, beta=1.0, n_per_month=12, month_noise=1.2))
        assert r["n_trades"] == 120
        assert r["t_alpha"] < 3.0
        assert r["status"] != "pass"

    def test_tiny_alpha_fails_economic_floor(self):
        """W2 회귀: 유의하지만 경제성 없는 α(0.1% < 0.3%)는 탈락."""
        r = score_pairs(_pairs(alpha=0.1, beta=1.0, n_per_month=20))
        assert r["checks"]["alpha_material"] is False
        assert r["gate_passed"] is False

    def test_insufficient_trades(self):
        pairs = _pairs(alpha=2.0, beta=1.0)[: GATE_MIN_TRADES - 1]
        r = score_pairs(pairs)
        assert r["status"] == "insufficient"
        assert r["checks"]["enough_trades"] is False

    def test_insufficient_clusters(self):
        """시기 다양성 부족(군집 < 최소): 거래가 많아도 보류."""
        base = _pairs(alpha=2.0, beta=1.0)
        two_months = [(n, x, MONTHS[i % 2]) for i, (n, x, _) in enumerate(base)]
        r = score_pairs(two_months)
        assert r["n_clusters"] == 2 < GATE_MIN_CLUSTERS
        assert r["status"] == "insufficient"
        assert r["checks"]["enough_clusters"] is False

    def test_insufficient_down_window_is_held(self):
        """W1 회귀: 하락창 표본 < 최소(10)면 강한 알파라도 보류."""
        cycle = [1.0, 2.5, 4.0, 6.0]  # 거의 전부 양수 바스켓
        pairs = [(cycle[i % 4] + 2.0 + _wiggle(i), cycle[i % 4], MONTHS[i % 10])
                 for i in range(56)]
        # 하락 거래를 최소 미만(4건)만 섞는다
        pairs += [(-1.0 + 2.0 + _wiggle(i), -1.0, MONTHS[i % 10]) for i in range(4)]
        r = score_pairs(pairs)
        assert 0 < r["down"]["n"] < GATE_MIN_DOWN_TRADES
        assert r["status"] == "insufficient"
        assert r["checks"]["enough_down"] is False


@pytest.mark.unit
class TestDeflation:
    """s1-v3: 누적 시도 수 N 에 맞춘 동적 합격선(다중검정 보정)."""

    def test_threshold_floor_and_monotone(self):
        # N=1 단일검정은 HLZ 하한 3.0, N 이 커질수록 단조 증가, 결코 3.0 밑은 없음
        assert deflated_t_threshold(1) == GATE_MIN_T_ALPHA
        assert deflated_t_threshold(24) == GATE_MIN_T_ALPHA  # N≈38 전까진 floor 가 지배
        seq = [deflated_t_threshold(n) for n in (1, 100, 400, 3106)]
        assert all(b >= a for a, b in zip(seq, seq[1:]))
        assert deflated_t_threshold(100) > GATE_MIN_T_ALPHA
        assert deflated_t_threshold(100) == pytest.approx(3.28, abs=0.05)
        assert deflated_t_threshold(400) == pytest.approx(3.66, abs=0.05)

    def test_dsr_benchmark_grows_with_n(self):
        assert dsr_benchmark_t(1) is None
        assert dsr_benchmark_t(3106) > dsr_benchmark_t(100) > 0

    def test_bootstrap_threshold_engaged(self):
        """score_pairs 가 정규임계가 아니라 wild-cluster 부트스트랩 임계로 판정."""
        r = score_pairs(_pairs(alpha=2.0, beta=1.0), n_trials=1)
        m = r["multiplicity"]
        assert m["t_crit_method"] == "wild_bootstrap"
        assert m["t_crit"] >= GATE_MIN_T_ALPHA          # floor 의도 보존
        assert "t_crit_normal" in m
        # 진짜 강한 알파는 부트스트랩 임계도 넘어 통과
        assert r["t_alpha"] >= m["t_crit"]
        assert r["status"] == "pass"

    def test_genuine_alpha_blocked_when_n_large(self):
        """N=1 에서 통과하는 진짜 알파라도, 누적 N 이 충분히 크면 합격선이
        관측 t 를 추월해 탈락한다 — '천천히 반복'이 아니라 N 자체가 임계를 올림."""
        # 현실적 잡음(amp=6)으로 t≈6.4 인 진짜 알파 — 무잡음이면 t 가 수백이라
        # 임계가 추월할 수 없다(실제 홀드아웃엔 항상 잡음이 있다).
        cycle = [-3.0, -1.5, 1.0, 2.5, 4.0, 6.0]
        pairs = []
        i = 0
        for mi, mo in enumerate(MONTHS):
            for j in range(8):
                x = cycle[(mi + j) % len(cycle)]
                noise = 6.0 * (((i * 7 + 3) % 11) - 5) / 5.0  # 결정적 ±6
                pairs.append((2.0 + 1.0 * x + noise, x, mo))
                i += 1
        base = score_pairs(pairs, n_trials=1)
        assert base["status"] == "pass"
        t_obs = base["t_alpha"]
        # 관측 t 를 넘기는 합격선을 만드는 최소 N 을 찾는다 (단조 증가이므로 존재)
        n = 10
        while deflated_t_threshold(n) <= t_obs and n < 10 ** 18:
            n *= 10
        big = score_pairs(pairs, n_trials=n)
        assert big["multiplicity"]["t_crit"] > t_obs
        assert big["checks"]["alpha_significant"] is False
        assert big["status"] == "fail"
        # FWER p 는 N 과 함께 커진다 (운으로 이 t 이상이 나올 확률)
        assert big["multiplicity"]["fwer_p"] > base["multiplicity"]["fwer_p"]


@pytest.mark.unit
class TestScoringWindow:
    """채점 창은 급등 컷오프(2025-06-30)를 넘지 못한다 — 금지 구간 채점 방지(s1-v3)."""

    @staticmethod
    def _empty_loader(_tk):
        return None, {}

    def test_window_end_clamped_to_cutoff(self):
        # 컷오프를 넘는 끝을 줘도 강제로 2025-06-30 으로 클램프
        r = score({"name": "x"}, [], self._empty_loader, lambda a, b: None,
                  w0="", w1="2026-12-31")
        assert r["window"][1] == SCORE_DATE_HI

    def test_window_default_ends_at_cutoff(self):
        # 기본 호출(끝 미지정)도 컷오프에서 끝남 — 급등 구간 미채점
        r = score({"name": "x"}, [], self._empty_loader, lambda a, b: None)
        assert r["window"] == ["", SCORE_DATE_HI]
