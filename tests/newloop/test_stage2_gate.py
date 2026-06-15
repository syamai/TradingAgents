"""Stage2 게이트(s2-v1) 측정 엔진 테스트 — 투입자본 기준 시장중립 회귀.

합성 일별 시계열(포트폴리오 일수익 r_p, 시장 r_m, 투입비중 w)로 통계 핵심을
검증한다. 적대 감사가 재현한 파훼(방향성 junk 가짜통과·희석 실력 가짜탈락·
현금오염 하락검사)를 회귀로 고정한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from tradingagents.newloop import ledger
from tradingagents.newloop.stage2_gate import (
    GATE_VERSION,
    S2_ALPHA_ANN_MIN,
    S2_BETA_MAX,
    S2_MIN_INVESTED_DAYS,
    hac_ols,
    score_portfolio,
)


def _full(alpha_daily: float, beta: float, n: int = 250, idio: float = 0.004, seed: int = 0):
    """항상 100% 투입(w=1): r_p = α + β·r_m + 잡음."""
    rng = np.random.default_rng(seed)
    r_m = rng.normal(0.0005, 0.012, n)
    r_p = alpha_daily + beta * r_m + rng.normal(0.0, idio, n)
    return r_p, r_m, np.ones(n)


@pytest.mark.unit
class TestHacOls:
    def test_recovers_alpha_beta(self):
        r_p, r_m, _ = _full(alpha_daily=0.001, beta=1.3)
        a, _t_a, b, _t_b, n = hac_ols(r_p, r_m, maxlags=5)
        assert a == pytest.approx(0.001, abs=0.0006)
        assert b == pytest.approx(1.3, abs=0.1)
        assert n == 250

    def test_degenerate_returns_none(self):
        assert hac_ols([1.0, 2.0], [1.0, 2.0], maxlags=2) is None
        assert hac_ols([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], maxlags=2) is None


@pytest.mark.unit
class TestScorePortfolio:
    def test_pure_beta_no_alpha_fails(self):
        r_p, r_m, w = _full(alpha_daily=0.0, beta=1.5)
        r = score_portfolio(r_p, r_m, w)
        assert r["beta"] == pytest.approx(1.5, abs=0.12)
        assert r["checks"]["alpha_significant"] is False
        assert r["status"] == "fail"

    def test_genuine_alpha_passes(self):
        r_p, r_m, w = _full(alpha_daily=0.0015, beta=1.0)
        r = score_portfolio(r_p, r_m, w)
        assert r["t_alpha"] >= 3.0
        assert r["alpha_ann_pct"] > 0
        assert r["checks"]["beta_controlled"] is True
        assert r["status"] == "pass"

    def test_high_beta_blocked(self):
        r_p, r_m, w = _full(alpha_daily=0.0015, beta=2.5)
        r = score_portfolio(r_p, r_m, w)
        assert r["beta"] > S2_BETA_MAX
        assert r["checks"]["beta_controlled"] is False
        assert r["status"] == "fail"

    def test_insufficient_invested_days(self):
        r_p, r_m, w = _full(alpha_daily=0.0015, beta=1.0, n=250)
        w = np.zeros(250)
        w[: S2_MIN_INVESTED_DAYS - 1] = 1.0  # 투입일 최소 미만
        r = score_portfolio(r_p, r_m, w)
        assert r["n_invested_days"] == S2_MIN_INVESTED_DAYS - 1
        assert r["status"] == "insufficient"
        assert r["checks"]["enough_days"] is False


@pytest.mark.unit
class TestAuditRegressions:
    """적대 감사가 실제 재현한 오측정을 투입자본 기준 측정이 닫는지 고정."""

    def test_directional_junk_blocked(self):
        """대부분-현금 + 모멘텀(상승 후 투입), 실력 0(β 노출만): 가짜 통과 금지.

        구 전체달력 회귀는 시장캡처를 절편으로 흘려 α≈27%·통과시켰다.
        투입자본 회귀는 β 가 시장캡처를 흡수해 α≈0 → 비통과.
        """
        n = 400
        rng = np.random.default_rng(11)
        r_m = rng.normal(0.001, 0.012, n)
        prev_up = np.concatenate([[False], r_m[:-1] > 0])  # 상승 다음날 투입(모멘텀)
        w = np.where(prev_up, 0.3, 0.0)
        sleeve = 1.2 * r_m + rng.normal(0.0, 0.003, n)      # β 노출, 알파 0
        r_p = w * sleeve
        r = score_portfolio(r_p, r_m, w)
        assert r["beta"] == pytest.approx(1.2, abs=0.2)
        assert abs(r["alpha_ann_pct"]) < 15        # 절편으로 새던 ~27% 가 사라짐
        assert r["checks"]["alpha_significant"] is False
        assert r["status"] != "pass"

    def test_diluted_skill_recovered(self):
        """투입은 드물지만(≈10%) 투입자본당 강한 실력: 가짜 탈락 금지.

        구 전체달력 회귀는 현금일이 절편을 희석해 t≈1.6 으로 탈락시켰다.
        투입자본 회귀는 현금을 빼고 봐 진짜 알파를 살린다.
        """
        n = 500
        rng = np.random.default_rng(5)
        r_m = rng.normal(0.0005, 0.012, n)
        invest = rng.random(n) < 0.12
        w = np.where(invest, 0.3, 0.0)
        sleeve = 0.004 + 1.0 * r_m + rng.normal(0.0, 0.004, n)  # 투입자본 0.4%/day 실력
        r_p = w * sleeve
        r = score_portfolio(r_p, r_m, w)
        assert r["n_invested_days"] >= S2_MIN_INVESTED_DAYS
        assert r["deployment"] < 0.1               # 대부분 현금
        assert r["alpha_ann_pct"] > 0
        assert r["t_alpha"] >= 3.0
        assert r["status"] == "pass"

    def test_down_check_sees_invested_collapse(self):
        """투입 하락일의 베타 위장(하락장 β 급증)을 검사가 본다.

        구 버전은 현금일(r_p=0) 때문에 하락일 초과수익이 구조적 +로 깔려
        위장을 못 봤다. 투입일만 보면 하락 붕괴가 음(-)으로 드러난다.
        """
        n = 400
        rng = np.random.default_rng(3)
        r_m = rng.normal(0.0003, 0.012, n)
        invest = rng.random(n) < 0.5
        w = np.where(invest, 0.3, 0.0)
        beta_t = np.where(r_m < 0, 2.6, 1.0)              # 하락장에서 민감도 급증
        sleeve = beta_t * r_m + rng.normal(0.0, 0.003, n)
        r_p = w * sleeve
        r = score_portfolio(r_p, r_m, w)
        assert r["down"]["n"] >= 1
        assert r["down"]["mean_excess_pct"] < 0          # 하락 붕괴가 음으로 보임(현금오염 제거)
        assert r["status"] != "pass"

    def test_deflation_raises_bar_with_many_candidates(self):
        r_p, r_m, w = _full(alpha_daily=0.0015, beta=1.0)
        base = score_portfolio(r_p, r_m, w, n_trials=1)
        many = score_portfolio(r_p, r_m, w, n_trials=500)
        assert many["multiplicity"]["t_crit"] > base["multiplicity"]["t_crit"]


@pytest.mark.unit
class TestBootstrapAndDiagnostics:
    """재감사 확정 결함 3개: MBB 임계·overlay 죽은연산·미분산 진단."""

    def test_mbb_inflates_threshold_for_fat_tailed_small_sample(self):
        """소표본·중첩보유 fat-tail junk(알파0): 이동블록 부트스트랩이 정규임계
        보다 합격선을 올려 가짜통과를 막는다(정규는 FWER 0.96 누출)."""
        n = 60
        rng = np.random.default_rng(7)
        r_m = rng.normal(0.0005, 0.012, n)
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.5 * e[i - 1] + rng.standard_t(4) * 0.006
        r_p = 1.0 * r_m + e               # 진짜 알파 0
        w = np.ones(n)
        r = score_portfolio(r_p, r_m, w, n_trials=1, max_hold_days=20)
        m = r["multiplicity"]
        assert m["t_crit_method"] == "mbb_bootstrap"
        assert m["t_crit"] > m["t_crit_normal"]      # 정규보다 엄격
        assert m["t_crit"] > 4.0                      # 소표본 fat-tail 반영
        assert r["status"] != "pass"

    def test_concentration_diagnostic_flags_undiversified(self):
        """단일종목 지배(k=1 과반)면 미분산 경고 + drop1 민감도 노출."""
        n = 250
        rng = np.random.default_rng(1)
        r_m = rng.normal(0.0005, 0.012, n)
        r_p = 0.0008 + 1.0 * r_m + rng.normal(0.0, 0.004, n)
        w = np.ones(n)
        k = np.where(rng.random(n) < 0.6, 1, 4)      # 60% 단일종목
        r = score_portfolio(r_p, r_m, w, n_trials=1, k=k)
        c = r["concentration"]
        assert c is not None
        assert c["undiversified"] is True
        assert c["frac_solo"] > 0.5
        assert c["drop1_alpha_ann_pct"] is not None

    def test_overlay_no_longer_multiplies(self):
        """k 미전달이면 concentration=None — 진단은 옵션, 판정엔 무관."""
        r_p, r_m, w = _full(alpha_daily=0.0015, beta=1.0)
        r = score_portfolio(r_p, r_m, w, n_trials=1)
        assert r["concentration"] is None
        assert r["status"] == "pass"


@pytest.mark.unit
class TestCostAwareThreshold:
    """s2-v2: 비용 인지 α 하한(3%/년) — 유의해도 비용 미만이면 경제성 탈락."""

    def test_version_and_floor(self):
        assert GATE_VERSION == "s2-v2"
        assert S2_ALPHA_ANN_MIN == 3.0

    def test_significant_subcost_alpha_fails_material(self):
        # α 일간 0.0001 → 연율 ≈ 2.52% < 3% 하한. 잡음 작아 t 는 유의하지만 경제성 탈락.
        r_p, r_m, w = _full(alpha_daily=0.0001, beta=1.0, idio=0.00015)
        r = score_portfolio(r_p, r_m, w)
        assert 0 < r["alpha_ann_pct"] < S2_ALPHA_ANN_MIN
        assert r["checks"]["alpha_significant"] is True
        assert r["checks"]["alpha_material"] is False
        assert r["status"] == "fail"


@pytest.mark.unit
class TestStage2BudgetSharing:
    """#4: Stage2 채점이 '<family>-s2' 키로 같은 누적 N(예산)에 합산된다."""

    def test_separate_s2_key_shares_budget(self, tmp_path):
        db = str(tmp_path / "ledger.db")
        ledger.register_family("famX", "h", db_path=db)
        ledger.record_submission(
            "famX",
            [{"strategy_ref": "s1", "gate_version": "s1-v3", "status": "pass", "result": {}}],
            db_path=db)
        assert ledger.cumulative_trials(db_path=db) == 1
        # Stage1 family 는 통과로 잠겨 재제출 불가
        ok1, _ = ledger.can_submit("famX", db_path=db)
        assert ok1 is False
        # Stage2 는 '<family>-s2' 별도 키 — 신규라 제출 가능
        ledger.register_family("famX-s2", "[S2] famX", db_path=db)
        ok2, _ = ledger.can_submit("famX-s2", db_path=db)
        assert ok2 is True
        ledger.record_submission(
            "famX-s2",
            [{"strategy_ref": "s1", "gate_version": GATE_VERSION, "status": "pass", "result": {}}],
            db_path=db)
        # 누적 N 이 Stage1+Stage2 합산 → 예산 공유 (#4)
        assert ledger.cumulative_trials(db_path=db) == 2
