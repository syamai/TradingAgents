"""interpretation 12 함수 — 빈 입력 안전 + 키 결과로 합리적 출력."""
from __future__ import annotations

import pandas as pd
import pytest

import dashboard.interpretation as itp
from dashboard.advanced_analysis import compute_advanced_report
from dashboard.correlation_analysis import compute_correlation_report
from tests.dashboard.test_correlation_analysis import _sample_holdings_signed


# === 빈 입력 / 안전성 ===

@pytest.mark.unit
class TestSafeOnEmpty:
    @pytest.mark.parametrize("fn", [
        itp.interpret_cumulative, itp.interpret_concurrent,
        itp.interpret_up_down, itp.interpret_lag,
        itp.interpret_regime, itp.interpret_level,
    ])
    def test_empty_correlation_report(self, fn):
        empty = compute_correlation_report(pd.DataFrame(), ticker="005930")
        out = fn(empty)
        assert isinstance(out, str)
        assert len(out) > 0
        assert "해석 불가" in out or "데이터 없음" in out

    @pytest.mark.parametrize("fn", [
        itp.interpret_adf, itp.interpret_granger, itp.interpret_var_irf,
        itp.interpret_cointegration, itp.interpret_mi, itp.interpret_rolling,
    ])
    def test_empty_advanced_report(self, fn):
        empty = compute_advanced_report(pd.DataFrame())
        out = fn(empty)
        assert isinstance(out, str)
        assert len(out) > 0


# === sample data — 기본 동작 확인 ===

@pytest.fixture
def corr_report():
    df = _sample_holdings_signed(60)
    return compute_correlation_report(df, ticker="005930", company_name="삼성전자")


@pytest.fixture
def adv_report():
    df = _sample_holdings_signed(200)
    return compute_advanced_report(df)


@pytest.mark.unit
class TestCorrelationInterpretations:
    def test_cumulative_mentions_top_buyer_or_seller(self, corr_report):
        out = itp.interpret_cumulative(corr_report)
        # _sample_holdings_signed 는 retail(-100*delta), foreign_registered(+100*delta)
        # 누적이 부호 보존이라 평균이 0 근처이지만 어느 쪽으로든 정렬됨.
        assert "비중" in out
        assert "%" in out

    def test_concurrent_identifies_strong(self, corr_report):
        out = itp.interpret_concurrent(corr_report)
        # retail = -1, foreign_registered = +1 → 매우 강한 상관
        assert "강한" in out
        assert "개인" in out or "외국인(등록)" in out

    def test_up_down_classifies_pattern(self, corr_report):
        out = itp.interpret_up_down(corr_report)
        assert "역행" in out or "추종" in out

    def test_lag_handles_no_lead(self, corr_report):
        out = itp.interpret_lag(corr_report)
        # 랜덤워크 데이터라 lag 효과 없는 게 정상
        assert ("선도" in out or "chasing" in out or "lag" in out
                or "단기 신호" in out)


@pytest.mark.unit
class TestAdvancedInterpretations:
    def test_adf_mentions_return_and_close(self, adv_report):
        out = itp.interpret_adf(adv_report)
        assert "return" in out
        assert "close" in out
        assert "정상" in out

    def test_granger_outputs_lead_or_chase(self, adv_report):
        out = itp.interpret_granger(adv_report)
        assert "예측" in out or "chasing" in out

    def test_var_irf_output_present_when_fit(self, adv_report):
        out = itp.interpret_var_irf(adv_report)
        # 정상 적합 시 차수·horizon 또는 fit 실패 메시지
        assert "차수" in out or "fit" in out or "실패" in out

    def test_cointegration_mentions_count(self, adv_report):
        out = itp.interpret_cointegration(adv_report)
        assert "공적분" in out

    def test_mi_describes_top(self, adv_report):
        out = itp.interpret_mi(adv_report)
        assert "비선형" in out or "MI" in out or "Pearson" in out or "약함" in out

    def test_rolling_summarizes_stability(self, adv_report):
        out = itp.interpret_rolling(adv_report)
        assert "안정" in out or "변동" in out or "regime" in out or "window" in out


# === 렌더된 마크다운에 해석 줄 포함 ===

@pytest.mark.unit
class TestMarkdownContainsInterpretation:
    def test_corr_markdown_has_quoted_interpretation_lines(self):
        from dashboard.correlation_analysis import render_markdown
        df = _sample_holdings_signed(60)
        report = compute_correlation_report(df, ticker="005930", company_name="삼성전자")
        md = render_markdown(report)
        # 6 개 섹션 모두 `> **해석**:` 줄 포함
        assert md.count("> **해석**:") == 6

    def test_advanced_markdown_has_quoted_interpretation_lines(self):
        from dashboard.advanced_analysis import render_advanced_markdown
        df = _sample_holdings_signed(200)
        adv = compute_advanced_report(df)
        md = render_advanced_markdown(adv)
        # 6 섹션 + ADF의 inline `*해석*` 별개. `> **자동 해석**:` 줄 카운트
        assert md.count("> **자동 해석**:") == 6
