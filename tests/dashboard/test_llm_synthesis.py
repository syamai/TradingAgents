"""llm_synthesis — 컨텍스트 빌더 + LLM 호출 mock."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dashboard.advanced_analysis import compute_advanced_report
from dashboard.correlation_analysis import compute_correlation_report
from dashboard.llm_synthesis import (
    DEFAULT_MODEL, DEFAULT_PROVIDER, _build_user_message, synthesize_conclusion,
)
from tests.dashboard.test_correlation_analysis import _sample_holdings_signed


@pytest.fixture
def sample_report():
    df = _sample_holdings_signed(60)
    return compute_correlation_report(df, ticker="005930",
                                       company_name="삼성전자", market="KOSPI")


@pytest.fixture
def sample_advanced():
    df = _sample_holdings_signed(200)
    return compute_advanced_report(df)


# === 컨텍스트 빌더 ===

@pytest.mark.unit
class TestBuildUserMessage:
    def test_includes_header_and_sections(self, sample_report):
        msg = _build_user_message(sample_report)
        assert "삼성전자" in msg
        assert "(005930)" in msg
        assert "KOSPI" in msg
        # 6 섹션 라벨 포함
        for label in ["1. 누적", "2. 동시", "3. 상승일", "4. Lag",
                      "5. Regime", "6. Level"]:
            assert label in msg

    def test_advanced_sections_when_provided(self, sample_report, sample_advanced):
        msg = _build_user_message(sample_report, sample_advanced)
        # 7-1 ~ 7-6 표시
        for label in ["7-1. ADF", "7-2. Granger", "7-3. VAR",
                      "7-4. Cointegration", "7-5. Mutual", "7-6. Rolling"]:
            assert label in msg

    def test_no_advanced_block_when_omitted(self, sample_report):
        msg = _build_user_message(sample_report)
        assert "7-1" not in msg
        assert "정교한 분석" not in msg


# === LLM 호출 (mock) ===

@pytest.mark.unit
class TestSynthesizeConclusion:
    def test_empty_report_short_circuits(self):
        out = synthesize_conclusion({}, None)
        assert "분석 데이터가 없어" in out

    def test_zero_days_short_circuits(self):
        out = synthesize_conclusion({"n_days": 0}, None)
        assert "분석 데이터가 없어" in out

    def test_calls_factory_with_provider_and_model(self, sample_report):
        with patch("dashboard.llm_synthesis.create_llm_client") as mk:
            client = MagicMock()
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="## 🧠 최종\n\n좋습니다.")
            client.get_llm.return_value = llm
            mk.return_value = client

            out = synthesize_conclusion(sample_report, provider="ollama",
                                         model="gemma4:26b-a4b")

            mk.assert_called_once_with("ollama", "gemma4:26b-a4b")
            assert "좋습니다" in out
            assert out.startswith("## 🧠")

    def test_adds_header_if_missing(self, sample_report):
        with patch("dashboard.llm_synthesis.create_llm_client") as mk:
            client = MagicMock()
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="본문만 있음 — 헤더 없음")
            client.get_llm.return_value = llm
            mk.return_value = client

            out = synthesize_conclusion(sample_report)
            assert out.startswith("## 🧠 최종 종합 의견")
            assert "본문만 있음" in out

    def test_exception_returns_graceful_message(self, sample_report):
        with patch("dashboard.llm_synthesis.create_llm_client") as mk:
            mk.side_effect = ConnectionError("Ollama server not reachable")
            out = synthesize_conclusion(sample_report)
            assert "## 🧠 최종 종합 의견" in out
            assert "LLM 호출 실패" in out
            assert "ConnectionError" in out

    def test_empty_llm_response_handled(self, sample_report):
        with patch("dashboard.llm_synthesis.create_llm_client") as mk:
            client = MagicMock()
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="   ")
            client.get_llm.return_value = llm
            mk.return_value = client

            out = synthesize_conclusion(sample_report)
            assert "응답이 비어" in out

    def test_default_provider_model_used_when_none(self, sample_report):
        with patch("dashboard.llm_synthesis.create_llm_client") as mk:
            client = MagicMock()
            llm = MagicMock()
            llm.invoke.return_value = MagicMock(content="## 🧠 X")
            client.get_llm.return_value = llm
            mk.return_value = client

            synthesize_conclusion(sample_report)
            mk.assert_called_once_with(DEFAULT_PROVIDER, DEFAULT_MODEL)
