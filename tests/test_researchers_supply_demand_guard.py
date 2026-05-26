"""Researchers (bull/bear)의 supply_demand_report 인용 가드 회귀 테스트.

`supply_demand_report`가 빈 문자열이거나 ``<not applicable: ...>`` /
``<unavailable: ...>`` 같은 마커 문자열이면 LLM이 "데이터 부재 = 약점"으로
해석할 위험이 있다. 가드(``startswith("<")`` 검사)로 prompt에서 supply/demand
블록 자체를 제거하는지 확인.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher


def _state(supply_demand_report: str) -> dict:
    return {
        "investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": "", "count": 0,
        },
        "market_report": "m", "sentiment_report": "s",
        "news_report": "n", "fundamentals_report": "f",
        "supply_demand_report": supply_demand_report,
        "company_of_interest": "005930.KS",
        "asset_type": "stock",
    }


def _capture_llm():
    """Return (llm, captured) where captured["prompt"] is set on invoke."""
    llm = MagicMock()
    captured = {}
    def _invoke(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return AIMessage(content="argument")
    llm.invoke.side_effect = _invoke
    return llm, captured


@pytest.mark.unit
class TestBullSupplyDemandGuard:
    def test_real_report_included(self):
        llm, cap = _capture_llm()
        node = create_bull_researcher(llm)
        node(_state("외국인 7일 연속 순매수, 누적 +1.2조원..."))
        assert "Supply/Demand signals" in cap["prompt"]
        assert "외국인 7일 연속" in cap["prompt"]

    def test_empty_report_block_omitted(self):
        llm, cap = _capture_llm()
        node = create_bull_researcher(llm)
        node(_state(""))
        assert "Supply/Demand signals" not in cap["prompt"]

    def test_not_applicable_marker_block_omitted(self):
        llm, cap = _capture_llm()
        node = create_bull_researcher(llm)
        node(_state("<not applicable: non-KR ticker>"))
        assert "Supply/Demand signals" not in cap["prompt"]
        assert "not applicable" not in cap["prompt"]

    def test_unavailable_marker_block_omitted(self):
        llm, cap = _capture_llm()
        node = create_bull_researcher(llm)
        node(_state("<unavailable: KIS_APP_KEY missing>"))
        assert "Supply/Demand signals" not in cap["prompt"]

    def test_missing_key_does_not_crash(self):
        llm, cap = _capture_llm()
        node = create_bull_researcher(llm)
        state = _state("")
        del state["supply_demand_report"]  # 키 자체 없는 경우
        node(state)
        assert "Supply/Demand signals" not in cap["prompt"]


@pytest.mark.unit
class TestBearSupplyDemandGuard:
    def test_real_report_included(self):
        llm, cap = _capture_llm()
        node = create_bear_researcher(llm)
        node(_state("외국인 5일 연속 순매도, 공매도 비중 12%..."))
        assert "Supply/Demand signals" in cap["prompt"]
        assert "5일 연속 순매도" in cap["prompt"]

    def test_not_applicable_marker_block_omitted(self):
        llm, cap = _capture_llm()
        node = create_bear_researcher(llm)
        node(_state("<not applicable: non-KR ticker>"))
        assert "Supply/Demand signals" not in cap["prompt"]

    def test_missing_key_does_not_crash(self):
        llm, cap = _capture_llm()
        node = create_bear_researcher(llm)
        state = _state("")
        del state["supply_demand_report"]
        node(state)
        assert "Supply/Demand signals" not in cap["prompt"]
