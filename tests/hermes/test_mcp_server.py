"""MCP 서버 e2e — 도구 등록·메타데이터 검증.

stdio transport 자체 검증은 별도 통합 환경 (Hermes 와의 실제 통신) 에서.
이 단위 테스트는 FastMCP 인스턴스에 도구가 *등록되어 있고 메타데이터가
정상* 임을 확인. ``@mcp.tool()`` 데코레이터가 함수 자체는 그대로 노출하므로
직접 호출도 가능 — wrap 한 ``run_analyst`` / ``compute_correlation`` 의
unit test 가 그 행동을 이미 검증.
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.mark.unit
class TestMcpServerToolRegistration:
    def test_module_imports_without_side_effects(self):
        # 분석가 LLM 인스턴스는 lazy — import 만으로는 Ollama 호출 X.
        from tradingagents.hermes import mcp_server
        assert mcp_server.mcp.name == "trading-ai-hermes"
        # G4: 5 분석가 + 4 통계 + 4 가설/라벨 = 13 도구.
        for tool_name in (
            "analyst_supply_demand", "analyst_sentiment", "analyst_market",
            "analyst_news", "analyst_fundamentals",
            "compute_correlation", "compute_trend", "compute_advanced",
            "get_holdings_window",
            "save_analysis", "add_feedback", "list_hypotheses",
            "get_hypothesis_with_labels",
        ):
            assert callable(getattr(mcp_server, tool_name)), tool_name

    def test_registers_expected_tools(self):
        from tradingagents.hermes.mcp_server import mcp

        async def _list():
            return await mcp.list_tools()

        tools = asyncio.run(_list())
        names = {t.name for t in tools}
        assert names == {
            "analyst_supply_demand", "analyst_sentiment", "analyst_market",
            "analyst_news", "analyst_fundamentals",
            "compute_correlation", "compute_trend", "compute_advanced",
            "get_holdings_window",
            "save_analysis", "add_feedback", "list_hypotheses",
            "get_hypothesis_with_labels",
        }

    def test_tool_descriptions_are_non_empty(self):
        from tradingagents.hermes.mcp_server import mcp

        async def _list():
            return await mcp.list_tools()

        tools = asyncio.run(_list())
        for t in tools:
            assert t.description, f"{t.name} 의 description 누락"
            assert len(t.description) > 30, (
                f"{t.name} description 이 너무 짧음 — Hermes 가 도구 선택 시 "
                f"docstring 참조하므로 풍부해야 함"
            )
