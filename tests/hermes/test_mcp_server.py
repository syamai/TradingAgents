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
        # 5 분석가 + 4 통계 + 4 가설/라벨 + 3 전략 = 16 도구.
        for tool_name in (
            "analyst_supply_demand", "analyst_sentiment", "analyst_market",
            "analyst_news", "analyst_fundamentals",
            "compute_correlation", "compute_trend", "compute_advanced",
            "get_holdings_window",
            "save_analysis", "add_feedback", "list_hypotheses",
            "get_hypothesis_with_labels",
            "backtest_strategy", "save_strategy", "list_strategies",
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
            "backtest_strategy", "save_strategy", "list_strategies",
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


def _valid_strategy_spec() -> dict:
    return {
        "spec_version": 1, "name": "t", "direction": "long",
        "entry": {"all_of": [
            {"signal": "net_streak", "subject": "foreign_registered",
             "min_days": 3, "sign": "buy"},
        ]},
        "exit": {"max_hold_days": 20, "stop_loss_pct": 5.0},
    }


def _fake_result(gate_passed=True) -> dict:
    m = {
        "n_trades": 60, "win_rate": 0.62, "avg_net_ret_pct": 1.2,
        "avg_hold_days": 8.5, "cum_return_pct": 35.0, "sharpe": 1.4,
        "mdd_pct": -12.0, "avg_excess_ret_pct": 2.1,
    }
    return {"in_sample": m, "out_sample": dict(m), "gate_passed": gate_passed,
            "universe_size": 2, "n_in": 1, "n_out": 1}


@pytest.mark.unit
class TestStrategyTools:
    def test_backtest_strategy_returns_result(self, monkeypatch):
        from tradingagents.hermes import mcp_server
        monkeypatch.setattr(mcp_server, "_universe", lambda: ["005930", "000660"])
        monkeypatch.setattr(
            mcp_server, "run_universe_backtest",
            lambda spec, tickers, **kw: _fake_result(),
        )
        out = mcp_server.backtest_strategy(_valid_strategy_spec())
        assert out["gate_passed"] is True
        assert out["in_sample"]["win_rate"] == 0.62

    def test_backtest_strategy_invalid_spec_raises(self):
        from tradingagents.hermes import mcp_server
        bad = _valid_strategy_spec()
        bad["entry"]["all_of"][0]["min_days"] = 6  # off-grid
        with pytest.raises(ValueError):
            mcp_server.backtest_strategy(bad)

    def test_save_strategy_persists_then_dedups(self, monkeypatch, tmp_path):
        from tradingagents.hermes import mcp_server
        from tradingagents.hermes.strategy_store import StrategyStore
        store = StrategyStore(root=tmp_path)
        monkeypatch.setattr(mcp_server, "_strategy_store", lambda: store)
        monkeypatch.setattr(mcp_server, "_universe", lambda: ["005930", "000660"])
        monkeypatch.setattr(
            mcp_server, "run_universe_backtest",
            lambda spec, tickers, **kw: _fake_result(),
        )
        first = mcp_server.save_strategy(_valid_strategy_spec())
        assert first["duplicate"] is False
        assert first["gate_passed"] is True
        # 동일 로직 재저장 → 재백테스트 없이 duplicate.
        second = mcp_server.save_strategy(_valid_strategy_spec(), name="other")
        assert second["duplicate"] is True
        assert second["strategy_id"] == first["strategy_id"]

    def test_list_strategies_filters_gate(self, monkeypatch, tmp_path):
        from tradingagents.hermes import mcp_server
        from tradingagents.hermes.strategy_store import StrategyStore
        store = StrategyStore(root=tmp_path)
        monkeypatch.setattr(mcp_server, "_strategy_store", lambda: store)
        spec_a = _valid_strategy_spec()
        spec_b = _valid_strategy_spec()
        spec_b["entry"]["all_of"][0]["min_days"] = 5
        store.save(spec_a, _fake_result(gate_passed=True))
        store.save(spec_b, _fake_result(gate_passed=False))
        assert len(mcp_server.list_strategies()) == 2
        assert len(mcp_server.list_strategies(gate_passed=True)) == 1
