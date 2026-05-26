"""Unit tests for supply_demand_analyst.

LLM 호출이 들어가는 prefetch 통합은 단계 6 수동 검증에서 다룬다.
여기서는 비한국 게이트, 포맷터, _safe wrapper만 검증.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.analysts import supply_demand_analyst as sda
from tradingagents.dataflows.kis_auth import KisCredentialError


# === 비한국 게이트: LLM 호출 0회로 즉시 N/A ===

@pytest.mark.unit
class TestNonKoreanGate:
    def test_us_ticker_returns_na(self):
        llm = MagicMock()
        node = sda.create_supply_demand_analyst(llm)
        out = node({"company_of_interest": "NVDA", "trade_date": "2026-05-27"})
        assert out["supply_demand_report"].startswith("<not applicable:")
        assert "non-KR ticker" in out["supply_demand_report"]
        assert out["messages"] == []
        llm.invoke.assert_not_called()

    def test_japan_ticker_returns_na(self):
        llm = MagicMock()
        node = sda.create_supply_demand_analyst(llm)
        out = node({"company_of_interest": "7203.T", "trade_date": "2026-05-27"})
        assert out["supply_demand_report"].startswith("<not applicable:")

    def test_korean_kospi_passes_gate(self, monkeypatch):
        """`.KS` ticker should NOT short-circuit — should reach _run_prefetch."""
        called = {"prefetch": 0}
        def fake_prefetch(state, llm):
            called["prefetch"] += 1
            return {"messages": [], "supply_demand_report": "stub"}
        monkeypatch.setattr(sda, "_run_prefetch", fake_prefetch)

        llm = MagicMock()
        node = sda.create_supply_demand_analyst(llm)
        node({"company_of_interest": "005930.KS", "trade_date": "2026-05-27"})
        assert called["prefetch"] == 1

    def test_kosdaq_passes_gate(self, monkeypatch):
        called = {"prefetch": 0}
        monkeypatch.setattr(sda, "_run_prefetch",
                            lambda state, llm: called.__setitem__("prefetch", 1) or
                                                {"messages": [], "supply_demand_report": "x"})
        node = sda.create_supply_demand_analyst(MagicMock())
        node({"company_of_interest": "035720.KQ", "trade_date": "2026-05-27"})
        assert called["prefetch"] == 1

    def test_bare_6digit_passes_gate(self, monkeypatch):
        """6자리 코드만 들어와도 is_korean_ticker True — gate 통과."""
        called = {"prefetch": 0}
        monkeypatch.setattr(sda, "_run_prefetch",
                            lambda state, llm: called.__setitem__("prefetch", 1) or
                                                {"messages": [], "supply_demand_report": "x"})
        node = sda.create_supply_demand_analyst(MagicMock())
        node({"company_of_interest": "005930", "trade_date": "2026-05-27"})
        assert called["prefetch"] == 1


# === _safe wrapper ===

@pytest.mark.unit
class TestSafeWrapper:
    def test_success_passes_through(self):
        assert sda._safe(lambda x: x * 2, 5) == 10

    def test_credential_error_specific_message(self):
        def boom():
            raise KisCredentialError("KIS_APP_KEY missing")
        result = sda._safe(boom)
        assert result.startswith("<unavailable: KIS credentials not configured")

    def test_generic_exception_wrapped(self):
        def boom():
            raise RuntimeError("network down")
        result = sda._safe(boom)
        assert result.startswith("<unavailable:")
        assert "network down" in result

    def test_passes_args_kwargs(self):
        def f(a, b, c=0):
            return a + b + c
        assert sda._safe(f, 1, 2, c=3) == 6


# === block formatters ===

_INVESTOR_SAMPLE = [
    {
        "date": "2026-05-27", "close": 75300,
        "foreign_qty": -123456, "institution_qty": 234567, "retail_qty": -111111,
        "foreign_amount": -9302092800, "institution_amount": 17668994100,
        "retail_amount": -8366901300,
    },
    {
        "date": "2026-05-26", "close": 75100,
        "foreign_qty": 100000, "institution_qty": -50000, "retail_qty": -50000,
        "foreign_amount": 7510000000, "institution_amount": -3755000000,
        "retail_amount": -3755000000,
    },
]

_PROGRAM_SAMPLE = [
    {"date": "2026-05-27", "close": 75300, "net_qty": 150000, "net_amount": 11295000000},
    {"date": "2026-05-26", "close": 75100, "net_qty": -80000, "net_amount": -6008000000},
]

_SHORT_SAMPLE = [
    {
        "date": "2026-05-27", "close": 75300,
        "short_qty": 55555, "short_volume_ratio": 3.21,
        "short_amount": 4181626500, "short_amount_ratio": 3.05,
    },
]


@pytest.mark.unit
class TestFormatInvestor:
    def test_rows_render(self):
        block = sda._format_investor(_INVESTOR_SAMPLE)
        assert "(2 rows" in block
        assert "2026-05-27" in block
        assert "+234,567" in block  # institution_qty positive sign
        assert "-123,456" in block  # foreign_qty negative
        assert "75,300" in block    # close

    def test_unavailable_string_passthrough(self):
        assert sda._format_investor("<unavailable: x>") == "<unavailable: x>"

    def test_empty_list_returns_no_rows_marker(self):
        block = sda._format_investor([])
        assert "<no investor-trend rows" in block

    def test_signed_zero_renders(self):
        rows = [{
            "date": "2026-05-27", "close": 100,
            "foreign_qty": 0, "institution_qty": 0, "retail_qty": 0,
            "foreign_amount": 0, "institution_amount": 0, "retail_amount": 0,
        }]
        block = sda._format_investor(rows)
        assert "+0" in block  # signed format


@pytest.mark.unit
class TestFormatProgram:
    def test_rows_render(self):
        block = sda._format_program(_PROGRAM_SAMPLE)
        assert "(2 rows" in block
        assert "+150,000" in block
        assert "-80,000" in block

    def test_unavailable_passthrough(self):
        assert sda._format_program("<unavailable: y>") == "<unavailable: y>"

    def test_empty_marker(self):
        assert sda._format_program([]).startswith("<no program-trading")


@pytest.mark.unit
class TestFormatShort:
    def test_rows_render(self):
        block = sda._format_short(_SHORT_SAMPLE)
        assert "(1 rows" in block
        assert "3.21" in block  # vol_ratio_pct
        assert "55,555" in block

    def test_unavailable_passthrough(self):
        assert sda._format_short("<unavailable: z>") == "<unavailable: z>"

    def test_empty_marker(self):
        assert sda._format_short([]).startswith("<no short-interest")


# === system message assembly ===

@pytest.mark.unit
class TestSystemMessage:
    def test_includes_all_three_blocks(self):
        msg = sda._build_system_message(
            ticker="005930.KS",
            start_date="2026-05-20", end_date="2026-05-27",
            investor_block="INVESTOR_BLOCK_X",
            program_block="PROGRAM_BLOCK_Y",
            short_block="SHORT_BLOCK_Z",
        )
        assert "INVESTOR_BLOCK_X" in msg
        assert "PROGRAM_BLOCK_Y" in msg
        assert "SHORT_BLOCK_Z" in msg
        assert "<start_of_investor_trend>" in msg
        assert "<end_of_investor_trend>" in msg
        assert "<start_of_program_trading>" in msg
        assert "<start_of_short_interest>" in msg

    def test_includes_ticker_and_window(self):
        msg = sda._build_system_message(
            ticker="035720.KQ",
            start_date="2026-05-20", end_date="2026-05-27",
            investor_block="a", program_block="b", short_block="c",
        )
        assert "035720.KQ" in msg
        assert "2026-05-20" in msg
        assert "2026-05-27" in msg

    def test_unavailable_marker_propagates(self):
        """A fetcher failure marker should appear verbatim — the LLM is told
        in the prompt to flag <unavailable> blocks. The downstream guard in
        bull/bear is on supply_demand_report level (the final LLM output),
        not on the individual block. Here we only verify block placement."""
        msg = sda._build_system_message(
            ticker="005930.KS",
            start_date="2026-05-20", end_date="2026-05-27",
            investor_block="<unavailable: KIS down>",
            program_block="ok",
            short_block="ok",
        )
        assert "<unavailable: KIS down>" in msg
