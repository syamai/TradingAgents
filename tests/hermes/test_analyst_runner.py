"""AnalystRunner 단독 호출 wrap 검증.

PRD #1 의 모듈 #1 (AnalystRunner) 테스트. 외부 동작 (LangGraph state 의존성
부재, 한국/비한국 ticker 분기, 노드 출력 키 매핑) 만 단언, 내부 구현 디테일은
단언하지 않는다.

KIS API 외부 의존성은 monkeypatch 로 격리. LLM 은 FakeListChatModel 로 mock.
"""
from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tradingagents.hermes.analyst_runner import (
    list_analysts,
    run_analyst,
)


@pytest.mark.unit
class TestListAnalysts:
    def test_returns_five_analysts_sorted(self):
        names = list_analysts()
        assert names == [
            "fundamentals", "market", "news", "sentiment", "supply_demand",
        ]


@pytest.mark.unit
class TestRunAnalystValidation:
    def test_unknown_analyst_raises_value_error(self):
        fake = FakeListChatModel(responses=["mock"])
        with pytest.raises(ValueError, match="unsupported analyst"):
            run_analyst("005930.KS", "2026-05-29", "unknown", llm=fake)  # type: ignore[arg-type]


@pytest.mark.unit
class TestSupplyDemandNonKoreanTicker:
    def test_non_korean_ticker_returns_not_applicable_without_llm_call(self):
        # 비한국 ticker 면 supply_demand 분석가는 LLM 호출 없이 즉시 N/A 마커 반환.
        # FakeListChatModel 의 responses 가 빈 리스트여도 호출 자체가 안 나므로 통과.
        fake = FakeListChatModel(responses=[])
        report = run_analyst(
            "AAPL", "2026-05-29", "supply_demand", llm=fake,
        )
        assert report.startswith("<not applicable")
        assert "KOSPI/KOSDAQ only" in report


@pytest.mark.unit
class TestSupplyDemandKoreanTicker:
    def test_korean_ticker_with_kis_unavailable_still_invokes_llm(
        self, monkeypatch,
    ):
        # 한국 ticker. KIS API 호출은 monkeypatch 로 unavailable 마커 강제 →
        # 분석가는 빈 데이터 prompt 로 LLM 호출 → mock 응답 반환.
        # 검증 포인트: state 의존성 없이 단독 호출 성공 + LLM chain 도달.
        from tradingagents.dataflows import kis_api

        def _fake_unavailable(*args, **kwargs):
            raise RuntimeError("KIS not configured in test env")

        monkeypatch.setattr(kis_api, "fetch_investor_trend", _fake_unavailable)
        monkeypatch.setattr(kis_api, "fetch_program_trading", _fake_unavailable)
        monkeypatch.setattr(kis_api, "fetch_short_interest", _fake_unavailable)

        fake = FakeListChatModel(responses=["# Mock supply/demand report\n중립"])
        report = run_analyst(
            "005930.KS", "2026-05-29", "supply_demand", llm=fake,
        )
        assert report == "# Mock supply/demand report\n중립"
