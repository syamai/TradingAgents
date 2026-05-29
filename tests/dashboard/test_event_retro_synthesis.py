"""event_retro_synthesis.py — LLM 통합 호출 + 룰베이스 fallback 검증."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from dashboard import event_retro_synthesis as ers
from dashboard.event_retro_synthesis import (
    CLUSTER_ENUM, _classify_by_keywords, _parse_llm_json,
    _fallback_classify, _recompute_decomposition,
    synthesize_event_retro, render_synthesis_markdown,
)


# === 룰베이스 키워드 분류 =====================================================
class TestClassifyByKeywords:
    def test_company_specific(self):
        assert _classify_by_keywords("DB손해보험 자사주 매입 발표") == "company_specific"
        assert _classify_by_keywords("배당 결정") == "company_specific"

    def test_earnings_rating(self):
        assert _classify_by_keywords("KB증권 목표가 상향") == "earnings_rating"
        assert _classify_by_keywords("4Q24 영업이익 발표") == "earnings_rating"

    def test_macro_shock(self):
        assert _classify_by_keywords("트럼프 관세 100% 검토") == "macro_shock"
        assert _classify_by_keywords("FOMC 금리 동결") == "macro_shock"

    def test_industry_regulation(self):
        assert _classify_by_keywords("K-ICS 비율 기준 변경") == "industry_regulation"
        assert _classify_by_keywords("손해율 악화") == "industry_regulation"

    def test_no_match_returns_none(self):
        assert _classify_by_keywords("아무 관련 없는 뉴스") is None
        assert _classify_by_keywords("") is None


# === JSON 파싱 ===============================================================
class TestParseLlmJson:
    def test_bare_json(self):
        text = '{"clusters": {"2025-03-27": "company_specific"}}'
        out = _parse_llm_json(text)
        assert out["clusters"]["2025-03-27"] == "company_specific"

    def test_fenced_json(self):
        text = '```json\n{"clusters": {"a": "drift"}}\n```'
        out = _parse_llm_json(text)
        assert out["clusters"]["a"] == "drift"

    def test_extra_text_around(self):
        text = "여기 결과입니다:\n{\"clusters\": {\"x\": \"macro_shock\"}}\n끝."
        out = _parse_llm_json(text)
        assert out["clusters"]["x"] == "macro_shock"

    def test_invalid_returns_none(self):
        assert _parse_llm_json("not json at all") is None
        assert _parse_llm_json("") is None


# === 룰베이스 fallback =======================================================
class TestFallbackClassify:
    def test_assigns_drift_when_no_news(self):
        report = {
            "timeline": [
                {"date": "2025-03-27", "pct": -7.0, "matched_news": []},
            ],
        }
        out = _fallback_classify(report)
        assert out["clusters"]["2025-03-27"] == "drift"

    def test_assigns_cluster_from_news(self):
        report = {
            "timeline": [
                {"date": "2025-03-27", "pct": -7.0,
                 "matched_news": [{"title": "DB손해보험 자사주 소각 결정",
                                   "snippet": ""}]},
                {"date": "2025-04-07", "pct": -4.5,
                 "matched_news": [{"title": "트럼프 관세 100% 검토",
                                   "snippet": ""}]},
            ],
        }
        out = _fallback_classify(report)
        assert out["clusters"]["2025-03-27"] == "company_specific"
        assert out["clusters"]["2025-04-07"] == "macro_shock"
        # synthesis_paragraphs 는 5 enum 모두 placeholder
        assert set(out["synthesis_paragraphs"].keys()) == set(CLUSTER_ENUM)


# === 분해 재계산 ==============================================================
class TestRecomputeDecomposition:
    def test_drift_residual(self):
        # total -10%, 이벤트 합 -8% (-3 + -5) → drift = -2%
        timeline = [
            {"date": "2025-03-27", "pct": -3.0, "cluster": "company_specific"},
            {"date": "2025-04-07", "pct": -5.0, "cluster": "macro_shock"},
        ]
        out = _recompute_decomposition(timeline, total_return_pct=-10.0)
        assert out["company_specific"] == -3.0
        assert out["macro_shock"] == -5.0
        assert out["drift"] == -2.0   # -10 - (-3 + -5)
        # 합계 = total
        assert abs(sum(out.values()) - (-10.0)) < 0.001

    def test_unknown_cluster_treated_as_drift(self):
        timeline = [
            {"date": "2025-03-27", "pct": -3.0, "cluster": "BOGUS"},
        ]
        out = _recompute_decomposition(timeline, total_return_pct=-5.0)
        # BOGUS 는 drift 로 fall through, drift residual = -5 - 0 = -5
        assert out["drift"] == -5.0


# === synthesize_event_retro 흐름 ============================================
class TestSynthesizeEventRetro:
    def _base_report(self):
        return {
            "meta": {"ticker": "005830", "company_name": "DB손해보험",
                     "start": "2025-02-14", "end": "2025-04-14"},
            "period_summary": {"total_return_pct": -10.0, "max_drawdown_pct": -17.5},
            "timeline": [
                {"date": "2025-03-27", "pct": -7.0,
                 "is_price_jump": True, "is_volume_anomaly": False,
                 "is_period_anchor": False,
                 "matched_news": [{"title": "자사주 소각", "snippet": ""}]},
                {"date": "2025-04-07", "pct": -4.5,
                 "is_price_jump": True, "is_volume_anomaly": False,
                 "is_period_anchor": False,
                 "matched_news": [{"title": "트럼프 관세", "snippet": ""}]},
            ],
            "warnings": [], "macro": {},
        }

    def test_llm_success_path(self):
        json_resp = (
            '{"clusters": {"2025-03-27": "company_specific",'
            '"2025-04-07": "macro_shock"},'
            '"synthesis_paragraphs": {'
            '"earnings_rating": "본 기간 해당 카테고리 이벤트 없음.",'
            '"company_specific": "자사주 관련 회사 고유 이슈로 해석됨.",'
            '"industry_regulation": "본 기간 해당 카테고리 이벤트 없음.",'
            '"macro_shock": "관세 정책 충격으로 해석됨.",'
            '"drift": "본 기간 해당 카테고리 이벤트 없음."}}'
        )
        fake_result = MagicMock()
        fake_result.content = json_resp
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = fake_result
        fake_client = MagicMock()
        fake_client.get_llm.return_value = fake_llm

        with patch.object(ers, "create_llm_client", return_value=fake_client):
            # provider=anthropic 으로 single 모드 강제 (mock 1 응답)
            out = synthesize_event_retro(
                self._base_report(), provider="anthropic", mode="single",
            )

        assert out["timeline"][0]["cluster"] == "company_specific"
        assert out["timeline"][1]["cluster"] == "macro_shock"
        assert out["synthesis"]["method"] == "llm_single"
        # 분해 재계산 확인
        assert out["decomposition"]["company_specific"] == -7.0
        assert out["decomposition"]["macro_shock"] == -4.5
        # -10 - (-7 - 4.5) = -10 + 11.5 = 1.5
        assert abs(out["decomposition"]["drift"] - 1.5) < 0.01

    def test_llm_failure_fallback(self):
        with patch.object(ers, "create_llm_client",
                          side_effect=RuntimeError("API key missing")):
            out = synthesize_event_retro(self._base_report())
        assert out["synthesis"]["method"] == "rule_based_fallback"
        # 룰베이스 키워드 매칭 동작 확인
        assert out["timeline"][0]["cluster"] == "company_specific"
        assert out["timeline"][1]["cluster"] == "macro_shock"
        # warnings 누적
        assert any("LLM 호출 실패" in w for w in out["warnings"])

    def test_invalid_json_fallback(self):
        fake_result = MagicMock()
        fake_result.content = "이건 JSON이 아닙니다"
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = fake_result
        fake_client = MagicMock()
        fake_client.get_llm.return_value = fake_llm

        with patch.object(ers, "create_llm_client", return_value=fake_client):
            out = synthesize_event_retro(
                self._base_report(), provider="anthropic", mode="single",
            )
        assert out["synthesis"]["method"] == "rule_based_fallback"
        assert any("LLM JSON 출력 오류" in w for w in out["warnings"])

    def test_empty_timeline_passthrough(self):
        report = {
            "meta": {"ticker": "X"},
            "timeline": [],
        }
        out = synthesize_event_retro(report)
        assert out["timeline"] == []
        # synthesis 키 없음 (조기 return)
        assert "synthesis" not in out


# === render =================================================================
class TestRender:
    def test_renders_5_sections(self):
        report = {
            "synthesis": {
                "method": "llm",
                "paragraphs": {c: f"{c} 단락" for c in CLUSTER_ENUM},
            },
        }
        md = render_synthesis_markdown(report)
        for c in CLUSTER_ENUM:
            assert f"({c})" in md
            assert f"{c} 단락" in md

    def test_fallback_notes_method(self):
        report = {
            "synthesis": {
                "method": "rule_based_fallback",
                "paragraphs": {c: "x" for c in CLUSTER_ENUM},
            },
        }
        md = render_synthesis_markdown(report)
        assert "LLM 합성 실패" in md

    def test_no_synthesis_empty(self):
        assert render_synthesis_markdown({}) == ""
