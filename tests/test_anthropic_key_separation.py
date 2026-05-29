"""Tests for trading-ai vs Hermes ANTHROPIC key separation.

trading-ai picks up ``TRADINGAGENTS_ANTHROPIC_API_KEY`` first; Hermes
keeps using bare ``ANTHROPIC_API_KEY`` via langchain's auto-lookup.
This lets the two systems run on separate keys with independent rate
limits and billing.
"""

import pytest

from tradingagents.llm_clients import anthropic_client as mod


def _capture_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedChatAnthropic",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


@pytest.mark.unit
class TestAnthropicKeySeparation:
    def test_prefix_env_var_used_when_no_explicit_key(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_ANTHROPIC_API_KEY", "ta-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "hermes-key")
        captured = _capture_kwargs(monkeypatch)

        mod.AnthropicClient(model="claude-sonnet-4-6").get_llm()

        assert captured["kwargs"]["api_key"] == "ta-key"

    def test_explicit_api_key_kwarg_wins_over_prefix_env(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_ANTHROPIC_API_KEY", "ta-key")
        captured = _capture_kwargs(monkeypatch)

        mod.AnthropicClient(
            model="claude-sonnet-4-6", api_key="explicit",
        ).get_llm()

        assert captured["kwargs"]["api_key"] == "explicit"

    def test_no_prefix_env_falls_back_to_langchain_lookup(self, monkeypatch):
        # 폴백 = api_key kwarg 가 NormalizedChatAnthropic 에 안 들어감 →
        # langchain 이 ANTHROPIC_API_KEY 자동 lookup. 우리 코드 단에서는
        # api_key 키가 llm_kwargs 에 없는지만 검증.
        monkeypatch.delenv("TRADINGAGENTS_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "hermes-key")
        captured = _capture_kwargs(monkeypatch)

        mod.AnthropicClient(model="claude-sonnet-4-6").get_llm()

        assert "api_key" not in captured["kwargs"]

    def test_empty_prefix_env_falls_back_to_langchain(self, monkeypatch):
        # 빈 문자열은 미설정과 동일 취급.
        monkeypatch.setenv("TRADINGAGENTS_ANTHROPIC_API_KEY", "")
        captured = _capture_kwargs(monkeypatch)

        mod.AnthropicClient(model="claude-sonnet-4-6").get_llm()

        assert "api_key" not in captured["kwargs"]

    def test_prefix_env_does_not_affect_model_or_base_url(self, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_ANTHROPIC_API_KEY", "ta-key")
        captured = _capture_kwargs(monkeypatch)

        mod.AnthropicClient(
            model="claude-sonnet-4-6",
            base_url="https://example.com",
        ).get_llm()

        assert captured["kwargs"]["model"] == "claude-sonnet-4-6"
        assert captured["kwargs"]["base_url"] == "https://example.com"
        assert captured["kwargs"]["api_key"] == "ta-key"
