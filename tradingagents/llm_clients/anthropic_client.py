import os
import re
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
)

# trading-ai 전용 키 환경변수. 설정되어 있으면 ``ANTHROPIC_API_KEY`` (Hermes
# 가 자체적으로 픽업) 와 분리되어 두 시스템이 별도 한도·모니터링을 가질 수
# 있다. 미설정 시 langchain 의 ``ANTHROPIC_API_KEY`` 자동 lookup 으로 폴백.
_TRADINGAGENTS_KEY_ENV = "TRADINGAGENTS_ANTHROPIC_API_KEY"

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+
# and Sonnet 4.5+ only. Haiku (any version shipped to date) 400s with
# ``"This model does not support the effort parameter"`` (#831). Future
# ``claude-{opus,sonnet}-X-Y`` releases inherit effort support via the
# forward-compat pattern below; future Haiku stays excluded by default.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
}
_EFFORT_PATTERN = re.compile(r"^claude-(opus|sonnet)-\d+-\d+$")


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    return model_lc in _EFFORT_EXACT or bool(_EFFORT_PATTERN.match(model_lc))


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance.

        키 lookup 우선순위:
          1. ``self.kwargs["api_key"]`` (호출자 명시)
          2. ``TRADINGAGENTS_ANTHROPIC_API_KEY`` 환경변수 (trading-ai 전용)
          3. 폴백: langchain 의 ``ANTHROPIC_API_KEY`` 자동 lookup
             (Hermes 와 공유되는 디폴트 키)
        """
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # 호출자가 api_key 명시 안 했고 prefix 환경변수가 있으면 그 키 주입.
        # 명시한 경우 (위 passthrough 루프에서 이미 복사) 는 건드리지 않는다.
        if "api_key" not in llm_kwargs:
            tk = os.environ.get(_TRADINGAGENTS_KEY_ENV)
            if tk:
                llm_kwargs["api_key"] = tk

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
