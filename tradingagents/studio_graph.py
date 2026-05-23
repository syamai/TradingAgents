"""LangGraph Studio entry point.

Returns a compiled graph that ``langgraph dev`` exposes through its API
server, which Studio (smith.langchain.com/studio) connects to.

The graph itself is built by ``TradingAgentsGraph`` exactly like a normal
CLI run; this module only owns provider configuration so Studio doesn't
require any extra setup.  The defaults mirror ``run_gemma.py`` (local
Ollama with gemma4:26b-a4b); switch them or set ``TRADINGAGENTS_*`` env
vars in ``.env`` to point at a different provider.

Studio supplies its own in-memory checkpointer at the API-server layer,
so the compiled graph here is returned without one.
"""

from __future__ import annotations

import os

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _build_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    # DEFAULT_CONFIG is OpenAI-by-default, so use plain assignment (not
    # setdefault) — these keys always exist and must be overridden to
    # point Studio at the local Ollama server.
    cfg["llm_provider"] = "ollama"
    cfg["backend_url"] = "http://localhost:11434/v1"
    cfg["deep_think_llm"] = "gemma4:26b-a4b"
    cfg["quick_think_llm"] = "gemma4:26b-a4b"
    cfg["max_debate_rounds"] = 1
    cfg["max_risk_discuss_rounds"] = 1
    cfg["output_language"] = "Korean"

    runtime_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".runtime")
    cfg["results_dir"] = os.path.join(runtime_dir, "logs")
    cfg["data_cache_dir"] = os.path.join(runtime_dir, "cache")
    cfg["memory_log_path"] = os.path.join(runtime_dir, "memory", "trading_memory.md")
    return cfg


def make_graph():
    """Factory called once by ``langgraph dev`` on server start."""
    ta = TradingAgentsGraph(config=_build_config())
    return ta.graph
