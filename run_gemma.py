import os
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(PROJECT_DIR, ".runtime")

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["backend_url"] = "http://localhost:11434/v1"
config["deep_think_llm"] = "gemma4:26b-a4b"
config["quick_think_llm"] = "gemma4:26b-a4b"
# analyst_mode 기본값 "auto" — 모델명에 gemma가 있으면 자동으로 prefetch 모드로 전환.
# 다른 모델/API로 갈 때는 "tool_calling"으로 명시하거나 그냥 auto에 맡기면 됨.
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["output_language"] = "Korean"
config["results_dir"] = os.path.join(RUNTIME_DIR, "logs")
config["data_cache_dir"] = os.path.join(RUNTIME_DIR, "cache")
config["memory_log_path"] = os.path.join(RUNTIME_DIR, "memory", "trading_memory.md")

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("005930.KS", "2026-05-22")
print("=== DECISION ===")
print(decision)
