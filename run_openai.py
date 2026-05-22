"""
OpenAI GPT 모델로 TradingAgents 실행.
결과물은 .runtime/openai/ 하위에 분리 저장된다 (로컬 모델 결과와 충돌 없음).
"""
import os
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(PROJECT_DIR, ".runtime", "openai")

# .env 직접 로드 — 프레임워크는 os.environ만 읽으므로 명시적 로드 필요.
env_file = Path(PROJECT_DIR) / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5.5"        # 최종 종합/판단 — 최신 프론티어, 1M context
config["quick_think_llm"] = "gpt-5.4-mini"  # 분석가/토론 — 빠르고 tool calling 강함
config["backend_url"] = None                # OpenAI 기본 엔드포인트 (api.openai.com/v1)
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["output_language"] = "Korean"

# analyst_mode "auto" → OpenAI는 gemma 패턴이 아니므로 자동으로 tool_calling 사용.
# (옵션 E 분기 검증의 일환)
config["analyst_mode"] = "auto"

config["results_dir"] = os.path.join(RUNTIME_DIR, "logs")
config["data_cache_dir"] = os.path.join(RUNTIME_DIR, "cache")
config["memory_log_path"] = os.path.join(RUNTIME_DIR, "memory", "trading_memory.md")

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("005930.KS", "2026-05-22")
print("=== DECISION ===")
print(decision)
