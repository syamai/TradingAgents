# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fork of **TauricResearch/TradingAgents** (multi-agent LLM trading framework on LangGraph) with three independent extensions layered on top:

1. **Korean-market supply/demand pipeline** — KIS (Korea Investment & Securities) investor/program/short tables, DART filings, Bank-of-Korea macro, Naver/Alpha-Vantage feeds.
2. **Streamlit dashboard** (`dashboard/`) — multi-ticker holdings/correlation/trend/advanced analysis + LLM synthesis.
3. **Hermes Agent integration** (`tradingagents/hermes/` + `hermes_assets/`) — Phase 1 of issue #1. Wraps the 5 LangGraph analysts as MCP tools, persists hypotheses to SQLite, labels them post-hoc against KOSPI relative return. The LangGraph trading flow (NVDA-style US-equity main.py path) is **never modified by Hermes work — wrap only**.

The two pipelines share `tradingagents/agents/analysts/*_analyst.py`, `tradingagents/dataflows/`, and `tradingagents/llm_clients/`.

## Commands

Package manager: **`uv`** (lockfile `uv.lock`). All commands below assume the repo root as CWD.

### Run

```bash
# LangGraph trading flow (US-style, single ticker)
uv run python main.py                       # NVDA example baked into main.py
tradingagents                               # interactive CLI (installed via project.scripts)
tradingagents analyze --checkpoint          # resume via LangGraph checkpoints
tradingagents analyze --clear-checkpoints   # reset before run

# Streamlit dashboard (Korean tickers)
uv run streamlit run dashboard/app.py       # default http://localhost:8501

# Hermes MCP server (stdio, invoked by Hermes — not run manually)
uv run python -m tradingagents.hermes.mcp_server

# Hermes daily labeling cron (LLM-free)
uv run python -m tradingagents.hermes.labeling_cron --dry-run
uv run python -m tradingagents.hermes.labeling_cron --today 2026-06-12
```

### Test

```bash
uv run pytest                               # full suite
uv run pytest tests/hermes/                 # Hermes integration tests only
uv run pytest tests/dashboard/              # dashboard tests
uv run pytest -k test_supply_demand_analyst # by name match
uv run pytest -m unit                       # by marker (unit / integration / smoke)
uv run pytest tests/hermes/test_cache.py::TestSetGetRoundTrip -v   # single class
```

Markers (`pyproject.toml`): `unit`, `integration`, `smoke`. `--strict-markers` is on, so undeclared markers fail.

### Hermes invocations (one-shot CLI)

```bash
# Sanity (requires ANTHROPIC_API_KEY in shell env)
set -a; source .env; set +a
hermes -z "PING" --yolo

# End-to-end stock analysis (saves hypotheses to ~/.tradingagents/hermes/hypotheses.db)
hermes -z "삼성전자 005930.KS 봐줘" -s stock-analyst --yolo

# Diagnose silent failure: hermes -z exits 0 with no stdout on API errors.
# Check ~/.hermes/sessions/request_dump_*.json's `error` field, or:
sqlite3 ~/.hermes/state.db "SELECT id, message_count, output_tokens FROM sessions ORDER BY started_at DESC LIMIT 3;"
# message_count=1 + output_tokens=0 → API call never landed (rate limit, auth, etc.)
```

## Architecture

### Two LLM-orchestration surfaces that coexist

| Layer | Purpose | LLM provider (default) | Entry |
|---|---|---|---|
| **LangGraph TradingAgentsGraph** | US-style flow: analysts → researchers (bull/bear debate) → trader → risk manager → portfolio manager. Returns BUY/HOLD/SELL. | OpenAI/Anthropic/Google/etc. via `tradingagents/llm_clients/` factory. | `tradingagents/graph/trading_graph.py`, `main.py`, `cli/main.py`. |
| **Hermes stock-analyst skill** | Korean stocks, swing horizon (2–8 weeks). Hermes calls trading-ai's MCP tools, applies 27 domain seeds, emits a hypotheses JSON, persists it, then collects user feedback + auto KOSPI-relative labels later. | Hermes itself = Claude Sonnet 4.6. Underlying *analysts* still run on local Ollama gemma (cost 0). | `tradingagents/hermes/mcp_server.py` (stdio), Hermes invokes it. |

The 5 analyst factories (`create_*_analyst(llm)`) live in `tradingagents/agents/analysts/`. They return LangGraph-state-driven closures. **Both surfaces call the same factories** — Hermes runs them without a full LangGraph by stuffing a minimal state dict (`{company_of_interest, trade_date, messages: []}`) into `tradingagents/hermes/analyst_runner.py`.

### Korean-market data flow

```
KIS API ─► kis_api.py ──► kis_history_store.py ──► holdings table
                          (Parquet + SQLite, 5y+)        │
                                                         ▼
                                              dashboard/{holdings_chart,
                                                         correlation_analysis,
                                                         trend_analysis,
                                                         advanced_analysis}.py
                                                         │
                                                         ▼
                                            Streamlit UI  OR  Hermes MCP wrap
                                                              (tradingagents/hermes/
                                                               statistics_tools.py)
```

`kis_history_store.py` is the single source of truth for raw + derived Korean holdings. It normalizes `005930.KS` / `005930.KQ` / `005930` to a 6-digit key. Raw `investor` writes trigger automatic re-materialization of derived `holdings`.

### Hermes Phase 1 module map (8 modules)

| Module | File | Role |
|---|---|---|
| AnalystRunner | `hermes/analyst_runner.py` | Call one of 5 analysts without a LangGraph instance |
| AnalystOutputCache | `hermes/cache.py` | SQLite, composite PK `(ticker, date, analyst, model_id, prompt_version)` |
| HypothesisStore | `hermes/hypothesis_store.py` | JSON-schema-validated hypothesis records + labels (4 kinds: `user_immediate`/`user_followup`/`auto_relative`/`auto_absolute`) |
| LabelingScheduler | `hermes/labeling.py` + `labeling_cron.py` | LLM-free post-hoc labeling via `fetch_kospi`. Thresholds: 2w ±2%, 4w ±3%, 8w ±5% (linear interpolation for 3/5/6/7w). Holiday advance via `_close_at_or_after`. |
| MCP server | `hermes/mcp_server.py` | FastMCP with 13 tools (5 analysts + 4 stats + 4 hypothesis/label) |
| Seed memory | `hermes_assets/memories/MEMORY.md` | 27 domain seeds, single file split by `§` |
| Skill | `hermes_assets/skills/finance/stock-analyst/SKILL.md` | v0.3.0 workflow: 4-tool standard callset → JSON → `save_analysis` → feedback commands |
| Feedback command | `mcp_server.add_feedback` | Auto-routes to `user_immediate` vs `user_followup` by 14-day threshold from `as_of_date` |

### Persistent state locations

- `~/.tradingagents/cache/` — analyst response cache (short-window, 7d). Override with `TRADINGAGENTS_CACHE_DIR`.
- `~/.tradingagents/kis_history/` — KIS history store (5y+, Parquet + SQLite).
- `~/.tradingagents/hermes/cache.db` — AnalystOutputCache (composite PK).
- `~/.tradingagents/hermes/hypotheses.db` — HypothesisStore (hypothesis_records + hypothesis_labels).
- `~/.tradingagents/memory/trading_memory.md` — LangGraph decision log (Trading flow). Override with `TRADINGAGENTS_MEMORY_LOG_PATH`.
- `~/.tradingagents/cache/checkpoints/<TICKER>.db` — LangGraph resume checkpoints (opt-in).
- `~/.hermes/` — Hermes agent home (memories, skills, sessions, state.db). Repo-managed mirrors live in `hermes_assets/`.

### Configuration knobs

- `TRADINGAGENTS_LLM_PROVIDER` / `TRADINGAGENTS_DEEP_THINK_LLM` / `TRADINGAGENTS_QUICK_THINK_LLM` / `TRADINGAGENTS_LLM_BACKEND_URL` — `default_config.py` reads these in-place so model choice is `.env`-driven.
- `HERMES_ANALYST_PROVIDER` / `HERMES_ANALYST_MODEL` / `HERMES_ANALYST_BASE_URL` — override the Ollama gemma default for analysts invoked via the MCP server.
- `OLLAMA_BASE_URL` — point LangGraph flow at a remote Ollama.
- API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `KIS_APP_KEY`, `KIS_APP_SECRET`, `DART_API_KEY`, `ALPHA_VANTAGE_API_KEY`, …) — `.env` is auto-loaded by `tradingagents/__init__.py`.

### ANTHROPIC key separation (trading-ai vs Hermes)

To keep rate limits and billing independent for the two systems:

| Env var | Used by | Resolution |
|---|---|---|
| `TRADINGAGENTS_ANTHROPIC_API_KEY` | trading-ai (LangGraph flow, dashboard LLM synthesis, search_aggregator) | `anthropic_client.AnthropicClient.get_llm` injects this into `ChatAnthropic(api_key=…)` explicitly. |
| `ANTHROPIC_API_KEY` | Hermes (hypothesis synthesis, `/feedback`, `/labels`, memory evolution) — and the trading-ai fallback when the prefix variable is unset. | langchain's auto-lookup; Hermes also reads it from env or its own `auth.json`. |

Lookup order in `anthropic_client.py`: explicit `api_key` kwarg → `TRADINGAGENTS_ANTHROPIC_API_KEY` → langchain's `ANTHROPIC_API_KEY` fallback. See `tests/test_anthropic_key_separation.py` for the contract.

## Conventions

- **Default language: Korean.** New code comments, docstrings, error messages, commit subjects, and PR descriptions are written in Korean. Symbols (function/class names) stay English. Recent commit history shows the canonical `feat(scope): 한국어 본문` shape.
- **Korean tickers use the `.KS` (KOSPI) / `.KQ` (KOSDAQ) yfinance suffix** at the API boundary; `kis_history_store._norm_ticker` strips them to 6-digit for storage.
- **Hermes work must not modify the 5 analyst files or the LangGraph TradingAgentsGraph.** Wrap, don't edit. Hermes Phase 1 PRD locks `tradingagents/agents/analysts/*.py` and `tradingagents/graph/*.py` as out-of-scope. The Streamlit dashboard is also out-of-scope for Hermes changes.
- **Don't add an automated sync from `~/.hermes/{memories,skills}/` back to `hermes_assets/`.** Phase 2 introduces a human approval gate for Hermes's self-evolving memory; auto-sync would defeat it. Run the `cp` manually when promoting a vetted change.
- **`statsmodels` (advanced_analysis, Granger, VAR) hates `inf`/NaN.** Pre-IPO `close=0` rows produce `±inf` via `pct_change`. Filter at entry: `df[df["close"] > 0].replace([np.inf, -np.inf], np.nan).dropna(subset=["price_change_pct"])`.
- **`hermes -z` returns exit 0 on API errors** (rate limit, expired key). Never trust the exit code alone — check `~/.hermes/sessions/request_dump_*.json`'s `error` field or `state.db` `sessions.output_tokens`.
- **2025-06 이후 데이터는 테스트(OOS/검증) 데이터로 쓰지 않는다.** 2025-06 이후 약 1년간 한국 증시가 비정상적으로 급등해(생존자 풀 동일가중 누적 ~18.8배 vs KOSPI ~4.3배의 상당분이 이 구간), 이 구간을 OOS/검증에 넣으면 성과가 구조적으로 과대평가된다. walk-forward·time-split 의 **OOS 윈도우 상한을 2025-06-30** 으로 두고, 그 이후 데이터는 채점에서 제외해 **forward 관찰 전용**으로 남긴다. (walk-forward 에선 최신 데이터가 OOS 이므로, 데이터 그리드를 2025-06-30 에서 잘라 채점하면 충족. in-sample 도 이 상한 이전만 사용.)
- **전문용어·약어는 사용자 응답에서 처음 쓸 때 반드시 풀어서 설명한다.** 백테스트/평가 지표를 약어만 던지지 말 것 (예: "ir_med 1.63" → "ir_med(walk-forward OOS 창들의 시장초과 IR 중앙값) 1.63"). 자주 쓰는 용어 풀이:
  - **IR (Information Ratio, 정보비율)** = 초과수익 평균 ÷ 초과수익 변동성(연율화). 본 프로젝트에선 KOSPI 대비 초과수익 기준.
  - **ir_med / ir_min** = walk-forward 여러 OOS 창의 시장초과 IR의 **중앙값 / 최솟값**.
  - **walk-forward** = 학습구간(IS)→검증구간(OOS) 창을 시간순으로 굴리며 반복 검증(레짐 운 제거).
  - **IS / OOS** = In-Sample(학습 구간) / Out-Of-Sample(표본 외 = 미래 검증 구간).
  - **xsec / time split** = 종목 해시 분할(횡단면 일반화) / 시간 분할(시기 일반화).
  - **MDD** = Maximum Drawdown(최대 낙폭, %). **win_rate** = 수익 거래 비율. **sharpe** = 위험조정 수익(수익/변동성).
  - **fair gate(공정 게이트)** = 채택 합격선. 모든 OOS 창 시장초과 IR>0 **AND** 중앙 IR>0.5(`GATE_EXCESS_MIN_IR`).

## Issue tracker

Active PRD: **syamai/trading-agents-ai issue #1** (Hermes integration Phase 1). Use `gh --repo syamai/trading-agents-ai` — the bare `gh` defaults to the upstream `TauricResearch/TradingAgents` (different repo) because that's where the fork's `origin` doesn't carry the issue.
