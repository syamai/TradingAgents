# hermes_assets

Hermes Agent 통합용 자산 — repo 안에 버전 관리, 사용 시 `~/.hermes` 와
`~/.local/bin` 으로 복사·등록한다. Phase 1 PRD (이슈 #1) 구현 산물.

## 구성

| 경로 | 배포 위치 | 용도 |
|---|---|---|
| `memories/MEMORY.md` | `~/.hermes/memories/MEMORY.md` | 27 시드 도메인 룰 |
| `skills/finance/stock-analyst/SKILL.md` | `~/.hermes/skills/finance/stock-analyst/SKILL.md` | 한국 종목 스윙 분석 워크플로우 v0.3.0 |
| `bin/trading-ai-mcp` | `~/.local/bin/trading-ai-mcp` | MCP stdio 서버 entry wrapper |

## 신규 머신 부팅 시 배포

```bash
# 1. 자산 복사
mkdir -p ~/.hermes/memories ~/.hermes/skills/finance/stock-analyst ~/.local/bin
cp hermes_assets/memories/MEMORY.md ~/.hermes/memories/
cp hermes_assets/skills/finance/stock-analyst/SKILL.md ~/.hermes/skills/finance/stock-analyst/
cp hermes_assets/bin/trading-ai-mcp ~/.local/bin/
chmod +x ~/.local/bin/trading-ai-mcp

# 2. MCP 서버 등록 (Hermes 설정에 자동 추가)
hermes mcp add trading-ai-hermes --command trading-ai-mcp
# 프롬프트: "Enable all 13 tools?" → Y

# 3. 라벨링 크론 등록 (옵션 — 일별 자동 KOSPI 상대 라벨링)
crontab -e
# 추가:
# 0 9 * * * cd ~/Source/trading-ai && uv run python -m tradingagents.hermes.labeling_cron >> ~/.tradingagents/hermes/cron.log 2>&1

# 4. 환경 변수 (분석가 LLM — 디폴트 Ollama gemma4)
# 별도 설정 불요. 다른 모델 쓰려면:
# export HERMES_ANALYST_PROVIDER=anthropic
# export HERMES_ANALYST_MODEL=claude-sonnet-4-6
```

## 배포 검증

```bash
# Sanity ping
hermes -z "ping" --yolo

# 가설 분석 e2e
hermes -z "삼성전자 005930.KS 봐줘" -s stock-analyst --yolo
```

## 자산 갱신 흐름

`~/.hermes` 의 원본을 직접 편집한 뒤 다시 repo 로 동기화:

```bash
cp ~/.hermes/memories/MEMORY.md hermes_assets/memories/
cp ~/.hermes/skills/finance/stock-analyst/SKILL.md hermes_assets/skills/finance/stock-analyst/
```

자동화는 의도적으로 두지 않음 — Hermes 의 자기 메모리 진화 (Phase 2 변경
제안 게이트) 와 충돌 방지.
