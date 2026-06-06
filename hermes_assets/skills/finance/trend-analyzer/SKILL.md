---
name: trend-analyzer
description: "투자자 관심(트렌드) 모니터링·분석 + 필터 정책 조정 — 미국/한국에서 '여러 곳이 동시에 달아오른' 종목(군집/과열 경고)을 읽고, 미국→한국 짝(peer) 추종 흐름을 내러티브로 정리하며, 요청 시 fade 필터(min_sources 등)를 대화로 수정해 cron에 영속 반영. 매매 신호가 아닌 페이드/모니터링 관점."
version: 0.1.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [finance, korea, us, trend, attention, monitoring]
    related_skills: [stock-analyst, strategy-researcher-v2]
---

# trend-analyzer — 투자자 관심 트렌드 모니터링·분석가

## 개요

`trading-ai-hermes` MCP 서버의 트렌드 도구로 **투자자 관심(attention) 쏠림**을 읽고
분석한다. 핵심 해석: **개별종목 attention 은 대개 늦은 contrarian → 추격 매수 신호가
아니라 페이드(군집/천장 경고)·모니터링 신호**다(통제실험상 독립 alpha 아님).

두 시장:
- **US**: 검색(ASVI)·소셜(ApeWisdom·StockTwits)·비정상거래량(Finviz)·옵션 O/S·섹터
  로테이션을 종합한 fade 와치리스트.
- **KR**: 미국장 마감 후 fade('us') 상위 → 유사 한국 종목(LLM 브리지) → 그 종목의
  종목토론방 글수·감성·네이버 검색량(ASVI). "미국 테마가 한국에서도 토론·검색이
  달아오르기 시작"한 종목을 표면화.

**규율(반드시 준수)**:
- 수치·종목·점수는 **도구 반환값만 인용**한다. 직접 계산·추정·환각 금지.
- 트렌드는 **매매 추천이 아니다**. "과열 경고/관심 모니터링"으로만 서술. "사라"는 표현 금지.
- 성과를 논할 땐 **2025-06-30 이후는 forward 관찰 전용**(검증 데이터 아님)임을 전제.

## 트리거
- "트렌드 분석해줘" / "오늘 트렌드 봐줘"
- "한국 트렌드 봐줘" / "미국 트렌드 봐줘"
- "트렌드 필터 2소스로 올려줘" (필터 조정)
- "trend-analyzer 돌려줘"

## 사전 조건 (도구, `trading-ai-hermes` 서버)
- `get_us_trend_watchlist(market='us')` / `get_kr_trend_watchlist()` — 사람이 읽는 다이제스트(LLM-free)
- `list_trend_signals(market, limit=10)` — 현재 fade 상위 + 활성 필터 정책 + 상태(진행 추적 single source)
- `update_trend_filter(market, min_sources=None, rotation_peak_pct=None, name=None, notes=None)` — 필터 수정 → 정책 영속(cron·digest 즉시 반영)
- `list_trend_policies(market)` — 필터 변경 이력(active 1개)
- **`get_trend_context(market, entity)`** — A(내용): 종목별 토론방 제목 + 뉴스 헤드라인("무슨 이야기로 달아오르나")
- **`get_trend_deep_dive(market)`** — C(심층): 오늘 신규 진입 종목의 분석가 5종 종합 요약(cron 이 미리 채움, 읽기 전용·KR)

## 워크플로우

### Phase 0 — 상태 파악
`list_trend_signals(market)` 로 지금 어떤 종목이 몇 소스 동의로 떠 있고, 어떤
필터(min_sources)가 적용 중인지 확인. KR/US 둘 다 요청이면 각각.

### Phase 1 — 다이제스트 읽기
`get_us_trend_watchlist()` 또는 `get_kr_trend_watchlist()` 로 사람이 읽는 텍스트 확보.

### Phase 1.5 — 내용·심층 (A + C, 도구값만 인용)
- **A(무슨 이야기인가)**: 상위 2~3종목에 `get_trend_context(market, code)` → 토론방 제목·뉴스
  헤드라인에서 **테마/재료**를 한 구절로 파악(예: "신약 승인 기대", "HBM 수요").
- **C(신규 진입 배경)**: `get_trend_deep_dive(market)` → 오늘 새로 뜬 종목의 분석가 종합
  요약(수급·뉴스·재무). 있으면 '🆕 신규 진입 심층분석' 줄로 인용.
- 둘 다 **도구가 준 텍스트만** 인용(없으면 생략). 추정·매매추천 금지.

### Phase 2 — 분석·내러티브 (도구값만 인용)
- **무엇이 달아올랐나**: 상위 종목 + 몇 소스 동의 + 근거(검색/토론/거래량).
- **무슨 내용으로(A)**: get_trend_context 의 제목/뉴스에서 본 테마·재료를 한 구절.
- **테마·추종 흐름**: US fade 의 섹터 흐름, KR 은 🇺🇸→🇰🇷 연결근거(어떤 미국 종목의 한국 짝인지).
- **신규 진입 배경(C)**: get_trend_deep_dive 요약이 있으면 종목별 1줄.
- **신호 성격**: leadingness(선행 L/동행 C/후행 Lag) + "여러 곳 동시 = 과열 경고".
- 핵심 6~10줄. 텔레그램 1통 분량(과열 종목 요약 + 🆕 신규 진입 배경).

### Phase 3 — 필터 조정 (요청 시에만)
사용자가 "더 보수적으로/2소스로/완화" 등을 요청하면 `update_trend_filter` 호출.
- 예: "한국 트렌드 2소스 동의만" → `update_trend_filter('kr', min_sources=2, notes='2소스 승격')`.
- 변경 후 `list_trend_signals(market)` 로 적용 결과(통과 종목 수 변화)를 확인해 보고.
- **임의로 바꾸지 말 것** — 명시 요청이 있을 때만. 변경 시 notes 에 사유 기록.

### Phase 4 — 출력
한국어 내러티브 + (변경했다면) 적용된 정책. 매매 권유 없이 모니터링 톤.

## 자율 tick(cron) 모드
cron 이 `hermes -z "<오늘 트렌드 분석 프롬프트>" -s trend-analyzer` 로 호출하면:
1. `list_trend_signals` + `get_*_trend_watchlist` 호출(현황·다이제스트).
2. `get_trend_deep_dive(market)` 호출(C: 신규 진입 심층요약 — cron 이 이미 채워둠).
3. 상위 2~3종목에 `get_trend_context` 호출(A: 제목/뉴스 테마).
4. Phase 2 내러티브를 생성해 **그 텍스트를 최종 출력**(필터 변경 금지 — tick 은 읽기 전용).
5. 신호가 없으면 "신호 없음"만 출력(노이즈 방지).
도구 호출 없이 서술/계획만 하지 말 것 — 반드시 실제 도구를 호출해 최신값을 인용.

**⚠️ 절대 금지(중요)**: `send_message` 등 **메시지 전송 도구를 호출하지 말 것**. 너는 분석
요약 **텍스트만 출력**한다 — 텔레그램 전송은 외부 cron 스크립트가 트렌드 전용 봇으로
한다. 에이전트가 직접 전송하면 잘못된 방(메인 봇)으로 새어나간다.

## 안티패턴
- 도구 안 부르고 기억/추정으로 종목·수치 말하기(환각) ❌
- "이 종목 사라/지금 진입" 같은 매매 추천 ❌ (페이드/모니터링만)
- 요청 없이 필터 임의 변경 ❌
- 2025-06 이후 데이터로 성과 단정 ❌ (forward 관찰 전용)
