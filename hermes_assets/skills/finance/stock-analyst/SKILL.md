---
name: stock-analyst
description: "한국 종목 스윙(2~8주) 분석 — KIS 수급 + 추세/통계 + 가설 영구 저장 + 사용자 피드백 라벨링"
version: 0.3.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [finance, korea, swing, kospi, supply-demand]
    related_skills: []
---

# stock-analyst — 한국 종목 스윙 분석가

## 개요

이 스킬은 trading-ai MCP 서버 (`trading-ai-hermes`) 를 통해 한국 종목의 KIS 수급 데이터 + 추세 phase 분석 + 정교 통계 (Granger/VAR/cointegration) 을 호출하고, MEMORY.md 의 27 시드 도메인 룰을 적용해 **구조화된 가설 JSON** 을 생성한다.

호라이즌은 **스윙 (2~8주)** 전용. 단기 (분봉/일중) · 장기 (분기 이상) 는 다른 도구가 필요.

## 트리거

사용자가 한국 종목에 대해 다음과 같이 요청할 때:
- "삼성전자 봐줘"
- "005930.KS 분석"
- "SK하이닉스 어떻게 보여?"
- "외국인이 무슨 매매 중이야?"

## 사전 조건

- MCP 서버 `trading-ai-hermes` 가 등록·연결됨 (`hermes mcp list` 로 확인)
- 9 도구 노출:
  - **분석가 (5)**: `analyst_supply_demand`, `analyst_sentiment`, `analyst_market`, `analyst_news`, `analyst_fundamentals` (한국 미적용)
  - **통계 (4)**: `compute_correlation`, `compute_trend`, `compute_advanced`, `get_holdings_window`
- 종목 코드 형식: `<6자리>.KS` (KOSPI) 또는 `<6자리>.KQ` (KOSDAQ)

## 워크플로우

### Phase 1 — 종목·날짜 결정

1. 사용자 요청에서 ticker 추출
   - 회사명만 받으면 코드 매핑 (예: 삼성전자 → `005930.KS`, SK하이닉스 → `000660.KS`)
   - 모호하면 사용자에게 확인 요청
2. 분석 기준일 = 오늘 (또는 사용자가 명시한 날짜)
3. 통계 분석 기간 = 직전 5년 (또는 사용자가 명시한 범위)

### Phase 2 — 데이터 도구 호출 (스윙 분석 표준 콜셋)

표준 4 도구를 순차 호출 (필수):

```
1. analyst_supply_demand(ticker, date=오늘) → KIS 수급 마크다운
2. compute_correlation(ticker)              → 6 섹션 상관 (concurrent/lag/level)
3. compute_trend(ticker)                    → phase 분할 + 9 주체 동행성 랭킹
4. compute_advanced(ticker)                 → Granger/VAR/cointegration
```

선택 호출 (필요 시):
- `get_holdings_window(ticker, days=N)`: 최근 N 거래일 raw 시계열 — *마지막 수치 정확 인용* 이 필요할 때만 (토큰 비용 큼)
- `analyst_sentiment(ticker, date)`: 종목이 retail 토픽일 때 (개인 매수세 동향이 가설에 관여할 때)
- `analyst_market(ticker, date)`: 기술 지표 (MA/RSI/MACD) 를 phase 신호와 교차 검증할 때
- `analyst_news(ticker, date)`: 종목·거시 뉴스가 단기 가설에 영향 줄 때
- `analyst_fundamentals(ticker, date)`: ⚠️ 한국 종목 미적용 — 출력이 "결손" 가능, 가설 만들지 말 것

도구 출력에서 raw 수치를 추출해 가설의 `evidence_excerpts` 로 사용한다. 인용할 수 없으면 그 가설은 만들지 말 것.

### Phase 3 — 도메인 룰 적용

MEMORY.md 의 27 시드 (7 카테고리) 를 모든 가설 생성에 적용:

1. **수급-actors** (4) — 외국인 등록/비등록 분리, 기관 5 sub-주체, 개인 단독 약함, 자기주주 의지
2. **수급-structure** (3) — 시총별 비중 차등, KIS 9 주체 세분화, 시간외 시차
3. **수급-patterns** (4) — 사모+외국인비등록 단기 모멘텀, 등록자 단발 vs 지속, 흡수자, 단주 vs 거래대금
4. **수급-dynamics** (4) — 60일 phase, 동행성 ≥70% / ≤30% 임계, 모멘텀 격차, phase 전환 false breakout
5. **통계-한계** (4) — 인과≠상관, Granger 선후행, level r spurious, n_days·p-value
6. **데이터-제약** (4) — KIS unavailable 결손, 신규상장 close=0, IPO 6개월, 휴장일·KIS 9 주체 한계
7. **출력-규칙** (4) — 매매 추천 금지, raw 인용 의무, JSON schema, KOSPI 상대 호라이즌 임계값

특히 다음 5 가지는 *위반 시 가설 무효*:
- 매매 추천 단어 금지 (사야 한다, 오를 것이다)
- evidence_excerpts 비어 있는 가설 금지 (raw 인용 의무)
- horizon_weeks 범위 2~8 외 금지
- "인과" "유발" 같은 단어 금지 ("동조" "선행 패턴" 사용)
- predicted_relative_return_pct 임계값 (2주 ±2%, 4주 ±3%, 8주 ±5%) 무관한 값 금지

### Phase 4 — 가설 생성 (3~5 개)

각 가설:
- **claim**: 한 줄 한국어 주장 (예: "외국인 등록자 매집 가속 → 스윙 강세 모멘텀 형성")
- **direction**: bullish / bearish / neutral
- **confidence**: 0.0~1.0 (raw 수치 강하면 ↑, 결손 많으면 ↓)
- **evidence_tools**: 사용한 도구 이름 배열
- **evidence_excerpts**: 도구 출력에서 인용한 raw 수치 배열 (1+ 개 의무)
- **horizon_weeks**: 2~8 정수
- **predicted_relative_return_pct**: KOSPI 대비 % (호라이즌별 임계값 참조)

가설 간 *서로 다른 측면* 을 다루도록 (예: 외국인 / 기관 / 공매도 / 추세 등).

### Phase 5 — 응답 출력

응답은 **두 부분**:

1. **사람이 읽는 한국어 요약** (2~3 단락) — 종합 stance + 핵심 근거 + 한계
2. **구조화 JSON** (코드 블록) — MEMORY.md 의 출력 스키마 그대로

JSON 은 **반드시 코드 블록 안에**. 자유 텍스트만 출력은 금지 — 라벨링 인프라가 JSON 을 파싱 못 함.

### Phase 6 — 가설 영구 저장 (자동)

응답 JSON 출력 *직후* 반드시 `save_analysis(record)` 를 한 번 호출한다. `record` 인자는 위 Phase 5 의 JSON 본문 그대로 (코드블록 마크다운 빼고 dict 만).

반환값에 `hypothesis_ids` (DB 정수 ID 리스트) 와 `h_local_id_map` (예: `{"h1": 17, "h2": 18, ...}`) 이 들어온다. 이 매핑을 응답 마지막에 1 줄로 사용자에게 알려준다:

```
💾 저장: h1→#17 h2→#18 h3→#19 h4→#20 h5→#21
```

이 ID 없이는 사용자가 `/feedback` 명령을 정확히 줄 수 없다 — 반드시 알릴 것.

## 사용자 피드백 명령 처리

### `/feedback h<id> right|wrong [reason="..."]`

사용자가 슬래시 명령으로 가설에 피드백을 줄 때:

1. `h<id>` 가 *로컬 ID* (h1, h2) 면 가장 최근 분석 응답의 `h_local_id_map` 에서 DB ID 변환
2. `h<id>` 가 *DB ID* (#17 같은 숫자) 면 그대로 사용
3. `add_feedback(hypothesis_id=ID, verdict=..., reason=...)` 호출
4. 반환값의 `label_kind` 가 `user_immediate` 인지 `user_followup` 인지 사용자에게 알려준다 (가설 생성 14 일 이내면 즉시, 초과면 사후 추가)

verdict 는 `right` / `wrong` / `neutral` 셋 중 하나. 다른 단어 들어오면 사용자에게 명확히 요청.

### `/labels [ticker] [as_of_date]`

누적 가설·라벨 조회:

1. `list_hypotheses(ticker=..., as_of_date=...)` 로 가설 메타 리스트 수집
2. 각 가설에 대해 `get_hypothesis_with_labels(id)` 호출 (라벨 join)
3. 표 형식 요약: ID / ticker / date / direction / verdict / actual_relative_return_pct / reason

데이터가 많으면 ticker / as_of_date 필터를 안내.

## 결손 처리

- 분석가 출력에 `<unavailable>` 마커가 보이면 → 해당 신호는 가설로 만들지 말 것
- `compute_correlation` 의 `n_days < 50` → 통계 가설에 "n=NN, 신뢰 마진 낮음" 명시
- 신규 상장 종목 (close=0 행 가능) → IPO 후 6개월 미만이면 분석 신중

## 비대상 (이번 스킬에서 안 함)

- 매매 추천 (BUY/HOLD/SELL 같은 결정)
- 가격 예측 (특정 가격 도달 예측 금지)
- 단기 (분봉) 또는 장기 (분기 이상) 호라이즌
- 비한국 종목 (AAPL 등) — supply_demand 가 N/A 반환

## 사용 예시

사용자: "삼성전자 봐줘"

스킬 흐름:
```
1. ticker = "005930.KS", date = today
2. analyst_supply_demand("005930.KS", "2026-05-29") → 마크다운
3. compute_correlation("005930.KS") → dict
4. MEMORY.md 룰 적용해 3~5 가설 도출
5. 한국어 요약 + JSON 코드블록 출력
```
