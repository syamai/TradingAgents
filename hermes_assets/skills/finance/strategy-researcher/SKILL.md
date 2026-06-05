---
name: strategy-researcher
description: "수급 룰 매매전략 자율 연구 — 파라미터화 spec 제안→5년 백테스트→게이트 판정→변이 반복→검증 전략 영구화"
version: 0.1.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [finance, korea, backtest, strategy, supply-demand]
    related_skills: [stock-analyst]
---

# strategy-researcher — 수급 룰 매매전략 자율 연구가

## 개요

이 스킬은 trading-ai MCP 서버 (`trading-ai-hermes`) 의 백테스트 도구로 **파라미터화된 수급 룰 매매전략**을 자율 연구한다. 한 종목 분석(stock-analyst)이 아니라, **종목군 5년 데이터에서 통계적으로 수익률·승률·위험조정이 우수한 매매기법을 발굴·검증**하는 것이 목적.

루프: 후보 전략 spec 제안 → `backtest_strategy` 로 종목군 백테스트 → 채택 게이트 판정 → 미달이면 파라미터 변이/신규 → in-sample 통과 시 out-of-sample 검증 → 양쪽 통과 시 `save_strategy` 로 영구화. 백테스트 *계산*은 결정론적 엔진이 하고, Hermes 는 **전략 제안·변이·판정만** 한다 (수익률을 직접 계산하지 말 것 — 환각).

## 트리거

- "수급 전략 연구해줘"
- "돈 되는 매매기법 찾아줘"
- "외국인 수급으로 백테스트 돌려봐"
- "전략 더 발굴해줘" (이전 연구 이어서)

## 사전 조건

- MCP 서버 `trading-ai-hermes` 등록·연결됨
- 전략 도구 3종:
  - `backtest_strategy(spec, universe=None)` — 탐색용 백테스트 (저장 X). in/out-sample 메트릭 + gate_passed 반환
  - `save_strategy(spec, name=None)` — 내부 재백테스트 후 영구 저장. 동일 로직은 duplicate
  - `list_strategies(gate_passed=None)` — 누적 시도/통과 조회 (진행 추적의 single source)
- KIS 히스토리에 종목군(약 199 종목, 5년) 적재돼 있어야 함

## 전략 spec 스키마

```json
{
  "spec_version": 1,
  "name": "<사람이 읽는 식별자>",
  "direction": "long",
  "entry": { "all_of": [ <신호>, ... ] },          // AND, 1~3개
  "exit": {
    "signal_all_of": [ <신호>, ... ],              // 옵션, 0~2개 (매수 주체의 매도 전환 등)
    "stop_loss_pct": <STOP_GRID>,                  // 옵션
    "take_profit_pct": <TP_GRID>,                  // 옵션
    "max_hold_days": <HOLD_GRID>                    // 필수
  }
}
```

### 신호 어휘 (5종) — 전부 scale-free (종목군 일반화)

| signal | 의미 | 파라미터 (그리드) |
|---|---|---|
| `net_streak` | `{subject}_net_qty` 연속 N일 동일 부호 | `subject`, `min_days∈{2,3,4,5,7,10}`, `sign∈{buy,sell}` |
| `pct_threshold` | `{subject}_pct` (누적 영향력 비중%) 임계 | `subject`, `op∈{>=,<=}`, `value∈{10,20,30,40,50}` |
| `pct_delta` | `{subject}_pct` 의 N일 변화량 | `subject`, `window∈{1,3,5,10,20}`, `op`, `value∈{2,5,10}` |
| `net_vol_ratio` | N일 누적 net_qty / N일 누적 volume | `subject`, `window∈{1,3,5,10,20}`, `op`, `value∈{0.05,0.1,0.2}` |
| `price_filter` | 종가 vs N일 이동평균 위치 | `mode∈{above_ma,below_ma}`, `window∈{5,20,60}` |

- `subject` ∈ `foreign_registered, foreign_unregistered, pension, private_equity, investment_trust, securities, bank, insurance, retail, other_corp, foreign`. **"institution" 같은 통합 카테고리 금지** (sub-주체 분리 의무).
- 청산 그리드: `stop_loss_pct∈{3,5,8,10}`, `take_profit_pct∈{5,8,10,15,20}`, `max_hold_days∈{5,10,20,40,60}`.
- **수치는 그리드 값만** — 그리드 밖이면 `backtest_strategy` 가 ValueError. 변이는 그리드 인접값으로만 (과최적화 방어).

## 채택 게이트

`backtest_strategy` 가 반환하는 `gate_passed` 는 다음을 **in-sample·out-of-sample 양쪽** 충족할 때만 True:

- 승률 (`win_rate`) ≥ 0.60
- 샤프 (`sharpe`) ≥ 1.2
- 최대낙폭 (`mdd_pct`) ≥ -20% (더 얕아야)
- 거래수 (`n_trades`) ≥ 50
- in/out 승률 격차 |in − out| ≤ 0.10 (과적합 탐지)

수익률(`cum_return_pct`)·KOSPI 초과(`avg_excess_ret_pct`)는 *측정만* — 게이트 아님. 단순 우상향장 편승 방지를 위해 참고는 하되, 채택은 승률+위험조정으로 한다.

## 워크플로우

### Phase 0 — 진행 상태 조회

`list_strategies()` 로 이미 시도한 전략 수·통과 수를 먼저 확인한다. 컨텍스트가 끊겨도 여기서 이어간다. 통과 3개 도달했으면 종료.

### Phase 1 — 후보 룰 제안

MEMORY.md 의 `[전략-시드-*]` 룰을 적용해 후보 spec 2~4개 제안:

- 외국인등록 지속매수 (`net_streak` buy min_days 5~10 + `pct_threshold`)
- 사모+외국인비등록 동반 단기 모멘텀
- 비중 추세 (`pct_delta`)
- 거래량 강도 (`net_vol_ratio`)
- **추세동조 필터 (`price_filter above_ma`) 를 entry 에 AND** — long-only 거의 필수
- 청산: `max_hold_days` (필수) + `stop_loss_pct` (권장 5~8%)

### Phase 2 — in-sample 백테스트

각 후보 → `backtest_strategy(spec)`. **반환의 `in_sample` 메트릭만 본다** (out_sample 은 Phase 5 까지 참조 금지).

### Phase 3 — in-sample 게이트 판정

`in_sample` 이 게이트(승률·샤프·MDD·거래수) 미달이면 Phase 4, 통과면 Phase 5.

### Phase 4 — 파라미터 변이 / 신규

미달 원인을 보고 그리드 **인접값**으로 변이:
- 거래수 부족 → 조건 완화 (min_days↓, 신호 수↓)
- 승률 낮음 → 추세필터 추가, stop_loss 조정, 보유기간 단축
- 샤프 낮음/MDD 깊음 → stop_loss 강화, 추세필터
- **out_sample 메트릭으로 변이하지 말 것** (holdout 누수). 변이도 매 회 반드시 저장 (Phase 6) → 같은 변이 반복 차단.

### Phase 5 — out-of-sample 검증

in-sample 통과 후보의 같은 결과에서 `out_sample` + `gate_passed`(in/out 격차 포함) 확인. `gate_passed=True` 면 Phase 6. False 면 **폐기** (out 으로 재튜닝 금지) — 새 가설로 Phase 1 복귀.

### Phase 6 — 저장

- 게이트 통과: `save_strategy(spec, name=...)` → `strategy_id` 받아 사용자에게 보고. (내부 재백테스트로 메트릭 무결성 보장.)
- 게이트 미통과 후보도 `save_strategy` 로 저장 — 연구 로그 + 시도 카운트 + 동일 변이 재시도 차단 (duplicate 반환).

### Phase 7 — 종료조건

다음 중 하나 충족 시 종료하고 결과 요약:
- 게이트 통과 전략 **3개** 저장, 또는
- 총 시도 **30회** 도달

종료 시 `list_strategies(gate_passed=True)` 로 채택 전략 표 출력 (name / 신호 요약 / in·out 승률·샤프·MDD·누적수익률).

## 출력 형식

각 통과 전략 보고:
```
✅ #<id> <name>
   진입: <신호 요약>  / 청산: max_hold=N, stop_loss=X%
   in : 승률 62% 샤프 1.4 MDD -12% 누적 +35% (거래 71)
   out: 승률 60% 샤프 1.3 MDD -14% 누적 +28% (거래 33)
```

## 결손 처리 / 주의

- `n_trades` 가 0 이거나 매우 적으면 신호가 너무 빡빡 — 조건 완화
- `sharpe` 가 null 이면 거래 2건 미만 → 게이트 자동 fail
- survivorship bias (현재 상장 종목만) 존재 — 절대수익보다 승률·위험조정 게이트로 일부 완충 (한계 인지)
- 다중 시도(M회) 중 우연 통과 위험 — in/out 격차 게이트로 완화하나 통과 ≠ 미래 보장. "백테스트 기준" 임을 명시

## 비대상

- 단일 종목 1회 분석 (→ `stock-analyst` 스킬)
- short(공매도) 전략 — long-only 만 (1차)
- 분봉/일중 또는 분기 이상 호라이즌
- 실거래 주문 실행 (백테스트 검증까지만)
