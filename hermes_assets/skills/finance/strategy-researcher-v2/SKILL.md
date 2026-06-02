---
name: strategy-researcher-v2
description: "수급/기술 매매전략 자율 연구 v2 — 이미 실패한 메커니즘(눌림목·모멘텀·저변동·저공매도)·통제(2025-06 컷오프·ETF제외·유동성·다중검정)를 숙지한 채 구조적으로 새로운 전략/신규 지표를 제안·백테스트·검증. 숫자만 바꾼 재시도 금지, 신규 지표 신설 우선."
version: 0.1.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [finance, korea, backtest, strategy, supply-demand, v2]
    related_skills: [strategy-researcher, stock-analyst]
---

# strategy-researcher-v2 — 수급 룰 매매전략 자율 연구가 (v2)

## References
- `references/v2-manual-comparison-patterns.md` — manual experiment takeaways; update when a family shows stable failure/success patterns worth reusing.
- `references/v2-resume-after-interruption.md` — how to verify whether a v2 autonomous loop is actually running, identify the resume point from DB/process evidence, and restart safely with single-writer discipline.
- `references/v2-autoswitch-ops.md` — cron-driven autoswitch loop operations: 80% quota handoff to LLM-free mode, what evidence proves a tick really advanced, and post-400 search-direction guidance.
- `references/v2-status-line-monitoring.md` — quota-safe Telegram one-line heartbeat pattern using a script-only (`no_agent`) cron for long-running v2 research loops.
- `references/v2-portfolio-policy.md` — current Korean-stock portfolio-combination policy for v2 backtests: 90% stock exposure, 10% cash, 30 target names, 3% base per-name weight, 5% cap, and interpretation pitfalls when MDD changes after policy updates.
- `references/v2-portfolio-rerank-workflow.md` — how to rerank saved v2 strategies under the current portfolio-aware engine, including 4-way robust Sharpe ranking, quick top-30 seed reruns, long full-DB background reruns, and CSV/JSON reporting.

## 개요

`trading-ai-hermes-v2` MCP 서버의 v2 백테스트 도구로 **수급/기술 매매전략**을 자율 연구한다.
목표는 단순 게이트 통과가 아니라, **이미 실패한 메커니즘·편향을 피해 구조적으로 새로운 전략
(필요시 신규 지표 신설)을 제안하고 통제(2025-06 컷오프·ETF제외·유동성·다중검정) 하에서 검증**
하는 것. **통제 후 살아남는 엣지가 없다는 결론도 유효한 결과**다(억지 통과보다 가치 있음).

루프: spec/신규지표 제안 → `backtest_strategy_v2` → 게이트 판정 → (제한적)변이/새 메커니즘 →
`save_strategy_v2` 영속화. 백테스트 *계산*은 결정론적 엔진이 한다 — **Hermes 는 제안·판정만**
(수익률·메트릭을 직접 계산/추정하지 말 것, 환각 금지. 반드시 도구 반환값만 인용).

## 트리거
- "수급 전략 v2 연구해줘"
- "신규 신호로 게이트 통과 전략 찾아줘"
- "strategy-researcher-v2 돌려줘"

## 사전 조건 (도구 3종, `trading-ai-hermes-v2` 서버)
- `backtest_strategy_v2(spec, universe=None)` — 탐색 백테스트(저장 X). in/out 메트릭 + gate_passed 반환
- `save_strategy_v2(spec, name=None)` — 내부 재백테스트 후 `strategies_v2.db` 저장. 동일 로직은 duplicate
- `list_strategies_v2(gate_passed=None)` — 누적 시도/통과 (진행 추적 single source)

## 핵심 맥락 — 이미 실패한 것 / 반드시 통제할 것 (2026-06 갱신, 반드시 숙지)

이 유니버스(KOSPI/KOSDAQ 수급 long-only)에서 광범위 탐색이 끝났고, **통제를 제대로 걸면 검증된 *거래 가능* 엣지가 아직 없다.** 같은 메커니즘을 숫자만 바꿔 재시도하는 것은 노이즈 채굴이다. 아래를 숙지하고 **구조적으로 새로운 가설**만 제안하라.

### 이미 검증되어 기각된 메커니즘 — 재제안 금지 (숫자만 바꾼 변형 포함)
- **눌림목/역추세 (`price_drop`) 가족** — 그리드 100개+ 소진, 깊은 MDD. 손절·익절·보유 숫자 변형은 무의미.
- **가격 모멘텀 (`price_return`)** — 횡단면·시계열 모두 2025-06 컷오프 적용 시 게이트 탈락(약세연도 음수).
- **저변동성 (`realized_vol`)** — 임계·횡단면·forward 전부 시장 하회. 한국 유니버스에 프리미엄 부재 확정.
- **저공매도압력 (`short_ratio`)** — 통과처럼 보였으나 **illiquidity 아티팩트**: 유동성 필터(상대 백분위·고정 50억·시기일관 중앙배수) 3방식 모두 게이트 탈락 = 저유동 종목이 만든 가짜 수익.

### 채점 시 반드시 적용할 통제 (빼면 전부 착시)
1. **2025-06-30 컷오프** — 이후 한국증시 비정상 급등(생존풀 동일가중 ~18.8배 vs KOSPI ~4.3배)은 OOS/검증서 제외(forward 관찰 전용). 포함하면 거의 모든 long 전략이 통과하는 착시.
2. **ETF 제외** — 유니버스의 ETF(KODEX/TIGER 등 12종)는 공매도·수급 구조가 달라 stock 전략을 오염. 반드시 제외.
3. **유동성 필터** — 저유동(거래 불가) 종목 제외(임팩트코스트). "거래 가능한 종목군에서도 살아남는가"가 진짜 질문.
4. **다중검정 규율** — 누적 시행이 정당화 한도(~147)를 이미 크게 초과. 통과 *수* 가 아니라 *구조적 신규성* 으로 진행. 숫자 변형으로 게이트 넘기기 = 우연 통과(p-hacking).
5. **생존편향** — 현생존 종목만 → 과거 절대수익은 낙관 상한. 시장초과 IR(상대)로만 판단, 진짜 검증은 forward 뿐.

### 진짜 다음 수 = 새 지표(신호) 생성 후 검증 (숫자 변형 ❌)
기존 신호 조합은 사실상 소진. **새 지표를 만들어 어휘에 추가하고 검증하라:**
- **기술적 지표 검증**: 기존에 없는 기술 신호(예: RSI·스토캐스틱식 오실레이터, 밴드폭/변동성 돌파, 거래량 급증·정체, 갭, 신고가 근접 등)를 *trailing·look-ahead 0* 으로 신규 구현 후 검증(학술적으론 약하나 직접 채택/기각 판정).
- **신규 수급 지표 생성**: 기존 net_qty/pct/streak 를 넘는 *파생* 수급 신호 — 주체 간 집중도(허핀달), 수급-가격 선행/지연, 매집 가속(2차미분), 수급 분산도, 외국인·기관·개인 합의/이격, 특정주체 비중의 추세 전환 등 — 을 새로 정의·구현·검증.
- 신규 신호는 `strategy_spec_v2`/`backtest_engine_v2` 코드 추가 필요(아래 "V2 코드 업데이트" 규율 + 백업 의무). **숫자 그리드 변형보다 신호 신설을 항상 우선.**

## 전략 신규성 규율 (숫자만 바꾼 재시도 금지 — 매 제안 전 점검)
1. `list_strategies_v2()` 로 **기존 시도의 신호 조합(메커니즘)** 파악 — 개수가 아니라 *어떤 신호 타입 조합이 이미 쓰였나*.
2. 새 후보는 기존과 **메커니즘이 달라야** 한다: (a) 새 신호 타입, (b) 다른 신호 타입 조합, 또는 (c) 다른 경제적 논리. **같은 신호 타입 집합에서 그리드 숫자만 바꾼 spec 제안 금지**(spec_hash 중복은 엔진이 막지만 near-duplicate 도 피하라).
3. 각 후보에 **한 줄 경제적 논리**(왜 시장을 초과하나)를 붙여라. "손절을 더 줄여서"는 논리가 아니다.
4. 위 "기각된 메커니즘 가족"은 재제안 금지.

## v2 전략 spec 스키마

```json
{
  "spec_version": 2,
  "name": "<사람이 읽는 식별자>",
  "direction": "long",
  "entry": { "all_of": [ <신호>, ... ] },          // AND, 1~3개
  "exit": {
    "signal_all_of": [ <신호>, ... ],              // 옵션, 0~2개
    "stop_loss_pct": <STOP_GRID>,                  // 옵션 (권장)
    "take_profit_pct": <TP_GRID>,                  // 옵션 (권장)
    "max_hold_days": <HOLD_GRID>                    // 필수
  }
}
```

### 신호 어휘 (11종 = 기존 8 + 검증·기각 3). 수치는 그리드 값만 (위반 시 ValueError)

| signal | 의미 | 파라미터 (그리드) |
|---|---|---|
| `price_drop` ⭐ | 최근 W일 종가 수익률 ≤ -X% (역추세 진입) | `window∈{3,5,10,20}`, `value∈{3,5,8,10,15}` |
| `trend_slope` ⭐ | `{subject}_net_qty` 누적합 W일 변화 방향 (수급 레짐) | `subject`, `window∈{10,20,60}`, `direction∈{up,down}` |
| `rolling_corr` ⭐ | `{subject}_net_qty` vs `price_change_pct` W일 trailing r ≥ min_r | `subject`, `lookback∈{20,60,120}`, `min_r∈{0.1,0.2,0.3,0.5}` |
| `net_streak` | `{subject}_net_qty` 연속 N일 동일 부호 | `subject`, `min_days∈{2,3,4,5,7,10}`, `sign∈{buy,sell}` |
| `pct_threshold` | `{subject}_pct` 임계 | `subject`, `op∈{>=,<=}`, `value∈{10,20,30,40,50}` |
| `pct_delta` | `{subject}_pct` N일 변화량 | `subject`, `window∈{1,3,5,10,20}`, `op`, `value∈{2,5,10}` |
| `net_vol_ratio` | N일 net_qty / N일 volume | `subject`, `window∈{1,3,5,10,20}`, `op`, `value∈{0.05,0.1,0.2}` |
| `price_filter` | 종가 vs N일 이평 위치 | `mode∈{above_ma,below_ma}`, `window∈{5,20,60}` |
| `short_ratio` ⚠️기각 | 공매도 거래량 비중 W일 평균 (illiquidity 아티팩트로 기각) | `window∈{1,3,5,10,20}`, `op`, `value∈{5,10,15,20,30}` |
| `price_return` ⚠️기각 | W일 가격 모멘텀 (컷오프 후 탈락) | `window∈{20,60,120,250}`, `op`, `value∈{0,5,10,20,30}` |
| `realized_vol` ⚠️기각 | W일 실현변동성 (저변동성 부재) | `window∈{20,60,120}`, `op`, `value∈{20,30,40,50}` |

> ⚠️기각 3종은 이미 검증되어 기각된 메커니즘 — 어휘엔 있으나 **재제안 금지**(핵심 맥락 참조). 어휘 목록은 이게 전부이며, 새 메커니즘은 **신규 신호 신설**로만 가능.

- `subject` ∈ `foreign_registered, foreign_unregistered, pension, private_equity, investment_trust, securities, bank, insurance, retail, other_corp, foreign`. **통합 카테고리 금지** (sub-주체 분리).
- 청산 그리드: `stop_loss_pct∈{3,5,8,10}`, `take_profit_pct∈{5,8,10,15,20}`, `max_hold_days∈{5,10,20,40,60}`.
- ⚠️ `rolling_corr` 는 무방향이 아니라 양의 동조만 — 방향은 `net_streak buy` 와 AND 로 부여.

## 채택 게이트 (v2) — 2단계: 빠른 스크린 → 진짜 채택 기준

**1단계 빠른 스크린** (`backtest_strategy_v2` 의 `gate_passed`, in/out 양쪽): 승률>0.5 · 샤프>1.0 · MDD≥-20% · 거래≥50 · in/out 승률격차≤0.10. **이건 약한 1차 필터일 뿐 채택 기준이 아니다.**

**2단계 진짜 채택 기준 (fair gate)** — 1단계 통과 후 반드시:
- **walk-forward × 시장초과 IR**: 모든 OOS 윈도우에서 KOSPI 대비 초과 IR>0 **AND** 중앙 IR>0.5 (`run_walk_forward_validation`). 단일분할 레짐운·시장베타 착시 제거.
- **위 "핵심 맥락" 통제 적용**: 2025-06-30 컷오프 + ETF 제외 + 유동성 필터. 이걸 통과해야 "거래 가능 후보".
- 빠른 스크린만 통과하고 fair gate·통제에서 떨어지는 전략은 **채택 아님**(저공매도가 그 예 — 1단계는 통과했으나 유동성 통제에서 붕괴).

수익률·절대 누적은 생존편향 상한이라 채택 근거 아님 — 시장초과 IR 로만 판단.

## 포트폴리오 결합 정책 / 결과 해석

현재 v2 백테스트는 한국 개별주 포트폴리오 원칙을 반영해 active 종목을 단순 100% 동일가중하지 않는다. 기본 정책은 주식 90% + 현금 10%, 목표 30종목, 종목당 기본 3%, 종목당 최대 5%, 총 주식 노출 90% 상한이다. active 종목 수가 부족하면 남는 비중은 현금 0% 수익률로 둔다.

결과 해석 시 특히 MDD 변화에 주의한다. 포트폴리오 정책 반영 전 결과와 비교하면, MDD 개선이 전략 로직 개선이 아니라 결합 방식 변경 때문일 수 있다. 사용자가 “가장 좋은 결과”, “상위 10개”, “MDD가 왜 좋아졌나”처럼 비교를 요청하면 `portfolio_policy` 응답 필드 또는 `references/v2-portfolio-policy.md`를 확인하고, 현재 정책이 적용된 결과인지 명시하라.

사용자가 “최상위 결과 N개를 포트폴리오까지 다시”, “상위 후보 재검증”, “가장 좋은 결과를 현재 엔진으로 다시 뽑아”라고 하면 저장 DB의 과거 메트릭을 그대로 보고하지 말고, 현재 포트폴리오-aware 엔진으로 재백테스트하라. 빠른 응답은 기존 저장 메트릭 기준 상위 N개를 seed로 골라 재계산하고, 랭킹은 `min(xsec_in/out/time_in/out Sharpe)`를 우선해 한 구간만 좋은 전략을 낮춘다. 전체 DB 재랭크는 오래 걸리므로 background + notify_on_complete로 돌리고, 우선 top-N 결과와 CSV/JSON 산출물을 보고한다. 상세 절차는 `references/v2-portfolio-rerank-workflow.md` 참고.

## 워크플로우

### Phase 0 — 진행 상태
`list_strategies_v2()` 로 누적 시도 수·통과 수 확인. 컨텍스트 끊겨도 여기서 이어간다.
gate_passed=1 이 이미 있으면 즉시 보고 후 종료.

### Phase 1 — 후보 제안 (구조적 신규성 우선)
위 "전략 신규성 규율"을 적용해 후보 spec 2~4개 제안. **각 후보는 기존과 다른 메커니즘 + 한 줄 경제적 논리**를 갖춰야 한다. 숫자만 다른 변형·기각된 가족(눌림목/모멘텀/저변동/저공매도) 재제안 금지. 기존 신호로 새 메커니즘이 안 나오면 **신규 지표(기술·수급) 신설을 우선 제안**(코드 추가 경로).

- **제안 전 그리드 유효성 확인**: `price_drop.value=9` 같은 값은 v2 엔진이 거절한다. 모든 수치는 허용 그리드에 맞추고, 애매하면 가장 가까운 유효값(예: 8 또는 10) 대체안을 함께 제시.
- 사용자가 "3개 다 해"처럼 즉시 실행을 원하면 **무효 spec 을 그대로 실행하지 말고** "원안 그리드 밖 → 대체안 X로 실행"을 먼저 명시한 뒤 유효 spec 으로 진행.

### Phase 2 — 백테스트
각 후보 → `backtest_strategy_v2(spec)`. **반환의 `in_sample` 메트릭만 먼저 본다**
(out_sample 은 Phase 5 까지 변이 근거로 쓰지 말 것 — holdout 누수).

### Phase 3 — in-sample 게이트 판정
in_sample 이 승률>0.5·샤프>1.0·MDD≥-20%·거래≥50 미달이면 Phase 4, 통과면 Phase 5.

### Phase 4 — 변이 (제한적 — 같은 메커니즘 미세조정 최소화)
**숫자 변형은 한 메커니즘당 1~2회로 제한.** 명백한 단일 구현 결함(예: 거래수 부족 → 조건 1회 완화)만 인접 그리드로 보정하고, 그 이상은 **변이를 멈추고 Phase 1 의 새 메커니즘/새 지표로 전환**한다. 같은 신호 조합을 숫자만 바꿔 반복하면 다중검정만 악화(통과해도 우연).
- **out_sample 로 변이 금지**(holdout 누수). 변이도 매 회 `save_strategy_v2` 저장(중복 차단).
- 기각된 가족(눌림목·모멘텀·저변동·저공매도)의 변이는 금지 — 새 우물을 파라.

### Phase 5 — out-of-sample 검증
in 통과 후보의 `out_sample` + `gate_passed`(격차 포함) 확인. True 면 Phase 6.
False 면 **폐기**(out 으로 재튜닝 금지) → Phase 1 새 가설.
- ⚠️ 빠른 스크린 `gate_passed`=True 는 "채택"이 아니다 — "채택 게이트" 2단계(walk-forward 시장초과 IR + 2025-06 컷오프·ETF·유동성 통제)까지 통과해야 거래 가능 후보. 스크린만 통과한 건 illiquidity 등 착시 의심.

### Phase 6 — 저장 / 보고
- 게이트 통과: `save_strategy_v2(spec, name=...)` → `strategy_id` 받아 **즉시 보고**.
- 미통과 후보도 `save_strategy_v2` 저장 (연구 로그 + 시도 카운트 + duplicate 차단).
- 사용자가 "이번 루프에서 가장 좋은 3개", "#180 왜 제일 좋냐", "주변 변형 다 해"처럼
  **수동 후속 비교**를 요청하면, 변이 생성에는 계속 in-sample 만 쓰되, **보고용 순위표는 out-sample
  샤프/MDD/거래수 균형**으로 별도 정리해도 된다. 단 이 비교는 *설명용*이며, out-sample 을 보고 다시
  파라미터를 미세조정하는 근거로 쓰면 안 된다.
- 세션별 상위 후보/주변 변형 결과는 `references/v2-manual-comparison-patterns.md`에 요약해 두고,
  이후 비슷한 사용자가 "왜 이 후보가 최고인가"를 물을 때 비교 프레임으로 재사용한다.

### Phase 7 — 종료조건
- **fair gate + 통제(컷오프·ETF·유동성)까지 통과한 거래 가능 후보 발견** — 빠른 스크린 gate_passed 만으론 부족, 또는
- **시도 예산 도달** (다중검정 한도 고려, 무한 숫자변형 금지). 이때 **"통제 후 살아남는 엣지 없음"도 정당한 종료 결과**.
도달 시 결과 요약 — 통과 후보뿐 아니라 어떤 메커니즘군을 새로 시도했고 왜 기각됐는지 함께 보고.

## 보고 조건 / 100개 이후 연속 자율 루프
- **gate_passed=1 발견 시 즉시 보고** (다음 tick 기다리지 말 것).
- 미발견 시 정기 보고는 **사용자가 명시적으로 heartbeat를 원할 때만** 켠다. 기본값은 조용한 milestone-only 알림이다. 30분 status-line cron을 만들었다면 목표 도달/통과/사용자 중단 시 autoswitch cron과 함께 pause/remove 하라.
- **100개 평가 완료 시 gate_passed=1 이 없으면 종료하지 말고**, 먼저 기존 전략들을 분석해서 새 전략군을 설계한 뒤 추가 자율 루프를 진행한다.
- 추가 루프 진행 전 보고는 가능하면 한 번만 한다: 현재 평가 수, 통과 수, 최선 후보, 실패 병목, 다음 루프 방향.
- Telegram 보고는 가능하면 LLM/provider 호출(`hermes -z ... send_message`)에 의존하지 말고, `TELEGRAM_BOT_TOKEN` + Bot API `sendMessage` 같은 결정론적 경로를 우선 사용한다. provider 인증 실패가 보고 경로까지 막는 것을 방지하기 위함이다. 단, script 내부 직접 Bot API 발송은 `deliver=local`로도 막히지 않으므로 stop 상태를 매 tick 반복 발송하지 않도록 one-shot sentinel 또는 cron pause를 반드시 둔다.
- 기존 신호 조합 한계가 명확하면(=숫자 변형 소진) **신규 신호 신설(V2 코드 업데이트)을 우선**한다. 레시피 (기 추가된 `short_ratio`/`price_return`/`realized_vol` 가 템플릿):
  1. `strategy_spec_v2.py`: 그리드 상수 + `_validate_signal_v2` 검증 분기 추가.
  2. `backtest_engine_v2.py` `_eval_signal_v2`: **trailing(rolling/shift)·look-ahead 0** 평가 분기 추가. 미래·당일 종가 이후 정보 사용 절대 금지.
  3. 횡단면 팩터면 `xsec_backtest.py` `_char_series`(랭킹 특성)도 추가 고려.
  4. **코드 수정 전 대상 파일 `.bak-YYYYmmdd-HHMMSS` 백업 필수**(백업 없는 수정 금지). 수정 후 `validate_spec_v2` + 간단 백테스트로 최소 검증 후 보고.
- 코드 업데이트 후에는 가능한 최소 검증(스키마/간단 백테스트/도구 호출)을 수행하고, 그 결과를 다음 루프 시작 전에 보고한다.
- **100개 평가 완료 시 최종 요약**은 gate_passed=1 발견 또는 사용자가 지정한 최대 batch 도달 시에만 최종 종료 요약으로 낸다. 그 전에는 "중간 batch 요약 + 추가 루프 시작 보고"로 취급한다.

## 자율루프 운영 안전장치
- **중요: 이 skill 자체는 "상주 백그라운드 루프"가 아니다.** SKILL.md 는 연구 절차를 정의할 뿐이며, 실제 자율루프가 돌고 있는지는 별도 실행 주체(cron / controller script / 장시간 Hermes worker)가 있어야 한다. 사용자가 "지금 strategy-researcher-v2 자율루프가 돌고 있나?"라고 물으면 문서 존재만으로 그렇다고 답하지 말고, **(1) 프로세스, (2) cron/launchd, (3) `strategies_v2.db` 최근 mtime·증가 추세, (4) 세션/리포트 흔적**을 함께 확인해 실제 실행 여부를 판정한다.
- 실행 중인 v2 루프/컨트롤러는 원칙적으로 **단일 writer** 로 유지한다. 여러 `hermes_v2_loop`, `hermes_v2_controller`, 또는 장시간 `strategy-researcher-v2` worker 가 같은 `strategies_v2.db` 를 동시에 쓰면 SQLite lock, 중복 평가, 코드 업데이트 충돌이 생길 수 있다.
- 루프 재시작 시 batch 번호는 고정값에서 시작하지 말고 DB count 기준으로 복원한다: `batch = count / TARGET_BATCH + 1`. 예: 이미 266개면 batch 3, target 300부터 이어간다.
- 각 tick 전후 `strategies_v2.db` 의 전략 수를 비교한다. 하위 Hermes 프로세스가 0 exit 여도 저장 수가 증가하지 않으면 실패로 간주한다.
- `MAX_NO_PROGRESS_TICKS` 같은 연속 무진행 한도를 둔다. N회 연속 저장 수 증가가 없으면 Telegram으로 중단 보고 후 루프를 멈춘다. 무진행 상태에서 tick을 빠르게 반복하는 무한 루프를 금지한다.
- 운영 점검에서 사용자가 "80%부터는 LLM 없이 도는 거지?"처럼 **자동전환 로직 자체의 실제 동작**을 묻는다면, 설계 문구만 인용하지 말고 cron/DB/process/detail-log 증거를 우선 확인하라. 특히 autoswitch 요약 로그가 이전 run의 `stop_target_reached` 상태를 아직 보여줄 수 있으므로, 실제 진행 여부는 `strategies_v2.db` 증가와 `hermes_v2_tick_detail.log`/프로세스 증거로 판정한다. 세부 운영 메모는 `references/v2-autoswitch-ops.md` 참고.
- 400개+ 평가 뒤에도 gate_passed=0 이면, 기존 `price_drop + trend_slope(foreign_registered)` near-miss 가족의 exit-only 미세조정보다 **entry family 전환**을 우선한다. 특히 `investment_trust / insurance / bank / private_equity / foreign_unregistered` 와 market-regime/flow/rotation 계열을 먼저 소진하고, FR contrarian 재반복은 뒤로 미뤄라.

## 출력 형식 (통과 전략)
```
🎯 #<id> <name>
   진입: <신호 요약>  / 청산: stop_loss=X% take_profit=Y% max_hold=N
   in : 승률 .. 샤프 .. MDD ..% 누적 ..% (거래 ..)
   out: 승률 .. 샤프 .. MDD ..% 누적 ..% (거래 ..)
```

## 주의
- `sharpe` null = 거래 2건 미만 → 게이트 자동 fail.
- **survivorship bias (현생존 종목만)**: 과거 절대수익은 낙관 상한. 시장초과 IR(상대)로만 판단, forward 만 클린. 항상 "백테스트 기준"임을 명시.
- **illiquidity 함정**: 빠른 스크린 통과 전략이 저유동 종목 픽일 수 있음(저공매도 사례). 유동성 통제 후에도 살아남는지 반드시 확인.
- **다중검정 한도 초과 상태**: 통과를 "발견"으로 과대해석 금지. 숫자 변형 통과는 거의 우연. 구조적 신규성·forward 만 신뢰 근거.
- 메트릭/수익률을 **직접 계산하지 말 것** — 반드시 도구 반환값만 인용(환각 금지).

## 비대상
- 단일 종목 1회 분석 (→ `stock-analyst`)
- 기존 5종 신호만 쓰는 strict 게이트 연구 (→ `strategy-researcher` v1)
- short/공매도, 분봉/일중, 실거래 주문
