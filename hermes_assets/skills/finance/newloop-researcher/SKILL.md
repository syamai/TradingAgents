---
name: newloop-researcher
description: "newloop 가설 생성·자가학습 tick — 원장의 실패 교훈을 읽고 구조적으로 새로운 가설 family 1개를 사전등록·설계해 Stage1 게이트(시장 도움 제거 채점)에 응시. Stage1 통과 시 Stage2(포트폴리오 시장중립) 채점→통과 시 Stage3 forward 관찰 등록까지 자동 연결하고 결과를 보고. 베타 위장(상승장 고민감 종목 선택)은 게이트가 절편 α로 적발하므로 시장중립적 우위 가설 우선. 고갈 선언 시 생성 중단."
version: 0.2.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [finance, korea, newloop, stage1-gate, preregistration, self-learning]
    related_skills: [strategy-researcher-v2]
---

# newloop-researcher — 가설 생성·자가학습 tick

새 자율 루프의 생성 측. **1 tick = family 최대 1개.** 검증 측(Stage1 게이트)은
`tradingagents/newloop/`에 동결되어 있고 이 스킬은 그것을 절대 수정하지 않는다.

## 배경 (왜 이 모양인가)
- 구 루프는 3,106개를 만들었지만 통계적으로 운과 구분되는 전략이 0개였다.
  원인: 개수 채우기 + 베타 위장(출렁임 큰 종목이 상승장에서 실력처럼 보임).
- 새 게이트는 거래단위 회귀 `net = α + β·basket`의 **절편 α**로만 합격을 준다
  — 시장이 제자리일 때도 버는가. 승률·총수익은 게이트 금지.
- 따라서 "오르는 장에서 더 오르는" 가설은 만들어봤자 α=0으로 떨어진다.

## 절차

### 0. 작업 디렉토리
`cd /Users/selab/Source/trading-ai` (uv 명령 전제)

### 1. 교훈 로딩 (자가학습 입력)
```bash
uv run python -m tradingagents.newloop.lessons
```
- 출력에 **"⛔ 고갈 선언"**이 있으면 → **새 가설을 생성하지 말 것.**
  고갈 사실·누적 표를 그대로 보고하고 tick 종료. (신호공간 소진은 정당한 결론)
- 표의 기존 family 가설·탈락 양식을 숙지 — 같은 구조 재탕 금지.

### 2. 새 가설 설계 (사전등록 규율)
- `family_key`: kebab-case 신규 — 기존 표의 키와 중복·유사 변형 금지.
- 가설 한 줄: **경제적 근거 필수** — "누가 왜 돈을 잃어주는가"가 들어가야 함.
- 기존 family들과 **신호 조합 수준에서** 구조적으로 달라야 함.
- 탈락 양식이 "베타 위장"인 family가 있으면: 돌파·고변동 선호 구성을 피하고
  시장중립적 우위(수급 괴리·상대강도·개인물량 흡수·페어성)를 우선.
- spec 형식: `tradingagents/hermes/strategy_spec_v2.py`의 v2 스키마
  (`entry.all_of` 신호 + `exit` sl/tp/max_hold). 신호 어휘도 그 파일 참조.
- 변형 8~12개(한도 24): 핵심 파라미터 2~3축만 그리드.

### 3. spec 저장 → Stage1 응시
```bash
# specs를 JSON 배열로 저장
/tmp/newloop_spec_<family_key>.json

uv run python -m tradingagents.newloop.stage1_gate \
  --family <family_key> --hypothesis "<가설 한 줄>" \
  --spec-file /tmp/newloop_spec_<family_key>.json \
  --out artifacts/newloop/stage1_<family_key>_$(date +%Y%m%d).json
```
- `제출 거부`(한도 소진 등)가 나오면 **그대로 보고** — family 키를 바꿔 우회
  재응시 금지(시험지 마모 방지 규율의 핵심).
- `spec 무효`면 슬롯이 소모되지 않았으므로 spec 수정 후 재시도 가능.

### 3.5 Stage1 통과 시 — Stage2 채점 → forward 등록 (자동 연결)
Stage1 결과 JSON(`artifacts/newloop/stage1_<family>_<date>.json`)의 `results[]` 에서
**`status == "pass"` 인 spec 만** 추린다. 하나도 없으면 이 단계 전체 생략
(`insufficient`·`fail` 은 절대 넘기지 않음). pass 가 있을 때만 아래를 순서대로 실행:

1. 통과 spec 들만 모아 `/tmp/newloop_s2_<family>.json`(JSON 배열)로 저장.
2. Stage2(포트폴리오 시장중립) 채점 — Stage1 과 같은 홀드아웃 예산을 `<family>-s2` 키로 공유:
   ```bash
   uv run python -m tradingagents.newloop.stage2_gate \
     --family <family> --spec-file /tmp/newloop_s2_<family>.json \
     --out artifacts/newloop/stage2_<family>_$(date +%Y%m%d).json
   ```
   `제출 거부`(예산/한도 소진)면 그대로 보고하고 종료 — 우회 금지.
3. Stage2 결과(`artifacts/newloop/stage2_<family>_<date>.json`)에서 다시 `status == "pass"`
   인 spec 만 추려 `/tmp/newloop_fwd_<family>.json` 저장 → forward(Stage3) 관찰 등록:
   ```bash
   uv run python -m tradingagents.newloop.forward --register --family <family> \
     --spec-file /tmp/newloop_fwd_<family>.json \
     --out artifacts/newloop/forward_<family>_$(date +%Y%m%d).json
   ```
   forward 는 **채점이 아니라 2025-06-30 이후 관찰**(예산 비소비). 등록 후 4주 경과해야
   판정 자격이 생기므로, 등록 직후 forward 메트릭은 표본 부족이 정상이다.

### 4. 보고 (Telegram)
- family·가설·결과표(ref / n / 군집 / t(α) / β / 하락창 / FWERp / 판정)
- 다중검정 상태: **누적 채점 N / 예산(HOLDOUT_TRIAL_BUDGET) 잔여**, 이번 채점이
  직면한 **동적 합격선 t(α)≥t_crit(N)**. 합격선은 t≥3.0 고정이 아니라 누적 N 이
  클수록 오른다 — "느리게 반복"으로는 못 피한다(N 자체가 임계를 올림).
- 원장 상태(제출 #/한도)와 시험지 마모(누적 family·spec 수)
- **Stage1 pass 발생 시**: Step 3.5 의 Stage2 결과(α연율·β·t(α)·t임계·하락t·판정)와,
  Stage2 도 통과면 forward 등록 사실(관찰 시작·판정 최소 4주)을 함께 보고.
  "🎯 Stage1→Stage2 통과 — forward 관찰 진입" 강조 + 사람 확인 요청.
  insufficient는 탈락이 아님(표본/하락창 부족) — 구분해 보고.
- **예산 소진(잔여 0) 또는 거부 시**: 그대로 보고 — 새 홀드아웃 수집은 사람 결정.

## 금지
- 게이트·원장·홀드아웃 코드 및 임계 수정 (동결 — 변경은 사람이 버전 상향으로만)
- 홀드아웃 스토어(`kis_history_fresh_exam`) 직접 조회·탐색 활용
- 운영 DB(strategies_v2.db 등) 쓰기
- 같은 tick에 family 2개 이상
- 게이트 결과를 보고 spec을 미세조정해 같은 제출 안에서 반복 응시하는 행위
