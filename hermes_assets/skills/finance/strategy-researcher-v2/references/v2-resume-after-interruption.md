# v2 연구 루프 중단 후 재개 체크리스트

이 참고문서는 `strategy-researcher-v2`를 실제 운영 중 끊겼을 때, "스킬이 있으니 루프가 돌고 있다"고 가정하지 않고 **실행 흔적 기준으로 재개 판단**하는 절차를 정리한다.

## 언제 쓰나
- 사용자가 "자율루프가 실제로 돌고 있나"를 묻는 경우
- provider/auth/429 문제 뒤에 연구가 어디서 멈췄는지 확인해야 하는 경우
- 수동 1회 재개와 장기 cron 재개를 순차적으로 붙여야 하는 경우

## 소스 오브 트루스 우선순위
1. `strategies_v2.db`의 총 count / last id / last created_at
2. 현재 살아 있는 process / cron / launchd 흔적
3. session 기록과 보고서(.md)에서 마지막 family 방향 확인
4. 스킬 문구 자체는 참고용일 뿐, 실행 증거가 아니다

## 실제 판단 규칙
- `process/cron/launchd`에 연구 worker가 없으면 **현재 상주 루프는 없는 것**으로 본다.
- `strategies_v2.db`의 count가 증가하지 않고 마지막 저장 시각이 과거면 **중단된 상태**로 본다.
- 최근 보고서가 특정 family를 "고갈/폐기"로 결론냈다면, 재개는 그 family 미세조정보다 **다음 family 전환**부터 시작한다.
- 현재 별도 worker가 없으면 single-writer 조건은 충족된 것으로 보고 재개 가능성을 높게 본다.

## 이번 세션에서 확인된 유효 패턴
- `foreign_registered` contrarian family는 MDD 병목이 깊어, 재개 시 `private_equity` / `investment_trust`로 subject 전환하는 것이 자연스러웠다.
- 재개는 먼저 **정확히 3개 수동 평가+저장**으로 안전하게 확인하고,
  이후 **batch cron**을 붙이는 2단계가 안정적이었다.
- batch 재개 prompt는 아래 보호 규칙을 넣는 편이 좋다:
  - `list_strategies_v2(brief=True)`를 count single source로 사용
  - `passed > 0`이면 stop
  - `total >= batch ceiling`이면 stop
  - tick당 정확히 3개 새 전략만 저장
  - duplicate면 대체 후보를 즉시 시도
  - child loop / parallel writer 금지

## 운영 주의
- provider 429는 재인증 문제가 아니라 사용량 제한일 수 있으므로, auth 문제와 분리해서 다뤄야 한다.
- Telegram에서 보이는 "자율루프" 질문에는 스킬 존재 여부가 아니라 **실제 프로세스/DB/cron 증거**를 먼저 제시한다.
- 결과 보고는 `starting total -> saved ids -> ending total` 형태가 재개 확인에 가장 명확하다.
