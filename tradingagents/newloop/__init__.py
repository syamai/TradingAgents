"""newloop — forward-only 승격 원칙의 새 자율 루프 (기존 hermes와 격리).

기존 시스템(hermes)과의 경계:
  - 기존 코드는 import 만 한다(엔진 _simulate_v2, 스토어 로더). 수정 금지.
  - 기존 DB(strategies_v2.db 등)는 읽기전용 입력. 쓰기 금지.
  - 자체 상태는 ~/.tradingagents/newloop/ 에만 둔다.

구성:
  - stage1_gate  : 1차 관문 — 시장 도움 제거 채점(회귀 절편 α) + 기계 판정
  - ledger       : 사전등록·재응시 제한 원장 (append-only)
  - holdout      : 동결된 홀드아웃 시험지(종목 매니페스트) 로딩
"""
