# v2 포트폴리오 재랭크 워크플로우

## 언제 쓰나
사용자가 "가장 좋은 결과 N개", "최상위 결과를 포트폴리오까지 다시", "상위 후보 재검증"처럼 기존 `strategies_v2.db` 저장 전략을 현재 엔진/포트폴리오 정책으로 다시 비교하라고 할 때 사용한다.

## 핵심 원칙
- 먼저 현재 포트폴리오 결합 정책이 적용되는지 확인한다. 현재 정책은 `stock_weight=0.9`, `cash_weight=0.1`, `target_positions=30`, `max_single_weight=0.05`.
- 저장 DB의 과거 메트릭은 후보 선별용으로만 쓰고, 최종 보고 숫자는 현재 엔진으로 재백테스트한 값만 인용한다.
- 랭킹은 한 구간만 좋은 전략을 피하기 위해 4-way robust 기준을 우선한다:
  - `min(xsec_in_sharpe, xsec_out_sharpe, time_in_sharpe, time_out_sharpe)` 내림차순
  - tie-break: 4-way 평균 Sharpe, 최악 MDD, time-out Sharpe
- `gate=False`라도 상위 후보를 보고할 수 있지만, "통과 전략 없음"을 먼저 명시한다.
- 거래 수가 매우 적은 후보는 MDD/Sharpe가 좋아 보여도 표본 부족으로 별도 주석을 단다.

## 빠른 실행 패턴
세션에서 만든 재랭크 스크립트 예시 경로:

```bash
cd /Users/selab/Source/trading-ai
RERANK_LIMIT=30 uv run python scripts/rerank_v2_portfolio_top30.py
```

- `RERANK_LIMIT=30`: 저장 DB의 기존 4-way Sharpe 기준 상위 30개만 빠르게 재검증한다.
- `RERANK_LIMIT=0`: 전체 저장 전략을 재검증한다. 오래 걸리므로 background + notify_on_complete로 실행한다.

## 산출물 패턴
스크립트는 다음 디렉터리에 결과를 남긴다.

```text
artifacts/strategy_v2_portfolio_rerank/
  top30_portfolio_<UTC>.csv
  top30_portfolio_<UTC>.json
  all_portfolio_<UTC>.jsonl
```

보고 시 CSV를 `MEDIA:/absolute/path.csv`로 첨부하고, JSON 원본도 함께 첨부하면 좋다.

## 보고 템플릿
- 포트폴리오 정책 요약
- 후보/Universe/랭킹 기준
- 최종 gate 통과 수
- Top N 핵심 지표:
  - id/name
  - 4-way 최소 Sharpe
  - 평균 Sharpe
  - 최악 MDD
  - xsec in/out Sharpe·MDD·trades
  - time in/out Sharpe·MDD·trades
- 해석:
  - 최근 time-out만 좋은 전략인지
  - 과거 time-in이 약한지
  - MDD 병목인지
  - 거래 수 부족인지

## 주의
- 포트폴리오 정책 반영 후 MDD 개선은 전략 로직 개선이 아니라 결합 방식 변경 때문일 수 있다.
- 전체 1000개+ 재검증은 오래 걸린다. 대화형 응답에는 먼저 top-30 seed 재검증을 반환하고, 전체 재랭크는 background로 걸어 완료 알림을 받는 방식이 안전하다.
