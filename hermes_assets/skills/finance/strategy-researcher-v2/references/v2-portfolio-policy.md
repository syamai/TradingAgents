# v2 백테스트 포트폴리오 결합 정책

## 배경

한 세션에서 사용자가 한국 개별주 포트폴리오 원칙을 업로드했고, v2 백테스트 엔진에 단일 종목 100% 노출 방식 대신 현실적인 포트폴리오 결합 규칙을 반영했다. 이후 v2 전략 결과를 해석할 때 이 정책을 전제로 삼아야 한다.

## 현재 적용된 핵심 규칙

- 한국 개별주 기본 주식 비중: 90%
- 현금 비중: 10%
- 목표 보유 종목 수: 30개
- 단일 종목 기본 비중: 90% / 30 = 3%
- 단일 종목 최대 비중: 5%
- active 종목 수가 부족하면 남는 비중은 현금 0% 수익률로 처리
- active 종목 수가 많아도 총 주식 노출은 90%를 넘기지 않음

## 해석상 주의

기존 active 종목 단순 동일가중 방식에서는 active 종목이 1개뿐이면 해당 종목에 100% 투자한 것처럼 포트폴리오 수익률과 MDD가 계산됐다. 새 정책에서는 active 1종목이면 기본 3%만 노출되고 나머지는 현금이다.

예시:

```text
active 1종목, 해당 종목 -8%
→ 포트폴리오 손실 약 -0.24% (3% × -8%)
→ 나머지 97%는 현금
```

active 30종목이 모두 -8%면:

```text
30개 × 3% = 주식 90%
→ 포트폴리오 손실 -7.2%
```

## 코드 적용 지점

- `tradingagents/hermes/backtest_engine.py`
  - `KOREA_STOCK_PORTFOLIO_POLICY`
  - `_combine_korea_stock_portfolio()`
- `tradingagents/hermes/backtest_engine_v2.py`
  - v2 종목분할 백테스트 결합 방식
- `tradingagents/hermes/strategy_validation.py`
  - time split 검증 결합 방식
- `tradingagents/hermes/mcp_server_v2.py`
  - MCP 응답에 `portfolio_policy` 포함

검증 예시:

```bash
uv run pytest tests/hermes/test_backtest_engine.py tests/hermes/test_strategy_v2.py -q
```

## 보고 방식

v2 결과를 보고할 때는 가능하면 `portfolio_policy`가 적용됐는지 언급한다. 특히 과거 결과와 MDD가 크게 달라졌다면, 전략 자체가 개선된 것이 아니라 포트폴리오 결합 정책이 바뀐 영향일 수 있음을 구분해서 설명한다.

## 아직 미반영된 포트폴리오 제약

아래 제약은 종목 메타데이터가 필요해서 아직 엔진에 일반 적용되지 않았다.

- 업종별 25% 상한
- 코스닥 20~30% 제한
- 테마/적자 성장주 5~10% 제한
- 시총 구간별 비중
- 관리종목/투자경고 제외

필요 메타 예시:

```text
ticker
market: KOSPI / KOSDAQ
sector_group
market_cap_bucket
is_theme_or_high_risk
is_warning_or_managed
```

향후 이 메타가 붙으면 업종/시장/시총 제한까지 포트폴리오 결합기에 추가한다.
