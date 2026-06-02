[수급-actors-외국인] 한국 종목 외국인 수급은 등록자(registered, 장기 자금) + 비등록자(unregistered, 단기 헤지펀드 비중 큼) 로 분리. 등록자 매수 = 장기 신뢰 신호, 비등록자 매수 = 단기 투기. 둘이 같은 방향이면 강한 신호, 괴리면 단기/장기 시각차. 가설 작성 시 외국인을 단일 entity 로 다루지 말 것 — 등록자/비등록자 분리 인용 의무.

§

[수급-actors-기관] KIS 기관 = 5 sub-주체. 연기금(pension) = 가장 안정·장기, 사모(private_equity) = 단기 알고리즘, 보험(insurance) = 장기 보유, 은행(bank) = 짧은 사이클, 투자신탁(investment_trust) = 펀드 흐름. "기관" 통합 분석은 신호 노이즈 큼 — sub-주체별로 분리해 봐야 차별성 드러남.

§

[수급-actors-개인] 개인(retail) 은 KOSPI/KOSDAQ 거래대금 50%+ 차지하는 비중 큰 주체. 단 등락 추세 예측력 약함 — 단기 모멘텀 추격 + 흡수 양면. retail 단독 매수가 강세 신호 아님 — 외국인·기관과 함께 봐야 의미. retail 매도 + 외국인 매수 동시 = 흡수자 패턴.

§

[수급-actors-자기주주] 자기주식(treasury_stock) 매수 = 회사 의지로 가격 방어. 6 개월 lockup 등 규제 영향. 빈도 낮지만 신호 강함. KIS 데이터에 별도 카테고리로 노출되며, 발표 후 첫 영업일 매수 강함 — 가설에 인용 시 "자기주식 매수 발표 + KIS 첫 영업일 net +X 만주" 구체화.

§

[수급-structure-시가총액-비중] 대형주(KOSPI 200) 는 외국인·기관 합산 비중 60~80% — 외국인 등록자 동향이 결정적. 중소형주는 개인·사모 비중 큼 — 모멘텀 추격 강함, 가격 변동성도 큼. 종목 시가총액에 따라 어떤 주체의 가설 가중치를 높일지 차등 — 대형주는 외국인등록, 중소형주는 사모·개인.

§

[수급-structure-주체-숫자] KIS investor API 는 9 주체 (foreign_registered, foreign_unregistered, foreign_combined, pension, private_equity, insurance, bank, investment_trust, retail) 노출. 거래소 일반 데이터(외국인/기관/개인 3 분류) 보다 세분화. 가설 작성 시 KIS 의 sub-주체 구분을 활용 — "기관 매수" 같은 통합 표현 피할 것.

§

[수급-structure-시간외] KIS 데이터는 정규장 + 시간외 단일 거래 합산. 시간외 대량 거래(외국인 블록딜 등)는 다음 영업일 net 에 반영 — 가격 갭(전일 종가 대비 시초가) 과 수급 시점 격차 발생 가능. "갭상승/하락 다음날 외국인 net 증감" 패턴 분석 시 이 시차 인지 필수.

§

[수급-patterns-사모-외국인비등록] 사모(private_equity) + 외국인비등록(foreign_unregistered) 동시 매수 = 단기 모멘텀 강함. 둘 다 단기 자금 + 알고리즘 비중 큼 → 스윙 호라이즌(2~4주) 에 가장 유의미. 5+ 영업일 지속되면 신호 신뢰도 ↑. 다만 두 주체 모두 빠른 차익 실현 가능 — 진입은 동조성 강할 때, 청산은 모멘텀 격차 발생 시.

§

[수급-patterns-등록자-단발-vs-지속] 외국인 등록 net_qty 단일 큰 매수(예: 하루 +50 만주) = 인덱스 리밸런싱 또는 ETF 유입 가능 (단발 — 추세 X). 5+ 영업일 연속 매수 + 가격 추세 동조 = 진짜 장기 매집 (추세 O). 가설 강도 차이 큼 — 단발 매수만 인용해 강세 가설 만들면 false positive.

§

[수급-patterns-흡수자] phase 가격 추세 ↑ 인데 특정 주체 net_qty < 0 (역행 매도) = "흡수자". phase 종료 시점에 흡수자 매도가 멈추면 다음 phase 시작(추세 전환 또는 가속) 가능성. compute_trend 의 ``absorbers`` 필드로 확인 — phase 별 절댓값 net_value 큰 역행 주체 자동 추출.

§

[수급-patterns-단주-vs-거래대금] net_qty(단주 수량) 와 net_value(거래대금) 의 격차 = 거래 단가 신호. 외국인 단주는 적은데 거래대금 큼 = 고가 매수 (확신·강한 의지). 단주는 큰데 거래대금 비례 적음 = 저가 분할 매수 (조용한 매집). 가설 인용 시 net_qty 만 보지 말고 net_value 도 함께 확인.

§

[수급-dynamics-60일-phase] trend_analysis 의 phase 분할 디폴트 윈도우 60 영업일 (~3개월). 60 일 미만 phase = noise 가능 — 인접 phase 와 병합해 재판단 검토. 적응 윈도우(adaptive_window) 사용 시 종목별 자동 조정 — compute_trend 출력의 ``slope_window_days`` 확인. 윈도우 작으면 빠른 신호·노이즈 큼, 크면 둔감·견고.

§

[수급-dynamics-동행성-임계값] 한 주체의 가격 phase 동행성 (agreement_pct): 100% 완전동조, 50% 무관, 0% 완전역행. |agreement-50| 가 클수록 추세 연관성 명확. ≥ 70% (강한 동조) 또는 ≤ 30% (강한 역행) 이면 스윙 가설 강 신호. 50% 근처(45~55%) 는 가격 추세와 무관 — 가설 만들지 말 것.

§

[수급-dynamics-모멘텀-격차] 외국인 등록 60일 모멘텀 vs 사모 60일 모멘텀 비교 = 장단기 자금 차이. 등록자 (+) + 사모 (-) = 장기 매집 + 단기 차익 실현 (스윙 강세 지속 가능성 높음). 등록자 (-) + 사모 (+) = 단기 추격 + 장기 이탈 (스윙 마지막 국면 가능성). 두 주체 부호 격차가 의미 있는 신호.

§

[수급-dynamics-phase-전환] cum_qty phase 가 down → up 으로 전환되는 시점이 스윙 진입 후보. 다만 첫 5~10 영업일은 false breakout 위험 — concurrent r 동조성 + 다른 주체 정렬도 함께 확인. compute_trend 의 phase 시작 시점 (start_date) 인용 시 "전환 후 N 영업일 경과 — 추세 검증 단계" 같은 시간 인식 표현.

§

[통계-한계-인과-vs-상관] r=0.7 강한 상관이라도 *인과 아님*. 두 시계열이 같은 외부 요인(KOSPI 트렌드, 금리, 환율) 에 동시에 반응한 spurious 상관일 수 있음. 가설에 "인과" "유발" 같은 단어 쓰지 말고 "동조" "역행" "선행 패턴" 같은 관찰 표현 사용.

§

[통계-한계-Granger] Granger 검정의 "예측력" = 시계열 선후행 패턴, 인과 아님. p<0.05 라도 X 가 Y 를 일으킨다는 증거 아님 — X 의 과거가 Y 예측에 도움 된다는 정도. compute_advanced 의 granger 결과 인용 시 "Granger 선행성" 같은 보수적 표현, "인과" 절대 금지.

§

[통계-한계-level-r-spurious] cum_qty(누적 보유) vs close(가격) 의 level correlation 은 둘 다 trending 이면 가짜로 높게 나옴 (random walk 두 개 r=0.6+ 흔함). 차분(net_qty / return) correlation 이 훨씬 신뢰. compute_correlation 의 ``level`` 섹션 r 만 인용해 강세 가설 만들면 false positive 위험.

§

[통계-한계-p-value-n] p-value 는 표본 크기에 강하게 의존 — n=50 의 p=0.04 와 n=1000 의 p=0.04 는 effect size 가 천지차. compute_advanced 인용 시 n_days 도 함께 인용 ("n=1224, p=0.001" 처럼). n_days < 50 이면 Granger/VAR/cointegration 신뢰 마진 낮음 — 가설 evidence 에 "n=NN 으로 신뢰 마진 낮음" 명시.

§

[데이터-제약-KIS-unavailable] analyst_supply_demand 또는 KIS API 출력에 "<unavailable: ...>" 마커 보이면 해당 신호는 가설로 만들지 말 것. 결손 인정 + "데이터 부재로 평가 불가" 명시. 추정으로 메우면 할루시네이션. 예: "공매도 데이터 unavailable → 약세 압력 평가 불가, 가설 만들지 않음" 명시.

§

[데이터-제약-신규상장-close0] 신규 상장 종목은 상장 전 데이터에 close=0 행 가능 (KIS 데이터 특성). compute_trend / compute_advanced 는 close=0 행 자동 제외. 외부 인용 시 IPO 후 6 개월 미만이면 phase 분석 신뢰 마진 낮음 표기. close=0 보이는 종목은 신중하게.

§

[데이터-제약-IPO-6개월] IPO 후 6 개월 미만 종목 (예: LG에너지솔루션 직후) 은 phase 분할·동행성 분석에서 신뢰 마진 낮음. 가설 confidence ↓ (0.5 이하 권장). 임시 흐름이 영구 패턴으로 오인될 위험. IPO 후 시간이 충분히 (12+ 개월) 지난 종목에 비해 가설 강도 차등.

§

[데이터-제약-휴장일-주체수] 한국 거래소 휴장일 (음력 명절, 대선일 등) 은 데이터 비어 있음 — labeling 시점이 휴장일이면 다음 거래일로 이동(LabelingScheduler 처리). KIS 9 주체 외 hedge fund 단독 카테고리 없음 — 외국인비등록에 합쳐짐. ETF/index fund 도 외국인등록에 통합 — 분리 분석 불가.

§

[출력-규칙-매매-금지] 종목 분석 결론에서 매매 추천 금지. "사야 한다 / 오를 것이다" 같은 단정 표현 안 됨. 보수적 hedge 표현 ("시사한다", "약한 증거", "~로 해석 가능", "가능성") 사용. 가격 도달 예측도 금지 — "X 원 도달" 표현 X. 분석가 역할이지 트레이더 아님.

§

[출력-규칙-raw-인용] 모든 주장은 raw 수치 인용 동반 ("외국인 등록 net_qty 60일 +120만주" "concurrent r=0.5417 (p<0.001, n=1224)" 처럼). 인용 없는 결론은 할루시네이션 의심 → confidence ↓. 데이터 결손도 "<unavailable: ...>" 명시. evidence_excerpts 비어 있는 가설 = 만들지 말 것 (schema 위반).

§

[출력-스키마] 종목 분석 최종 응답은 항상 다음 JSON 구조 (사람 요약 위 또는 아래에 코드 블록). 자유 텍스트만 출력하지 말 것 — 라벨링 인프라가 JSON 을 파싱 못 함.
```json
{
  "ticker": "<예: 005930.KS>",
  "as_of_date": "<YYYY-MM-DD>",
  "overall_stance": "<bullish | moderately_bullish | neutral | moderately_bearish | bearish>",
  "overall_confidence": <0.0~1.0>,
  "hypotheses": [
    {
      "id": "h1",
      "claim": "<짧은 한국어 주장>",
      "direction": "<bullish | bearish | neutral>",
      "confidence": <0.0~1.0>,
      "evidence_tools": ["<예: analyst_supply_demand, compute_correlation>"],
      "evidence_excerpts": ["<raw 수치 인용 1>", "<raw 수치 인용 2>"],
      "horizon_weeks": <2~8 정수>,
      "predicted_relative_return_pct": <KOSPI 대비 예상 수익률 %>
    }
  ]
}
```
각 hypothesis 는 evidence_excerpts 비어 있으면 안 됨. horizon_weeks 는 스윙 범위 2~8 만.

§

[학습-기준] 가설 적중 평가 = KOSPI 상대 수익률 기준 (절대 아닌). 호라이즌별 임계값: 2주 ±2%, 4주 ±3%, 8주 ±5%. direction=bullish & 상대 +임계값↑ = right. neutral & |상대| ≤ 임계값/2 = right. bearish & 상대 -임계값↓ = right. 그 외 = wrong. predicted_relative_return_pct 는 이 호라이즌·임계값에 맞춰 *합리적* 값으로 (지어내지 말 것 — 근거 약하면 confidence 낮추고 임계값 근처로).

§

[전략-시드-외국인등록-지속매수] strategy-researcher 의 1순위 후보. foreign_registered net_streak buy min_days 5~10 = 장기 자금 지속 유입. 대형주(외국인등록 비중 60~80%)에서 스윙 강세 베이스 시그널. pct_threshold(foreign_registered >= 30~40) 와 AND 결합 시 "지배적 주체의 지속 매집" 으로 신호 강도 ↑. 단발성 매수(min_days 2~3)는 노이즈 — 지속성이 핵심.

§

[전략-시드-사모-외국인비등록-동반] private_equity + foreign_unregistered 두 단기 자금이 동시 net_streak buy = 단기 모멘텀. entry.all_of 에 두 신호 AND. max_hold_days 10~20(2~4주) 짧게 — 둘 다 빠른 차익 실현 성향이라 보유 길면 모멘텀 소멸. 중소형주에서 특히 유의미.

§

[전략-시드-비중추세] pct_delta(window 10~20, op >=, value 5~10) = 특정 주체의 누적 영향력 비중이 상승 추세. "그 주체가 시장을 장악해 가는 중" 의 정량 표현. net_streak(부호) 보다 느리지만 추세 지속성 포착. 비중 하락(op <=)은 long 진입 신호로 부적합.

§

[전략-시드-추세동조필터] price_filter above_ma(window 20~60) 를 entry 에 AND 로 추가 = 하락장 진입 회피. 수급 신호가 좋아도 전체 추세가 꺾이면 승률 급락 — MEMORY 의 "phase 전환 false breakout" 위험. 하락 구간 진입을 거르면 보통 승률·샤프 동시 개선. long-only 전략의 거의 필수 보조 필터.

§

[전략-시드-거래량강도] net_vol_ratio(window 5~10, op >=, value 0.1~0.2) = net 매수가 거래량 대비 의미있는 규모. 절대 주수는 종목 간 비교 불가(시총 차이) — 거래량 대비 비율이 scale-free 라 종목군 일반화에 적합. "단주 vs 거래대금 격차" 시드의 정량판. 작은 net 매수(ratio 0.05 미만)는 신호로 약함.

§

[전략-시드-청산규칙] exit 는 max_hold_days(필수, 무한보유 방지) + stop_loss_pct(권장, 5~8% — 한국 변동성 고려) 조합이 기본. take_profit 는 모멘텀을 일찍 끊어 누적수익 낮출 수 있어 신중(없거나 10~20%). 수급 청산(exit.signal_all_of: 매수 주체의 net_streak sell)은 "진입 논리가 깨졌을 때" 청산 — 가격 손절과 상호보완.

§

[전략-시드-과최적화경계] in/out-sample 양쪽 게이트 통과가 채택 조건(승률≥60%·샤프≥1.2·MDD≥-20%·거래≥50). in 만 통과하면 곡선맞춤 의심 — out 메트릭으로 파라미터를 *재튜닝하지 말 것*(holdout 누수). 승률 in/out 격차 >10%p 면 폐기. 그리드 밖 값은 validate_spec 가 거부하므로 변이는 그리드 인접값으로만. 시도 30회 또는 통과 3개에서 종료.
