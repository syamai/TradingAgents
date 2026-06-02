# v2 manual comparison patterns

Session-derived notes for user-facing comparison after autonomous or semi-manual v2 research.

## 1) Validate manual mutations against the engine grid before presenting them as executable

Common pitfall:
- Human-proposed value feels local/obvious but is invalid for the engine.
- Example: `price_drop.value=9` is invalid because allowed grid is `{3,5,8,10,15}`.

Preferred handling:
1. Keep the human-readable intent (`8%보다 조금 더 깊은 눌림` 같은 설명) separate from the executable spec.
2. Present the nearest valid executable alternatives up front.
3. If the user says "다 해", do not send the invalid spec first and discover the error later; substitute explicitly.

## 2) Safe reporting split: generation vs explanation

Keep these separate:
- **Generation / mutation logic**: use in-sample only.
- **Explanation / ranking for the user**: it is acceptable to compare saved candidates with out-sample Sharpe, MDD, win rate, and trade count.

This preserves holdout discipline while still answering user questions like:
- "이번 루프에서 가장 좋은 3개 알려줘"
- "왜 #180이 제일 좋아?"
- "주변 3개 변형 중 뭐가 제일 낫지?"

## 3) Interpreting local variants around a strong candidate

Observed neighborhood around candidate `#180 diptr-fr-d10_8-ts60-tp8sl8h20`:
- Tightening stop loss from 8 to 5 hurt the strategy materially.
- Shortening `max_hold_days` from 20 to 10 improved risk-adjusted performance.
- Increasing `take_profit_pct` from 8 to 10 improved Sharpe and slightly improved MDD, but lowered win rate more than the shorter-hold variant.

Useful pattern:
- Around contrarian + trend-filter hybrids, **shorter hold / modestly wider profit capture** may help more than tighter stops.
- Do not generalize as a rule; treat as a local exploration heuristic around similar specs.

## 4) Suggested wording for user-facing comparison

- "현재 1위": best balance among Sharpe, MDD, and trade count
- "샤프 최고": highest out-sample Sharpe, even if win rate fell
- "전면 악화": worse on most key metrics vs baseline

Always mention the baseline candidate explicitly and compare deltas rather than absolute numbers alone.

## 5) Second-ring neighborhood around `#329` (`tp8 sl8 h10`)

Observed follow-up probes with the same entry family `price_drop(10,8) + trend_slope(foreign_registered,60,up)`:
- `tp10 sl8 h10`: out Sharpe `1.3061`, out MDD `-31.0355`, out win `0.5273`.
- `tp5 sl8 h10`: out win improved to `0.5779`, but out Sharpe fell to `1.1918` and out MDD worsened to `-32.4265`.
- `tp8 sl10 h10`: out win `0.5618`, out Sharpe `1.2533`, out MDD `-30.9935`.

Comparison vs baseline local winner `#329` (`out win 0.5516 / sharpe 1.4415 / mdd -30.8475`):
- None beat `#329` on the combined balance of Sharpe + MDD.
- Tightening or loosening exits alone did not solve the remaining bottleneck; MDD stayed near `-31%`.
- Useful heuristic: once a local best already has shorter hold, the next meaningful search step is often **entry-risk filtering** (e.g. `price_filter above_ma`, `rolling_corr`, or milder `price_drop`) rather than more exit-only tweaks.

## 6) Entry-risk filtering around the same FR contrarian family can improve optics without fixing the bottleneck

Observed follow-up around `price_drop + trend_slope(foreign_registered)`:
- Milder dips (`10d/5%`, `5d/5%`) improved in-sample balance somewhat versus `10d/8%`, but MDD still stayed around `-36%` to `-39%`.
- `rolling_corr(fr,60,0.2)` occasionally improved out-sample MDD versus baseline, but in-sample still missed the gate, so it should not drive mutation logic.
- `price_filter above_ma20` was too restrictive and often collapsed trade quality.
- `price_filter above_ma5` created a dangerous pattern: strong in-sample (`win 0.5281 / sharpe 1.2604`) but severe out-of-sample collapse (`sharpe 0.2396`, `mdd -52.2815`).

Useful takeaway:
- In this family, entry filtering alone may **polish** Sharpe/win rate but does not reliably solve the deep-drawdown structure.
- When MDD remains stuck far below `-20%` after milder dips + corr filters, the better next move is often **changing the subject family** (`private_equity`, `investment_trust`, etc.) rather than further polishing the same foreign-registered contrarian template.
