# v2 status-line monitoring

Compact monitoring pattern for the autonomous `strategy-researcher-v2` loop when the user wants lightweight Telegram progress without spending additional LLM quota.

## When to use
- The loop runs for hours/days and the user wants a terse heartbeat.
- You need a quota-safe status channel even when the main provider is near limit.
- The user asks questions like "지금 몇 개까지 갔어?" or wants progress every 30m.

## Pattern
Use a **script-only cron** (`no_agent=True`) that reads local state and prints a single status line. Deliver it to `origin` so the update appears in the active Telegram chat without invoking an LLM.

Recommended ingredients for the line:
- total evaluated / target batch
- gate-passed count
- current best near-miss (id + in/out MDD or other bottleneck metric)
- most recent strategy id/name fragment
- autoswitch cron last run time / status
- next scheduled run time

User-facing Telegram output should favor **readable Korean labels** over compact ops abbreviations. Avoid terse strings like `v2 508/800 | pass 0 | near ...` unless the user explicitly asks for ultra-compact output. Prefer multi-line Korean such as:

```text
전략 연구 v2 상태 요약
- 진행률: 820/1004개 평가 완료, 남은 184개
- 게이트 통과 전략: 0개
- 가장 아까운 후보: #482 ... — MDD in/out -29.0%/-31.5%, Sharpe 1.14/1.08
- 마지막 평가: #821 ... — 미통과, Sharpe in/out -0.43/0.13, MDD -82.1%/-79.7%
- 자동 연구 cron: 최근 실행 <timestamp>, 상태 정상/오류
- 다음 자동 연구 실행: <timestamp>
```

Keep the target synchronized with the autoswitch loop. If the autoswitch script uses `V2_TARGET` or has been extended beyond the previous target, do **not** hardcode an old value like `800` in the status-line script; read `V2_TARGET` or update both scripts together.

## Why this matters
- It keeps monitoring alive when usage-limit protection has already switched the main loop toward deterministic execution.
- It separates **control-plane observability** from the research loop itself.
- It gives fast human confirmation of progress without opening logs first.

## Verification
After adding the script-only status cron, verify:
- the cron job exists and is scheduled at the intended interval
- manual script execution prints the expected one-line summary
- at least one delivered message appears in the target chat
- the line updates when DB count or cron timing changes

## Design note
Prefer this script-only status path over an LLM-generated summary for routine heartbeats. Use richer LLM summaries only for milestone reports (e.g. new near-miss family, gate pass, code-update rationale).

## Anti-spam rule
Status-line monitoring is useful only when the user explicitly wants periodic heartbeats. The default for trading-v2 ops should be **quiet milestone-only** messaging, because repeated routine Telegram updates are disruptive.

- Do not create or leave enabled a 30-minute status cron unless the user asked for ongoing status pings.
- If the target is reached, pause/remove the status cron together with the autoswitch cron; do not keep sending "목표 초과 달성" summaries.
- If a script-only cron sends Telegram directly via Bot API, `deliver=local` does not prevent messages. Silence requires disabling the direct send path or pausing the cron.
- On user frustration about notification frequency, stop the notifications first, then explain.
