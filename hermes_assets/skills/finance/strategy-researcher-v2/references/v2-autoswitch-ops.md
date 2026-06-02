# v2 autoswitch ops

Operational notes for the cron-driven `trading-v2-autoswitch` loop that alternates between LLM and LLM-free execution based on Codex usage/quota.

## 1) Dual-mode behavior

The autoswitch tick script is designed so that:
- **below 80% primary usage** → run an LLM batch via Hermes + `strategy-researcher-v2`
- **at or above 80% primary usage** → run the deterministic **LLM-free** engine (`python -m tradingagents.hermes.strategy_research_v2`)

When a user asks "80%부터는 LLM 없이 도는 거지?", do not answer from assumption alone if verification is practical. Confirm against the active script/cron wiring and, if needed, a recent tick log or process evidence.

## 2) Verify the loop by evidence, not design intent

For this loop, strong live evidence is:
- cron job exists and is scheduled (`trading-v2-autoswitch`)
- recent `last_run_at` / `last_status=ok`
- DB count increased after a forced tick
- `hermes_v2_tick_detail.log` shows which mode actually ran
- transient processes match the selected mode:
  - LLM mode: `hermes -z ... -s strategy-researcher-v2` + `mcp_server_v2`
  - LLM-free mode: `uv run python -m tradingagents.hermes.strategy_research_v2`

Do not treat an old top-level summary log alone as proof of current behavior if it may not have been refreshed yet.

## 3) Practical pitfall: summary log can lag while the detail log and DB already moved

A forced cron run may succeed and add new strategies before `hermes_v2_tick.log` is re-read or refreshed in your session. If the summary log still shows an old `stop_target_reached` message, verify with:
- current strategy count / max id in `strategies_v2.db`
- running process list
- `hermes_v2_tick_detail.log`

Use the DB delta as the decisive proof that the tick actually advanced.

## 4) Post-400 search policy

After a large number of evaluations (roughly 400+), repeating the classic `price_drop + trend_slope(foreign_registered)` contrarian family tends to rediscover the same shape:
- win/sharpe can pass
- MDD remains stuck around `-29% ~ -39%`
- exit-only tweaks rarely solve the gate

Better follow-on direction:
- prioritize non-FR families and underexplored subjects
  - `investment_trust`
  - `insurance`
  - `bank`
  - `private_equity`
  - `foreign_unregistered`
- prefer market-regime / flow / rotation families before legacy dip-only repetition
- if editing the deterministic generator, back up the file first with a timestamped `.bak-YYYYmmdd-HHMMSS`

## 5) LLM prompt steering for autoswitch batches

When the LLM branch is still active, explicitly steer away from FR-contrarian exit-only mutations. Good instruction pattern:
- remind it that the current bottleneck is MDD, not win/sharpe
- say the best FR near-miss family is already saturated
- request new entry families centered on `private_equity`, `investment_trust`, `insurance`, `bank`, or `foreign_unregistered`
- prefer `rolling_corr + price_filter + milder price_drop` or other entry-risk filters over more exit-only local tweaks

## 6) Batch sizing and timeout discipline

The autoswitch cron may have a script timeout (observed pattern: 120s). Increasing `V2_LLM_BATCH` improves throughput when quota is plentiful, but large batches can partially save strategies and then exit with a timeout/error. Treat this as a control-plane problem, not necessarily research failure:
- verify DB count/max id to see whether strategies were saved despite cron `error`
- if `last_status=error` repeats after increasing the batch, reduce `V2_LLM_BATCH` or raise the cron/script timeout if supported
- keep status-line messaging honest: report `상태 오류` when the cron timed out, but also show current DB progress
- when extending the target (e.g. +200 strategies), update both the autoswitch target and any status-line target source; prefer reading `V2_TARGET` in helper scripts instead of hardcoding `800`

## 7) Notification hygiene — stop conditions must not spam Telegram

A `stop_target_reached` or `stop_already_passed` tick is a terminal/milestone state, not a reason to notify every schedule interval. If the cron remains enabled after the target is reached, an "important" filter that matches `status: stop_` will send the same Telegram message repeatedly. Avoid this pattern.

Preferred behavior:
- On target/pass reached, **send at most one final notification**, then pause/remove the autoswitch cron or write a durable "final sent" sentinel so later ticks stay silent.
- If the cron job uses `deliver: local` but the script calls Telegram Bot API directly, remember that `local` does **not** suppress those direct messages; silence must be enforced inside the script or by pausing the job.
- Routine status-line crons should be opt-in and low frequency; default to quiet milestone-only notifications unless the user explicitly asks for a heartbeat.
- If the user asks "왜 이렇게 자주 보내?" or shows annoyance about repeated trading-v2 notifications, immediately pause both the autoswitch and status-line notification crons first, then explain the cause briefly.
- When the user asks for completion ETA, verify whether the background process still exists and whether output files are still updating before estimating. Do not imply a test is still running from stale process IDs or old partial files.
