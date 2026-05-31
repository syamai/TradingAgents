#!/bin/bash
# trading-ai v2 자율 연구 — 자동전환 단발 tick (Hermes cron no_agent script).
#
# cron 이 30분 반복을 담당하므로 이 스크립트는 1 tick 만 수행하고 종료한다.
# Codex 실시간 사용률(primary 5h)을 보고 모드를 자동 전환:
#   used <  80%  → LLM 배치(Hermes gpt-5.5, strategy-researcher-v2)
#   used >= 80%  → LLM-free 결정론 루프(python -m strategy_research_v2, 토큰 0)
# stdout 은 local cron output 파일로 남기고, 요약은 Telegram Bot API 로 직접 전송한다.
# (gateway shutdown 타이밍 delivery error 우회)
#
# 종료조건: gate_passed>=1 OR total>=TARGET(기본 1004) → 보고만 하고 추가 평가 안 함.
# 단일 writer 보장: lock 파일로 이전 tick 미완 시 skip(좀비 재spawn 방지).
set -uo pipefail
export PATH="/Users/selab/.hermes/hermes-agent/venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
set -a
source "$HOME/.hermes/.env" 2>/dev/null || true
source /Users/selab/Source/trading-ai/.env 2>/dev/null || true
set +a
cd /Users/selab/Source/trading-ai 2>/dev/null || exit 0

DB="$HOME/.tradingagents/hermes/strategies_v2.db"
QUOTA="$HOME/.tradingagents/hermes/hermes_quota.py"
SUMMARY_LOG="$HOME/.tradingagents/hermes/hermes_v2_tick.log"
DETAIL_LOG="$HOME/.tradingagents/hermes/hermes_v2_tick_detail.log"
LOCK="$HOME/.tradingagents/hermes/.autoswitch.lock"
TARGET="${V2_TARGET:-1004}"
# cron 120초 제한 여유 확보용 기본값. 더 많이 돌릴 땐 env 로 override.
LLM_BATCH="${V2_LLM_BATCH:-7}"
GATEWAY_PID="$PPID"
TELEGRAM_TARGET_CHAT="${V2_TELEGRAM_CHAT_ID:-339701919}"
TELEGRAM_TARGET_THREAD="${V2_TELEGRAM_THREAD_ID:-${TELEGRAM_CRON_THREAD_ID:-}}"

now(){ date '+%Y-%m-%d %H:%M:%S %Z'; }
count(){ sqlite3 "$DB" "SELECT COUNT(*) FROM strategies;" 2>/dev/null || echo 0; }
passes(){ sqlite3 "$DB" "SELECT COALESCE(SUM(gate_passed),0) FROM strategies;" 2>/dev/null || echo 0; }
maxid(){ sqlite3 "$DB" "SELECT COALESCE(MAX(id),0) FROM strategies;" 2>/dev/null || echo 0; }
send_telegram_summary(){
  local text="$1"
  [ -n "${TELEGRAM_BOT_TOKEN:-}" ] || return 0
  [ -n "${TELEGRAM_TARGET_CHAT:-}" ] || return 0
  [ -n "$text" ] || return 0
  local truncated="$text"
  if [ "${#truncated}" -gt 3800 ]; then
    truncated="${truncated:0:3800}
...(truncated)"
  fi
  local api="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage"
  if [ -n "${TELEGRAM_TARGET_THREAD:-}" ]; then
    curl -fsS -X POST "$api" \
      --data-urlencode "chat_id=${TELEGRAM_TARGET_CHAT}" \
      --data-urlencode "message_thread_id=${TELEGRAM_TARGET_THREAD}" \
      --data-urlencode "text=${truncated}" \
      >/dev/null || true
  else
    curl -fsS -X POST "$api" \
      --data-urlencode "chat_id=${TELEGRAM_TARGET_CHAT}" \
      --data-urlencode "text=${truncated}" \
      >/dev/null || true
  fi
}
report_and_maybe_send(){
  local payload notify_mode
  payload="$(cat)"
  printf '%s\n' "$payload" | tee "$SUMMARY_LOG"

  notify_mode="${V2_AUTOSWITCH_NOTIFY:-important}"
  case "$notify_mode" in
    always)
      send_telegram_summary "$payload"
      ;;
    never)
      ;;
    important|*)
      if printf '%s' "$payload" | grep -Eq 'status: failed|status: stop_|gate winners:'; then
        send_telegram_summary "$payload"
      fi
      ;;
  esac
}
top3(){
  sqlite3 -noheader -separator ' | ' "$DB" "SELECT substr(name,1,30), 'win '||round(in_win_rate,2), 'shp '||round(in_sharpe,2), 'mdd '||round(in_mdd,1), 'n'||in_n_trades FROM strategies WHERE in_n_trades>=50 ORDER BY in_sharpe DESC LIMIT 3;" 2>/dev/null
}
new_rows(){
  sqlite3 -noheader -separator ' | ' "$DB" "SELECT id, substr(name,1,60), 'in_shp '||round(in_sharpe,2), 'out_shp '||round(out_sharpe,2), 'mdd '||round(out_mdd,1) FROM strategies WHERE id > $1 ORDER BY id;" 2>/dev/null
}
gate_rows(){
  sqlite3 -noheader -separator ' | ' "$DB" "SELECT name, 'in win '||round(in_win_rate,2)||' shp '||round(in_sharpe,2)||' mdd '||round(in_mdd,1), 'out win '||round(out_win_rate,2)||' shp '||round(out_sharpe,2) FROM strategies WHERE gate_passed=1 LIMIT 3;" 2>/dev/null
}

# gateway 가 띄운 이전/현재 tick의 v2 MCP 서버만 정리한다.
# 현재 대화 세션(다른 PPID)에서 쓰는 mcp_server_v2 는 건드리지 않는다.
gateway_owned_v2_pids(){
  ps -axo pid=,ppid=,command= | awk -v gw="$GATEWAY_PID" '
    index($0, "mcp_server_v2") {
      pid=$1; ppid=$2; p[pid]=ppid
    }
    END {
      for (pid in p) if (p[pid] == gw) mark[pid]=1
      changed=1
      while (changed) {
        changed=0
        for (pid in p) if (!mark[pid] && mark[p[pid]]) { mark[pid]=1; changed=1 }
      }
      for (pid in mark) print pid
    }'
}

cleanup_gateway_v2(){
  pids="$(gateway_owned_v2_pids | tr '\n' ' ' | xargs 2>/dev/null || true)"
  [ -z "$pids" ] && return 0
  kill $pids 2>/dev/null || true
  sleep 1
  remaining="$(gateway_owned_v2_pids | tr '\n' ' ' | xargs 2>/dev/null || true)"
  [ -z "$remaining" ] || kill -9 $remaining 2>/dev/null || true
}

# --- 단일 writer lock (이전 tick 실행 중이면 skip) ---
if [ -f "$LOCK" ]; then
  pid=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "(이전 tick PID $pid 실행 중 — 이번 tick skip)"
    exit 0
  fi
fi
echo $$ > "$LOCK"
# 스케줄러가 타임아웃 시 그룹에 SIGTERM 을 보내므로 TERM/INT 에도 정리가 돌아야
# lock 제거 + MCP 회수가 실행된다(예전엔 EXIT 만 → SIGKILL 에 우회돼 stale lock).
# _cleanup_once 재진입 가드로 EXIT 와 중복 실행 방지(cleanup_gateway_v2 는 멱등).
_cleaned=0
_cleanup_once(){ [ "$_cleaned" = 1 ] && return 0; _cleaned=1; rm -f "$LOCK"; cleanup_gateway_v2; }
trap '_cleanup_once' EXIT
trap '_cleanup_once; exit 143' TERM
trap '_cleanup_once; exit 130' INT

CNT=$(count); PASS=$(passes); MAX_BEFORE=$(maxid)
START_TS="$(now)"
START_EPOCH=$(date +%s)
cleanup_gateway_v2

# --- 종료조건 1: gate 통과 ---
if [ "${PASS:-0}" -ge 1 ]; then
  {
    echo "[trading-ai v2 autoswitch] $(now)"
    echo "status: stop_already_passed"
    echo "progress: eval=$CNT/$TARGET pass=$PASS"
    echo "gate winners:"; gate_rows
  } | report_and_maybe_send
  exit 0
fi

# --- 종료조건 2: target 도달 ---
if [ "${CNT:-0}" -ge "$TARGET" ]; then
  {
    echo "[trading-ai v2 autoswitch] $(now)"
    echo "status: stop_target_reached"
    echo "progress: eval=$CNT/$TARGET pass=0"
    echo "top3(in_sharpe, trades>=50):"; top3
  } | report_and_maybe_send
  exit 0
fi

# --- 자동전환 판정 ---
Q=$(python3 "$QUOTA" 2>/dev/null)
MODE=$(echo "$Q" | awk '{print $1}')
PRI=$(echo "$Q" | awk '{print $2}')
SEC_OR_RESET=$(echo "$Q" | awk '{print $3}')
[ -z "$MODE" ] && MODE="available"
RUN_EXIT=0

if [ "$MODE" = "available" ]; then
  P="strategy-researcher-v2 스킬 사용. 서로 다른 새 v2 전략 ${LLM_BATCH}개만 평가/저장하고 종료. 반드시 도구를 실제 호출(서술/계획 금지).
1) list_strategies_v2(brief=true) 1회로 현황 확인(중복 회피).
2) 매 전략: backtest_strategy_v2(spec) → 즉시 save_strategy_v2(spec, name=...). 이를 ${LLM_BATCH}회(서로 다른 spec).
3) 현재 804개+/통과 0개에서 핵심 병목은 MDD다. 기존 최상위 near-miss 는 대체로 price_drop + trend_slope(foreign_registered,60,up) 계열이며 win/sharpe 는 통과하지만 MDD 가 대개 -29%~-39%에 머문다. 같은 FR contrarian 가족을 exit 만 바꿔 반복하지 말 것.
4) 이번 배치는 entry family 를 바꿔라: private_equity / investment_trust / insurance / bank / foreign_unregistered 중심, 또는 rolling_corr + price_filter + milder price_drop 의 3요소 조합 우선.
5) 리스크 억제 우선: stop_loss 3~5, take_profit 5~8, max_hold 5~10, price_drop 3~5 선호. above_ma5 는 out 붕괴 이력이 있으니 피하고, 필요하면 above_ma60 또는 rolling_corr>=0.2 같은 더 견고한 필터를 우선.
6) gate_passed=true 면 즉시 상세 보고.
메트릭은 도구 반환값만 인용(직접 계산 금지)."
  hermes -z "$P" -s strategy-researcher-v2 --provider openai-codex -m gpt-5.5 --yolo > "$DETAIL_LOG" 2>&1 || RUN_EXIT=$?
  MODE_LABEL="LLM(Codex gpt-5.5)"
  QUOTA_LABEL="primary=${PRI:-?}% secondary=${SEC_OR_RESET:-?}%"
else
  # cron 120s 제한 → 한 tick 당 소량(V2_LLMFREE_MAX, 기본 8)만 평가해 timeout 회피.
  # preload(199종목)≈18s + 8 eval≈9s ≈ 27s. 누적 진행은 다음 tick 들이 이어감.
  LLMFREE_MAX="${V2_LLMFREE_MAX:-8}"
  V2_LLMFREE_MAX="$LLMFREE_MAX" uv run python -m tradingagents.hermes.strategy_research_v2 > "$DETAIL_LOG" 2>&1 || RUN_EXIT=$?
  MODE_LABEL="LLM-free(max ${LLMFREE_MAX}/tick)"
  QUOTA_LABEL="primary=${PRI:-?}% reset_in=${SEC_OR_RESET:-?}s"
fi

cleanup_gateway_v2

AFTER=$(count)
APASS=$(passes)
MAX_AFTER=$(maxid)
ADDED=$((AFTER - CNT))
DURATION=$(( $(date +%s) - START_EPOCH ))
if [ "$RUN_EXIT" -eq 0 ]; then
  STATUS_LABEL="ok"
else
  STATUS_LABEL="failed"
fi

{
  echo "[trading-ai v2 autoswitch] $START_TS -> $(now)"
  echo "status: $STATUS_LABEL"
  echo "mode: $MODE_LABEL"
  echo "quota: $QUOTA_LABEL"
  echo "duration_sec: $DURATION"
  echo "progress: eval=$AFTER/$TARGET pass=$APASS (delta=+$ADDED)"

  if [ "$ADDED" -gt 0 ]; then
    echo "new strategies:"
    new_rows "$MAX_BEFORE"
  else
    echo "new strategies: none"
  fi

  if [ "${APASS:-0}" -ge 1 ]; then
    echo "gate winners:"; gate_rows
  else
    echo "top3(in_sharpe, trades>=50):"; top3
  fi

  if [ "$RUN_EXIT" -ne 0 ]; then
    echo "detail tail (last 40 lines):"
    tail -n 40 "$DETAIL_LOG" 2>/dev/null || true
  else
    echo "detail log: $DETAIL_LOG"
  fi
} | report_and_maybe_send

if [ "$RUN_EXIT" -ne 0 ]; then
  exit "$RUN_EXIT"
fi

exit 0
