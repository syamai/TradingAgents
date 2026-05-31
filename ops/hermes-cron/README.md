# hermes autoswitch cron — 타임아웃 supervision 패치 보관

`~/.hermes`(NousResearch hermes-agent + 스크립트 + config)에 적용한 cron 타임아웃
정리 버그 수정의 **버전관리 사본 + 재적용 절차**다. trading-ai 레포와는 별개
코드베이스(hermes-agent)이므로 git 브랜치로 합칠 수 없어 patch/사본으로 보관한다.

> ⚠ 이건 **백업/문서**다. 실제 동작하는 fix 는 `~/.hermes` 에 이미 적용·실행 중이며,
> 이 디렉터리와 **자동 연동되지 않는다**. `hermes update` 등으로 ~/.hermes 가 갱신돼
> fix 가 사라지면 아래 절차로 수동 재적용한다.

## 배경 버그 (supervision)
hermes-agent 의 cron 스케줄러(`cron/scheduler.py` `_run_job_script`)가 잡 스크립트를
`subprocess.run(timeout=)` 로 돌렸는데, 타임아웃 시 **직속 bash 자식만 SIGKILL** 했다.
그 결과 스크립트가 띄운 worker/MCP 서버가 **orphan(PPID=1)으로 살아남아 DB 를 계속
쓰고**, uncatchable SIGKILL 이 스크립트의 EXIT trap 을 우회해 **stale lock**(죽은 PID)이
남았다. autoswitch tick(`trading-v2-autoswitch`, every 10m, no_agent)이 BATCH=7 로
120s 를 계속 초과해 매 tick `last_status=error` 인데 전략은 증가하는 현상으로 표출.

## 변경 3가지
| 파일(이 디렉터리) | 실제 위치 | 내용 |
|---|---|---|
| `0001-fix-cron-...process-group-no.patch` | `~/.hermes/hermes-agent/cron/scheduler.py` | `subprocess.run` → `Popen(start_new_session=True)` + 타임아웃 시 그룹 **SIGTERM→10s grace→SIGKILL**(`_terminate_process_group`). timeout 값·캡처·redact·win32 경로 보존. |
| `hermes_v2_autoswitch_tick.sh` | `~/.hermes/scripts/hermes_v2_autoswitch_tick.sh` | trap `EXIT` → `EXIT/TERM/INT` + 재진입 가드(`_cleanup_once`). SIGTERM 에도 lock 제거·`cleanup_gateway_v2` 실행. |
| `config.cron.snippet.yaml` | `~/.hermes/config.yaml` `cron:` 블록 | `script_timeout_seconds: 540` 추가. |

근거 커밋: hermes-agent `acd9aaa0c`(base `769ee86cd`). 패치는 `git am` 으로 재적용 가능.

## 타임아웃 값(540s) 측정 근거
600s 임시 한도에서 BATCH=7 완주 tick 실측: **66 / 234 / 274 / 421 s**(편차 큼, 최대 421s).
단순 ×1.5(=631s)는 10분 주기(600s)를 넘어 부적절 → **관측 최대 × ~1.3 = 540s** 로 확정
(주기 미만이라 다음 tick 과 겹침 없음, 초과 시엔 위 패치로 graceful clean-kill).

## 검증 결과(적용 후)
- 타임아웃 tick 에서 stale lock·orphan **모두 사라짐**(라이브 확인).
- 540s 적용 후 tick `status=ok`(예: 296s 완주), `last_error=None`.

## 재적용 절차 (~/.hermes 갱신으로 fix 유실 시)
```bash
# 1) scheduler.py 패치 (충돌 시 수동 해결 — 아래 '한계' 참고)
cd ~/.hermes/hermes-agent
git am < /Users/selab/Source/trading-ai/ops/hermes-cron/0001-fix-cron-*.patch
#   (또는: git apply <patch>  # 커밋 없이 워킹트리만)

# 2) tick 스크립트
cp /Users/selab/Source/trading-ai/ops/hermes-cron/hermes_v2_autoswitch_tick.sh \
   ~/.hermes/scripts/hermes_v2_autoswitch_tick.sh

# 3) config.yaml cron: 블록에 script_timeout_seconds: 540 추가
#    (config.cron.snippet.yaml 참고 — 전체 config 는 시크릿 포함이라 미보관)

# 4) 게이트웨이 reload
launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway

# 5) 확인
~/.hermes/hermes-agent/venv/bin/python -c "import sys;sys.path.insert(0,'/Users/selab/.hermes/hermes-agent');import importlib;print(importlib.import_module('cron.scheduler')._get_script_timeout())"  # 540
```

## 한계
- **자동 연동 아님** — 위 절차로 수동 재적용해야 함.
- scheduler 패치는 base `769ee86cd` 기준. 현재 upstream main 이 **312 커밋 앞**이라 그쪽
  `scheduler.py` 가 바뀌었으면 `git am` 충돌 가능 → 수동 병합 필요.
- 근본 유지는 **NousResearch/hermes-agent 업스트림 PR** 이 정석(현재 미제출). PR 거절돼도
  이 로컬 패치/실행 fix 에는 영향 없음.
