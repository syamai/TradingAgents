"""자가학습 컨텍스트 빌더 — 원장의 실패 기록을 다음 가설 생성의 입력으로.

루프의 '학습'은 LLM 가중치가 아니라 이 모듈이 담당한다: 지금까지 시도한
family·가설·탈락 사유를 요약해 생성 에이전트의 프롬프트에 주입함으로써
같은 가설의 재탕과 같은 실패 양식의 반복을 막는다.

고갈 선언: 최근 EXHAUSTION_K개 family가 연속으로 통과 없이 소진되면
"이 신호공간에 선택 알파 없음"을 정당한 종료로 선언한다 — 루프는 새 가설
생성을 멈추고 사람에게 보고해야 한다.

사용: uv run python -m tradingagents.newloop.lessons   # 마크다운 출력
"""

from __future__ import annotations

import json
import sqlite3

from tradingagents.newloop.ledger import LEDGER_DB, MAX_SUBMISSIONS, VARIANTS_MAX

EXHAUSTION_K = 5  # 연속 소진 family 수 — 도달 시 신규 생성 중단·보고


def _failure_mode(results: list[dict]) -> str:
    """family의 채점 결과들에서 지배적 탈락 양식을 분류한다."""
    modes = []
    for r in results:
        res = r.get("result", {})
        checks = res.get("checks", {})
        beta = res.get("beta")
        if res.get("status") == "insufficient":
            modes.append("표본/하락창 부족")
        elif checks.get("down_not_broken") is False and beta is not None and beta > 1.3:
            modes.append("베타 위장(고민감 종목 선택, 하락창 붕괴)")
        elif checks.get("alpha_significant") is False:
            modes.append("알파 비유의(시장 도움 제거 후 실력 없음)")
        elif checks.get("alpha_material") is False:
            modes.append("알파 비경제적(슬리피지 버퍼 이하)")
        else:
            modes.append("기타")
    # 최빈 양식
    return max(set(modes), key=modes.count) if modes else "기록 없음"


def build_lessons(db_path: str | None = None) -> dict:
    con = sqlite3.connect(f"file:{db_path or LEDGER_DB}?mode=ro", uri=True)
    fams = con.execute(
        "SELECT family_key, hypothesis, registered_at, submissions, max_submissions, status "
        "FROM families ORDER BY registered_at"
    ).fetchall()
    out_fams = []
    for key, hyp, reg, subs, mx, status in fams:
        rows = con.execute(
            "SELECT strategy_ref, status, result_json FROM gate_results WHERE family_key=?",
            (key,),
        ).fetchall()
        results = [{"strategy_ref": r[0], "status": r[1], "result": json.loads(r[2])} for r in rows]
        best = None
        for r in results:
            t = r["result"].get("t_alpha")
            if t is not None and (best is None or t > best["result"].get("t_alpha", -9e9)):
                best = r
        out_fams.append({
            "family_key": key, "hypothesis": hyp, "registered_at": reg,
            "submissions": f"{subs}/{mx}", "status": status,
            "n_specs_scored": len(results),
            "best_t_alpha": best["result"].get("t_alpha") if best else None,
            "best_alpha_pct": best["result"].get("alpha_pct") if best else None,
            "best_beta": best["result"].get("beta") if best else None,
            "failure_mode": _failure_mode(results) if status != "passed" else "통과",
        })

    # 고갈 판정 — 최근 K개 family가 전부 통과 없이 소진
    recent = [f for f in out_fams if f["status"] in ("exhausted", "passed")][-EXHAUSTION_K:]
    exhausted = (len(recent) >= EXHAUSTION_K
                 and all(f["status"] == "exhausted" for f in recent))

    total_specs = sum(f["n_specs_scored"] for f in out_fams)
    return {
        "families": out_fams,
        "exhausted": exhausted,
        "exhaustion_rule": f"최근 {EXHAUSTION_K}개 family 연속 무통과 소진 시 신규 생성 중단",
        "exam_wear": {"total_families": len(out_fams), "total_specs_scored": total_specs,
                      "note": "홀드아웃 시험지는 쓸수록 닳는다 — 채점 총량을 항상 보고"},
        "limits": {"max_submissions_per_family": MAX_SUBMISSIONS,
                   "max_variants_per_submission": VARIANTS_MAX},
    }


def to_markdown(lessons: dict) -> str:
    L = ["## newloop 교훈 (자가학습 컨텍스트)", ""]
    if lessons["exhausted"]:
        L += ["**⛔ 고갈 선언 상태 — 새 가설을 생성하지 말 것.** "
              f"({lessons['exhaustion_rule']})", ""]
    w = lessons["exam_wear"]
    L += [f"- 시험지 마모: family {w['total_families']}개, 누적 채점 spec {w['total_specs_scored']}개",
          f"- 한도: family당 제출 {lessons['limits']['max_submissions_per_family']}회, "
          f"제출당 변형 {lessons['limits']['max_variants_per_submission']}개", ""]
    if lessons["families"]:
        L += ["| family | 가설 | 제출 | 상태 | best t(α) | best β | 탈락 양식 |",
              "|---|---|---|---|---|---|---|"]
        for f in lessons["families"]:
            L.append(f"| {f['family_key']} | {f['hypothesis'][:40]} | {f['submissions']} "
                     f"| {f['status']} | {f['best_t_alpha']} | {f['best_beta']} | {f['failure_mode']} |")
    else:
        L.append("(아직 시도된 family 없음)")
    L += ["", "### 생성 규칙 (요약)",
          "- 위 family들과 **구조적으로 다른** 가설만 — 같은 신호 조합의 파라미터 재탕 금지",
          "- '베타 위장' 양식이 반복되면: 시장 민감도를 키우는 방향(돌파·고변동 선호)을 피하고 "
          "시장중립적 구성(상대 강도, 페어성, 수급 괴리)을 우선",
          "- 가설에는 경제적 근거 한 줄 필수 — 누가 왜 돈을 잃어주는가",
          "- 통과(passed) family가 있으면 그 *인접* 구조 탐색이 우선"]
    return "\n".join(L)


if __name__ == "__main__":
    print(to_markdown(build_lessons()))
