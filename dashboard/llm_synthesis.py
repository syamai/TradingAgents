"""12 분석 섹션 해석을 통합해 최종 종합 의견을 만드는 LLM 호출.

로컬 LLM(Ollama 등 OpenAI 호환 API) 사용. 디폴트는 ``gemma4:26b-a4b``.
환경변수 ``DASHBOARD_SYNTHESIS_PROVIDER`` / ``DASHBOARD_SYNTHESIS_MODEL`` 로
override.

규칙 기반 ``interpretation`` 모듈이 만든 12 줄 해석 + 종목 기본 정보를
LLM 에 컨텍스트로 주고 자연어 종합. LLM 은 패턴 통합·맥락 해석에 강점,
규칙 기반은 통계 정확성 보장 — 둘을 분리해 LLM 환각 영향 줄임.

매매 추천은 시스템 프롬프트에서 금지. 통계 한계(인과 ≠ 상관 등)를
명시하도록 지시.
"""
from __future__ import annotations

import os
import textwrap
from typing import Optional

from dashboard import interpretation as itp
from tradingagents.llm_clients.factory import create_llm_client

DEFAULT_PROVIDER = os.environ.get("DASHBOARD_SYNTHESIS_PROVIDER", "ollama")
DEFAULT_MODEL = os.environ.get("DASHBOARD_SYNTHESIS_MODEL", "gemma4:26b-a4b")

_SYSTEM_PROMPT = textwrap.dedent("""
당신은 시계열 통계 분석 결과를 종합 해석하는 한국인 금융 데이터 분석가입니다.
사용자가 한 종목의 5년치 수급-가격 분석 결과 12 섹션을 자연어로 정리해
드릴 것입니다. 이를 읽고 **통합 결론**을 한국어 마크다운으로 작성하세요.

### 출력 형식
1. 첫 줄: `## 🧠 최종 종합 의견` 헤더
2. 2~4 단락의 **통합 서술** — 누가 누구를 매매하고, 어떤 시간 구조와 인과
   구조를 보이는지, 어떤 점이 일반적이고 어떤 점이 특이한지
3. **핵심 포인트** 3~6 개의 bullet list
4. 마지막 한 줄: `> ⚠️ 본 분석은 과거 데이터의 통계적 패턴 관찰일 뿐
   투자 자문이 아닙니다. 인과 관계나 미래 가격을 보장하지 않습니다.`

### 작성 규칙
- **매매 추천·가격 예측 금지**. "사야 한다", "오를 것이다" 같은 단정 표현 안 됨.
- **보수적 표현 사용**. "시사한다", "~의 약한 증거", "~로 해석 가능" 같은
  hedge 표현. 단정짓지 마세요.
- **통계 한계 명시**. 인과 ≠ 상관, Granger 의 "예측력"의 의미, level r 의
  spurious 위험, 표본 크기에 따른 p-value 한계 등을 적절히 언급.
- 12 섹션 각각을 나열하지 말고 **공통 패턴과 특이점**으로 묶어 서술.
- 표나 코드 블록 사용 자제 — 단락 위주.
- 외래어보다 한국어 우선 (예: "역행 매매", "추세 추종").
- 분량은 250~400자 (단락 부분), bullet 은 각 짧게 1~2 줄.
""").strip()


def _build_user_message(report: dict, advanced: Optional[dict] = None) -> str:
    """report + advanced → LLM 입력용 컨텍스트 문자열."""
    name = report.get("company_name") or "(이름 없음)"
    ticker = report.get("ticker") or "?"
    market = report.get("market") or "-"
    n = report.get("n_days", 0)
    w = report.get("window") or {}
    pr = report.get("price") or {}

    header = (
        f"종목: {name} ({ticker}) · {market}\n"
        f"기간: {w.get('start')} ~ {w.get('end')} ({n}일)\n"
        f"종가: {pr.get('start')} → {pr.get('end')} "
        f"({pr.get('total_return_pct', 0):+.1f}%) · "
        f"일별 σ {pr.get('daily_std_pct', 0):.2f}%"
    )

    sections: list[str] = ["[상관 분석 6 섹션]"]
    sections.append("1. 누적 매수/매도: " + itp.interpret_cumulative(report))
    sections.append("2. 동시 상관: " + itp.interpret_concurrent(report))
    sections.append("3. 상승일/하락일 평균: " + itp.interpret_up_down(report))
    sections.append("4. Lag(CCF): " + itp.interpret_lag(report))
    sections.append("5. Regime 변화: " + itp.interpret_regime(report))
    sections.append("6. Level r: " + itp.interpret_level(report))

    if advanced and advanced.get("adf"):
        sections.append("")
        sections.append("[정교한 분석 6 섹션]")
        sections.append("7-1. ADF 정상성: " + itp.interpret_adf(advanced))
        sections.append("7-2. Granger 인과: " + itp.interpret_granger(advanced))
        sections.append("7-3. VAR + IRF: " + itp.interpret_var_irf(advanced))
        sections.append("7-4. Cointegration: " + itp.interpret_cointegration(advanced))
        sections.append("7-5. Mutual Information: " + itp.interpret_mi(advanced))
        sections.append("7-6. Rolling correlation: " + itp.interpret_rolling(advanced))

    return header + "\n\n" + "\n".join(sections)


def synthesize_conclusion(
    report: dict,
    advanced: Optional[dict] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """report + advanced(옵션) → LLM 종합 의견 마크다운.

    실패(네트워크·모델 부재 등) 시 graceful 메시지 반환 — 예외 던지지 않음.
    """
    if not report or report.get("n_days", 0) == 0:
        return "## 🧠 최종 종합 의견\n\n*분석 데이터가 없어 종합 의견을 생성할 수 없습니다.*"

    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL

    user_msg = _build_user_message(report, advanced)

    try:
        client = create_llm_client(provider, model)
        llm = client.get_llm()
        result = llm.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
        content = getattr(result, "content", str(result))
        if not content or not content.strip():
            return (
                "## 🧠 최종 종합 의견\n\n"
                f"*LLM({provider}/{model}) 응답이 비어 있습니다.*"
            )
        text = content.strip()
        # LLM 이 헤더를 안 붙였으면 추가
        if not text.lstrip().startswith("## "):
            text = "## 🧠 최종 종합 의견\n\n" + text
        return text
    except Exception as exc:
        return (
            "## 🧠 최종 종합 의견\n\n"
            f"*LLM 호출 실패({provider}/{model}): "
            f"{type(exc).__name__}: {exc}*\n\n"
            f"위 12 섹션 자동 해석을 직접 읽고 종합해 주세요."
        )
