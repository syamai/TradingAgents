"""한국 종목 식별/변환 헬퍼.

trading-ai 라우터(`interface.py`)와 통합 wrapper(`yahoo_naver.py`)에서 공용으로 쓰는
유틸. 외부 호출 없음, 순수 문자열 처리.
"""
from __future__ import annotations

import re


# Yahoo Finance가 사용하는 한국 거래소 접미사.
#   .KS = KOSPI (코스피)
#   .KQ = KOSDAQ (코스닥)
# Yahoo는 KONEX 종목을 별도 코드로 다루지 않으므로 두 가지만 지원한다.
_KO_SUFFIXES = (".KS", ".KQ")

_SIX_DIGIT = re.compile(r"^\d{6}$")


def is_korean_ticker(ticker: str) -> bool:
    """`.KS`/`.KQ` 접미사를 가진 한국 종목인지 판단.

    예외 케이스: 사용자가 접미사 없이 6자리 코드만 넘기는 경우(예: "005930")도
    한국 종목으로 인식한다. 다른 곳에서 들어오는 입력 표기가 일관되지 않을
    가능성을 흡수하기 위함.
    """
    if not isinstance(ticker, str):
        return False
    t = ticker.strip().upper()
    if t.endswith(_KO_SUFFIXES):
        return True
    return bool(_SIX_DIGIT.match(t))


def to_naver_code(ticker: str) -> str:
    """Yahoo 표기 → 네이버 금융 6자리 종목코드.

    `005930.KS` → `005930`, `005930` → `005930`.
    한국 종목이 아닌 경우 ValueError. 호출 전에 `is_korean_ticker`로 가드한다.
    """
    if not isinstance(ticker, str):
        raise ValueError(f"ticker must be str, got {type(ticker).__name__}")
    t = ticker.strip().upper()
    if t.endswith(_KO_SUFFIXES):
        t = t[:-3]
    if not _SIX_DIGIT.match(t):
        raise ValueError(f"not a Korean ticker code: {ticker!r}")
    return t


def combine_sources(
    *,
    yahoo_block: str,
    korean_block: str,
    korean_label: str,
    method: str,
) -> str:
    """야후 + 한국 소스 두 블록을 헤더로 명시해 단일 문자열로 합친다.

    LLM이 두 소스를 구분해서 인용할 수 있도록 `<start_of_*>/<end_of_*>` 마커를
    사용한다. 이는 `sentiment_analyst.py`가 이미 채택한 패턴과 동일하다.

    한쪽 블록이 비어 있거나 `<unavailable: ...>` 형태인 경우에도 그대로 표기해서
    LLM에게 어느 소스가 누락됐는지 알려준다.
    """
    yahoo_block = yahoo_block.strip() if yahoo_block else "<unavailable: empty>"
    korean_block = korean_block.strip() if korean_block else "<unavailable: empty>"
    return (
        f"## Combined data for `{method}` — Yahoo Finance + {korean_label}\n\n"
        f"<start_of_yahoo>\n{yahoo_block}\n<end_of_yahoo>\n\n"
        f"<start_of_{korean_label.lower()}>\n{korean_block}\n<end_of_{korean_label.lower()}>\n"
    )
