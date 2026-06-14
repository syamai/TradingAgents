"""KIS history store tickers 의 증권유형(security_type) 분류·영속화.

유니버스는 네이버 시가총액 상위 스크래핑이라 ETF·ETN(KODEX·TIGER·… 지수상품)이
섞여 들어온다. 이들은 종목선정(팩터 전략) 대상이 아니므로 분류해 제외용으로 표시한다.

권위 분류원: yfinance ``quoteType`` (ETF/ETN 모두 'ETF' 로 분류됨). ETN 은 종목명에
'ETN' 이 들어가므로 우선 구분해 'ETN' 으로, 나머지 'ETF'→'ETF', 'EQUITY'→'STOCK'.

결과는 tickers.security_type 에 적재되고 ``strategy_research_v2._preload`` 가
('ETF','ETN') 을 유니버스에서 제외한다. **유니버스 재수집·확장 후 재실행**하면 된다.

사용:
    uv run python scripts/classify_security_types.py            # 미분류만
    uv run python scripts/classify_security_types.py --refresh  # 전체 재분류
    uv run python scripts/classify_security_types.py 069500 005930  # 명시 종목
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradingagents.dataflows.kis_history_store import KisHistoryStore  # noqa: E402

# 한국 ETF 발행 브랜드 토큰 — 실제 회사명엔 안 나타나는 확정 식별자(라틴 대문자/접두).
# yfinance 가 한국 '종목'을 MUTUALFUND 로 대량 오분류하므로 이름 브랜드를 1차 분류원으로,
# yfinance quoteType=='ETF' 는 브랜드 못 잡은 ETF 안전망으로만 쓴다.
_ETF_BRAND = re.compile(
    r"KODEX|TIGER|ACE |ARIRANG|KBSTAR|KOSEF|KINDEX|HANARO|SOL |RISE |PLUS |"
    r"TIMEFOLIO|TIME |1Q |KIWOOM|히어로즈|마이다스|네비게이터|WON |FOCUS |BNK |마이티")


def classify(code: str, name: str | None) -> str:
    """(code, name) → 'STOCK'/'ETF'/'ETN'. ETN(이름)·ETF(브랜드)·ETF(yfinance) 순,
    그 외는 STOCK (yahoo 의 MUTUALFUND/NONE 오분류는 무시)."""
    nm = name or ""
    if "ETN" in nm:
        return "ETN"
    if _ETF_BRAND.search(nm):
        return "ETF"
    for suffix in (".KS", ".KQ"):
        try:
            qt = yf.Ticker(code + suffix).get_info().get("quoteType")
        except Exception:        # noqa: BLE001 — 네트워크/심볼 결손은 STOCK 기본값
            qt = None
        if qt == "ETF":
            return "ETF"
        if qt:                   # EQUITY/MUTUALFUND/NONE 등 — 종목으로 간주
            break
    return "STOCK"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tickers", nargs="*", help="명시 종목. 비우면 store 전체.")
    p.add_argument("--refresh", action="store_true", help="이미 분류된 종목도 재분류.")
    p.add_argument("--workers", type=int, default=8, help="동시 조회 수(기본 8).")
    args = p.parse_args(argv)

    store = KisHistoryStore()
    names = store.ticker_names()
    types = store.ticker_security_types()
    candidates = args.tickers or store.list_tickers()
    targets = [t for t in candidates if args.refresh or not types.get(t)]
    print(f"전체 {len(candidates)} / 분류 대상 {len(targets)} (미분류 {sum(1 for t in candidates if not types.get(t))})")
    if not targets:
        return 0

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for code, st in zip(targets, ex.map(lambda c: classify(c, names.get(c)), targets)):
            store.set_security_type(code, st)
            counts[st] = counts.get(st, 0) + 1

    print("분류 결과:", {k: counts[k] for k in sorted(counts)})
    excluded = counts.get("ETF", 0) + counts.get("ETN", 0)
    print(f"→ 유니버스 제외 대상(ETF+ETN): {excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
