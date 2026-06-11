"""홀드아웃 시험지 로딩 — 매니페스트에 동결된 종목·격리 스토어만 사용.

시험지 규칙:
  - 이 종목들은 전략 탐색(discovery)에 절대 쓰지 않는다.
  - 채점 호출은 ledger 에 기록되어 시험지 마모(반복 노출)를 추적한다.
"""

from __future__ import annotations

import json
import os

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout_manifest.json")


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def holdout_universe():
    """(tickers, loader, basket_ret) — 격리 스토어 기반 홀드아웃 유니버스.

    스토어는 인스턴스 생성 시점에 KIS_HISTORY_DIR 를 읽으므로, 로딩 동안만
    격리 경로로 강제하고 끝나면 복원한다 — env 가 프로세스 전역에 남으면
    이후 코드의 스토어 접근이 전부 시험지를 가리키게 되는 누출이 생긴다.
    """
    m = load_manifest()
    prev = os.environ.get("KIS_HISTORY_DIR")
    os.environ["KIS_HISTORY_DIR"] = os.path.expanduser(m["store_dir"])
    try:
        from tradingagents.dataflows.kis_history_store import KisHistoryStore
        from tradingagents.hermes.forward_test import ETF_CODES
        from tradingagents.hermes.strategy_research_v2 import _preload

        manifest_codes = {t["code"] for t in m["tickers"]}
        raw = [t for t in KisHistoryStore().list_tickers()
               if str(t)[:6] in manifest_codes and str(t)[:6] not in ETF_CODES]
        tickers, loader, _kf, _uf = _preload(raw)

        px: dict[str, dict[str, float]] = {}
        for tk in tickers:
            df, _ = loader(tk)
            if df is None or df.empty:
                continue
            d = df[["date", "close"]].copy()
            d["date"] = d["date"].astype(str)
            px[tk] = dict(zip(d["date"], d["close"].astype(float)))
    finally:
        if prev is None:
            os.environ.pop("KIS_HISTORY_DIR", None)
        else:
            os.environ["KIS_HISTORY_DIR"] = prev

    def basket_ret(d0: str, d1: str):
        """같은 보유구간(d0→d1) 동일가중 바스켓 수익(%) — '아무거나 들고 있기' 대조군."""
        rs = [mm[d1] / mm[d0] - 1 for mm in px.values() if d0 in mm and d1 in mm and mm[d0] > 0]
        return 100 * sum(rs) / len(rs) if rs else None

    return tickers, loader, basket_ret
