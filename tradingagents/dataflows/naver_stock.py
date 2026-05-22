"""네이버 금융 OHLCV 어댑터.

엔드포인트: ``api.finance.naver.com/siseJson.naver``
응답은 표준 JSON이 아니라 Python 리터럴 형태의 ``[[...], [...], ...]`` 문자열이라
``ast.literal_eval``로 안전하게 파싱한다. 일반 ``json.loads``는 키워드 없는
배열의 trailing comma 등으로 실패할 수 있다.

출력 포맷은 yfinance의 ``get_YFin_data_online``과 동일한 헤더 + CSV 본문
구조로 맞춘다 — LLM이 두 소스를 같은 스키마로 비교할 수 있게.

부산물로 ``fetch_naver_ohlcv_df``를 제공해 ``naver_indicator``가 stockstats
계산에 재활용한다.
"""
from __future__ import annotations

import ast
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from .korean_utils import to_naver_code

_BASE = "https://api.finance.naver.com/siseJson.naver"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}
_TIMEOUT = 5.0


def _fmt(d: str) -> str:
    """YYYY-MM-DD → YYYYMMDD (네이버 API 형식)."""
    return d.replace("-", "")


def fetch_naver_ohlcv_df(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """네이버 차트 API로 OHLCV를 가져와 DataFrame으로 반환.

    네이버 응답 컬럼 순서:
        ``[Date(YYYYMMDD), Open, High, Low, Close, Volume, ForeignerRatio]``

    반환 DataFrame은 ``Date``를 ``DatetimeIndex``로 두고 컬럼 이름은
    yfinance와 호환되도록 ``Open/High/Low/Close/Volume``로 통일한다 (외인 비율은 버림).
    데이터 없으면 빈 DataFrame.
    """
    code = to_naver_code(symbol)
    resp = requests.get(
        _BASE,
        params={
            "symbol": code,
            "requestType": 1,
            "startTime": _fmt(start_date),
            "endTime": _fmt(end_date),
            "timeframe": "day",
        },
        headers=_HEADERS,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()

    text = resp.text.strip()
    if not text or text == "[]":
        return pd.DataFrame()

    try:
        raw = ast.literal_eval(text)
    except (ValueError, SyntaxError) as e:
        raise RuntimeError(f"naver chart parse failed: {e}") from e

    if not raw or len(raw) < 2:
        return pd.DataFrame()

    header, *rows = raw
    df = pd.DataFrame(rows, columns=header)

    # 표준화
    df = df.rename(columns={
        "날짜": "Date", "시가": "Open", "고가": "High",
        "저가": "Low", "종가": "Close", "거래량": "Volume",
    })
    if "Date" not in df.columns:
        # 첫 컬럼이 날짜인 경우(영문 헤더가 안 올 때)
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep]
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")

    return df


def fetch_naver_ohlcv(symbol: str, start_date: str, end_date: str) -> str:
    """yfinance 출력 형식과 동일한 헤더 + CSV 문자열로 반환."""
    df = fetch_naver_ohlcv_df(symbol, start_date, end_date)
    if df.empty:
        return f"<no Naver OHLCV for {symbol} between {start_date} and {end_date}>"

    csv_string = df.to_csv()
    header = (
        f"# Stock data for {symbol} from {start_date} to {end_date} (source: Naver Finance)\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string
