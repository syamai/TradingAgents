from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from tradingagents.dataflows import market_history


@pytest.mark.unit
class TestMarketHistoryCacheDeterminism:
    def test_historical_range_uses_existing_cache_even_when_ttl_expired(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("TRADINGAGENTS_MARKET_CACHE_REFRESH", raising=False)
        cache_p = Path(tmp_path) / "market" / "usdkrw_2020-01-01_2020-01-03.parquet"
        cache_p.parent.mkdir(parents=True)
        expected = pd.DataFrame({"date": ["2020-01-01"], "close": [1111.0]})
        expected.to_parquet(cache_p, index=False)
        old = time.time() - 86400 * 30
        os.utime(cache_p, (old, old))

        class BoomTicker:
            def __init__(self, symbol):
                raise AssertionError(f"network fetch should not run for historical cache: {symbol}")

        monkeypatch.setattr(market_history.yf, "Ticker", BoomTicker)

        out = market_history.fetch_usdkrw("2020-01-01", "2020-01-03")

        pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)

    def test_force_refresh_bypasses_historical_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("TRADINGAGENTS_MARKET_CACHE_REFRESH", "1")
        cache_p = Path(tmp_path) / "market" / "usdkrw_2020-01-01_2020-01-03.parquet"
        cache_p.parent.mkdir(parents=True)
        pd.DataFrame({"date": ["2020-01-01"], "close": [1111.0]}).to_parquet(cache_p, index=False)

        calls = []

        class FakeTicker:
            def __init__(self, symbol):
                calls.append(symbol)

            def history(self, *args, **kwargs):
                return pd.DataFrame({
                    "Date": pd.to_datetime(["2020-01-01"]),
                    "Close": [2222.0],
                })

        monkeypatch.setattr(market_history.yf, "Ticker", FakeTicker)

        out = market_history.fetch_usdkrw("2020-01-01", "2020-01-03")

        assert calls == ["KRW=X"]
        assert float(out["close"].iloc[0]) == 2222.0
