"""트렌드 신호 수집·저장·랭킹 단위 테스트.

핵심 검증:
  - snapshot-lock: 같은 (asof,market,source,entity,metric) 행은 멱등하고 절대
    덮어쓰지 않는다(Google Trends 소급 재스케일 누출 차단의 토대).
  - point-in-time read: asof_date_lte 로 과거만 읽힌다(look-ahead 차단).
  - fade 랭킹: ≥2 소스 동의 종목만(봇·pump 단일소스 스파이크 배제).
  - fetcher graceful-degrade: 첫 호출 실패면 ([], False).
"""
import pytest

from tradingagents.dataflows.trend_store import TrendStore
from tradingagents.dataflows.trends import apewisdom
from tradingagents.dataflows.trends.base import COINCIDENT, SignalRow
from tradingagents.hermes import trend_collect
from tradingagents.hermes.trend_rank import fade_ranking, format_digest

pytestmark = pytest.mark.unit


def _row(entity, source, *, asof="2026-06-05", abn=1.0, rank=1, market="us", metric="m"):
    return SignalRow(
        release_date=asof, asof_date=asof, market=market, entity=entity,
        source=source, metric=metric, leadingness=COINCIDENT,
        raw_value=None, abnormal_value=abn, rank=rank,
    )


def test_snapshot_lock_idempotent(tmp_path):
    s = TrendStore(root=tmp_path)
    rows = [_row("NVDA", "apewisdom")]
    assert s.write(rows) == 1
    assert s.write(rows) == 0  # 같은 snapshot_hash 재적재 무시


def test_snapshot_lock_no_overwrite(tmp_path):
    # 같은 키면 값이 달라도 최초 관측을 유지(소급 재스케일 차단)
    s = TrendStore(root=tmp_path)
    s.write([_row("NVDA", "apewisdom", abn=1.0)])
    s.write([_row("NVDA", "apewisdom", abn=999.0)])
    df = s.read(market="us")
    assert len(df) == 1
    assert df.iloc[0]["abnormal_value"] == 1.0


def test_read_pointintime_filter(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_row("A", "apewisdom", asof="2026-06-01")])
    s.write([_row("B", "apewisdom", asof="2026-06-05")])
    past = s.read(market="us", asof_date_lte="2026-06-03")
    assert set(past["entity"]) == {"A"}


def test_last_snapshot_date(tmp_path):
    s = TrendStore(root=tmp_path)
    assert s.last_snapshot_date("us", "apewisdom") is None
    s.write([_row("A", "apewisdom", asof="2026-06-01")])
    s.write([_row("B", "apewisdom", asof="2026-06-05")])
    assert s.last_snapshot_date("us", "apewisdom") == "2026-06-05"


def test_fade_requires_two_sources(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([
        _row("NVDA", "apewisdom", rank=1),
        _row("NVDA", "finviz_unusual", rank=2),
        _row("AAPL", "apewisdom", rank=3),  # 단일 소스 → 제외
    ])
    fr = fade_ranking("us", store=s, top_n=10)
    assert list(fr["entity"]) == ["NVDA"]
    assert fr.iloc[0]["n_sources"] == 2


def test_digest_empty_when_no_consensus(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_row("AAPL", "apewisdom")])
    assert "없음" in format_digest("us", store=s)


def test_apewisdom_parsing(monkeypatch):
    fake = {
        "results": [
            {"rank": 1, "ticker": "nvda", "mentions": 300, "mentions_24h_ago": 100},
            {"rank": 2, "ticker": "low", "mentions": 5, "mentions_24h_ago": 1},
        ]
    }
    monkeypatch.setattr(apewisdom, "_http_get_json", lambda *a, **k: fake)
    rows, ok = apewisdom.collect_apewisdom(
        asof_date="2026-06-05", min_mentions=10, pages=1
    )
    assert ok
    assert [r.entity for r in rows] == ["NVDA"]  # min_mentions 미달 'low' 제외
    assert rows[0].abnormal_value == pytest.approx((300 - 100) / 100)
    assert rows[0].leadingness == "C"


def test_apewisdom_fetch_fail(monkeypatch):
    monkeypatch.setattr(apewisdom, "_http_get_json", lambda *a, **k: None)
    assert apewisdom.collect_apewisdom(pages=1) == ([], False)


def test_collect_unknown_source():
    assert trend_collect.run("nope", dry_run=True)["status"] == "error"


def test_collect_dry_run_no_write(monkeypatch):
    monkeypatch.setattr(
        apewisdom, "_http_get_json",
        lambda *a, **k: {"results": [
            {"rank": 1, "ticker": "NVDA", "mentions": 300, "mentions_24h_ago": 100}
        ]},
    )
    r = trend_collect.run("apewisdom", asof_date="2026-06-05", dry_run=True)
    assert r["status"] == "ok"
    assert r["written"] == 0
    assert r["collected"] == 1


def test_signalrow_validation():
    with pytest.raises(ValueError):
        SignalRow("d", "d", "us", "NVDA", "s", "m", "BOGUS")  # 잘못된 leadingness
    with pytest.raises(ValueError):
        SignalRow("d", "d", "jp", "NVDA", "s", "m", COINCIDENT)  # 잘못된 market
