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
from tradingagents.dataflows.trends import apewisdom, google_trends, stocktwits_delta
from tradingagents.dataflows.trends.base import COINCIDENT, SignalRow
from tradingagents.hermes import trend_collect
from tradingagents.hermes.trend_rank import (
    attention_by_sector,
    fade_ranking,
    format_digest,
    rotation_alerts,
    rotation_ranking,
)

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


def _sector_row(entity, mom, rank, asof="2026-06-05"):
    return SignalRow(
        release_date=asof, asof_date=asof, market="us", entity=entity,
        entity_type="sector", source="sector_rotation", metric="rs_momentum",
        leadingness="L", raw_value=None, abnormal_value=mom, rank=rank,
    )


def test_entity_type_stored_and_migrated(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_sector_row("Technology", 10.0, 1)])
    df = s.read(market="us", source="sector_rotation")
    assert df.iloc[0]["entity_type"] == "sector"
    # 기본값: ticker 행도 정상
    s.write([_row("NVDA", "apewisdom")])
    assert s.read(market="us", source="apewisdom").iloc[0]["entity_type"] == "ticker"


def test_rotation_ranking(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_sector_row("Technology", 10.0, 1), _sector_row("Utilities", -9.0, 2)])
    rot = rotation_ranking("us", store=s)
    assert list(rot["entity"]) == ["Technology", "Utilities"]
    assert rot.iloc[0]["rs_momentum"] == 10.0


def test_fade_excludes_sectors(tmp_path):
    # 섹터는 fade_ranking 에서 제외(entity_type 필터) — 개별종목만 군집 판정
    s = TrendStore(root=tmp_path)
    s.write([
        _row("NVDA", "apewisdom"), _row("NVDA", "finviz_unusual"),
        _sector_row("Technology", 10.0, 1),
    ])
    fr = fade_ranking("us", store=s, top_n=10)
    assert list(fr["entity"]) == ["NVDA"]


def test_digest_sector_section(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_sector_row("Technology", 11.5, 1)])
    d = format_digest("us", store=s)
    assert "섹터 로테이션" in d and "Technology" in d


def test_hot_candidates(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([
        _row("NVDA", "apewisdom", abn=5.0),
        _row("AAPL", "finviz_unusual", abn=3.0),
        _sector_row("Technology", 10.0, 1),  # 섹터는 universe 후보에서 제외
    ])
    cands = s.hot_candidates("us", limit=5)
    assert "Technology" not in cands
    assert cands[:2] == ["NVDA", "AAPL"]  # abnormal 큰 순


def test_universe_sources_empty_when_no_universe():
    # universe 없으면(첫 tick) 빈 결과 + fetch_ok True(실패 아님)
    assert google_trends.collect_google_trends(universe=None) == ([], True)
    assert stocktwits_delta.collect_stocktwits_delta(universe=None) == ([], True)


def test_stocktwits_delta_parsing(monkeypatch):
    monkeypatch.setattr(
        stocktwits_delta, "collect_stocktwits_messages",
        lambda tk, **k: ([{"entities": {"sentiment": {"basic": "Bullish"}}}] * 5, True),
    )
    rows, ok = stocktwits_delta.collect_stocktwits_delta(
        universe=["nvda"], asof_date="2026-06-05"
    )
    assert ok
    assert rows[0].entity == "NVDA"
    assert rows[0].raw_value == 5.0
    assert rows[0].source == "stocktwits_delta"
    assert rows[0].leadingness == "C"


def test_stocktwits_delta_fetch_fail(monkeypatch):
    # 모든 종목 fetch 실패 → 빈(실패 아님, universe 는 있었음)
    monkeypatch.setattr(
        stocktwits_delta, "collect_stocktwits_messages",
        lambda tk, **k: ([], False),
    )
    assert stocktwits_delta.collect_stocktwits_delta(universe=["NVDA"]) == ([], True)


def test_rotation_alert_peak(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_sector_row("Technology", 11.5, 1)])  # ≥10% → 과열
    a = rotation_alerts("us", store=s)
    assert any("과열" in x and "Technology" in x for x in a)


def test_rotation_alert_crossover(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_sector_row("Energy", -2.0, 1, asof="2026-06-04")])  # 전일 음
    s.write([_sector_row("Energy", 3.0, 1, asof="2026-06-05")])   # 오늘 양 → 전환
    a = rotation_alerts("us", asof_date="2026-06-05", store=s)
    assert any("양전환" in x and "Energy" in x for x in a)


def test_rotation_alert_none(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([_sector_row("Utilities", -5.0, 1)])  # 과열 아님 + 전일 없음
    assert rotation_alerts("us", store=s) == []


def test_attention_by_sector(tmp_path, monkeypatch):
    import tradingagents.dataflows.trends.sector_map as sm
    monkeypatch.setattr(sm, "get_sectors", lambda tks: {str(t).upper(): "Technology" for t in tks})
    s = TrendStore(root=tmp_path)
    s.write([
        _row("NVDA", "apewisdom"), _row("NVDA", "finviz_unusual"),
        _row("AVGO", "apewisdom"), _row("AVGO", "finviz_unusual"),
    ])
    out = attention_by_sector("us", store=s)
    assert set(out.get("Technology", [])) == {"NVDA", "AVGO"}
