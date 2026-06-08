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
from tradingagents.dataflows.trends import (
    apewisdom,
    google_trends,
    options_os,
    stocktwits_delta,
)
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
    assert "업종 흐름" in d and "Technology" in d


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
    assert any("약세→강세 전환" in x and "Energy" in x for x in a)


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


def test_options_os_empty_no_universe():
    assert options_os.collect_options_os(universe=None) == ([], True)


def test_options_os_parsing(monkeypatch):
    import pandas as pd
    import yfinance

    class FakeChain:
        def __init__(self):
            self.calls = pd.DataFrame({"volume": [100, 200]})
            self.puts = pd.DataFrame({"volume": [50, 50]})

    class FakeTicker:
        options = ["2026-06-06", "2026-06-13"]

        def __init__(self, tk):
            pass

        def option_chain(self, exp):
            return FakeChain()

        def history(self, period="1d"):
            return pd.DataFrame({"Volume": [100000]})

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    rows, ok = options_os.collect_options_os(
        universe=["nvda"], asof_date="2026-06-05", max_expiries=2
    )
    assert ok
    assert rows[0].entity == "NVDA"
    # opt_vol=(300+100)*2만기=800, stk=100000 → O/S=0.008
    assert abs(rows[0].abnormal_value - 0.008) < 1e-4
    assert rows[0].source == "options_os" and rows[0].leadingness == "C"


def test_rrg_quadrant():
    from tradingagents.hermes.trend_rank import _rrg_quadrant
    assert _rrg_quadrant(105, 5) == "Leading"     # 강(≥100) + 상승
    assert _rrg_quadrant(95, 5) == "Improving"    # 약 + 상승
    assert _rrg_quadrant(105, -5) == "Weakening"  # 강 + 하락
    assert _rrg_quadrant(95, -5) == "Lagging"     # 약 + 하락
    assert _rrg_quadrant(None, 5) == "Improving"  # ratio 없으면 약 취급


def test_fade_evidence_and_tier(tmp_path):
    s = TrendStore(root=tmp_path)
    s.write([
        SignalRow("2026-06-05", "2026-06-05", "us", "NVDA", "apewisdom",
                  "mention_momentum_24h", COINCIDENT, raw_value=300, abnormal_value=2.5, rank=1),
        SignalRow("2026-06-05", "2026-06-05", "us", "NVDA", "options_os",
                  "os_ratio", COINCIDENT, raw_value=1000, abnormal_value=0.02, rank=1),
    ])
    fr = fade_ranking("us", store=s, top_n=10)
    row = fr.iloc[0]
    assert row["tier"] == "🟡 관심 쏠림 시작"  # 2소스
    ev = row["evidence"]
    assert any("레딧 언급" in e for e in ev)  # 근거에 raw 신호값
    assert any("옵션 거래 쏠림" in e for e in ev)
    assert any("학술 Barber" in e for e in ev)  # ApeWisdom 소셜 herding
    assert any("학술 Johnson" in e for e in ev)  # 옵션 O/S
    assert any("기여도" in e for e in ev)  # 소스별 기여도


# === 한국판(naver_board + US→KR 브리지 연결근거) ===

def test_naver_board_counts_asof(monkeypatch):
    from tradingagents.dataflows.trends import naver_board
    posts = [
        {"date": "2026.06.05 10:00"}, {"date": "2026.06.05 11:00"},
        {"date": "2026.06.05 12:00"}, {"date": "2026.06.04 09:00"},  # 전일 → 제외
    ]
    monkeypatch.setattr(
        naver_board, "collect_naver_discussion", lambda code, **kw: (None, posts)
    )
    rows, ok = naver_board.collect_naver_board(
        universe=["005930"], asof_date="2026-06-05"
    )
    assert ok and len(rows) == 1
    r = rows[0]
    assert r.entity == "005930" and r.raw_value == 3.0  # 당일 3건만
    assert r.source == "naver_board" and r.leadingness == "C" and r.market == "kr"


def test_naver_board_empty_universe():
    from tradingagents.dataflows.trends import naver_board
    assert naver_board.collect_naver_board(universe=None) == ([], True)


def test_naver_board_all_fail(monkeypatch):
    from tradingagents.dataflows.trends import naver_board
    monkeypatch.setattr(
        naver_board, "collect_naver_discussion", lambda code, **kw: ("err", [])
    )
    out = naver_board.collect_naver_board(universe=["005930"], asof_date="2026-06-05")
    assert out == ([], False)


def test_digest_kr_single_source_and_provenance(monkeypatch, tmp_path):
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    s.write([SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                       "board_volume", COINCIDENT, raw_value=120, abnormal_value=120, rank=1)])
    monkeypatch.delenv("TREND_KR_MIN_SOURCES", raising=False)  # ambient env 비의존
    monkeypatch.setattr(
        trend_rank, "_kr_provenance",
        lambda codes, **kw: {"005930": {"us": "NVDA", "reason": "HBM 공급망", "name": "삼성전자"}},
    )
    d = trend_rank.format_digest("kr", store=s)
    assert "과열 주목 종목 [KR]" in d        # 단일 소스(min_sources=1) 통과
    assert "삼성전자(005930)" in d
    assert "🇺🇸NVDA" in d and "한국 짝" in d  # 연결근거
    assert "종목토론방 글 120건" in d
    assert "업종 흐름" not in d               # KR 은 섹터 로테이션 섹션 없음


def test_digest_kr_verdict_varies_with_price(monkeypatch, tmp_path):
    """enrich_price=True 면 KR '판정' 줄이 주가 5일 괴리로 종목마다 달라진다(상수 탈피)."""
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    s.write([
        SignalRow("2026-06-08", "2026-06-08", "kr", "005930", "naver_board",
                  "board_volume", COINCIDENT, raw_value=200, abnormal_value=200, rank=1),
        SignalRow("2026-06-08", "2026-06-08", "kr", "000660", "naver_board",
                  "board_volume", COINCIDENT, raw_value=190, abnormal_value=190, rank=2),
    ])
    monkeypatch.delenv("TREND_KR_MIN_SOURCES", raising=False)
    monkeypatch.setattr(trend_rank, "_kr_provenance", lambda codes, **kw: {})
    monkeypatch.setattr(
        trend_rank, "_kr_price",
        lambda code: {"change_5d_pct": -5.0} if code == "005930" else {"change_5d_pct": 6.0},
    )
    d = trend_rank.format_digest("kr", store=s, enrich_price=True)
    assert "페이드(천장) 경고 강화" in d        # 005930: 주가 5일 -5% → 페이드 경고
    assert "추격 과열 주의" in d                  # 000660: 주가 5일 +6% → 추격 과열
    assert "→ 판정:" in d and "신호 성격" not in d  # 결론이 종목별 '판정'으로 교체됨

    # 기본(off)이면 옛 상수 동작 유지 — 두 종목 모두 같은 '신호 성격'(상수)
    d0 = trend_rank.format_digest("kr", store=s, enrich_price=False)
    assert d0.count("막 달아오르는 중") == 2 and "판정" not in d0


def test_digest_kr_verdict_falls_back_when_no_price(monkeypatch, tmp_path):
    """enrich_price=True 라도 주가 조회 실패(None)면 기존 '신호 성격'으로 폴백."""
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    s.write([SignalRow("2026-06-08", "2026-06-08", "kr", "005930", "naver_board",
                       "board_volume", COINCIDENT, raw_value=200, abnormal_value=200, rank=1)])
    monkeypatch.delenv("TREND_KR_MIN_SOURCES", raising=False)
    monkeypatch.setattr(trend_rank, "_kr_provenance", lambda codes, **kw: {})
    monkeypatch.setattr(trend_rank, "_kr_price", lambda code: None)
    d = trend_rank.format_digest("kr", store=s, enrich_price=True)
    assert "신호 성격" in d and "판정" not in d


# === Phase 2: 토론방 감성 틸트 + 네이버 DataLab 검색량 + min_sources 승격 ===

def test_naver_board_sentiment_tilt(monkeypatch):
    from tradingagents.dataflows.trends import naver_board
    posts = [
        {"date": "2026.06.05 10:00", "up": 30, "down": 5},
        {"date": "2026.06.05 11:00", "up": 10, "down": 5},
        {"date": "2026.06.04 09:00", "up": 99, "down": 0},  # 전일 → 감성/글수 제외
    ]
    monkeypatch.setattr(
        naver_board, "collect_naver_discussion", lambda code, **kw: (None, posts)
    )
    rows, ok = naver_board.collect_naver_board(
        universe=["005930"], asof_date="2026-06-05"
    )
    vol = [r for r in rows if r.metric == "board_volume"]
    sent = [r for r in rows if r.metric == "board_sentiment"]
    assert ok and len(vol) == 1 and len(sent) == 1
    assert vol[0].rank == 1                      # 글수만 순위 부여
    assert sent[0].rank is None                  # 감성은 rank 없음(fade_score 무영향)
    # 당일 공감 40 / 비공감 10 → (40-10)/50 = 0.6
    assert abs(sent[0].abnormal_value - 0.6) < 1e-9
    assert sent[0].raw_value == 50.0             # 표본(공감+비공감)


def test_naver_board_sentiment_below_threshold(monkeypatch):
    from tradingagents.dataflows.trends import naver_board
    posts = [{"date": "2026.06.05 10:00", "up": 2, "down": 1}]  # 표본 3 < 5
    monkeypatch.setattr(
        naver_board, "collect_naver_discussion", lambda code, **kw: (None, posts)
    )
    rows, _ = naver_board.collect_naver_board(
        universe=["005930"], asof_date="2026-06-05"
    )
    assert [r.metric for r in rows] == ["board_volume"]  # 감성 행 없음


def test_naver_datalab_no_creds(monkeypatch):
    from tradingagents.dataflows.trends import naver_datalab
    monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)
    monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)
    assert naver_datalab.collect_naver_datalab(universe=["005930"]) == ([], False)


def test_naver_datalab_empty_universe():
    from tradingagents.dataflows.trends import naver_datalab
    assert naver_datalab.collect_naver_datalab(universe=None) == ([], True)


def test_naver_datalab_scope_error(monkeypatch):
    """DataLab 스코프 미인증(errorCode 024) → graceful degrade ([], False)."""
    from tradingagents.dataflows.trends import naver_datalab

    class _Resp:
        status_code = 200
        def json(self):
            return {"errorCode": "024", "errorMessage": "Scope Status Invalid"}

    monkeypatch.setenv("NAVER_CLIENT_ID", "x")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "y")
    monkeypatch.setattr(naver_datalab, "_resolve_names", lambda codes: {"005930": "삼성전자"})
    monkeypatch.setattr(naver_datalab.requests, "post", lambda *a, **k: _Resp())
    assert naver_datalab.collect_naver_datalab(
        universe=["005930"], asof_date="2026-06-05"
    ) == ([], False)


def test_naver_datalab_asvi_and_scale_invariance(monkeypatch):
    """ASVI = log(당주)−log(직전8주 중앙값); 선형 재스케일에 불변(snapshot-lock 근거)."""
    import math
    from tradingagents.dataflows.trends import naver_datalab

    monkeypatch.setenv("NAVER_CLIENT_ID", "x")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "y")
    monkeypatch.setattr(naver_datalab, "_resolve_names", lambda codes: {"005930": "삼성전자"})

    series = [20.0] * 8 + [60.0]            # baseline 20, 당주 60 → log(3)
    monkeypatch.setattr(naver_datalab, "_fetch_series", lambda *a, **k: series)
    rows, ok = naver_datalab.collect_naver_datalab(
        universe=["005930"], asof_date="2026-06-05"
    )
    assert ok and len(rows) == 1
    r = rows[0]
    assert r.source == "naver_datalab" and r.metric == "asvi" and r.leadingness == "L"
    assert abs(r.abnormal_value - round(math.log(3), 4)) < 1e-9
    assert r.rank == 1

    scaled = [v * 0.5 for v in series]     # 전체 0.5배 → 동일 ASVI
    monkeypatch.setattr(naver_datalab, "_fetch_series", lambda *a, **k: scaled)
    rows2, _ = naver_datalab.collect_naver_datalab(
        universe=["005930"], asof_date="2026-06-05"
    )
    assert abs(rows2[0].abnormal_value - r.abnormal_value) < 1e-9


def test_digest_kr_sentiment_line(monkeypatch, tmp_path):
    """KR 디지스트가 감성 틸트를 근거 줄로 표시(기여도·학술 없이 단독)."""
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    s.write([
        SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                  "board_volume", COINCIDENT, raw_value=120, abnormal_value=120, rank=1),
        SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                  "board_sentiment", COINCIDENT, raw_value=50, abnormal_value=0.6, rank=None),
    ])
    monkeypatch.delenv("TREND_KR_MIN_SOURCES", raising=False)
    monkeypatch.setattr(trend_rank, "_kr_provenance", lambda codes, **kw: {})
    d = trend_rank.format_digest("kr", store=s)
    assert "토론방 분위기 긍정 우세" in d
    assert "종목토론방 글 120건" in d


def test_digest_kr_min_sources_promotion(monkeypatch, tmp_path):
    """TREND_KR_MIN_SOURCES=2 승격 시 단일 소스 종목은 탈락(2소스 독립 동의만)."""
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    # 토론방 1소스만 — 승격 후 탈락해야
    s.write([SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                       "board_volume", COINCIDENT, raw_value=120, abnormal_value=120, rank=1)])
    monkeypatch.setattr(trend_rank, "_kr_provenance", lambda codes, **kw: {})
    monkeypatch.setenv("TREND_KR_MIN_SOURCES", "2")  # 런타임 env 플립
    d = trend_rank.format_digest("kr", store=s)
    assert "과열 주목 종목 [KR]" not in d   # 단일 소스 → 승격 게이트 탈락

    # 검색량(독립 2번째 소스) 추가 → 통과
    s.write([SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_datalab",
                       "asvi", "L", raw_value=60, abnormal_value=1.0, rank=1)])
    d2 = trend_rank.format_digest("kr", store=s)
    assert "과열 주목 종목 [KR]" in d2
    assert "검색량 급증" in d2 and "종목토론방 글 120건" in d2


def test_kr_min_sources_invalid_env(monkeypatch):
    """잘못된 env 값(빈문자열·비정수)은 import/호출을 깨지 않고 1 로 폴백."""
    from tradingagents.hermes import trend_rank
    for bad in ("", "abc", "2.5"):
        monkeypatch.setenv("TREND_KR_MIN_SOURCES", bad)
        assert trend_rank._kr_min_sources() == 1
    monkeypatch.setenv("TREND_KR_MIN_SOURCES", "2")
    assert trend_rank._kr_min_sources() == 2


def test_board_sentiment_zero_fade_score_and_n_sources(tmp_path):
    """board_sentiment(rank=None)는 fade_score·n_sources 에 영향 0(불변식 직접 검증)."""
    s = TrendStore(root=tmp_path)
    base = SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                     "board_volume", COINCIDENT, raw_value=120, abnormal_value=120, rank=1)
    s.write([base])
    fr1 = fade_ranking("kr", store=s, min_sources=1)
    score1 = float(fr1.iloc[0]["fade_score"])
    # 감성 행 추가(높은 abnormal_value)
    s.write([SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                       "board_sentiment", COINCIDENT, raw_value=99, abnormal_value=0.9, rank=None)])
    fr2 = fade_ranking("kr", store=s, min_sources=1)
    assert float(fr2.iloc[0]["fade_score"]) == score1        # fade_score 무변
    assert int(fr2.iloc[0]["n_sources"]) == 1                # 같은 source → 미증가


def test_naver_board_sentiment_min_votes_boundary(monkeypatch):
    """_MIN_VOTES=5 경계: 표본 4 → 감성 없음, 5 → 감성 있음(off-by-one 방어)."""
    from tradingagents.dataflows.trends import naver_board

    def run(up, down):
        posts = [{"date": "2026.06.05 10:00", "up": up, "down": down}]
        monkeypatch.setattr(
            naver_board, "collect_naver_discussion", lambda code, **kw: (None, posts)
        )
        rows, _ = naver_board.collect_naver_board(universe=["005930"], asof_date="2026-06-05")
        return [r.metric for r in rows]

    assert run(2, 2) == ["board_volume"]                      # 표본 4 < 5 → 감성 없음
    assert "board_sentiment" in run(3, 2)                     # 표본 5 → 감성 있음


def test_digest_kr_negative_sentiment(monkeypatch, tmp_path):
    """부정 우세 틸트(음수)도 디지스트에 표기."""
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    s.write([
        SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                  "board_volume", COINCIDENT, raw_value=120, abnormal_value=120, rank=1),
        SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                  "board_sentiment", COINCIDENT, raw_value=25, abnormal_value=-0.6, rank=None),
    ])
    monkeypatch.delenv("TREND_KR_MIN_SOURCES", raising=False)
    monkeypatch.setattr(trend_rank, "_kr_provenance", lambda codes, **kw: {})
    d = trend_rank.format_digest("kr", store=s)
    assert "토론방 분위기 부정 우세" in d


def test_naver_datalab_non_json_and_timeout_degrade(monkeypatch):
    """HTTP 비-JSON 응답·요청 예외 모두 graceful degrade([], False)."""
    import requests
    from tradingagents.dataflows.trends import naver_datalab
    monkeypatch.setenv("NAVER_CLIENT_ID", "x")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "y")
    monkeypatch.setattr(naver_datalab, "_resolve_names", lambda codes: {"005930": "삼성전자"})

    class _BadJson:
        status_code = 500
        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(naver_datalab.requests, "post", lambda *a, **k: _BadJson())
    assert naver_datalab.collect_naver_datalab(
        universe=["005930"], asof_date="2026-06-05") == ([], False)

    def _raise(*a, **k):
        raise requests.RequestException("timeout")
    monkeypatch.setattr(naver_datalab.requests, "post", _raise)
    assert naver_datalab.collect_naver_datalab(
        universe=["005930"], asof_date="2026-06-05") == ([], False)


def test_naver_datalab_asvi_true_median(monkeypatch):
    """짝수(8) baseline 의 중앙값은 true median(google_trends 와 동일 정의)."""
    import math
    from tradingagents.dataflows.trends import naver_datalab
    monkeypatch.setenv("NAVER_CLIENT_ID", "x")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "y")
    monkeypatch.setattr(naver_datalab, "_resolve_names", lambda codes: {"005930": "삼성전자"})
    # baseline [40,40,40,40,50,50,50,50] → true median 45, 당주 90 → log(90/45)=log2
    series = [40.0, 40.0, 40.0, 40.0, 50.0, 50.0, 50.0, 50.0, 90.0]
    monkeypatch.setattr(naver_datalab, "_fetch_series", lambda *a, **k: series)
    rows, ok = naver_datalab.collect_naver_datalab(universe=["005930"], asof_date="2026-06-05")
    assert ok and abs(rows[0].abnormal_value - round(math.log(2), 4)) < 1e-9


def test_trend_policy_set_get_merge(tmp_path):
    """정책 set→get, 부분 수정 시 미지정 필드 승계, 이력 보존."""
    s = TrendStore(root=tmp_path)
    assert s.get_active_policy("kr") is None
    p1 = s.set_policy("kr", min_sources=2, rotation_peak_pct=8.0, name="tight")
    assert p1["min_sources"] == 2 and p1["active"] == 1
    # min_sources 만 수정 → rotation_peak_pct 승계
    p2 = s.set_policy("kr", min_sources=1)
    assert p2["min_sources"] == 1 and p2["rotation_peak_pct"] == 8.0
    act = s.get_active_policy("kr")
    assert act["min_sources"] == 1 and act["id"] == p2["id"]   # 최신 active
    hist = s.list_policies("kr")
    assert len(hist) == 2 and sum(h["active"] for h in hist) == 1  # active 1개만


def _kr_vol(code, asof, abn=100, rank=1):
    return SignalRow(asof, asof, "kr", code, "naver_board", "board_volume",
                     COINCIDENT, raw_value=abn, abnormal_value=abn, rank=rank)


def test_trend_analyses_cache(tmp_path):
    """심층분석 캐시 save/get/list + 멱등 갱신."""
    s = TrendStore(root=tmp_path)
    assert s.get_analysis("kr", "005930", "2026-06-05") is None
    s.save_analysis("kr", "005930", "2026-06-05", "외인 순매수 + 신약 기대", name="삼성전자")
    got = s.get_analysis("kr", "005930", "2026-06-05")
    assert got["summary"].startswith("외인") and got["name"] == "삼성전자"
    s.save_analysis("kr", "005930", "2026-06-05", "갱신된 요약")  # 같은 키 → 갱신
    got2 = s.get_analysis("kr", "005930", "2026-06-05")
    assert got2["summary"] == "갱신된 요약"
    assert got2["name"] == "삼성전자"   # name 미지정 갱신은 기존 이름 보존(COALESCE, B2)
    assert len(s.list_analyses("kr", "2026-06-05")) == 1  # 멱등(중복 행 없음)


def test_latest_analysis_asof(tmp_path):
    """deep_dive 는 스냅샷이 아닌 '분석' 최신 날짜를 읽는다(B3)."""
    s = TrendStore(root=tmp_path)
    assert s.latest_analysis_asof("kr") is None
    s.save_analysis("kr", "005930", "2026-06-04", "어제")
    s.save_analysis("kr", "035720", "2026-06-06", "오늘")
    assert s.latest_analysis_asof("kr") == "2026-06-06"


def test_new_entrants_detection(tmp_path):
    """오늘 fade 에 있고 직전일 fade 에 없던 종목만 신규 진입."""
    from tradingagents.hermes import trend_deepen
    s = TrendStore(root=tmp_path)
    s.set_policy("kr", min_sources=1)              # 단일 소스로 통과(테스트 결정성)
    s.write([_kr_vol("000100", "2026-06-04", abn=90)])   # 어제부터 있던 종목
    s.write([_kr_vol("000100", "2026-06-05", abn=95),
             _kr_vol("035720", "2026-06-05", abn=80)])   # 035720 = 오늘 신규
    fresh = trend_deepen.new_entrants("kr", asof="2026-06-05", store=s)
    assert fresh == ["035720"]                     # 000100 은 어제 있었으니 제외


def test_deepen_disabled_and_market(tmp_path, monkeypatch):
    """TREND_DEEPEN=0 → 비활성, market!=kr → 스킵(분석가 KR 전용)."""
    from tradingagents.hermes import trend_deepen
    s = TrendStore(root=tmp_path)
    monkeypatch.setenv("TREND_DEEPEN", "0")
    assert trend_deepen.deepen("kr", store=s)["status"] == "disabled"
    monkeypatch.setenv("TREND_DEEPEN", "1")
    assert trend_deepen.deepen("us", store=s)["status"] == "skip_market"


def test_deepen_cache_hit_skips_analysts(tmp_path, monkeypatch):
    """캐시 hit 종목은 분석가·종합 호출 0(비용 절약)."""
    from tradingagents.hermes import trend_deepen
    import tradingagents.hermes.analyst_runner as ar
    s = TrendStore(root=tmp_path)
    s.set_policy("kr", min_sources=1)
    s.write([_kr_vol("035720", "2026-06-05")])
    s.save_analysis("kr", "035720", "2026-06-05", "이미 분석됨")   # 캐시 존재
    monkeypatch.setattr(trend_deepen, "analysis_targets", lambda *a, **k: (["035720"], "2026-06-05"))
    monkeypatch.setattr(ar, "run_analyst", lambda *a, **k: (_ for _ in ()).throw(AssertionError("호출되면 안 됨")))
    out = trend_deepen.deepen("kr", asof="2026-06-05", store=s)
    assert out["cached"] == 1 and out["analyzed"] == 0


def test_analysis_targets_staleness(tmp_path):
    """상위 fade 중 분석 없음/오래된 종목만 타깃(신선한 분석은 제외) — 확대 커버."""
    from tradingagents.hermes import trend_deepen
    s = TrendStore(root=tmp_path)
    s.set_policy("kr", min_sources=1)
    s.write([_kr_vol("035720", "2026-06-06", abn=90),
             _kr_vol("000660", "2026-06-06", abn=80, rank=2)])
    s.save_analysis("kr", "035720", "2026-06-06", "오늘 분석됨(신선)")   # 신선 → 제외
    s.save_analysis("kr", "000660", "2026-06-01", "5일 전(stale)")        # 오래됨 → 포함
    s.write([_kr_vol("005930", "2026-06-06", abn=70, rank=3)])      # 분석 없음 → 포함
    codes, asof = trend_deepen.analysis_targets("kr", "2026-06-06", freshness_days=3, store=s)
    assert "000660" in codes and "005930" in codes and "035720" not in codes


def test_entity_presence(tmp_path):
    """며칠째 상위(KST 기준) + first_seen — 시계열 맥락."""
    from datetime import datetime, timedelta, timezone
    kst = timezone(timedelta(hours=9))
    today = datetime.now(kst).date()
    s = TrendStore(root=tmp_path)
    for d in (today, today - timedelta(days=1), today - timedelta(days=10)):
        s.write([_kr_vol("005930", d.isoformat())])
    p = s.entity_presence("kr", "005930", lookback_days=7)
    assert p["days_present"] == 2                               # 오늘+어제(10일전 제외)
    assert p["first_seen"] == (today - timedelta(days=10)).isoformat()


def test_kr_price_fallback_and_nan(monkeypatch):
    """_kr_price: .KS 실패→.KQ 폴백, 둘 다 실패→None, NaN 거래량도 직렬화 안전(BLOCK)."""
    import json
    import pandas as pd
    import yfinance as yf
    import tradingagents.hermes.mcp_server as m

    def df(prices, vols):
        return pd.DataFrame({"Close": prices, "Volume": vols})

    class KqOnly:                       # .KS 빈값 → .KQ 성공
        def __init__(self, sym): self.sym = sym
        def history(self, period=None):
            return pd.DataFrame() if self.sym.endswith(".KS") \
                else df([100, 101, 102, 103, 104, 110], [10, 10, 10, 10, 10, 20])
    monkeypatch.setattr(yf, "Ticker", KqOnly)
    r = m._kr_price("035720")
    assert r and r["price"] == 110 and json.dumps(r)            # 폴백 + 직렬화

    class AllFail:
        def __init__(self, s): pass
        def history(self, period=None): return pd.DataFrame()
    monkeypatch.setattr(yf, "Ticker", AllFail)
    assert m._kr_price("999999") is None                       # 둘 다 실패

    class NanVol:                       # 거래량 NaN → NaN 없이 직렬화
        def __init__(self, s): self.s = s
        def history(self, period=None):
            return df([100, 101, 102, 103, 104, 110], [float("nan")] * 6) \
                if self.s.endswith(".KS") else pd.DataFrame()
    monkeypatch.setattr(yf, "Ticker", NanVol)
    r2 = m._kr_price("035720")
    assert r2 is not None and r2["volume_ratio"] == 0.0 and json.dumps(r2)


def test_market_context_shape(monkeypatch):
    """get_trend_market_context: divergence 항상 존재(non-kr도), MCP 직렬화 가능(BLOCK)."""
    import json
    import tradingagents.hermes.mcp_server as m
    monkeypatch.setattr(m, "_kr_price", lambda code: {
        "price": 2070000, "change_1d_pct": -9.9, "change_5d_pct": -9.6,
        "volume": 5778751, "volume_ratio": 1.0})
    r = m.get_trend_market_context("kr", "000660")
    assert r["divergence"] and "괴리" in r["divergence"] and json.dumps(r)
    r2 = m.get_trend_market_context("us", "NVDA")   # px 없음 → divergence None, 직렬화
    assert "divergence" in r2 and r2["divergence"] is None and json.dumps(r2)


def test_deepen_synthesis_fail_saves_fallback(tmp_path, monkeypatch):
    """종합 LLM 실패해도 분석가 결과로 폴백 저장 → 캐시되어 재분석 안 함(B4)."""
    from tradingagents.hermes import trend_deepen
    import tradingagents.hermes.analyst_runner as ar
    s = TrendStore(root=tmp_path)
    s.set_policy("kr", min_sources=1)
    s.write([_kr_vol("035720", "2026-06-05")])
    monkeypatch.setattr(trend_deepen, "analysis_targets", lambda *a, **k: (["035720"], "2026-06-05"))
    monkeypatch.setattr(ar, "run_analyst", lambda t, d, a, **k: f"{a} 보고서 내용 샘플")
    monkeypatch.setattr(trend_deepen, "_synthesis_llm",
                        lambda: (_ for _ in ()).throw(RuntimeError("LLM down")))
    out = trend_deepen.deepen("kr", asof="2026-06-05", store=s)
    assert out["analyzed"] == 1
    saved = s.get_analysis("kr", "035720", "2026-06-05")
    assert saved is not None and "보류" in saved["summary"]   # 폴백 저장됨
    # 재실행 시 캐시 hit(분석가 0)
    monkeypatch.setattr(ar, "run_analyst",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("재분석 금지")))
    assert trend_deepen.deepen("kr", asof="2026-06-05", store=s)["cached"] == 1


def test_deepen_time_budget_defers(tmp_path, monkeypatch):
    """시간예산 0 이면 신규 종목 시작 안 함(deferred), 분석가 0 호출(N11)."""
    from tradingagents.hermes import trend_deepen
    import tradingagents.hermes.analyst_runner as ar
    s = TrendStore(root=tmp_path)
    s.set_policy("kr", min_sources=1)
    s.write([_kr_vol("035720", "2026-06-05"), _kr_vol("000660", "2026-06-05", abn=80, rank=2)])
    monkeypatch.setattr(trend_deepen, "analysis_targets", lambda *a, **k: (["035720", "000660"], "2026-06-05"))
    monkeypatch.setattr(ar, "run_analyst",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("예산0이면 호출 금지")))
    out = trend_deepen.deepen("kr", asof="2026-06-05", time_budget_s=0.0, store=s)
    assert out["analyzed"] == 0 and out["deferred"] == 2


def test_set_policy_validates_inputs(tmp_path):
    """LLM 경계 입력 검증 — 잘못된 market/min_sources 는 ValueError(쓰레기 정책 차단)."""
    import pytest as _pt
    s = TrendStore(root=tmp_path)
    for bad_mkt in ("jp", "", "US", "kr "):
        with _pt.raises(ValueError):
            s.set_policy(bad_mkt, min_sources=2)
    for bad_ms in (0, -1, 100, True, 2.5):
        with _pt.raises(ValueError):
            s.set_policy("kr", min_sources=bad_ms)
    with _pt.raises(ValueError):
        s.set_policy("us", rotation_peak_pct=-5)
    # 정상값은 통과
    assert s.set_policy("kr", min_sources=2)["min_sources"] == 2
    assert s.set_policy("us", rotation_peak_pct=12.5)["rotation_peak_pct"] == 12.5
    # 잘못된 입력으로 active 가 오염되지 않음(여전히 1개)
    assert sum(p["active"] for p in s.list_policies("kr")) == 1


def test_resolve_min_sources_policy_over_env(tmp_path, monkeypatch):
    """min_sources 우선순위: active 정책 > env(KR) > 기본(KR1·US2)."""
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    monkeypatch.setenv("TREND_KR_MIN_SOURCES", "1")
    assert trend_rank._resolve_min_sources("kr", s) == 1    # 정책 없음 → env
    assert trend_rank._resolve_min_sources("us", s) == 2    # US 기본
    s.set_policy("kr", min_sources=2)
    assert trend_rank._resolve_min_sources("kr", s) == 2    # 정책이 env 덮어씀
    s.set_policy("us", min_sources=3)
    assert trend_rank._resolve_min_sources("us", s) == 3    # US도 정책 적용


def test_digest_kr_respects_policy(monkeypatch, tmp_path):
    """정책 min_sources=2 면 단일 소스 종목 탈락(env 없이 정책만으로)."""
    from tradingagents.hermes import trend_rank
    s = TrendStore(root=tmp_path)
    s.write([SignalRow("2026-06-05", "2026-06-05", "kr", "005930", "naver_board",
                       "board_volume", COINCIDENT, raw_value=120, abnormal_value=120, rank=1)])
    monkeypatch.setattr(trend_rank, "_kr_provenance", lambda codes, **kw: {})
    monkeypatch.delenv("TREND_KR_MIN_SOURCES", raising=False)
    s.set_policy("kr", min_sources=2)
    assert "과열 주목 종목 [KR]" not in trend_rank.format_digest("kr", store=s)


def test_naver_datalab_filters_nonpositive_asvi(monkeypatch):
    """검색이 평소 이하(ASVI<=0)면 신호 아님 → 행 미생성(단 fetch_ok=True)."""
    from tradingagents.dataflows.trends import naver_datalab
    monkeypatch.setenv("NAVER_CLIENT_ID", "x")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "y")
    monkeypatch.setattr(naver_datalab, "_resolve_names", lambda codes: {"005930": "삼성전자"})
    # 당주(30)가 baseline(60) 이하 → ASVI<0 → 표면화 안 함
    declining = [60.0] * 8 + [30.0]
    monkeypatch.setattr(naver_datalab, "_fetch_series", lambda *a, **k: declining)
    rows, ok = naver_datalab.collect_naver_datalab(universe=["005930"], asof_date="2026-06-05")
    assert rows == [] and ok is True   # 수집은 성공(fetch_ok), 검색 쏠림만 없음
