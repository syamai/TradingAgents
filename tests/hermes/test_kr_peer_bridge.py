"""US 관심 쏠림 종목 → 유사 한국 종목 브리지 단위 테스트.

핵심: (1) 캐시 멱등 — 같은 US 티커 재요청 시 LLM 재호출 0, (2) TTL 만료 시 캐시
무시, (3) LLM 출력 정제(6자리 검증·중복 제거·개수 상한), (4) kr_universe 가 US
fade 와치리스트를 유사 한국 코드로 변환.
"""
from __future__ import annotations

import pytest

from tradingagents.dataflows.trend_store import TrendStore
from tradingagents.dataflows.trends.base import SignalRow
from tradingagents.hermes.kr_peer_bridge import (
    KrPeerCache,
    _KrPeer,
    _KrPeerList,
    _clean_peers,
    kr_universe,
    map_us_to_kr,
)

pytestmark = pytest.mark.unit


class _FakeLLM:
    """with_structured_output → self, invoke → 프롬프트의 티커로 고정 응답."""

    model = "fake-model"

    def __init__(self, peers_by_ticker: dict):
        self._peers = peers_by_ticker
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.calls += 1
        for t, peers in self._peers.items():
            if t in prompt:
                return _KrPeerList(peers=[_KrPeer(**p) for p in peers])
        return _KrPeerList(peers=[])


def _us_row(entity, source, abn, rank=1):
    return SignalRow(
        "2026-06-05", "2026-06-05", "us", entity, source,
        "m", "C", abnormal_value=abn, rank=rank,
    )


def test_map_cache_miss_then_hit(tmp_path):
    fake = _FakeLLM({"NVDA": [{"code": "005930", "name": "삼성전자", "reason": "반도체"}]})
    cache = KrPeerCache(root=tmp_path)
    m1 = map_us_to_kr(["NVDA"], llm=fake, cache=cache)
    assert m1["NVDA"][0]["code"] == "005930"
    assert fake.calls == 1
    # 같은 티커 재요청 → 캐시 hit, LLM 재호출 없음
    m2 = map_us_to_kr(["NVDA"], llm=fake, cache=cache)
    assert m2["NVDA"][0]["name"] == "삼성전자"
    assert fake.calls == 1


def test_cache_ttl_expired(tmp_path):
    cache = KrPeerCache(root=tmp_path)
    cache.put("NVDA", [{"code": "005930", "name": "삼성전자", "reason": "x"}])
    with cache._conn() as conn:
        conn.execute(
            "UPDATE kr_peer_map SET cached_at=? WHERE us_ticker=?",
            ("2020-01-01T00:00:00Z", "NVDA"),
        )
    assert cache.get("NVDA", ttl_days=90) is None


def test_clean_peers_validates(tmp_path):
    peers = [
        {"code": "005930", "name": "삼성전자", "reason": "a"},
        {"code": "005930", "name": "dup", "reason": "b"},     # 중복
        {"code": "12345", "name": "bad", "reason": ""},        # 6자리 아님
        {"code": "000660", "name": "SK하이닉스", "reason": "HBM"},
    ]
    out = _clean_peers(peers, max_peers=5)
    assert [p["code"] for p in out] == ["005930", "000660"]


def test_clean_peers_caps(tmp_path):
    peers = [{"code": f"00000{i}", "name": str(i), "reason": ""} for i in range(1, 9)]
    assert len(_clean_peers(peers, max_peers=3)) == 3


def test_kr_universe_from_us_fade(tmp_path):
    ts = TrendStore(root=tmp_path / "trends")
    # NVDA 가 2소스 동의 → fade_ranking('us') 통과
    ts.write([
        _us_row("NVDA", "apewisdom", 2.0),
        _us_row("NVDA", "finviz_unusual", 1.0),
    ])
    fake = _FakeLLM({"NVDA": [{"code": "000660", "name": "SK하이닉스", "reason": "HBM"}]})
    cache = KrPeerCache(root=tmp_path)
    uni = kr_universe(store=ts, llm=fake, cache=cache, top_us=10)
    assert uni == ["000660"]


def test_reverse_provenance(tmp_path):
    cache = KrPeerCache(root=tmp_path)
    cache.put("NVDA", [{"code": "000660", "name": "SK하이닉스", "reason": "HBM"}])
    rev = cache.reverse(["000660"])
    assert rev["000660"]["us"] == "NVDA"
    assert rev["000660"]["name"] == "SK하이닉스"
