"""dart_industry.py — company.json mock + KSIC 매칭 + HARD_CODED fallback."""
from __future__ import annotations

from unittest.mock import patch, Mock

from tradingagents.dataflows import dart_industry
from tradingagents.dataflows.dart_industry import (
    CompanyInfo, Peer, fetch_company_info, find_peers, _match_induty,
    HARD_CODED_PEERS,
)


def _fake_resp(status="000", body=None):
    m = Mock()
    m.raise_for_status = lambda: None
    base = {"status": status, "corp_name": "DB손해보험",
            "stock_name": "DB손해보험", "induty_code": "65120",
            "corp_cls": "Y"}
    if body:
        base.update(body)
    m.json = lambda: base
    return m


class TestMatchInduty:
    def test_exact_match(self):
        assert _match_induty("65120", "65120")

    def test_prefix_3_match(self):
        # 651xx 모두 보험업
        assert _match_induty("65120", "65110", prefix_len=3)

    def test_no_match(self):
        assert not _match_induty("65120", "26120")

    def test_empty(self):
        assert not _match_induty("", "65120")
        assert not _match_induty("65120", "")


class TestFetchCompanyInfo:
    def test_normal(self, tmp_path):
        # 캐시 경로 isolated
        with patch.object(dart_industry, "_cache_path",
                          return_value=str(tmp_path / "cc.json")), \
             patch.object(dart_industry, "_api_key", return_value="dummy"), \
             patch("tradingagents.dataflows.dart_industry._resolve_corp_code",
                   return_value="00100114"), \
             patch("requests.get", return_value=_fake_resp()):
            info = fetch_company_info("005830")
        assert info is not None
        assert info.ticker == "005830"
        assert info.corp_code == "00100114"
        assert info.induty_code == "65120"
        assert info.corp_name == "DB손해보험"

    def test_no_key_returns_none(self, tmp_path):
        with patch.object(dart_industry, "_cache_path",
                          return_value=str(tmp_path / "cc.json")), \
             patch.object(dart_industry, "_api_key", return_value=None):
            info = fetch_company_info("005830")
        assert info is None

    def test_status_error_returns_none(self, tmp_path):
        with patch.object(dart_industry, "_cache_path",
                          return_value=str(tmp_path / "cc.json")), \
             patch.object(dart_industry, "_api_key", return_value="dummy"), \
             patch("tradingagents.dataflows.dart_industry._resolve_corp_code",
                   return_value="00100114"), \
             patch("requests.get",
                   return_value=_fake_resp(status="013")):
            info = fetch_company_info("005830")
        assert info is None

    def test_cache_hit_skips_api(self, tmp_path):
        cache_file = tmp_path / "cc.json"
        cache_file.write_text(
            '{"005830": {"ticker": "005830", "corp_code": "00100114",'
            '"corp_name": "DB손해보험", "induty_code": "65120",'
            '"stock_name": "DB", "corp_cls": "Y"}}',
            encoding="utf-8",
        )
        with patch.object(dart_industry, "_cache_path",
                          return_value=str(cache_file)), \
             patch.object(dart_industry, "_api_key", return_value="dummy"), \
             patch("requests.get") as mock_get:
            info = fetch_company_info("005830")
        assert info is not None
        assert info.induty_code == "65120"
        mock_get.assert_not_called()


class TestFindPeers:
    def test_hard_coded_priority(self, tmp_path):
        # 005830 은 HARD_CODED_PEERS 에 있음 → KSIC API 호출 없이 즉시 반환
        with patch.object(dart_industry, "_cache_path",
                          return_value=str(tmp_path / "cc.json")), \
             patch("tradingagents.dataflows.dart_industry._load_corp_map",
                   return_value={"000810": "C1", "001450": "C2",
                                 "000060": "C3", "000540": "C4"}):
            peers = find_peers("005830", n=4)
        assert len(peers) == 4
        assert all(p.source == "hard_coded" for p in peers)
        assert peers[0].ticker == "000810"
        assert peers[0].corp_code == "C1"

    def test_candidate_matching(self, tmp_path):
        # ticker 가 HARD_CODED 에 없으면 candidate_tickers 매칭
        # company_info 4건 mock
        infos = {
            "999999": CompanyInfo("999999", "X", "Target", "26120", None, "Y"),
            "111111": CompanyInfo("111111", "Y", "Peer 1", "26120", None, "Y"),
            "222222": CompanyInfo("222222", "Z", "Other", "65120", None, "Y"),
            "333333": CompanyInfo("333333", "W", "Peer 2", "26120", None, "Y"),
        }

        def fake_fetch(t, *, cache=True):
            return infos.get(t)

        with patch.object(dart_industry, "fetch_company_info",
                          side_effect=fake_fetch):
            peers = find_peers(
                "999999", n=4,
                candidate_tickers=["111111", "222222", "333333"],
            )
        assert len(peers) == 2
        assert {p.ticker for p in peers} == {"111111", "333333"}
        assert all(p.source == "ksic" for p in peers)

    def test_no_target_induty_returns_empty(self):
        with patch.object(dart_industry, "fetch_company_info",
                          return_value=None):
            peers = find_peers(
                "999999", n=4, candidate_tickers=["111111"],
            )
        assert peers == []

    def test_no_candidates_returns_empty(self):
        # HARD_CODED 에 없고 candidate 도 없으면 빈 리스트
        peers = find_peers("999999", n=4)
        assert peers == []
