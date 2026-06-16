"""종목별 주도세력 발굴 레이어 단위 테스트.

엔진(룩어헤드0·비용)은 기존 _simulate_v2 테스트가 커버하므로, 여기서는 신규
**결정 로직**(spec 빌더·시간분할·횡단 FDR·합격 매트릭스)과 **forward 어댑터**
(레지스트리 격리·멱등·ledger 비접촉)를 손계산/합성으로 검증한다.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.hermes.strategy_spec_v2 import validate_spec_v2
from tradingagents.newloop import leader_discovery as ld


@pytest.mark.unit
class TestMakeLeaderSpec:
    def test_valid_v2_spec(self):
        spec = ld.make_leader_spec("foreign_registered", entry_window=5, min_z=2.0)
        validate_spec_v2(spec)   # 던지면 실패
        assert spec["entry"]["all_of"][0]["signal"] == "flow_zscore"
        assert spec["entry"]["all_of"][0]["subject"] == "foreign_registered"
        ex = spec["exit"]["signal_all_of"][0]
        assert ex["signal"] == "flow_unwind" and ex["subject"] == "foreign_registered"
        assert spec["exit"]["max_hold_days"] == ld.EXIT_MAX_HOLD


@pytest.mark.unit
class TestBHSurvivors:
    def test_known_pvalues(self):
        # BH(α=0.10), m=4: 정렬 p=[.001,.04,.2,.5], 임계 .025/.05/.075/.10
        # .001<=.025(r1) ✓, .04<=.05(r2) ✓ → keep{0,2}; .2,.5 탈락.
        pvals = [0.001, 0.2, 0.04, 0.5]
        keep = ld._bh_survivors(pvals, 0.10)
        assert keep == {0, 2}

    def test_empty(self):
        assert ld._bh_survivors([], 0.10) == set()

    def test_none_survive(self):
        assert ld._bh_survivors([0.5, 0.6, 0.7], 0.10) == set()


@pytest.mark.unit
class TestSplitDates:
    def test_three_windows_with_embargo(self):
        dates = pd.bdate_range("2018-01-01", periods=400).astype(str)
        df = pd.DataFrame({"date": dates})
        sp = ld._split_dates(df)
        cut = int(400 * ld.IN_SAMPLE_PCT / 100)        # 280
        assert sp["is_lo"] == dates[0]
        assert sp["is_hi"] == dates[cut]
        # embargo 갭: oos_lo 는 cutoff 보다 DEFAULT_EMBARGO_DAYS 거래일 뒤.
        assert sp["oos_lo"] == dates[cut + ld.DEFAULT_EMBARGO_DAYS]
        assert sp["oos_hi"] == "2025-07-01"
        # IS 와 OOS 비겹침.
        assert sp["is_hi"] < sp["oos_lo"]

    def test_post_ceiling_excluded(self):
        # 2025-06-30 이후 날짜는 분할 모집단에서 제외(채점 금지 구간).
        dates = list(pd.bdate_range("2018-01-01", periods=350).astype(str)) + ["2025-12-31"]
        df = pd.DataFrame({"date": dates})
        sp = ld._split_dates(df)
        assert sp["is_lo"] <= ld.SCORE_DATE_HI
        assert sp["oos_hi"] == "2025-07-01"

    def test_insufficient_history(self):
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=100).astype(str)})
        assert ld._split_dates(df) is None


@pytest.mark.unit
class TestScore:
    def test_alpha_computed(self):
        # 2개월 군집, net 가 basket 평균을 양(+)으로 상회 → α>0, dict 키 존재.
        pairs = [
            (3.0, 1.0, "2021-01"), (4.0, 1.5, "2021-01"), (2.0, 0.5, "2021-01"),
            (5.0, 2.0, "2021-02"), (3.5, 1.0, "2021-02"), (-1.0, -2.0, "2021-02"),
        ]
        m = ld._score(pairs)
        assert m is not None
        assert m["n_trades"] == 6 and m["n_clusters"] == 2
        assert m["alpha_pct"] is not None and m["t_alpha"] is not None

    def test_too_few(self):
        assert ld._score([(1.0, 0.0, "2021-01")]) is None


@pytest.mark.unit
class TestFinalizeQualification:
    def _stock(self, tk, is_a, oos_a, oos_t, n=20, clusters=6):
        return {
            "ticker": tk, "leader_subject": "pension",
            "is_metrics": {"alpha_pct": is_a, "t_alpha": 4.0, "n_trades": 40, "n_clusters": 10},
            "oos_metrics": {"alpha_pct": oos_a, "t_alpha": oos_t,
                            "n_trades": n, "n_clusters": clusters},
            "n_trials": 36,
        }

    def test_sign_match_and_fdr(self):
        results = [
            self._stock("A", 1.0, 0.8, 3.5),    # IS+/OOS+ 강함 → qualified
            self._stock("B", 1.0, -0.5, -2.0),  # OOS 부호반전 → 탈락
            self._stock("C", 1.0, 0.6, 3.2, n=ld.OOS_MIN_TRADES - 1),  # OOS 거래수 floor 미달 → 탈락
        ]
        ld._finalize(results, n_eff=100, fdr_alpha=0.10)
        by = {r["ticker"]: r for r in results}
        assert by["A"]["qualified"] is True
        assert by["A"]["sign_match"] is True
        assert by["B"]["qualified"] is False
        assert by["B"]["sign_match"] is False
        assert by["C"]["qualified"] is False
        assert by["C"]["oos_floor_pass"] is False

    def test_missing_oos_not_qualified(self):
        results = [{"ticker": "X", "is_metrics": {"alpha_pct": 1.0}, "oos_metrics": None,
                    "n_trials": 36}]
        ld._finalize(results, n_eff=50)
        assert results[0]["qualified"] is False


@pytest.mark.unit
class TestForwardAdapter:
    def _map(self):
        spec = ld.make_leader_spec("pension", entry_window=5, min_z=2.0)
        return {
            "engine_version": "deadbeef",
            "stocks": [
                {"ticker": "005830", "market": "KOSPI", "name": "DB손해보험",
                 "leader_subject": "pension", "entry_spec": spec["entry"],
                 "exit_spec": spec["exit"], "is_metrics": {"alpha_pct": 1.0},
                 "oos_metrics": {"alpha_pct": 0.8}, "qualified": True},
                {"ticker": "000000", "leader_subject": "insurance",
                 "entry_spec": spec["entry"], "exit_spec": spec["exit"],
                 "qualified": False},   # 미합격 → 등록 안 됨
            ],
        }

    def test_register_only_qualified_and_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ld, "REGISTRY", tmp_path / "reg.json")
        added = ld.register_leaders(self._map(), registered_at="2026-06-16T00:00:00+09:00")
        assert added == 1
        reg = json.loads((tmp_path / "reg.json").read_text())
        assert [e["ticker"] for e in reg] == ["005830"]
        assert reg[0]["freeze_date"] == ld.SCORE_DATE_HI
        # 재등록은 동결 유지(멱등) — 0건 추가.
        assert ld.register_leaders(self._map(), registered_at="2026-07-01T00:00:00+09:00") == 0
        assert len(json.loads((tmp_path / "reg.json").read_text())) == 1

    def test_observe_eligibility_and_no_ledger(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ld, "REGISTRY", tmp_path / "reg.json")
        ld.register_leaders(self._map(), registered_at="2026-01-01T00:00:00+09:00")

        # 빈 df 로더 + 빈 kospi → fwd=None, 그러나 경과주수·자격은 계산.
        def fake_loader(_tk):
            return pd.DataFrame(), {"market": "KOSPI"}

        rows = ld.observe_leaders(now_iso="2026-03-01T00:00:00+09:00",
                                  loader=fake_loader, kospi_fetcher=lambda _a, _b: pd.DataFrame())
        assert len(rows) == 1
        r = rows[0]
        assert r["ticker"] == "005830"
        assert r["weeks_elapsed"] > ld.MIN_FORWARD_WEEKS
        assert r["eligible_for_decision"] is True
        assert r["fwd"] is None     # 데이터 없음 → 관찰 메트릭 없음(채점 아님)
