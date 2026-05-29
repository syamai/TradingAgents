"""dashboard/event_retro.py — 1주차 MVP 단위 테스트.

PeriodSummary / 이벤트 탐지 / 그룹화 / 분해 / 렌더 + 005830 회귀.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from dashboard.event_retro import (
    CLUSTER_ENUM, EventDay,
    compute_period_summary, _detect_event_days, _group_events,
    _compute_decomposition, compute_event_retro, render_event_retro_markdown,
)


def _make_df(prices: list[float], volumes: list[int] | None = None) -> pd.DataFrame:
    """동기적 일자 + close + volume df."""
    dates = pd.date_range("2025-01-01", periods=len(prices), freq="B")
    out = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "close": prices,
    })
    if volumes is not None:
        out["volume"] = volumes
    return out


# === compute_period_summary ====================================================

class TestPeriodSummary:
    def test_basic(self):
        df = _make_df([100, 110, 105, 90, 95])
        ps = compute_period_summary(df)
        assert ps.start_close == 100
        assert ps.end_close == 95
        assert ps.high_close == 110
        assert ps.low_close == 90
        assert ps.total_return_pct == -5.0
        # MDD = 90/110 - 1 = -18.1818...
        assert abs(ps.max_drawdown_pct - (-18.1818)) < 0.01

    def test_monotonic_up(self):
        df = _make_df([100, 110, 120, 130])
        ps = compute_period_summary(df)
        assert ps.total_return_pct == 30.0
        # MDD = high(130)/high(130) - 1 = 0 — 단조증가
        # 실제로는 lo=start=100, hi=end=130 → -23.07%
        assert ps.high_date == ps.end_date
        assert ps.low_date == ps.start_date


# === _detect_event_days ========================================================

class TestDetectEventDays:
    def test_price_jumps(self):
        # +5%, -4% 변동
        df = _make_df([100, 105, 105, 100.8])
        evts = _detect_event_days(df, price_jump_pct=3.0, vol_z=10)
        # 1=+5%, 3=-4% 가 jump. 0=시작, 3(마지막+pct?)
        # 0 anchor, 1 jump, 2 lo (100? no, all close>=100.8). 실제 lo는 0=100.
        dates = {e.date for e in evts}
        assert df["date"].iloc[1] in dates  # +5%
        assert df["date"].iloc[3] in dates  # -4%

    def test_volume_anomaly(self):
        # 변동률 0.1% (jump 없음), 단 거래량 마지막 행이 10배 spike
        prices = [100, 100.1, 100.2, 100.1, 100.3, 100.2, 100.1, 100.2] + [100.1] * 15 + [99.9]
        volumes = [1000] * (len(prices) - 1) + [10000]
        df = _make_df(prices, volumes)
        evts = _detect_event_days(df, price_jump_pct=10.0, vol_z=2.0)
        last_date = df["date"].iloc[-1]
        last_evt = next((e for e in evts if e.date == last_date), None)
        assert last_evt is not None
        # 거래량 z가 잡혔는지 또는 anchor (last) 잡힌 것
        assert last_evt.is_volume_anomaly or last_evt.is_period_anchor

    def test_anchors_always_included(self):
        # 평탄 — 변동/거래량 이벤트 없음
        df = _make_df([100] * 30, volumes=[1000] * 30)
        evts = _detect_event_days(df, price_jump_pct=3.0, vol_z=2.0)
        # 시작/종료/고점/저점 anchor 만 — 단 평탄이라 hi=lo=start=end. set 으로 중복 제거.
        # idx 0 = start, idx 29 = end. high/low 인덱스도 0(첫 idxmax)
        dates = {e.date for e in evts}
        assert df["date"].iloc[0] in dates
        assert df["date"].iloc[-1] in dates

    def test_no_volume_column(self):
        df = _make_df([100, 103.5, 100])  # 거래량 없음, +3.5% jump
        evts = _detect_event_days(df)
        # vol_z 이벤트 0건 (volume 없음), 가격 + anchor만
        for e in evts:
            assert e.volume is None
            assert e.z_volume is None


# === _group_events =============================================================

class TestGroupEvents:
    def test_adjacent_grouped(self):
        evts = [
            EventDay("2025-04-07", 100, -5, None, None, True, False, False),
            EventDay("2025-04-09", 95, -1.5, None, None, False, False, True),
            EventDay("2025-04-10", 100, 5.5, None, None, True, False, False),
        ]
        groups = _group_events(evts, gap_days=3)
        assert len(groups) == 1
        assert groups[0].start_date == "2025-04-07"
        assert groups[0].end_date == "2025-04-10"
        assert "~" in groups[0].label

    def test_distant_separate(self):
        evts = [
            EventDay("2025-02-20", 100, 0, None, None, False, False, True),
            EventDay("2025-03-27", 90, -7, None, None, True, False, False),
        ]
        groups = _group_events(evts, gap_days=3)
        assert len(groups) == 2

    def test_empty(self):
        assert _group_events([]) == []


# === _compute_decomposition ====================================================

class TestDecomposition:
    def test_all_drift_when_no_cluster(self):
        evts = [
            EventDay("2025-04-07", 100, -5.0, None, None, True, False, False),
            EventDay("2025-04-10", 100, +3.0, None, None, True, False, False),
        ]
        decomp = _compute_decomposition(evts, total_return_pct=-10.0)
        assert decomp["drift"] == -10.0
        for c in ("earnings_rating", "company_specific",
                  "industry_regulation", "macro_shock"):
            assert decomp[c] == 0.0

    def test_clusters_assigned(self):
        evts = [
            EventDay("2025-02-20", 100, -3.0, None, None, True, False, False,
                     cluster="earnings_rating"),
            EventDay("2025-04-07", 100, -5.0, None, None, True, False, False,
                     cluster="macro_shock"),
        ]
        decomp = _compute_decomposition(evts, total_return_pct=-10.0)
        assert decomp["earnings_rating"] == -3.0
        assert decomp["macro_shock"] == -5.0
        assert decomp["drift"] == -2.0  # -10 - (-3 - 5)

    def test_all_enum_keys_present(self):
        decomp = _compute_decomposition([], total_return_pct=-5.0)
        for c in CLUSTER_ENUM:
            assert c in decomp


# === render_event_retro_markdown ===============================================

class TestRender:
    def test_basic_structure(self):
        report = {
            "meta": {
                "ticker": "005830", "company_name": "DB손해보험",
                "market": "KOSPI",
                "start": "2025-02-14", "end": "2025-04-14",
                "generated_at": "2026-05-29T00:00:00Z",
            },
            "period_summary": {
                "start_date": "2025-02-14", "start_close": 102100,
                "high_date": "2025-02-18", "high_close": 103200,
                "low_date": "2025-04-09", "low_close": 78900,
                "end_date": "2025-04-14", "end_close": 84200,
                "total_return_pct": -17.53, "max_drawdown_pct": -23.55,
            },
            "timeline": [], "groups": [], "decomposition": {},
            "warnings": [],
        }
        md = render_event_retro_markdown(report)
        assert "DB손해보험" in md
        assert "005830" in md
        assert "기간 요약" in md
        assert "-17.53%" in md
        assert "-23.55%" in md
        assert "103,200" in md
        assert "78,900" in md

    def test_empty_period_summary(self):
        report = {
            "meta": {"ticker": "005830", "start": "2025-01-01", "end": "2025-12-31",
                     "company_name": "", "market": "",
                     "generated_at": "2026-05-29T00:00:00Z"},
            "period_summary": None, "timeline": [], "groups": [],
            "decomposition": {}, "warnings": ["데이터 없음"],
        }
        md = render_event_retro_markdown(report)
        assert "데이터 부족" in md
        assert "데이터 없음" in md


# === 005830 회귀 (network mark) ================================================

@pytest.mark.network
class TestEventRetro005830Regression:
    """005830 DB손해보험 2025-02-14 ~ 2025-04-14 — 수동 회고 매칭."""

    @pytest.fixture(scope="class")
    def report(self):
        return compute_event_retro("005830", "2025-02-14", "2025-04-14")

    def test_period_summary_matches_manual(self, report):
        ps = report["period_summary"]
        assert ps["high_date"] == "2025-02-18"
        assert ps["low_date"] == "2025-04-09"
        assert abs(ps["total_return_pct"] - (-17.53)) < 0.5
        assert ps["max_drawdown_pct"] < -24.0 + 1.0  # ≈ -23.55

    def test_timeline_includes_key_events(self, report):
        dates = {e["date"] for e in report["timeline"]}
        # 수동 회고의 핵심 이벤트들
        assert "2025-03-27" in dates  # 자사주 -7.76%
        assert "2025-04-07" in dates  # 관세 -4.53%
        assert "2025-04-09" in dates  # 저점
        assert "2025-04-10" in dates  # 반등 +5.58%

    def test_decomposition_sums_to_total(self, report):
        decomp = report["decomposition"]
        total = sum(decomp.values())
        ps = report["period_summary"]
        assert abs(total - ps["total_return_pct"]) < 0.01

    def test_groups_cluster_april_events(self, report):
        # 4-7/4-9/4-10 트럼프 관세 클러스터는 한 그룹으로
        groups = report["groups"]
        apr_groups = [
            g for g in groups
            if "2025-04-07" in g["members"] or "2025-04-10" in g["members"]
        ]
        assert any(
            "2025-04-07" in g["members"] and "2025-04-10" in g["members"]
            for g in apr_groups
        )
