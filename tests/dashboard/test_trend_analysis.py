"""dashboard.trend_analysis — phase detection + 회귀 + 흡수자 + 동행성."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.trend_analysis import (
    KEY_TREND_SUBJECTS,
    compute_absorbers,
    compute_trend_report,
    detect_phases,
    phase_regression,
    render_trend_markdown,
    trend_concordance,
)


def _make_df(
    n: int = 400,
    cum_pattern: str = "up_down",
    seed: int = 0,
) -> pd.DataFrame:
    """추세 패턴이 있는 합성 holdings 데이터.

    pattern:
      - ``"up_down"``  — 전반 +, 후반 -
      - ``"flat"``     — 변동 없음
      - ``"monotone"`` — 단조 증가
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    if cum_pattern == "up_down":
        half = n // 2
        cum = np.concatenate([
            np.cumsum(rng.normal(1000, 100, half)),
            np.cumsum(rng.normal(-1000, 100, n - half)) + 1000 * half,
        ])
    elif cum_pattern == "flat":
        cum = rng.normal(0, 50, n).cumsum() * 0.01
    elif cum_pattern == "monotone":
        cum = np.cumsum(rng.normal(500, 50, n))
    else:
        raise ValueError(cum_pattern)
    # close 는 cum 에 강하게 연동 (테스트 시 신호 명확) + 작은 노이즈
    close = 50000 + cum * 0.05 + rng.normal(0, 50, n)
    # net_qty = day-over-day diff of cum
    net = np.diff(cum, prepend=0)
    # 다른 9 주체 — retail 은 zero-sum 흡수자 역할 (foreign 반대 방향, 절대량 큼)
    other_cum = -cum * 1.0 + rng.normal(0, 100, n).cumsum() * 0.1
    other_net = np.diff(other_cum, prepend=0)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": close,
        "foreign_registered_cum_qty": cum * 0.6,
        "foreign_unregistered_cum_qty": cum * 0.4,
        "foreign_registered_net_qty": net * 0.6,
        "foreign_unregistered_net_qty": net * 0.4,
        "retail_cum_qty": other_cum,
        "retail_net_qty": other_net,
        "pension_cum_qty": np.zeros(n),
        "pension_net_qty": np.zeros(n),
        "private_equity_cum_qty": np.zeros(n),
        "private_equity_net_qty": np.zeros(n),
        "investment_trust_cum_qty": np.zeros(n),
        "investment_trust_net_qty": np.zeros(n),
        "securities_cum_qty": np.zeros(n),
        "securities_net_qty": np.zeros(n),
        "bank_cum_qty": np.zeros(n),
        "bank_net_qty": np.zeros(n),
        "insurance_cum_qty": np.zeros(n),
        "insurance_net_qty": np.zeros(n),
        "etc_corporate_cum_qty": np.zeros(n),
        "etc_corporate_net_qty": np.zeros(n),
    })
    return df


@pytest.mark.unit
class TestDetectPhases:
    def test_up_down_pattern_yields_at_least_two_phases(self):
        df = _make_df(n=400, cum_pattern="up_down")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        phases = detect_phases(df["foreign_cum_qty"])
        assert len(phases) >= 2
        # 첫 phase 는 up, 나중 phase 는 down 포함
        trends = [t for _, _, t in phases]
        assert "up" in trends
        assert "down" in trends

    def test_short_series_returns_single_phase(self):
        # 윈도우(60) + min_days(60) 미만 → 단일 phase fallback
        df = _make_df(n=80, cum_pattern="monotone")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        phases = detect_phases(df["foreign_cum_qty"])
        assert len(phases) == 1
        assert phases[0][0] == 0
        assert phases[0][1] == 80

    def test_monotone_pattern_single_up_phase(self):
        df = _make_df(n=300, cum_pattern="monotone")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        phases = detect_phases(df["foreign_cum_qty"])
        # 단조 증가 → 모든 phase up (또는 flat)
        trends = [t for _, _, t in phases]
        assert "down" not in trends

    def test_phases_cover_full_range(self):
        df = _make_df(n=400, cum_pattern="up_down")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        phases = detect_phases(df["foreign_cum_qty"])
        # phases 가 (0, n) 전체를 cover 해야 함
        assert phases[0][0] == 0
        assert phases[-1][1] == 400
        # phase 들이 인접 — 끝과 다음 시작이 같음
        for i in range(len(phases) - 1):
            assert phases[i][1] == phases[i + 1][0]


@pytest.mark.unit
class TestPhaseRegression:
    def test_strong_correlation(self):
        df = _make_df(n=300, cum_pattern="monotone")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        reg = phase_regression(df, 0, len(df), "foreign_cum_qty")
        # close = 50000 + cum * 0.01 + noise → r 매우 강함
        assert reg["pearson_r"] is not None
        assert reg["pearson_r"] > 0.9
        assert reg["r_squared"] is not None
        assert reg["n"] == 300

    def test_empty_segment_returns_none_r(self):
        df = _make_df(n=300)
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        reg = phase_regression(df, 0, 2, "foreign_cum_qty")
        assert reg["pearson_r"] is None
        assert reg["n"] == 2


@pytest.mark.unit
class TestComputeAbsorbers:
    def test_returns_role_per_subject(self):
        df = _make_df(n=300, cum_pattern="up_down")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        absorbers = compute_absorbers(df, 0, len(df), "foreign")
        # retail 은 cum 반대 방향이라 role 있음
        roles = {a["subject"]: a["role"] for a in absorbers}
        assert "retail" in roles
        assert roles["retail"] in ("absorber", "supplier", "same_side")

    def test_excludes_foreign_subregistries(self):
        df = _make_df(n=300, cum_pattern="up_down")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        absorbers = compute_absorbers(df, 0, len(df), "foreign")
        subjects = {a["subject"] for a in absorbers}
        # foreign 분석 시 등록/비등록 sub 는 제외 — 자기 자신 합
        assert "foreign_registered" not in subjects
        assert "foreign_unregistered" not in subjects

    def test_sorted_by_absolute_delta(self):
        df = _make_df(n=300, cum_pattern="up_down")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        absorbers = compute_absorbers(df, 0, len(df), "foreign")
        if len(absorbers) >= 2:
            deltas = [abs(a["delta"]) for a in absorbers]
            assert deltas == sorted(deltas, reverse=True)


@pytest.mark.unit
class TestTrendConcordance:
    def test_returns_percent_in_range(self):
        df = _make_df(n=400, cum_pattern="up_down")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        conc = trend_concordance(df, "foreign_cum_qty")
        assert conc["agreement_pct"] is not None
        assert 0 <= conc["agreement_pct"] <= 100
        assert conc["n"] > 0

    def test_high_concordance_when_cum_drives_price(self):
        # cum_pattern="monotone" 이면 close = 50000 + cum * 0.01 → 거의 1:1
        df = _make_df(n=400, cum_pattern="monotone")
        df = df.assign(
            foreign_cum_qty=df["foreign_registered_cum_qty"] + df["foreign_unregistered_cum_qty"]
        )
        conc = trend_concordance(df, "foreign_cum_qty")
        assert conc["agreement_pct"] >= 80


@pytest.mark.unit
class TestComputeTrendReport:
    def test_returns_subjects_dict(self):
        df = _make_df(n=400, cum_pattern="up_down")
        report = compute_trend_report(df)
        assert "subjects" in report
        # KEY_TREND_SUBJECTS = (foreign, retail)
        for s in KEY_TREND_SUBJECTS:
            assert s in report["subjects"]
            assert "phases" in report["subjects"][s]
            assert "trend_concordance" in report["subjects"][s]
            assert "label" in report["subjects"][s]

    def test_phase_dates_within_data_range(self):
        df = _make_df(n=400, cum_pattern="up_down")
        report = compute_trend_report(df)
        for subj in report["subjects"].values():
            for p in subj["phases"]:
                assert p["start_date"] >= df["date"].iloc[0]
                assert p["end_date"] <= df["date"].iloc[-1]
                assert p["n_days"] > 0


@pytest.mark.unit
class TestRenderTrendMarkdown:
    def test_contains_phase_table_header(self):
        df = _make_df(n=400, cum_pattern="up_down")
        report = compute_trend_report(df)
        md = render_trend_markdown(report)
        assert "## 8. 추세 분석" in md
        assert "Phase" in md
        assert "추세" in md
        # 자동 해석 마크
        assert "자동 해석" in md
