"""dashboard/holdings_chart.py 단위 테스트 — figure 구조, 상수, load round-trip."""
from __future__ import annotations

import pandas as pd
import pytest

from dashboard.holdings_chart import (
    SUBJECTS_ORDER, SUBJECT_LABELS, SUBJECT_COLORS,
    SUBS_10, ALL_SUBJECTS,
    load_holdings, make_figure, save_html,
)
from tradingagents.dataflows.kis_history_store import KisHistoryStore


# === 상수 사양 ===

@pytest.mark.unit
class TestConstants:
    def test_subject_labels_complete(self):
        assert set(SUBJECT_LABELS) == set(SUBJECTS_ORDER)

    def test_subject_colors_complete(self):
        assert set(SUBJECT_COLORS) == set(SUBJECTS_ORDER)

    def test_subjects_order_matches_all_subjects(self):
        assert set(SUBJECTS_ORDER) == set(ALL_SUBJECTS)
        assert len(SUBJECTS_ORDER) == 11

    def test_subs_10_subset_of_order(self):
        # 10 sub 모두 SUBJECTS_ORDER 에 포함되고, 추가로 외국인 통합 1개만 더 있어야 한다.
        assert set(SUBS_10).issubset(set(SUBJECTS_ORDER))
        assert set(SUBJECTS_ORDER) - set(SUBS_10) == {"foreign"}


# === make_figure 구조 ===

@pytest.mark.unit
class TestMakeFigure:
    def test_empty_input_returns_empty_figure(self):
        fig = make_figure(pd.DataFrame())
        assert len(fig.data) == 0
        assert fig.layout.height == 900

    def test_subplot_count_3_rows(self):
        # 빈 df 라도 layout 의 yaxis 개수가 row 수 + secondary y 만큼 생성된다.
        # secondary y row 1 → y, y2 / row 2 → y3 / row 3 → y4 (총 4 axis).
        fig = make_figure(_sample_holdings())
        yaxes = [a for a in dir(fig.layout) if a.startswith("yaxis")]
        assert len(yaxes) == 4

    def test_trace_count_matches_subjects(self):
        df = _sample_holdings()
        fig = make_figure(df)
        # 종가(1) + 가격변동률 bar(1) + 11 cum_qty(11) + 10 pct(10) = 23
        assert len(fig.data) == 23

    def test_visible_subjects_legendonly(self):
        from dashboard.holdings_chart import LEGEND_LABELS
        df = _sample_holdings()
        fig = make_figure(df, visible_subjects=["foreign", "retail"])
        # row 2 의 cum_qty trace 중 foreign·retail 만 True, 나머지 9 개는 'legendonly'.
        row2 = [t for t in fig.data if t.yaxis == "y3"]
        names_visible = [t.name for t in row2 if t.visible is True]
        # Row 2 trace.name 은 LEGEND_LABELS (짧은 라벨) 사용 — 외국인은 "통합"
        assert LEGEND_LABELS["foreign"] in names_visible
        assert LEGEND_LABELS["retail"] in names_visible
        assert len(names_visible) == 2

    def test_title_propagates(self):
        fig = make_figure(_sample_holdings(), title="삼성전자 005930")
        assert fig.layout.title.text == "삼성전자 005930"

    def test_invalid_pct_mode_raises(self):
        with pytest.raises(ValueError, match="pct_mode"):
            make_figure(_sample_holdings(), pct_mode="weekly")

    def test_pct_mode_changes_row3_title(self):
        cum = make_figure(_sample_holdings(), pct_mode="cumulative")
        daily = make_figure(_sample_holdings(), pct_mode="daily")
        # row 3 subplot title 은 layout.annotations[2].text (subplot_titles 순서 그대로)
        assert "누적" in cum.layout.annotations[2].text
        assert "일별" in daily.layout.annotations[2].text

    def test_pct_mode_daily_uses_net_qty(self):
        """일별 모드 row 3 첫 행의 비중이 net_qty 절댓값 분포와 일치."""
        from dashboard.holdings_chart import SUBS_10
        df = _sample_holdings()
        fig = make_figure(df, pct_mode="daily")
        # row 3 는 yaxis="y4"
        row3 = [t for t in fig.data if t.yaxis == "y4"]
        assert len(row3) == 10
        # _sample_holdings 첫 행: retail=100, foreign_registered=50 → 합=150
        # retail 비중 = 100/150 * 100 ≈ 66.67%, foreign_registered ≈ 33.33%
        # trace 순서는 SUBS_10 순서: foreign_registered, foreign_unregistered, ...
        first_row_pcts = {t.name: t.y[0] for t in row3}
        assert first_row_pcts["개인 (비중)"] == pytest.approx(66.6667, abs=0.01)
        assert first_row_pcts["외국인(등록) (비중)"] == pytest.approx(33.3333, abs=0.01)
        # 다른 sub 는 0
        zero_subs = [n for s in SUBS_10
                     for n in [SUBJECT_LABELS[s] + " (비중)"]
                     if s not in ("retail", "foreign_registered")]
        for name in zero_subs:
            assert first_row_pcts[name] == 0.0


# === load_holdings round trip ===

@pytest.mark.unit
class TestLoadHoldings:
    def test_round_trip_through_store(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("parquet", "sqlite"))
        # raw investor 2 rows → holdings hook 이 자동 materialize.
        store.write("005930", "investor", [
            _investor_row("2026-05-01", 100, retail_qty=100),
            _investor_row("2026-05-02", 101, retail_qty=-50),
        ])
        df, meta = load_holdings("005930", store=store)
        assert len(df) == 2
        assert list(df["retail_cum_qty"]) == [100, 50]
        # set_ticker_metadata 안 했으니 meta 는 빈 dict.
        assert meta == {}

    def test_meta_returned_when_set(self, tmp_path):
        store = KisHistoryStore(root=tmp_path, backends=("parquet", "sqlite"))
        store.write("005930", "investor",
                    [_investor_row("2026-05-01", 100, retail_qty=1)])
        store.set_ticker_metadata("005930", "삼성전자", "KOSPI")
        _, meta = load_holdings("005930", store=store)
        assert meta["company_name"] == "삼성전자"
        assert meta["market"] == "KOSPI"


# === save_html ===

@pytest.mark.unit
class TestSaveHtml:
    def test_writes_file(self, tmp_path):
        fig = make_figure(_sample_holdings())
        out = tmp_path / "sub" / "chart.html"
        path = save_html(fig, out)
        assert path.exists()
        # CDN 모드라 파일 자체는 작아야 하고, plotly.js src 가 포함되어야 한다.
        content = path.read_text(encoding="utf-8")
        assert "plotly" in content
        assert "cdn.plot.ly" in content or "cdn.plotly" in content


# === helpers ===

def _zero_subs() -> dict:
    out = {}
    for s in SUBS_10:
        out[f"{s}_qty"] = 0
    return out


def _investor_row(date: str, close: int, **overrides) -> dict:
    base = {"date": date, "close": close, **_zero_subs()}
    base.update(overrides)
    return base


def _sample_holdings() -> pd.DataFrame:
    """compute_holdings 를 거친 형태의 mini DataFrame."""
    from tradingagents.dataflows.kis_holdings import compute_holdings
    rows = [
        _investor_row("2026-05-01", 100, retail_qty=100,
                      foreign_registered_qty=50),
        _investor_row("2026-05-02", 105, retail_qty=-30, pension_qty=20),
        _investor_row("2026-05-03", 102, retail_qty=10, bank_qty=5),
    ]
    return compute_holdings(pd.DataFrame(rows))
