"""labeling_cron entry script 검증.

CLI argparse + run() + 카운트 dict 출력. 실제 LabelingScheduler 로직은
``test_labeling.py`` 에서 검증 — 여기는 진입점·dry_run·argparse 만.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.hermes import labeling_cron
from tradingagents.hermes.hypothesis_store import HypothesisStore


def _record(**overrides) -> dict:
    base = {
        "ticker": "005930.KS",
        "as_of_date": "2026-05-01",
        "overall_stance": "bullish",
        "overall_confidence": 0.7,
        "hypotheses": [{
            "id": "h1",
            "claim": "test",
            "direction": "bullish",
            "confidence": 0.7,
            "evidence_tools": ["t"],
            "evidence_excerpts": ["e"],
            "horizon_weeks": 4,
            "predicted_relative_return_pct": 3.0,
        }],
    }
    base.update(overrides)
    return base


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    """labeling_cron.run 이 디폴트 ``HypothesisStore()`` 를 만들 때 tmp 경로
    쓰도록 monkeypatch."""
    monkeypatch.setattr(
        "tradingagents.hermes.labeling_cron.HypothesisStore",
        lambda: HypothesisStore(root=tmp_path),
    )
    return HypothesisStore(root=tmp_path)


@pytest.mark.unit
class TestDryRun:
    def test_dry_run_counts_pending_by_horizon(self, isolated_store):
        isolated_store.save(_record(as_of_date="2026-05-01"))
        # horizon_weeks=2 가설 추가.
        rec2 = _record(as_of_date="2026-05-15")
        rec2["hypotheses"][0]["horizon_weeks"] = 2
        rec2["hypotheses"][0]["predicted_relative_return_pct"] = 2.0
        isolated_store.save(rec2)

        result = labeling_cron.run(dry_run=True)

        assert result["dry_run"] is True
        assert result["pending_count"] == 2
        # horizon_weeks 별 카운트.
        assert result["by_horizon"] == {"2": 1, "4": 1}

    def test_dry_run_empty_db(self, isolated_store):
        result = labeling_cron.run(dry_run=True)
        assert result["pending_count"] == 0
        assert result["by_horizon"] == {}


@pytest.mark.unit
class TestRun:
    def test_run_with_too_early_only(self, isolated_store, monkeypatch):
        isolated_store.save(_record(as_of_date="2026-05-01"))

        # fetcher 호출되면 안 됨 (too_early 면 가격 fetch 도 안 일어남).
        def boom(*a, **kw):
            raise AssertionError("fetcher should not be called for too_early")

        # LabelingScheduler 의 fetcher 디폴트 = fetch_close_series.
        # 디폴트 fetcher 가 호출되지 않도록 monkeypatch.
        monkeypatch.setattr(
            "tradingagents.hermes.labeling.fetch_close_series", boom,
        )

        result = labeling_cron.run(today="2026-05-10")

        # too_early=1, 나머지 0.
        assert result["too_early"] == 1
        assert result["labeled"] == 0


@pytest.mark.unit
class TestMain:
    def test_main_dry_run_prints_json_returns_zero(
        self, isolated_store, capsys,
    ):
        isolated_store.save(_record())

        rc = labeling_cron.main(["--dry-run"])

        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["dry_run"] is True
        assert parsed["pending_count"] == 1

    def test_main_today_override(self, isolated_store, capsys, monkeypatch):
        isolated_store.save(_record(as_of_date="2026-05-01"))
        monkeypatch.setattr(
            "tradingagents.hermes.labeling.fetch_close_series",
            lambda key, start, end: pd.DataFrame(columns=["date", "close"]),
        )

        rc = labeling_cron.main(["--today", "2026-05-10"])

        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        # 2026-05-10 시점에 2026-05-29 만기 가설 → too_early.
        assert parsed["too_early"] == 1
