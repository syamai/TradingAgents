"""forward arms 평가 JSON 환류 — 스키마·판정·기록 회귀 가드.

핵심 불변식:
  - build_payload 가 as_of/freeze_date/selected_strategy_ids/arms/promotion_picks/strategies
    스키마를 유지한다 (탐색 루프가 기계판독으로 의존).
  - 판정 3종: 거래 0건 → 모두 None(판정 불가). 거래 있으면 ①가중총실현>0
    ②KOSPI 초과 ③손절손실 미잠식이 불리언으로 산출된다.
  - write_json 은 일자별 파일(누적 보존) + latest 복사본을 동일 내용으로 기록한다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "forward_arms_eval",
    Path(__file__).resolve().parents[2] / "scripts" / "forward_arms_eval.py")
fae = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fae)


def _zero_agg(sids):
    return {sid: {"all": 0.0, "sl": 0.0, "n": 0, "sl_n": 0, "wins": 0} for sid in sids}


def _aggs(first_arm_trades=False):
    """ARMS 전체를 0건으로 채우고, 옵션으로 첫 arm 첫 슬리브에만 거래를 넣는다."""
    aggs = {arm: _zero_agg(sids) for arm, sids in fae.ARMS.items()}
    if first_arm_trades:
        arm, sids = next(iter(fae.ARMS.items()))
        aggs[arm][sids[0]] = {"all": 0.9, "sl": -0.3, "n": 4, "sl_n": 1, "wins": 3}
    return aggs


@pytest.mark.unit
class TestBuildPayload:
    def test_schema_and_selected_ids(self):
        p = fae.build_payload("2026-06-11", "2026-06-11", _aggs(), [], None, (None, None))
        assert set(p) == {"as_of", "freeze_date", "latest_data_date", "selected_strategy_ids",
                          "kospi", "arms", "promotion_picks", "strategies"}
        assert p["freeze_date"] == fae.FREEZE
        assert p["selected_strategy_ids"] == sorted({s for v in fae.ARMS.values() for s in v})
        assert len(p["arms"]) == len(fae.ARMS)
        for arm in p["arms"]:
            assert set(arm) == {"arm_key", "strategy_ids", "n_closed_trades", "n_stop_loss",
                                "weighted_realized_pct", "stop_loss_erosion_pct", "win_rate",
                                "kospi_return_pct", "verdicts"}
            assert set(arm["verdicts"]) == {"realized_positive", "beats_kospi",
                                            "stoploss_not_eroding"}

    def test_no_trades_verdicts_none(self):
        p = fae.build_payload("2026-06-11", "2026-06-11", _aggs(), [], 1.5, ("a", "b"))
        for arm in p["arms"]:
            assert arm["win_rate"] is None
            assert all(v is None for v in arm["verdicts"].values())
        assert p["strategies"] == []  # 거래 없는 슬리브는 상세 미포함

    def test_with_trades_verdicts_and_detail(self):
        p = fae.build_payload("2026-06-11", "2026-06-11", _aggs(first_arm_trades=True),
                              [], 0.1, ("2026-06-10", "2026-06-11"))
        arm = p["arms"][0]
        n_sleeves = len(arm["strategy_ids"])
        # 가중총실현 = 슬리브합 ÷ 슬리브수, 손절손실(-0.3)이 손절 외 이익(1.2)보다 작음 → 미잠식
        assert arm["n_closed_trades"] == 4 and arm["n_stop_loss"] == 1
        assert arm["weighted_realized_pct"] == pytest.approx(0.9 / n_sleeves, abs=1e-4)
        assert arm["win_rate"] == 0.75
        v = arm["verdicts"]
        assert v["realized_positive"] is True
        assert v["beats_kospi"] is (0.9 / n_sleeves > 0.1)
        assert v["stoploss_not_eroding"] is True
        assert len(p["strategies"]) == 1
        assert p["strategies"][0]["strategy_id"] == arm["strategy_ids"][0]

    def test_kospi_none_blocks_beats_kospi_only(self):
        p = fae.build_payload("2026-06-11", "2026-06-11", _aggs(first_arm_trades=True),
                              [], None, (None, None))
        v = p["arms"][0]["verdicts"]
        assert v["beats_kospi"] is None
        assert v["realized_positive"] is True


@pytest.mark.unit
class TestWriteJson:
    def test_daily_and_latest_written(self, tmp_path):
        p = fae.build_payload("2026-06-11", "2026-06-11", _aggs(), [], None, (None, None))
        daily = fae.write_json(p, out_dir=str(tmp_path))
        assert daily == str(tmp_path / "arms_eval_2026-06-11.json")
        loaded = json.loads(Path(daily).read_text(encoding="utf-8"))
        latest = json.loads((tmp_path / "arms_eval_latest.json").read_text(encoding="utf-8"))
        assert loaded == latest == p

    def test_other_dates_preserved(self, tmp_path):
        """일자별 파일 누적 — 다른 날짜 파일을 덮어쓰지 않는다 (FREEZE 변경 이력 보존)."""
        for d in ("2026-06-11", "2026-06-12"):
            fae.write_json(fae.build_payload(d, d, _aggs(), [], None, (None, None)),
                           out_dir=str(tmp_path))
        assert (tmp_path / "arms_eval_2026-06-11.json").exists()
        assert (tmp_path / "arms_eval_2026-06-12.json").exists()
        latest = json.loads((tmp_path / "arms_eval_latest.json").read_text(encoding="utf-8"))
        assert latest["as_of"] == "2026-06-12"
