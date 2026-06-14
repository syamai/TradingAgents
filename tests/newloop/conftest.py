"""newloop 테스트 공통 — 부트스트랩 재표본 수를 낮춰 속도 확보.

실사용 B=2000(임계 안정)이지만, 단위테스트는 방향성만 확인하므로 작은 B 로
충분하다. 시드 고정이라 결정적.
"""

import pytest

import tradingagents.newloop.stage1_gate as s1


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    monkeypatch.setattr(s1, "BOOTSTRAP_B", 400)
