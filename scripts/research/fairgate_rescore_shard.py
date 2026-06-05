"""공정 게이트 재채점 — 샤드 워커. argv: <shard_idx> <n_shards>.

shard i 는 (전역 인덱스 % n_shards == i) 인 spec 만 처리, 자기 DB
(~/.tradingagents/hermes_fairgate_sh{i}/strategies_v2.db)에 기록 → DB 락 충돌 0.
resumable(자기 샤드 DB 의 spec_hash 건너뜀).
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes.backtest_engine_v2 import run_universe_backtest_v2
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_spec import spec_hash
from tradingagents.hermes.strategy_validation import (
    engine_version, run_walk_forward_validation,
)
from tradingagents.hermes.strategy_store_v2 import StrategyStoreV2

SHARD = int(sys.argv[1])
NSHARD = int(sys.argv[2])
SRC = Path.home() / ".tradingagents/hermes/strategies_v2.db"
OUT_ROOT = Path.home() / f".tradingagents/hermes_fairgate_sh{SHARD}"

con = sqlite3.connect(str(SRC))
con.row_factory = sqlite3.Row
allspecs = [(r["id"], r["name"], json.loads(r["spec_json"]))
            for r in con.execute("SELECT id, name, spec_json FROM strategies ORDER BY id")]
con.close()
specs = [t for idx, t in enumerate(allspecs) if idx % NSHARD == SHARD]

store = StrategyStoreV2(root=OUT_ROOT)
with store._conn() as c:
    already = {r[0] for r in c.execute("SELECT spec_hash FROM strategies")}
todo = [(sid, name, spec) for (sid, name, spec) in specs
        if spec_hash(spec) not in already]
print(f"[sh{SHARD}] assigned={len(specs)} done={len(already)} todo={len(todo)}", flush=True)

raw = KisHistoryStore().list_tickers()
tickers, loader, kospi_fetcher, usdkrw_fetcher = _preload(raw)
ev = engine_version()
print(f"[sh{SHARD}] universe={len(tickers)} engine={ev} start", flush=True)

t0 = time.time()
done = err = 0
for sid, name, spec in todo:
    try:
        result = run_universe_backtest_v2(
            spec, tickers, loader=loader, kospi_fetcher=kospi_fetcher)
        wf = run_walk_forward_validation(
            spec, tickers, loader=loader, kospi_fetcher=kospi_fetcher)
        store.save(spec, result, name=name, wf_result=wf, engine_version=ev)
    except Exception as e:        # noqa: BLE001
        err += 1
        print(f"[sh{SHARD}] ERR src#{sid} {name[:24]}: {type(e).__name__}: {e}", flush=True)
    done += 1
    if done % 20 == 0 or done == len(todo):
        el = time.time() - t0
        eta = el / done * (len(todo) - done) / 60 if done else 0
        print(f"[sh{SHARD}] {done}/{len(todo)} ({el:.0f}s {el/max(done,1):.1f}/ea "
              f"eta {eta:.0f}min err={err})", flush=True)

print(f"[sh{SHARD}] DONE done={done} err={err} {(time.time()-t0)/60:.1f}min", flush=True)
