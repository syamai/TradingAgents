#!/usr/bin/env python3
"""Daily paper-trading report using only gate-passed v2 strategies.

- Universe: KIS holdings, ETF excluded.
- Strategy set: strategies_v2.db rows where gate_passed=1 only.
- Signal semantics: same v2 simulator as backtest, signal row i -> fill close[i+1].
- Output: Korean Telegram-friendly text; no orders are sent to a broker.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.kis_history_store import KisHistoryStore
from tradingagents.hermes import backtest_engine as bt
from tradingagents.hermes.backtest_engine_v2 import _market_overlay_multiplier, _simulate_v2
from tradingagents.hermes.forward_test import ETF_CODES
from tradingagents.hermes.strategy_research_v2 import _preload
from tradingagents.hermes.strategy_store import _row_to_strategy

DB = Path.home() / ".tradingagents/hermes/strategies_v2.db"
PAPER_DB = Path.home() / ".tradingagents/hermes/paper_trades.db"
STATE = Path.home() / ".tradingagents/hermes/papertrade_gate_passed_v2_state.json"
REPORT_DIR = Path.home() / ".tradingagents/hermes/papertrade_reports"
MAX_POSITIONS = 30
STOCK_WEIGHT = 0.90
BASE_WEIGHT = STOCK_WEIGHT / MAX_POSITIONS
MAX_SINGLE_WEIGHT = 0.05
INITIAL_CAPITAL_KRW = 100_000_000
TOP_SHOW = 20
# Selected 5-strategy paper-trading basket requested by user.
# 1518/1561 already contain KOSPI80 stock20 overlay in DB spec.
# 1841 contains FX overlay in DB spec. 1035/1081 get FX overlay at load time.
SELECTED_STRATEGY_IDS = [1561, 1518, 1035, 1081, 1841]
FX20_GE2_MA60_STOCK30 = {
    "type": "usdkrw_trailing_return_ma_scale",
    "window": 20,
    "op": ">=",
    "threshold_pct": 2.0,
    "ma_window": 60,
    "risk_stock_weight_pct": 30.0,
}
KOSPI80_STOCK20 = {
    "type": "kospi_trailing_return_scale",
    "window": 80,
    "op": "<=",
    "threshold_pct": 0.0,
    "risk_stock_weight_pct": 20.0,
}
OVERLAY_BY_ID = {
    1035: ("FX overlay", FX20_GE2_MA60_STOCK30),
    1081: ("FX overlay", FX20_GE2_MA60_STOCK30),
}


def load_passed() -> list[dict]:
    placeholders = ",".join("?" for _ in SELECTED_STRATEGY_IDS)
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            f"SELECT * FROM strategies WHERE gate_passed=1 AND id IN ({placeholders})",
            SELECTED_STRATEGY_IDS,
        ).fetchall()
    out = []
    for r in rows:
        d = _row_to_strategy(r)
        try:
            d["spec"] = json.loads(d["spec_json"])
        except Exception:
            continue
        overlay_info = OVERLAY_BY_ID.get(d["id"])
        if overlay_info:
            label, overlay = overlay_info
            d["spec"]["market_overlay"] = dict(overlay)
            d["overlay_label"] = label
            d["name"] = f"{d['name']} + {label}"
        elif d["spec"].get("market_overlay"):
            overlay_type = str(d["spec"].get("market_overlay", {}).get("type", ""))
            if overlay_type.startswith("kospi"):
                d["overlay_label"] = "KOSPI overlay (DB)"
            elif overlay_type.startswith("usdkrw"):
                d["overlay_label"] = "FX overlay (DB)"
            else:
                d["overlay_label"] = "market overlay (DB)"
        else:
            d["overlay_label"] = "none"
        out.append(d)
    rank = {sid: i for i, sid in enumerate(SELECTED_STRATEGY_IDS)}
    out.sort(key=lambda st: rank.get(st["id"], 999))
    return out


def fmt_pct(x: float | None, digits=2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}%"


def ticker_name(ticker: str, metas: dict[str, dict]) -> str:
    meta = metas.get(ticker) or {}
    name = meta.get("name") or meta.get("ticker_name") or ""
    return f"{ticker} {name}".strip()


def overlay_stock_weight(spec: dict, latest_date: str, kospi, usdkrw) -> float:
    overlay = spec.get("market_overlay")
    if not overlay:
        return 90.0
    idx = pd.Index(sorted(set(kospi["date"].astype(str).tolist()))) if kospi is not None else pd.Index([latest_date])
    mult = _market_overlay_multiplier(idx, kospi, overlay, usdkrw=usdkrw)
    if latest_date not in mult.index:
        val = float(mult.iloc[-1]) if len(mult) else 1.0
    else:
        val = float(mult.loc[latest_date])
    return round(90.0 * val, 2)


def fmt_krw(x: float | int) -> str:
    return f"{int(round(float(x))):,}원"


def _json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def init_paper_db() -> None:
    PAPER_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PAPER_DB) as con:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS paper_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generated_at TEXT NOT NULL UNIQUE,
                latest_data_date TEXT NOT NULL,
                mode TEXT NOT NULL,
                selected_strategy_ids_json TEXT NOT NULL,
                initial_capital_per_strategy_krw REAL NOT NULL,
                total_virtual_capital_krw REAL NOT NULL,
                report_json_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS paper_strategy_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
                strategy_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                overlay_label TEXT,
                initial_capital_krw REAL,
                target_stock_weight_pct REAL,
                cash_weight_pct REAL,
                per_name_weight_pct REAL,
                per_name_amount_krw REAL,
                target_invested_krw REAL,
                target_cash_krw REAL,
                open_tickers INTEGER,
                today_buys_count INTEGER NOT NULL,
                today_sells_count INTEGER NOT NULL,
                errors_json TEXT NOT NULL DEFAULT '[]',
                UNIQUE(run_id, strategy_id)
            );
            CREATE TABLE IF NOT EXISTS paper_target_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
                strategy_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                signals INTEGER,
                weight_pct REAL,
                amount_krw REAL,
                UNIQUE(run_id, strategy_id, ticker)
            );
            CREATE TABLE IF NOT EXISTS paper_trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL REFERENCES paper_runs(id) ON DELETE CASCADE,
                strategy_id INTEGER NOT NULL,
                event_type TEXT NOT NULL CHECK(event_type IN ('BUY','SELL')),
                ticker TEXT NOT NULL,
                strategy_name TEXT,
                entry_date TEXT,
                entry_price REAL,
                exit_date TEXT,
                exit_price REAL,
                gross_ret_pct REAL,
                net_ret_pct REAL,
                hold_days INTEGER,
                exit_reason TEXT,
                raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_paper_runs_date ON paper_runs(latest_data_date);
            CREATE INDEX IF NOT EXISTS idx_paper_events_date ON paper_trade_events(entry_date, exit_date);
            CREATE INDEX IF NOT EXISTS idx_paper_events_strategy ON paper_trade_events(strategy_id, ticker);
            CREATE INDEX IF NOT EXISTS idx_paper_positions_strategy ON paper_target_positions(strategy_id, ticker);
            """
        )


def save_snapshot_to_db(snapshot: dict, report_json_path: Path) -> int:
    init_paper_db()
    with sqlite3.connect(PAPER_DB) as con:
        cur = con.execute(
            """
            INSERT OR IGNORE INTO paper_runs (
                generated_at, latest_data_date, mode, selected_strategy_ids_json,
                initial_capital_per_strategy_krw, total_virtual_capital_krw, report_json_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["generated_at"],
                snapshot["latest_data_date"],
                snapshot["mode"],
                _json_dumps(snapshot["selected_strategy_ids"]),
                snapshot["initial_capital_per_strategy_krw"],
                snapshot["total_virtual_capital_krw"],
                str(report_json_path),
            ),
        )
        if cur.lastrowid:
            run_id = int(cur.lastrowid)
        else:
            run_id = int(con.execute("SELECT id FROM paper_runs WHERE generated_at=?", (snapshot["generated_at"],)).fetchone()[0])

        for r in snapshot["strategies"]:
            sid = int(r["strategy_id"])
            con.execute(
                """
                INSERT OR REPLACE INTO paper_strategy_snapshots (
                    run_id, strategy_id, name, overlay_label, initial_capital_krw,
                    target_stock_weight_pct, cash_weight_pct, per_name_weight_pct,
                    per_name_amount_krw, target_invested_krw, target_cash_krw,
                    open_tickers, today_buys_count, today_sells_count, errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sid,
                    r["name"],
                    r.get("overlay_label"),
                    r.get("initial_capital_krw"),
                    r.get("target_stock_weight_pct"),
                    r.get("cash_weight_pct"),
                    r.get("per_name_weight_pct"),
                    r.get("per_name_amount_krw"),
                    r.get("target_invested_krw"),
                    r.get("target_cash_krw"),
                    r.get("open_tickers"),
                    len(r.get("today_buys", [])),
                    len(r.get("today_sells", [])),
                    _json_dumps(r.get("errors", [])),
                ),
            )
            for pos in r.get("target_positions", []):
                con.execute(
                    """
                    INSERT OR REPLACE INTO paper_target_positions (
                        run_id, strategy_id, ticker, signals, weight_pct, amount_krw
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, sid, pos["ticker"], pos.get("signals"), pos.get("weight_pct"), pos.get("amount_krw")),
                )
            for event_type, rows in (("BUY", r.get("today_buys", [])), ("SELL", r.get("today_sells", []))):
                for tr in rows:
                    con.execute(
                        """
                        INSERT OR REPLACE INTO paper_trade_events (
                            run_id, strategy_id, event_type, ticker, strategy_name,
                            entry_date, entry_price, exit_date, exit_price,
                            gross_ret_pct, net_ret_pct, hold_days, exit_reason, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            sid,
                            event_type,
                            tr["ticker"],
                            tr.get("name"),
                            tr.get("entry_date"),
                            tr.get("entry_price"),
                            tr.get("exit_date"),
                            tr.get("exit_price"),
                            tr.get("gross_ret_pct"),
                            tr.get("net_ret_pct"),
                            tr.get("hold_days"),
                            tr.get("exit_reason"),
                            _json_dumps(tr),
                        ),
                    )
        return run_id


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    passed = load_passed()
    if not passed:
        print("페이퍼트레이딩: 선택 전략이 없습니다.")
        return 0

    raw = [tk for tk in KisHistoryStore().list_tickers() if str(tk)[:6] not in ETF_CODES]
    tickers, loader, kf, uf = _preload(raw)
    holdings = {}
    metas = {}
    latest_date = "0000-00-00"
    for tk in tickers:
        df, meta = loader(tk)
        holdings[tk] = df
        metas[tk] = meta or {}
        if df is not None and not df.empty:
            latest_date = max(latest_date, str(df["date"].iloc[-1]))

    kospi = kf(None, None)
    usdkrw = uf(None, None)

    strategy_reports = []
    all_errors = []

    for st in passed:
        sid = st["id"]
        spec = st["spec"]
        buys = []
        sells = []
        open_by_ticker: dict[str, list[dict]] = defaultdict(list)
        errors = []
        try:
            target_stock_weight = overlay_stock_weight(spec, latest_date, kospi, usdkrw)
        except Exception:
            target_stock_weight = 90.0

        for tk, df in holdings.items():
            try:
                trades, _daily, _active, _cdf = _simulate_v2(spec, df, kospi=kospi)
            except Exception as e:  # keep daily report robust
                if len(errors) < 3:
                    errors.append(f"#{sid} {tk}: {type(e).__name__} {e}")
                continue
            for tr in trades:
                ed = tr["entry_date"]
                xd = tr["exit_date"]
                is_forced_today = xd == latest_date and tr.get("exit_reason") == "forced_eod"
                row = {"ticker": tk, "sid": sid, "name": st["name"], **tr}
                if ed == latest_date:
                    buys.append(row)
                if xd == latest_date and not is_forced_today:
                    sells.append(row)
                if ed <= latest_date and (xd > latest_date or is_forced_today):
                    open_by_ticker[tk].append(row)

        selected = [tk for tk, _ in Counter({tk: len(v) for tk, v in open_by_ticker.items()}).most_common(MAX_POSITIONS)]
        per_name_weight = min(MAX_SINGLE_WEIGHT, (target_stock_weight / 100.0) / MAX_POSITIONS)
        per_name_amount = INITIAL_CAPITAL_KRW * per_name_weight
        invested_amount = per_name_amount * len(selected)
        target_positions = [
            {
                "ticker": tk,
                "signals": len(open_by_ticker[tk]),
                "weight_pct": round(per_name_weight * 100, 2),
                "amount_krw": round(per_name_amount),
            }
            for tk in selected
        ]
        rep = {
            "strategy_id": sid,
            "name": st["name"],
            "overlay_label": st.get("overlay_label", "none"),
            "initial_capital_krw": INITIAL_CAPITAL_KRW,
            "target_stock_weight_pct": target_stock_weight,
            "cash_weight_pct": round(100.0 - target_stock_weight, 2),
            "per_name_weight_pct": round(per_name_weight * 100, 2),
            "per_name_amount_krw": round(per_name_amount),
            "target_invested_krw": round(invested_amount),
            "target_cash_krw": round(INITIAL_CAPITAL_KRW - invested_amount),
            "open_tickers": len(open_by_ticker),
            "target_positions": target_positions,
            "today_buys": buys,
            "today_sells": sells,
            "errors": errors,
        }
        strategy_reports.append(rep)
        all_errors.extend(errors)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_data_date": latest_date,
        "mode": "per_strategy_independent_sleeves",
        "initial_capital_per_strategy_krw": INITIAL_CAPITAL_KRW,
        "total_virtual_capital_krw": INITIAL_CAPITAL_KRW * len(strategy_reports),
        "selected_strategy_ids": SELECTED_STRATEGY_IDS,
        "strategies": strategy_reports,
        "errors": all_errors[:10],
    }
    out_json = REPORT_DIR / f"papertrade_selected5_per_strategy_{latest_date}.json"
    snapshot["report_json_path"] = str(out_json)
    out_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    run_id = save_snapshot_to_db(snapshot, out_json)
    STATE.write_text(
        json.dumps(
            {"last_run_at": snapshot["generated_at"], "last_data_date": latest_date, "last_report": str(out_json), "last_db": str(PAPER_DB), "last_run_id": run_id, "mode": snapshot["mode"]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_buys = sum(len(r["today_buys"]) for r in strategy_reports)
    total_sells = sum(len(r["today_sells"]) for r in strategy_reports)
    lines = []
    lines.append("## 선정 5개 전략 독립 페이퍼트레이딩")
    lines.append(f"- 데이터 기준일: **{latest_date}**")
    lines.append(f"- 운용 방식: **전략별 독립 계좌 1억원씩**")
    lines.append(f"- 총 가상자산: **{fmt_krw(INITIAL_CAPITAL_KRW * len(strategy_reports))}**")
    lines.append(f"- 선정 ID: `{', '.join(str(x) for x in SELECTED_STRATEGY_IDS)}`")
    lines.append("- Overlay: `1561=KOSPI80(DB)`, `1518=KOSPI80(DB)`, `1035=FX20>=2%&>60MA`, `1081=FX20>=2%&>60MA`, `1841=FX20>=2%&>60MA(DB)`")
    lines.append(f"- 전체 신규 진입: **{total_buys}건** / 청산: **{total_sells}건**")
    lines.append("")

    for r in strategy_reports:
        lines.append(f"### #{r['strategy_id']} {r['name'][:52]}")
        lines.append(f"- 시작자산: **{fmt_krw(r['initial_capital_krw'])}**")
        lines.append(f"- Overlay: **{r['overlay_label']}**")
        lines.append(f"- 목표 주식노출: **{r['target_stock_weight_pct']:.1f}%** / 현금: **{r['cash_weight_pct']:.1f}%**")
        lines.append(f"- 목표 보유 후보: **{r['open_tickers']}개** → 최대 {MAX_POSITIONS}개, 종목당 **{r['per_name_weight_pct']:.2f}%** ({fmt_krw(r['per_name_amount_krw'])})")
        lines.append(f"- 목표 투입금: **{fmt_krw(r['target_invested_krw'])}** / 목표 현금: **{fmt_krw(r['target_cash_krw'])}**")
        lines.append(f"- 오늘 신규: **{len(r['today_buys'])}건** / 청산: **{len(r['today_sells'])}건**")
        for b in r["today_buys"][:5]:
            lines.append(f"  - BUY {ticker_name(b['ticker'], metas)} | 진입가 {b['entry_price']}")
        if len(r["today_buys"]) > 5:
            lines.append(f"  - ... 신규 외 {len(r['today_buys'])-5}건")
        for srow in r["today_sells"][:5]:
            lines.append(f"  - SELL {ticker_name(srow['ticker'], metas)} | 수익률 {fmt_pct(srow['net_ret_pct'])} | 사유 {srow.get('exit_reason')}")
        if len(r["today_sells"]) > 5:
            lines.append(f"  - ... 청산 외 {len(r['today_sells'])-5}건")
        if r["target_positions"]:
            lines.append("- 목표 보유 상위:")
            for i, pos in enumerate(r["target_positions"][:5], 1):
                lines.append(f"  {i}. {ticker_name(pos['ticker'], metas)} | 신호 {pos['signals']} | {pos['weight_pct']:.2f}% | {fmt_krw(pos['amount_krw'])}")
            if len(r["target_positions"]) > 5:
                lines.append(f"  - ... 외 {len(r['target_positions'])-5}개")
        else:
            lines.append("- 목표 보유: 없음")
        if r["errors"]:
            lines.append(f"- 주의: 일부 시뮬레이션 오류 샘플 `{r['errors'][0]}`")
        lines.append("")

    lines.append(f"상세 JSON: `{out_json}`")
    lines.append(f"매매일지 DB: `{PAPER_DB}` / run_id `{run_id}`")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
