#!/usr/bin/env python3
"""Export the latest paper-trading run to an Excel-compatible .xlsx file.

The project environment intentionally avoids extra Excel dependencies, so this
uses a tiny OOXML writer based on Python stdlib zipfile.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HOME = Path.home()
PAPER_DB = HOME / ".tradingagents/hermes/paper_trades.db"
STRATEGY_DB = HOME / ".tradingagents/hermes/strategies_v2.db"
DEFAULT_OUT_DIR = Path("/Users/selab/Source/trading-ai/artifacts/papertrade_reports")


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _load_run(run_id: int | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    con = _connect(PAPER_DB)
    if run_id is None:
        run = con.execute("SELECT * FROM paper_runs ORDER BY id DESC LIMIT 1").fetchone()
    else:
        run = con.execute("SELECT * FROM paper_runs WHERE id=?", (run_id,)).fetchone()
    if not run:
        raise SystemExit("No paper_runs found")
    rid = int(run["id"])
    snapshots = [dict(r) for r in con.execute("SELECT * FROM paper_strategy_snapshots WHERE run_id=? ORDER BY strategy_id", (rid,))]
    positions = [dict(r) for r in con.execute("SELECT * FROM paper_target_positions WHERE run_id=? ORDER BY strategy_id, ticker", (rid,))]
    events = [dict(r) for r in con.execute("SELECT * FROM paper_trade_events WHERE run_id=? ORDER BY event_type, strategy_id, ticker", (rid,))]
    return dict(run), snapshots, positions, events


def _fetch_stock_name(ticker: str) -> tuple[str, str]:
    code = str(ticker).zfill(6)
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", "ignore")
        m = re.search(r"<title>\s*([^<:]+?)\s*(?::|</title>)", data, re.S)
        name = html.unescape(m.group(1)).strip() if m else ""
        name = re.sub(r"\s+", " ", name)
        if not name or "네이버" in name or "Npay" in name:
            name = code
        return code, name
    except Exception:
        return code, code


def _load_name_cache(out_dir: Path, tickers: set[str]) -> dict[str, str]:
    cache_path = out_dir / "ticker_name_cache.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    missing = [str(t).zfill(6) for t in sorted(tickers) if not cache.get(str(t).zfill(6))]
    if missing:
        with ThreadPoolExecutor(max_workers=12) as ex:
            for code, name in ex.map(_fetch_stock_name, missing):
                cache[code] = name
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def _stock_name(cache: dict[str, str], ticker: str) -> str:
    code = str(ticker).zfill(6)
    return cache.get(code, code)


def _signal_summary(sig: Any) -> str:
    if not isinstance(sig, dict):
        return str(sig)
    s = sig.get("signal") or sig.get("type") or "?"
    items = [f"{k}={v}" for k, v in sig.items() if k not in ("signal", "type")]
    return s + ("(" + ", ".join(items) + ")" if items else "")


def _spec_summaries(spec_json: Any) -> tuple[str, str]:
    try:
        spec = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
    except Exception:
        return "", ""
    entry = spec.get("entry", {}).get("all_of", [])
    ex = spec.get("exit", {})
    exits = [_signal_summary(sig) for sig in (ex.get("signal_all_of", []) or [])]
    for key in ("stop_loss_pct", "take_profit_pct", "max_hold_days"):
        if key in ex and ex[key] is not None:
            exits.append(f"{key}={ex[key]}")
    return " AND ".join(_signal_summary(x) for x in entry), "; ".join(exits)


def _strategy_rows(selected_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not selected_ids:
        return {}
    qmarks = ",".join("?" for _ in selected_ids)
    con = _connect(STRATEGY_DB)
    out: dict[int, dict[str, Any]] = {}
    for r in con.execute(
        f"""
        SELECT id,name,spec_json,gate_passed,in_win_rate,in_sharpe,in_mdd,in_cum_return_pct,in_n_trades,
               out_win_rate,out_sharpe,out_mdd,out_cum_return_pct,out_n_trades,
               time_in_sharpe,time_out_sharpe,wf_excess_ir_median,wf_excess_ir_min,wf_n_windows,wf_gate_passed
        FROM strategies WHERE id IN ({qmarks})
        """,
        selected_ids,
    ):
        out[int(r["id"])] = dict(r)
    return out


def _build_sheets(run: dict[str, Any], snapshots: list[dict[str, Any]], positions: list[dict[str, Any]], events: list[dict[str, Any]], name_cache: dict[str, str]) -> dict[str, list[list[Any]]]:
    selected_ids = json.loads(run["selected_strategy_ids_json"])
    strat_rows = _strategy_rows([int(x) for x in selected_ids])
    buys = [e for e in events if e["event_type"] == "BUY"]
    sells = [e for e in events if e["event_type"] == "SELL"]
    name_by_sid = {s["strategy_id"]: s["name"] for s in snapshots}
    snap_by_sid = {s["strategy_id"]: s for s in snapshots}

    sell_by_sid: dict[int, dict[str, Any]] = {}
    for e in sells:
        sid = int(e["strategy_id"])
        amt = (snap_by_sid.get(sid) or {}).get("per_name_amount_krw") or 0
        pnl = amt * (e["net_ret_pct"] or 0) / 100.0
        d = sell_by_sid.setdefault(sid, {"count": 0, "wins": 0, "losses": 0, "net_ret_sum": 0.0, "gross_ret_sum": 0.0, "pnl": 0.0, "take_profit": 0, "stop_loss": 0, "other": 0})
        d["count"] += 1
        d["net_ret_sum"] += e["net_ret_pct"] or 0
        d["gross_ret_sum"] += e["gross_ret_pct"] or 0
        d["pnl"] += pnl
        if (e["net_ret_pct"] or 0) > 0:
            d["wins"] += 1
        elif (e["net_ret_pct"] or 0) < 0:
            d["losses"] += 1
        reason = e["exit_reason"] or "other"
        d[reason if reason in d else "other"] += 1

    summary = [
        ["항목", "값"],
        ["run_id", run["id"]],
        ["생성시각(UTC)", run["generated_at"]],
        ["기준 데이터일", run["latest_data_date"]],
        ["모드", run["mode"]],
        ["선정 전략 ID", ", ".join(map(str, selected_ids))],
        ["전략 수", len(selected_ids)],
        ["전략당 초기자본(원)", run["initial_capital_per_strategy_krw"]],
        ["총 가상자본(원)", run["total_virtual_capital_krw"]],
        ["총 보유종목 행 수", len(positions)],
        ["오늘 매수 건수", len(buys)],
        ["오늘 매도 건수", len(sells)],
        ["원본 JSON", run["report_json_path"]],
        ["", ""],
        ["전략별 현재 요약", "", "", "", "", "", "", ""],
        ["strategy_id", "name", "open_tickers", "target_invested_krw", "target_cash_krw", "today_buys", "today_sells", "overlay_label"],
    ]
    for s in snapshots:
        summary.append([s["strategy_id"], s["name"], s["open_tickers"], s["target_invested_krw"], s["target_cash_krw"], s["today_buys_count"], s["today_sells_count"], s["overlay_label"]])
    summary += [["", ""], ["금일 매도 결과 요약", "", "", "", "", "", "", "", "", ""], ["strategy_id", "strategy_name", "sell_count", "win_count", "loss_count", "win_rate_pct", "avg_net_ret_pct", "avg_gross_ret_pct", "est_realized_pnl_krw", "take_profit_count", "stop_loss_count", "other_exit_count"]]
    for sid in sorted(sell_by_sid):
        d = sell_by_sid[sid]
        cnt = d["count"]
        summary.append([sid, name_by_sid.get(sid), cnt, d["wins"], d["losses"], round(d["wins"] / cnt * 100, 2) if cnt else None, round(d["net_ret_sum"] / cnt, 4) if cnt else None, round(d["gross_ret_sum"] / cnt, 4) if cnt else None, round(d["pnl"], 0), d["take_profit"], d["stop_loss"], d["other"]])
    summary += [["", ""], ["금일 매도 상세", "", "", "", "", "", "", "", "", "", "", ""], ["strategy_id", "strategy_name", "ticker", "stock_name", "entry_date", "entry_price", "exit_date", "exit_price", "gross_ret_pct", "net_ret_pct", "hold_days", "exit_reason", "est_realized_pnl_krw"]]
    for e in sells:
        amt = (snap_by_sid.get(int(e["strategy_id"])) or {}).get("per_name_amount_krw") or 0
        summary.append([e["strategy_id"], e["strategy_name"], e["ticker"], _stock_name(name_cache, e["ticker"]), e["entry_date"], e["entry_price"], e["exit_date"], e["exit_price"], e["gross_ret_pct"], e["net_ret_pct"], e["hold_days"], e["exit_reason"], round(amt * (e["net_ret_pct"] or 0) / 100.0, 0)])

    strategy = [["strategy_id", "name", "gate_passed", "overlay_label", "initial_capital_krw", "target_stock_weight_pct", "cash_weight_pct", "per_name_weight_pct", "per_name_amount_krw", "target_invested_krw", "target_cash_krw", "open_tickers", "today_buys_count", "today_sells_count", "entry_summary", "exit_summary", "in_sharpe", "out_sharpe", "time_in_sharpe", "time_out_sharpe", "wf_excess_ir_median", "wf_excess_ir_min", "wf_n_windows", "wf_gate_passed", "in_mdd", "out_mdd", "in_cum_return_pct", "out_cum_return_pct", "in_n_trades", "out_n_trades", "errors_json"]]
    for s in snapshots:
        sr = strat_rows.get(int(s["strategy_id"]), {})
        entry_sum, exit_sum = _spec_summaries(sr.get("spec_json"))
        strategy.append([s["strategy_id"], s["name"], sr.get("gate_passed"), s["overlay_label"], s["initial_capital_krw"], s["target_stock_weight_pct"], s["cash_weight_pct"], s["per_name_weight_pct"], s["per_name_amount_krw"], s["target_invested_krw"], s["target_cash_krw"], s["open_tickers"], s["today_buys_count"], s["today_sells_count"], entry_sum, exit_sum, sr.get("in_sharpe"), sr.get("out_sharpe"), sr.get("time_in_sharpe"), sr.get("time_out_sharpe"), sr.get("wf_excess_ir_median"), sr.get("wf_excess_ir_min"), sr.get("wf_n_windows"), sr.get("wf_gate_passed"), sr.get("in_mdd"), sr.get("out_mdd"), sr.get("in_cum_return_pct"), sr.get("out_cum_return_pct"), sr.get("in_n_trades"), sr.get("out_n_trades"), s["errors_json"]])

    pos = [["strategy_id", "strategy_name", "ticker", "stock_name", "signals", "weight_pct", "amount_krw"]]
    for p in positions:
        pos.append([p["strategy_id"], name_by_sid.get(p["strategy_id"]), p["ticker"], _stock_name(name_cache, p["ticker"]), p["signals"], p["weight_pct"], p["amount_krw"]])

    def event_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
        out = [["event_type", "strategy_id", "strategy_name", "ticker", "stock_name", "entry_date", "entry_price", "exit_date", "exit_price", "gross_ret_pct", "net_ret_pct", "hold_days", "exit_reason", "est_realized_pnl_krw"]]
        for e in rows:
            amt = (snap_by_sid.get(int(e["strategy_id"])) or {}).get("per_name_amount_krw") or 0
            pnl = round(amt * (e["net_ret_pct"] or 0) / 100.0, 0) if e["event_type"] == "SELL" else None
            out.append([e["event_type"], e["strategy_id"], e["strategy_name"], e["ticker"], _stock_name(name_cache, e["ticker"]), e["entry_date"], e["entry_price"], e["exit_date"], e["exit_price"], e["gross_ret_pct"], e["net_ret_pct"], e["hold_days"], e["exit_reason"], pnl])
        return out

    explain = [
        ["항목", "설명"],
        ["전체이벤트", "오늘매수 시트와 오늘매도 시트를 합친 전체 거래 이벤트 로그입니다. event_type이 BUY면 신규 편입/매수 후보, SELL이면 청산/매도 결과입니다."],
        ["est_realized_pnl_krw", "전략별 per_name_amount_krw에 net_ret_pct를 곱한 추정 실현손익입니다. 실제 체결수량/슬리피지는 반영하지 않은 페이퍼트레이딩 추정치입니다."],
        ["보유종목", "기준일 현재 각 전략이 목표 보유로 잡은 종목 목록입니다. amount_krw는 목표 투입금액입니다."],
        ["금일 매도 결과 요약", "요약시트에 전략별 매도 건수, 승/패, 평균 수익률, 추정 실현손익, 청산 사유별 건수를 추가했습니다."],
    ]
    return {"요약": summary, "전략별": strategy, "보유종목": pos, "오늘매수": event_rows(buys), "오늘매도": event_rows(sells), "전체이벤트": event_rows(events), "설명": explain}


def _col_name(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell_xml(r: int, c: int, v: Any) -> str:
    ref = f"{_col_name(c)}{r}"
    if v is None:
        return f'<c r="{ref}"/>'
    if isinstance(v, bool):
        return f'<c r="{ref}" t="b"><v>{1 if v else 0}</v></c>'
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f'<c r="{ref}"><v>{v}</v></c>'
    txt = html.escape(str(v), quote=False)
    return f'<c r="{ref}" t="inlineStr"><is><t>{txt}</t></is></c>'


def _sheet_xml(rows: list[list[Any]]) -> str:
    maxcols = max((len(r) for r in rows), default=1)
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        '<sheetData>',
    ]
    for i, row in enumerate(rows, 1):
        out.append(f'<row r="{i}">' + "".join(_cell_xml(i, j, v) for j, v in enumerate(row, 1)) + "</row>")
    out.append("</sheetData>")
    out.append(f'<autoFilter ref="A1:{_col_name(maxcols)}{len(rows)}"/>')
    out.append("</worksheet>")
    return "".join(out)


def _write_xlsx(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets) + 1))
            + "</Types>",
        )
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + "".join(f'<sheet name="{html.escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheets, 1)) + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1)) + "</Relationships>")
        for i, rows in enumerate(sheets.values(), 1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows))
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
    if bad:
        raise SystemExit(f"Bad xlsx zip member: {bad}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--json", action="store_true", help="print JSON summary")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run, snapshots, positions, events = _load_run(args.run_id)
    tickers = {p["ticker"] for p in positions} | {e["ticker"] for e in events}
    cache = _load_name_cache(args.out_dir, tickers)
    sheets = _build_sheets(run, snapshots, positions, events, cache)
    selected_ids = json.loads(run["selected_strategy_ids_json"])
    latest = str(run["latest_data_date"])
    outfile = args.out_dir / f"papertrade_selected{len(selected_ids)}_{latest.replace('-', '')}_결과.xlsx"
    _write_xlsx(outfile, sheets)
    summary = {
        "xlsx": str(outfile),
        "run_id": int(run["id"]),
        "latest_data_date": latest,
        "selected_strategy_ids": selected_ids,
        "sheets": {name: max(len(rows) - 1, 0) for name, rows in sheets.items()},
        "size_bytes": outfile.stat().st_size,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(str(outfile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
