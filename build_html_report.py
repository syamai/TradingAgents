"""
.runtime/reports/<TICKER>/<DATE>/ 폴더의 마크다운 + 실시간 시장 데이터로
인터랙티브 대시보드 HTML을 생성한다.

생성되는 시각화:
  - KPI 카드 (가격, 변동률, RSI, 결정)
  - 가격 + 이동평균 + 볼린저 라인 차트
  - RSI 시계열 + 게이지
  - MACD 히스토그램
  - 거래량 바 차트
  - 감성 도넛 차트
  - Bull vs Bear 토론 카드
  - 리스크 3관점 (공격/보수/중립)
  - 분석 파이프라인 시각화

사용법:
    uv run python build_html_report.py                    # 005930.KS / 2026-05-22 기본값
    uv run python build_html_report.py NVDA 2024-05-10
"""
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

TICKER = sys.argv[1] if len(sys.argv) > 1 else "005930.KS"
DATE = sys.argv[2] if len(sys.argv) > 2 else "2026-05-22"
BASE = sys.argv[3] if len(sys.argv) > 3 else ".runtime"
MODEL_LABEL = sys.argv[4] if len(sys.argv) > 4 else "Gemma4 26B (MoE 4B active)"

src = Path(f"{BASE}/reports/{TICKER}/{DATE}")
if not src.exists():
    sys.exit(f"폴더 없음: {src}. 먼저 explode_log.py를 실행하세요.")


def read(rel):
    p = src / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ────────────────────────────────────────────────────────────
# 시장 데이터 수집
# ────────────────────────────────────────────────────────────
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators

start_date = (datetime.strptime(DATE, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")


def _safe_fetch(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return f"<error: {e}>"


def parse_ohlcv(csv_str):
    rows = []
    for line in csv_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Date,"):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "date": parts[0],
                "open":  float(parts[1]),
                "high":  float(parts[2]),
                "low":   float(parts[3]),
                "close": float(parts[4]),
                "volume": int(float(parts[5])),
            })
        except ValueError:
            continue
    return rows


def parse_indicator(s):
    out = []
    for line in s.splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2}):\s*(.+)", line.strip())
        if not m:
            continue
        date, val = m.group(1), m.group(2).strip()
        if "N/A" in val or "not" in val.lower():
            continue
        try:
            out.append({"date": date, "value": float(val)})
        except ValueError:
            continue
    return sorted(out, key=lambda x: x["date"])


ohlcv = parse_ohlcv(_safe_fetch(get_stock_data.func, TICKER, start_date, DATE))

indicators_to_fetch = ["rsi", "macd", "macds", "macdh", "boll", "boll_ub", "boll_lb",
                       "close_50_sma", "close_200_sma", "close_10_ema", "atr", "vwma"]
indicators = {}
for ind in indicators_to_fetch:
    indicators[ind] = parse_indicator(_safe_fetch(get_indicators.func, TICKER, ind, DATE))


# 최신 값 추출
def last_val(series, default=None):
    return series[-1]["value"] if series else default


latest_price = ohlcv[-1]["close"] if ohlcv else None
prev_price = ohlcv[-2]["close"] if len(ohlcv) >= 2 else None
price_change = ((latest_price - prev_price) / prev_price * 100) if (latest_price and prev_price) else None

latest_rsi = last_val(indicators["rsi"])
latest_macd = last_val(indicators["macd"])
latest_macds = last_val(indicators["macds"])
latest_volume = ohlcv[-1]["volume"] if ohlcv else None

# 통화 단위 추정
currency_symbol = "₩" if TICKER.endswith(".KS") or TICKER.endswith(".KQ") else "$"


# ────────────────────────────────────────────────────────────
# 텍스트에서 메타 추출
# ────────────────────────────────────────────────────────────
def extract_verdict(text):
    m = re.search(
        r"(Strong Buy|Buy|Overweight|Hold|Underweight|Sell|Strong Sell|매수|매도|비중\s*확대|비중\s*축소|보유)",
        text, re.IGNORECASE,
    )
    return m.group(0).strip() if m else "확인 필요"


def extract_sentiment(text):
    """sentiment.md에서 Bullish/Bearish/Neutral 라벨 추출."""
    m = re.search(r"\*\*방향[:\s]*(Bullish|Bearish|Neutral|Mixed|강세|약세|중립|혼조)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(Bullish|Bearish|Neutral|Mixed)\b", text)
    return m.group(1) if m else "Mixed"


verdict_text = read("05_portfolio/decision.md")
verdict = extract_verdict(verdict_text)
verdict_lower = verdict.lower()
if any(w in verdict_lower for w in ["buy", "overweight", "매수", "확대"]):
    verdict_class, verdict_emoji, verdict_color = "verdict-buy", "📈", "#10b981"
elif any(w in verdict_lower for w in ["sell", "underweight", "매도", "축소"]):
    verdict_class, verdict_emoji, verdict_color = "verdict-sell", "📉", "#ef4444"
else:
    verdict_class, verdict_emoji, verdict_color = "verdict-hold", "➡️", "#f59e0b"

sentiment_label = extract_sentiment(read("01_analysts/sentiment.md"))

# ────────────────────────────────────────────────────────────
# 섹션 정의 (텍스트 콘텐츠)
# ────────────────────────────────────────────────────────────
SECTIONS = [
    {"id": "market",       "stage": "1. 분석가",          "title": "기술적 분석",          "agent": "Market Analyst",            "icon": "📊", "color": "blue",   "content": read("01_analysts/market.md")},
    {"id": "sentiment",    "stage": "1. 분석가",          "title": "정서 분석",            "agent": "Sentiment Analyst",         "icon": "💬", "color": "blue",   "content": read("01_analysts/sentiment.md")},
    {"id": "news",         "stage": "1. 분석가",          "title": "뉴스 분석",            "agent": "News Analyst",              "icon": "📰", "color": "blue",   "content": read("01_analysts/news.md")},
    {"id": "fundamentals", "stage": "1. 분석가",          "title": "펀더멘털",             "agent": "Fundamentals Analyst",      "icon": "📈", "color": "blue",   "content": read("01_analysts/fundamentals.md")},
    {"id": "supply_demand","stage": "1. 분석가",          "title": "수급 분석 (KR)",       "agent": "Supply Demand Analyst",     "icon": "🇰🇷", "color": "blue",   "content": read("01_analysts/supply_demand.md")},
    {"id": "bull",         "stage": "2. 리서치 토론",     "title": "강세 논리",            "agent": "Bull Researcher",           "icon": "🐂", "color": "green",  "content": read("02_research/bull.md")},
    {"id": "bear",         "stage": "2. 리서치 토론",     "title": "약세 논리",            "agent": "Bear Researcher",           "icon": "🐻", "color": "red",    "content": read("02_research/bear.md")},
    {"id": "manager",      "stage": "2. 리서치 토론",     "title": "리서치 매니저 종합",   "agent": "Research Manager",          "icon": "🎓", "color": "purple", "content": read("02_research/manager.md")},
    {"id": "trader",       "stage": "3. 트레이더",        "title": "트레이더 1차 결정",    "agent": "Trader",                    "icon": "💼", "color": "orange", "content": read("03_trader/trader.md")},
    {"id": "aggressive",   "stage": "4. 리스크 토론",     "title": "공격적 관점",          "agent": "Aggressive Risk Analyst",   "icon": "⚡", "color": "red",    "content": read("04_risk/aggressive.md")},
    {"id": "conservative", "stage": "4. 리스크 토론",     "title": "보수적 관점",          "agent": "Conservative Risk Analyst", "icon": "🛡️", "color": "blue",   "content": read("04_risk/conservative.md")},
    {"id": "neutral",      "stage": "4. 리스크 토론",     "title": "중립 관점",            "agent": "Neutral Risk Analyst",      "icon": "⚖️", "color": "gray",   "content": read("04_risk/neutral.md")},
    {"id": "plan",         "stage": "5. 포트폴리오 매니저", "title": "투자 계획",          "agent": "Portfolio Manager (Plan)",  "icon": "📋", "color": "purple", "content": read("05_portfolio/plan.md")},
    {"id": "decision",     "stage": "5. 포트폴리오 매니저", "title": "최종 결정",          "agent": "Portfolio Manager (Decision)", "icon": "✅", "color": "purple", "content": read("05_portfolio/decision.md")},
]
stages = []
for s in SECTIONS:
    if s["stage"] not in stages:
        stages.append(s["stage"])


# ────────────────────────────────────────────────────────────
# 차트용 JS 데이터
# ────────────────────────────────────────────────────────────
def to_pairs(series):
    return [{"x": r["date"], "y": r["value"]} for r in series]


def to_price_pairs(rows, key):
    return [{"x": r["date"], "y": r[key]} for r in rows]


chart_data = {
    "labels": [r["date"] for r in ohlcv],
    "close":  to_price_pairs(ohlcv, "close"),
    "volume": [{"x": r["date"], "y": r["volume"]} for r in ohlcv],
    "ma50":   to_pairs(indicators["close_50_sma"]),
    "ma200":  to_pairs(indicators["close_200_sma"]),
    "ema10":  to_pairs(indicators["close_10_ema"]),
    "boll":   to_pairs(indicators["boll"]),
    "boll_ub": to_pairs(indicators["boll_ub"]),
    "boll_lb": to_pairs(indicators["boll_lb"]),
    "rsi":    to_pairs(indicators["rsi"]),
    "macd":   to_pairs(indicators["macd"]),
    "macds":  to_pairs(indicators["macds"]),
    "macdh":  to_pairs(indicators["macdh"]),
}

meta = {
    "ticker": TICKER,
    "date": DATE,
    "verdict": verdict,
    "verdict_color": verdict_color,
    "sentiment": sentiment_label,
    "currency": currency_symbol,
    "latest_price": latest_price,
    "price_change": price_change,
    "latest_rsi": latest_rsi,
    "latest_macd": latest_macd,
    "latest_macds": latest_macds,
    "latest_volume": latest_volume,
}

sections_json = json.dumps(SECTIONS, ensure_ascii=False)
chart_json = json.dumps(chart_data, ensure_ascii=False, default=str)
meta_json = json.dumps(meta, ensure_ascii=False, default=str)

# ────────────────────────────────────────────────────────────
# HTML
# ────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{TICKER} 분석 대시보드 — {DATE}</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  :root {{
    --bg: #0b1220;
    --panel: #131c2e;
    --panel-2: #1a2540;
    --border: #243149;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --accent: #38bdf8;
    --blue: #3b82f6;
    --green: #10b981;
    --red: #ef4444;
    --orange: #f59e0b;
    --purple: #a855f7;
    --gray: #6b7280;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, "SF Pro Display", "Segoe UI", "Noto Sans KR", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}
  .container {{ display: flex; min-height: 100vh; }}

  /* ── Sidebar ── */
  .sidebar {{
    width: 260px; background: var(--panel); border-right: 1px solid var(--border);
    padding: 20px 12px; position: sticky; top: 0; height: 100vh; overflow-y: auto;
  }}
  .ticker-card {{
    background: linear-gradient(135deg, #1e3a8a, #4c1d95);
    padding: 14px; border-radius: 10px; margin-bottom: 18px;
  }}
  .ticker-card .ticker {{ font-size: 22px; font-weight: 700; }}
  .ticker-card .date {{ font-size: 12px; color: rgba(255,255,255,0.7); margin-top: 2px; }}
  .stage-group {{ margin-bottom: 14px; }}
  .stage-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); padding: 6px 10px; font-weight: 600; }}
  .nav-item {{
    display: flex; align-items: center; gap: 8px; padding: 8px 10px;
    border-radius: 6px; cursor: pointer; font-size: 13px; color: var(--text);
    transition: all 0.15s; border-left: 3px solid transparent;
  }}
  .nav-item:hover {{ background: var(--panel-2); }}
  .nav-item.active {{ background: var(--panel-2); border-left-color: var(--accent); font-weight: 600; }}
  .nav-item .icon {{ font-size: 15px; }}
  .nav-section-link {{
    display: block; padding: 6px 10px; font-size: 12px; color: var(--text-dim);
    cursor: pointer; border-radius: 4px;
  }}
  .nav-section-link:hover {{ color: var(--accent); background: var(--panel-2); }}

  /* ── Main ── */
  .main {{ flex: 1; padding: 32px 44px; max-width: 1400px; }}

  /* Verdict banner */
  .verdict-banner {{
    padding: 22px 28px; border-radius: 14px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 22px; border: 1px solid;
  }}
  .verdict-buy {{ background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05)); border-color: var(--green); }}
  .verdict-sell {{ background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05)); border-color: var(--red); }}
  .verdict-hold {{ background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05)); border-color: var(--orange); }}
  .verdict-emoji {{ font-size: 48px; }}
  .verdict-text .label {{ font-size: 12px; color: var(--text-dim); letter-spacing: 2px; text-transform: uppercase; }}
  .verdict-text .value {{ font-size: 32px; font-weight: 800; margin-top: 4px; }}
  .verdict-text .subtitle {{ font-size: 13px; color: var(--text-dim); margin-top: 2px; }}

  /* KPI grid */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; position: relative; overflow: hidden;
  }}
  .kpi::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent);
  }}
  .kpi.green::before {{ background: var(--green); }}
  .kpi.red::before {{ background: var(--red); }}
  .kpi.orange::before {{ background: var(--orange); }}
  .kpi.purple::before {{ background: var(--purple); }}
  .kpi .label {{ font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; }}
  .kpi .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
  .kpi .sub {{ font-size: 12px; color: var(--text-dim); margin-top: 4px; }}
  .kpi.pos .value {{ color: var(--green); }}
  .kpi.neg .value {{ color: var(--red); }}

  /* Charts */
  .chart-grid {{ display: grid; gap: 14px; margin-bottom: 24px; }}
  .chart-grid.cols-2 {{ grid-template-columns: 2fr 1fr; }}
  .chart-grid.cols-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
  @media (max-width: 1100px) {{
    .chart-grid.cols-2, .chart-grid.cols-3 {{ grid-template-columns: 1fr; }}
  }}
  .chart-card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px; min-height: 120px;
  }}
  .chart-card.tall {{ min-height: 320px; }}
  .chart-card.short {{ min-height: 220px; }}
  .chart-title {{
    display: flex; justify-content: space-between; align-items: center;
    font-size: 14px; font-weight: 600; margin-bottom: 10px; color: var(--text);
  }}
  .chart-title .badge {{
    font-size: 11px; padding: 2px 8px; border-radius: 4px; background: var(--panel-2);
    color: var(--text-dim); font-weight: 500;
  }}
  .chart-canvas-wrap {{ position: relative; height: 240px; }}
  .chart-canvas-wrap.tall {{ height: 280px; }}
  .chart-canvas-wrap.short {{ height: 160px; }}

  /* RSI gauge */
  .rsi-gauge {{ display: flex; flex-direction: column; align-items: center; padding-top: 10px; }}
  .gauge-svg {{ width: 200px; height: 110px; }}
  .gauge-value {{ font-size: 32px; font-weight: 700; margin-top: -20px; }}
  .gauge-label {{ font-size: 12px; color: var(--text-dim); margin-top: 4px; }}
  .gauge-zones {{ display: flex; justify-content: space-between; width: 200px; font-size: 10px; color: var(--text-dim); margin-top: 4px; }}

  /* Debate cards */
  .debate-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{ .debate-grid {{ grid-template-columns: 1fr; }} }}
  .debate-card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px; position: relative;
  }}
  .debate-card.bull {{ border-top: 3px solid var(--green); }}
  .debate-card.bear {{ border-top: 3px solid var(--red); }}
  .debate-card h3 {{
    font-size: 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
  }}
  .debate-card.bull h3 {{ color: var(--green); }}
  .debate-card.bear h3 {{ color: var(--red); }}
  .debate-content {{ font-size: 13px; line-height: 1.6; max-height: 360px; overflow-y: auto; }}
  .debate-content::-webkit-scrollbar {{ width: 6px; }}
  .debate-content::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  .debate-content h1, .debate-content h2, .debate-content h3 {{ font-size: 14px; margin: 12px 0 6px; }}
  .debate-content p, .debate-content li {{ font-size: 13px; color: #cbd5e1; margin: 6px 0; }}
  .debate-content strong {{ color: #f1f5f9; }}
  .debate-content ul, .debate-content ol {{ margin-left: 18px; }}

  /* Risk triangle */
  .risk-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{ .risk-grid {{ grid-template-columns: 1fr; }} }}
  .risk-card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; border-top: 3px solid;
  }}
  .risk-card.aggressive {{ border-top-color: var(--red); }}
  .risk-card.conservative {{ border-top-color: var(--blue); }}
  .risk-card.neutral {{ border-top-color: var(--gray); }}
  .risk-card h3 {{ font-size: 15px; margin-bottom: 10px; display: flex; align-items: center; gap: 6px; }}
  .risk-card .risk-content {{ font-size: 12px; max-height: 240px; overflow-y: auto; color: #cbd5e1; }}

  /* Pipeline */
  .pipeline {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px; margin-bottom: 24px;
  }}
  .pipeline-title {{ font-size: 14px; font-weight: 600; margin-bottom: 14px; }}
  .pipeline-flow {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .pipeline-step {{
    flex: 1; min-width: 120px; text-align: center; padding: 10px 8px;
    background: var(--panel-2); border-radius: 8px; border: 1px solid var(--border);
  }}
  .pipeline-step .num {{
    width: 28px; height: 28px; line-height: 28px; border-radius: 50%;
    background: var(--accent); color: var(--bg); font-weight: 700; margin: 0 auto 6px; font-size: 12px;
  }}
  .pipeline-step .name {{ font-size: 12px; font-weight: 600; }}
  .pipeline-step .meta {{ font-size: 10px; color: var(--text-dim); margin-top: 2px; }}
  .pipeline-arrow {{ color: var(--accent); font-size: 18px; flex-shrink: 0; }}

  /* Section blocks (collapsible) */
  h2.section-h {{ font-size: 20px; margin: 28px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  .section {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    margin-bottom: 10px; overflow: hidden; transition: border-color 0.2s;
  }}
  .section.expanded {{ border-color: var(--accent); }}
  .section-header {{
    padding: 14px 18px; cursor: pointer; display: flex; align-items: center; gap: 12px;
  }}
  .section-header:hover {{ background: var(--panel-2); }}
  .section-header .icon {{
    width: 36px; height: 36px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;
  }}
  .icon-blue   {{ background: rgba(59,130,246,0.15);  color: var(--blue); }}
  .icon-green  {{ background: rgba(16,185,129,0.15);  color: var(--green); }}
  .icon-red    {{ background: rgba(239,68,68,0.15);   color: var(--red); }}
  .icon-orange {{ background: rgba(245,158,11,0.15);  color: var(--orange); }}
  .icon-purple {{ background: rgba(168,85,247,0.15);  color: var(--purple); }}
  .icon-gray   {{ background: rgba(107,114,128,0.15); color: var(--gray); }}
  .section-meta {{ flex: 1; }}
  .section-title {{ font-size: 15px; font-weight: 600; }}
  .section-agent {{ font-size: 11px; color: var(--text-dim); margin-top: 2px; }}
  .section-stage {{
    font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim);
    padding: 3px 8px; background: var(--bg); border-radius: 12px;
  }}
  .section-toggle {{ color: var(--text-dim); transition: transform 0.2s; }}
  .section.expanded .section-toggle {{ transform: rotate(90deg); }}
  .section-body {{ max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; }}
  .section.expanded .section-body {{ max-height: 80000px; transition: max-height 0.6s ease-in; }}
  .section-content {{ padding: 4px 24px 24px 24px; font-size: 14px; }}
  .section-content h1, .section-content h2, .section-content h3, .section-content h4 {{
    margin: 16px 0 8px; color: #f1f5f9;
  }}
  .section-content h1 {{ font-size: 18px; }}
  .section-content h2 {{ font-size: 16px; }}
  .section-content h3 {{ font-size: 15px; }}
  .section-content p {{ margin: 8px 0; color: #cbd5e1; }}
  .section-content ul, .section-content ol {{ margin: 8px 0 8px 22px; color: #cbd5e1; }}
  .section-content li {{ margin: 4px 0; }}
  .section-content strong {{ color: #f1f5f9; }}
  .section-content code {{
    background: var(--bg); padding: 2px 6px; border-radius: 4px;
    font-family: "SF Mono", monospace; font-size: 12px; color: var(--accent);
  }}
  .section-content table {{ border-collapse: collapse; margin: 10px 0; width: 100%; }}
  .section-content th, .section-content td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: left; font-size: 13px; }}
  .section-content th {{ background: var(--bg); font-weight: 600; }}

  /* Controls */
  .controls {{ display: flex; gap: 10px; margin-bottom: 14px; align-items: center; }}
  .btn {{
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }}
  .btn:hover {{ background: var(--panel-2); border-color: var(--accent); }}
  .search {{
    flex: 1; background: var(--panel); border: 1px solid var(--border);
    color: var(--text); padding: 8px 14px; border-radius: 6px; font-size: 13px; outline: none;
  }}
  .search:focus {{ border-color: var(--accent); }}
  .section.hidden {{ display: none; }}

  /* Sentiment badge */
  .sentiment-pill {{
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
  }}
  .sentiment-bullish, .sentiment-강세 {{ background: rgba(16,185,129,0.2); color: var(--green); }}
  .sentiment-bearish, .sentiment-약세 {{ background: rgba(239,68,68,0.2); color: var(--red); }}
  .sentiment-neutral, .sentiment-mixed, .sentiment-중립, .sentiment-혼조 {{ background: rgba(245,158,11,0.2); color: var(--orange); }}
</style>
</head>
<body>
<div class="container">
  <aside class="sidebar">
    <div class="ticker-card">
      <div class="ticker">{TICKER}</div>
      <div class="date">{DATE} · TradingAgents</div>
    </div>
    <div class="stage-group">
      <div class="stage-label">대시보드</div>
      <div class="nav-section-link" onclick="scrollTo2('dashboard')">📊 KPI</div>
      <div class="nav-section-link" onclick="scrollTo2('charts')">📈 차트</div>
      <div class="nav-section-link" onclick="scrollTo2('debate')">🥊 Bull vs Bear</div>
      <div class="nav-section-link" onclick="scrollTo2('risk')">⚖️ 리스크 토론</div>
      <div class="nav-section-link" onclick="scrollTo2('pipeline')">🔄 파이프라인</div>
    </div>
    <div id="nav-detail"></div>
  </aside>

  <main class="main">
    <!-- 결정 배너 -->
    <div id="dashboard" class="verdict-banner {verdict_class}">
      <div class="verdict-emoji">{verdict_emoji}</div>
      <div class="verdict-text">
        <div class="label">최종 결정</div>
        <div class="value">{verdict}</div>
        <div class="subtitle">{TICKER} · {DATE} · {MODEL_LABEL}</div>
      </div>
    </div>

    <!-- KPI 카드들 -->
    <div class="kpi-grid" id="kpi-grid"></div>

    <!-- 차트 그리드 -->
    <h2 class="section-h" id="charts">📈 시장 데이터 시각화</h2>
    <div class="chart-grid cols-2">
      <div class="chart-card tall">
        <div class="chart-title">가격 + 이동평균 + 볼린저 밴드 <span class="badge">3개월</span></div>
        <div class="chart-canvas-wrap tall"><canvas id="chart-price"></canvas></div>
      </div>
      <div class="chart-card tall">
        <div class="chart-title">RSI (모멘텀)</div>
        <div class="rsi-gauge" id="rsi-gauge"></div>
        <div class="chart-canvas-wrap short"><canvas id="chart-rsi"></canvas></div>
      </div>
    </div>
    <div class="chart-grid cols-3">
      <div class="chart-card short">
        <div class="chart-title">MACD <span class="badge">추세 모멘텀</span></div>
        <div class="chart-canvas-wrap short"><canvas id="chart-macd"></canvas></div>
      </div>
      <div class="chart-card short">
        <div class="chart-title">거래량 <span class="badge">Volume</span></div>
        <div class="chart-canvas-wrap short"><canvas id="chart-volume"></canvas></div>
      </div>
      <div class="chart-card short">
        <div class="chart-title">정서 분포</div>
        <div class="chart-canvas-wrap short"><canvas id="chart-sentiment"></canvas></div>
      </div>
    </div>

    <!-- Bull vs Bear -->
    <h2 class="section-h" id="debate">🥊 Bull vs Bear 토론</h2>
    <div class="debate-grid">
      <div class="debate-card bull">
        <h3>🐂 강세 논리 (Bull)</h3>
        <div class="debate-content" id="bull-content"></div>
      </div>
      <div class="debate-card bear">
        <h3>🐻 약세 논리 (Bear)</h3>
        <div class="debate-content" id="bear-content"></div>
      </div>
    </div>

    <!-- 리스크 3관점 -->
    <h2 class="section-h" id="risk">⚖️ 리스크 토론</h2>
    <div class="risk-grid">
      <div class="risk-card aggressive">
        <h3>⚡ 공격적</h3>
        <div class="risk-content" id="aggressive-content"></div>
      </div>
      <div class="risk-card neutral">
        <h3>⚖️ 중립</h3>
        <div class="risk-content" id="neutral-content"></div>
      </div>
      <div class="risk-card conservative">
        <h3>🛡️ 보수적</h3>
        <div class="risk-content" id="conservative-content"></div>
      </div>
    </div>

    <!-- 파이프라인 -->
    <h2 class="section-h" id="pipeline">🔄 분석 파이프라인</h2>
    <div class="pipeline">
      <div class="pipeline-title">5단계 의사결정 흐름</div>
      <div class="pipeline-flow" id="pipeline-flow"></div>
    </div>

    <!-- 원본 텍스트 -->
    <h2 class="section-h" id="raw">📄 원본 보고서 (전체 텍스트)</h2>
    <div class="controls">
      <input type="text" class="search" id="search" placeholder="🔍 보고서 내용 검색...">
      <button class="btn" onclick="expandAll()">모두 펼치기</button>
      <button class="btn" onclick="collapseAll()">모두 접기</button>
    </div>
    <div id="sections"></div>
  </main>
</div>

<script>
const SECTIONS = {sections_json};
const CHART_DATA = {chart_json};
const META = {meta_json};
const STAGES = {json.dumps(stages, ensure_ascii=False)};

// ────────────────────────────────────────────────────────────
// KPI 카드
// ────────────────────────────────────────────────────────────
function fmt(n, decimals=0) {{
  if (n === null || n === undefined) return '–';
  if (Math.abs(n) >= 1000000) return (n/1000000).toFixed(1) + 'M';
  if (Math.abs(n) >= 1000) return Math.round(n).toLocaleString();
  return n.toFixed(decimals);
}}

function rsiInterpret(v) {{
  if (v === null || v === undefined) return '–';
  if (v >= 70) return '과매수';
  if (v <= 30) return '과매도';
  if (v >= 60) return '강세';
  if (v <= 40) return '약세';
  return '중립';
}}

function macdSignal(macd, macds) {{
  if (macd === null || macds === null) return '–';
  return macd > macds ? '매수 시그널' : '매도 시그널';
}}

const kpis = [
  {{
    label: '현재가',
    value: META.latest_price !== null ? META.currency + fmt(META.latest_price) : '–',
    sub: META.price_change !== null ? (META.price_change >= 0 ? '▲' : '▼') + ' ' + Math.abs(META.price_change).toFixed(2) + '%' : '',
    cls: META.price_change !== null ? (META.price_change >= 0 ? 'pos' : 'neg') : ''
  }},
  {{
    label: 'RSI (14)',
    value: META.latest_rsi !== null ? META.latest_rsi.toFixed(2) : '–',
    sub: rsiInterpret(META.latest_rsi),
    cls: META.latest_rsi >= 70 ? 'red' : (META.latest_rsi <= 30 ? 'green' : 'orange')
  }},
  {{
    label: 'MACD',
    value: META.latest_macd !== null ? fmt(META.latest_macd, 0) : '–',
    sub: macdSignal(META.latest_macd, META.latest_macds),
    cls: (META.latest_macd > META.latest_macds) ? 'green' : 'red'
  }},
  {{
    label: '거래량',
    value: META.latest_volume !== null ? fmt(META.latest_volume) : '–',
    sub: '직전 거래일',
    cls: 'orange'
  }},
  {{
    label: '정서',
    value: META.sentiment,
    sub: 'Sentiment Analyst',
    cls: META.sentiment.toLowerCase().includes('bull') || META.sentiment.includes('강세') ? 'green' :
         (META.sentiment.toLowerCase().includes('bear') || META.sentiment.includes('약세') ? 'red' : 'orange')
  }},
  {{
    label: '최종 결정',
    value: META.verdict,
    sub: 'Portfolio Manager',
    cls: META.verdict_color === '#10b981' ? 'green' :
         (META.verdict_color === '#ef4444' ? 'red' : 'orange')
  }},
];

const kpiGrid = document.getElementById('kpi-grid');
kpis.forEach(k => {{
  const el = document.createElement('div');
  el.className = 'kpi ' + (k.cls || '');
  el.innerHTML = `
    <div class="label">${{k.label}}</div>
    <div class="value">${{k.value}}</div>
    <div class="sub">${{k.sub}}</div>
  `;
  kpiGrid.appendChild(el);
}});

// ────────────────────────────────────────────────────────────
// Chart.js 공통 설정
// ────────────────────────────────────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#243149';
Chart.defaults.font.family = "-apple-system, 'SF Pro Display', 'Noto Sans KR', sans-serif";

const dateAxis = {{
  type: 'time',
  time: {{ unit: 'week', tooltipFormat: 'yyyy-MM-dd', displayFormats: {{ week: 'M/d' }} }},
  grid: {{ color: 'rgba(36,49,73,0.5)' }},
  ticks: {{ font: {{ size: 10 }} }},
}};

// 가격 + MA + 볼린저
new Chart(document.getElementById('chart-price'), {{
  type: 'line',
  data: {{
    datasets: [
      {{
        label: '종가', data: CHART_DATA.close, borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.1)', borderWidth: 2, pointRadius: 0, tension: 0.1, fill: true,
      }},
      {{ label: 'MA50',  data: CHART_DATA.ma50,  borderColor: '#10b981', borderWidth: 1.2, pointRadius: 0, borderDash: [4,4] }},
      {{ label: 'MA200', data: CHART_DATA.ma200, borderColor: '#a855f7', borderWidth: 1.2, pointRadius: 0, borderDash: [4,4] }},
      {{ label: 'EMA10', data: CHART_DATA.ema10, borderColor: '#f59e0b', borderWidth: 1.2, pointRadius: 0 }},
      {{ label: 'BB Upper', data: CHART_DATA.boll_ub, borderColor: 'rgba(239,68,68,0.5)', borderWidth: 1, pointRadius: 0, borderDash: [2,4] }},
      {{ label: 'BB Lower', data: CHART_DATA.boll_lb, borderColor: 'rgba(239,68,68,0.5)', borderWidth: 1, pointRadius: 0, borderDash: [2,4] }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }}, tooltip: {{ backgroundColor: '#131c2e' }} }},
    scales: {{ x: dateAxis, y: {{ grid: {{ color: 'rgba(36,49,73,0.5)' }}, ticks: {{ callback: v => META.currency + v.toLocaleString() }} }} }}
  }}
}});

// RSI 시계열
new Chart(document.getElementById('chart-rsi'), {{
  type: 'line',
  data: {{
    datasets: [{{
      label: 'RSI', data: CHART_DATA.rsi,
      borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.15)',
      borderWidth: 2, pointRadius: 0, tension: 0.2, fill: true,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ...dateAxis, ticks: {{ font: {{ size: 9 }} }} }},
      y: {{
        min: 0, max: 100,
        grid: {{ color: ctx => ctx.tick.value === 70 || ctx.tick.value === 30 ? '#ef4444' : 'rgba(36,49,73,0.5)' }},
        ticks: {{ stepSize: 20, font: {{ size: 10 }} }}
      }}
    }}
  }}
}});

// MACD
new Chart(document.getElementById('chart-macd'), {{
  type: 'bar',
  data: {{
    datasets: [
      {{
        type: 'bar', label: 'Histogram', data: CHART_DATA.macdh,
        backgroundColor: ctx => (ctx.raw && ctx.raw.y >= 0) ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)',
      }},
      {{ type: 'line', label: 'MACD',   data: CHART_DATA.macd,  borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0 }},
      {{ type: 'line', label: 'Signal', data: CHART_DATA.macds, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, font: {{ size: 10 }} }} }} }},
    scales: {{ x: {{ ...dateAxis, ticks: {{ font: {{ size: 9 }} }} }}, y: {{ grid: {{ color: 'rgba(36,49,73,0.5)' }}, ticks: {{ font: {{ size: 10 }} }} }} }}
  }}
}});

// 거래량
new Chart(document.getElementById('chart-volume'), {{
  type: 'bar',
  data: {{
    datasets: [{{
      label: 'Volume', data: CHART_DATA.volume,
      backgroundColor: ctx => {{
        const i = ctx.dataIndex;
        const closes = CHART_DATA.close;
        if (i > 0 && closes[i] && closes[i-1]) {{
          return closes[i].y >= closes[i-1].y ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)';
        }}
        return 'rgba(148,163,184,0.5)';
      }}
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ...dateAxis, ticks: {{ font: {{ size: 9 }} }} }}, y: {{ grid: {{ color: 'rgba(36,49,73,0.5)' }}, ticks: {{ font: {{ size: 10 }}, callback: v => v >= 1e6 ? (v/1e6).toFixed(0) + 'M' : v.toLocaleString() }} }} }}
  }}
}});

// 정서 도넛
function countMatches(text, regex) {{
  if (!text) return 0;
  return (text.match(regex) || []).length;
}}
const sentimentText = (SECTIONS.find(s => s.id === 'sentiment') || {{}}).content || '';
const bullCount = countMatches(sentimentText, /Bullish|강세/gi);
const bearCount = countMatches(sentimentText, /Bearish|약세/gi);
const neutralCount = countMatches(sentimentText, /Neutral|Mixed|중립|혼조/gi);

new Chart(document.getElementById('chart-sentiment'), {{
  type: 'doughnut',
  data: {{
    labels: ['강세', '약세', '중립/혼조'],
    datasets: [{{
      data: [bullCount, bearCount, neutralCount],
      backgroundColor: ['rgba(16,185,129,0.7)', 'rgba(239,68,68,0.7)', 'rgba(245,158,11,0.7)'],
      borderColor: '#131c2e', borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false, cutout: '60%',
    plugins: {{ legend: {{ position: 'bottom', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }} }}
  }}
}});

// RSI 게이지
function renderRSIGauge() {{
  const rsi = META.latest_rsi;
  const el = document.getElementById('rsi-gauge');
  if (rsi === null) {{ el.innerHTML = '<div class="gauge-value">–</div>'; return; }}
  const angle = (rsi / 100) * 180 - 90;
  const color = rsi >= 70 ? '#ef4444' : (rsi <= 30 ? '#10b981' : '#f59e0b');
  el.innerHTML = `
    <svg class="gauge-svg" viewBox="0 0 200 110">
      <defs>
        <linearGradient id="gauge-bg" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#10b981"/>
          <stop offset="40%" stop-color="#f59e0b"/>
          <stop offset="70%" stop-color="#ef4444"/>
        </linearGradient>
      </defs>
      <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="url(#gauge-bg)" stroke-width="14" stroke-linecap="round"/>
      <line x1="100" y1="100" x2="${{100 + 70 * Math.cos((angle - 90) * Math.PI/180)}}" y2="${{100 + 70 * Math.sin((angle - 90) * Math.PI/180)}}" stroke="${{color}}" stroke-width="3" stroke-linecap="round"/>
      <circle cx="100" cy="100" r="5" fill="${{color}}"/>
    </svg>
    <div class="gauge-value" style="color:${{color}}">${{rsi.toFixed(1)}}</div>
    <div class="gauge-label">${{rsiInterpret(rsi)}}</div>
    <div class="gauge-zones"><span>0 과매도</span><span>50</span><span>과매수 100</span></div>
  `;
}}
renderRSIGauge();

// ────────────────────────────────────────────────────────────
// Bull / Bear / Risk 콘텐츠 주입
// ────────────────────────────────────────────────────────────
function fillSection(id, sectionId) {{
  const s = SECTIONS.find(x => x.id === sectionId);
  document.getElementById(id).innerHTML = marked.parse(s ? (s.content || '_(내용 없음)_') : '_(없음)_');
}}
fillSection('bull-content', 'bull');
fillSection('bear-content', 'bear');
fillSection('aggressive-content', 'aggressive');
fillSection('conservative-content', 'conservative');
fillSection('neutral-content', 'neutral');

// ────────────────────────────────────────────────────────────
// 파이프라인 시각화
// ────────────────────────────────────────────────────────────
const pipelineSteps = [
  {{ num: 1, name: '분석가 4명', meta: 'Market·Sentiment·News·Fundamentals' }},
  {{ num: 2, name: 'Bull vs Bear', meta: '리서치 매니저 종합' }},
  {{ num: 3, name: '트레이더', meta: '1차 결정안' }},
  {{ num: 4, name: '리스크 토론', meta: 'Aggressive·Conservative·Neutral' }},
  {{ num: 5, name: '포트폴리오 매니저', meta: '최종 결정: ' + META.verdict }},
];
const flow = document.getElementById('pipeline-flow');
pipelineSteps.forEach((step, i) => {{
  const el = document.createElement('div');
  el.className = 'pipeline-step';
  el.innerHTML = `<div class="num">${{step.num}}</div><div class="name">${{step.name}}</div><div class="meta">${{step.meta}}</div>`;
  flow.appendChild(el);
  if (i < pipelineSteps.length - 1) {{
    const a = document.createElement('div');
    a.className = 'pipeline-arrow';
    a.textContent = '→';
    flow.appendChild(a);
  }}
}});

// ────────────────────────────────────────────────────────────
// 원본 텍스트 섹션
// ────────────────────────────────────────────────────────────
const navDetail = document.getElementById('nav-detail');
STAGES.forEach(stage => {{
  const group = document.createElement('div');
  group.className = 'stage-group';
  group.innerHTML = `<div class="stage-label">${{stage}}</div>`;
  SECTIONS.filter(s => s.stage === stage).forEach(s => {{
    const item = document.createElement('div');
    item.className = 'nav-item';
    item.dataset.target = s.id;
    item.innerHTML = `<span class="icon">${{s.icon}}</span><span>${{s.title}}</span>`;
    item.onclick = () => {{
      document.getElementById('section-' + s.id).scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      expandSection(s.id);
    }};
    group.appendChild(item);
  }});
  navDetail.appendChild(group);
}});

const sectionsEl = document.getElementById('sections');
SECTIONS.forEach(s => {{
  const sec = document.createElement('div');
  sec.className = 'section';
  sec.id = 'section-' + s.id;
  sec.innerHTML = `
    <div class="section-header" onclick="toggleSection('${{s.id}}')">
      <div class="icon icon-${{s.color}}">${{s.icon}}</div>
      <div class="section-meta">
        <div class="section-title">${{s.title}}</div>
        <div class="section-agent">${{s.agent}}</div>
      </div>
      <div class="section-stage">${{s.stage}}</div>
      <div class="section-toggle">▶</div>
    </div>
    <div class="section-body">
      <div class="section-content">${{marked.parse(s.content || '_(내용 없음)_')}}</div>
    </div>
  `;
  sectionsEl.appendChild(sec);
}});

function toggleSection(id) {{
  const el = document.getElementById('section-' + id);
  el.classList.toggle('expanded');
}}
function expandSection(id) {{
  document.getElementById('section-' + id).classList.add('expanded');
}}
function expandAll()   {{ document.querySelectorAll('.section').forEach(s => s.classList.add('expanded')); }}
function collapseAll() {{ document.querySelectorAll('.section').forEach(s => s.classList.remove('expanded')); }}
function scrollTo2(id) {{
  document.getElementById(id).scrollIntoView({{ behavior: 'smooth', block: 'start' }});
}}

document.getElementById('search').addEventListener('input', e => {{
  const q = e.target.value.trim().toLowerCase();
  document.querySelectorAll('.section').forEach((sec, i) => {{
    const s = SECTIONS[i];
    const haystack = (s.title + ' ' + s.agent + ' ' + s.content).toLowerCase();
    if (!q || haystack.includes(q)) {{
      sec.classList.remove('hidden');
      if (q) sec.classList.add('expanded');
    }} else {{
      sec.classList.add('hidden');
    }}
  }});
}});
</script>
</body>
</html>
"""

output = src / "report.html"
output.write_text(HTML, encoding="utf-8")
print(f"✅ 생성 완료: {output}")
print(f"   브라우저에서 열기: open {output}")
