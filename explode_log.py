"""
full_states_log_*.json을 CLI가 만드는 방식대로 markdown 파일들로 분해.

사용법:
    uv run python explode_log.py                                  # 005930.KS 2026-05-22 기본값 (.runtime)
    uv run python explode_log.py NVDA 2024-05-10
    uv run python explode_log.py 005930.KS 2026-05-22 .runtime/openai   # 다른 base 디렉토리
"""
import json
import sys
from pathlib import Path

TICKER = sys.argv[1] if len(sys.argv) > 1 else "005930.KS"
DATE = sys.argv[2] if len(sys.argv) > 2 else "2026-05-22"
BASE = sys.argv[3] if len(sys.argv) > 3 else ".runtime"

src = Path(f"{BASE}/logs/{TICKER}/TradingAgentsStrategy_logs/full_states_log_{DATE}.json")
dst = Path(f"{BASE}/reports/{TICKER}/{DATE}")
dst.mkdir(parents=True, exist_ok=True)

d = json.loads(src.read_text())

(dst / "01_analysts").mkdir(exist_ok=True)
(dst / "02_research").mkdir(exist_ok=True)
(dst / "03_trader").mkdir(exist_ok=True)
(dst / "04_risk").mkdir(exist_ok=True)
(dst / "05_portfolio").mkdir(exist_ok=True)

def write(path, content):
    if content:
        (dst / path).write_text(content, encoding="utf-8")

write("01_analysts/market.md",       d.get("market_report"))
write("01_analysts/sentiment.md",    d.get("sentiment_report"))
write("01_analysts/news.md",         d.get("news_report"))
write("01_analysts/fundamentals.md", d.get("fundamentals_report"))

debate = d.get("investment_debate_state") or {}
write("02_research/bull.md",    debate.get("bull_history"))
write("02_research/bear.md",    debate.get("bear_history"))
write("02_research/manager.md", debate.get("judge_decision"))

write("03_trader/trader.md", d.get("trader_investment_decision"))

risk = d.get("risk_debate_state") or {}
write("04_risk/aggressive.md",    risk.get("aggressive_history"))
write("04_risk/conservative.md",  risk.get("conservative_history"))
write("04_risk/neutral.md",       risk.get("neutral_history"))

write("05_portfolio/plan.md",     d.get("investment_plan"))
write("05_portfolio/decision.md", d.get("final_trade_decision"))

complete = [
    f"# {TICKER} 분석 보고서 — {DATE}\n",
    "## 📊 Market Analyst", d.get("market_report", ""),
    "## 💬 Sentiment Analyst", d.get("sentiment_report", ""),
    "## 📰 News Analyst", d.get("news_report", ""),
    "## 📈 Fundamentals Analyst", d.get("fundamentals_report", ""),
    "## 🐂 Bull Researcher", debate.get("bull_history", ""),
    "## 🐻 Bear Researcher", debate.get("bear_history", ""),
    "## 🎓 Research Manager", debate.get("judge_decision", ""),
    "## 💼 Trader", d.get("trader_investment_decision", ""),
    "## ⚡ Aggressive Analyst", risk.get("aggressive_history", ""),
    "## 🛡️ Conservative Analyst", risk.get("conservative_history", ""),
    "## ⚖️ Neutral Analyst", risk.get("neutral_history", ""),
    "## 📋 Investment Plan", d.get("investment_plan", ""),
    "## ✅ Final Decision", d.get("final_trade_decision", ""),
]
(dst / "complete_report.md").write_text("\n\n".join(complete), encoding="utf-8")

print(f"✅ 저장 완료: {dst}")
for f in sorted(dst.rglob("*.md")):
    print(f"  {f.relative_to(dst)}")
