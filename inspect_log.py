"""
사용법:
    uv run python inspect_log.py                          # 전체 출력
    uv run python inspect_log.py market                   # 특정 섹션만
    uv run python inspect_log.py debate                   # Bull/Bear 토론
    uv run python inspect_log.py risk                     # 리스크 토론
    uv run python inspect_log.py <ticker> <date>          # 다른 종목/날짜
"""
import json
import sys
from pathlib import Path

TICKER = sys.argv[2] if len(sys.argv) > 2 else "005930.KS"
DATE = sys.argv[3] if len(sys.argv) > 3 else "2026-05-22"
SECTION = sys.argv[1] if len(sys.argv) > 1 else "all"

path = Path(f".runtime/logs/{TICKER}/TradingAgentsStrategy_logs/full_states_log_{DATE}.json")
d = json.loads(path.read_text())

SECTIONS = {
    "market":      ("📊 기술적 분석 (Technical Analyst)",       "market_report"),
    "sentiment":   ("💬 정서 분석 (Sentiment Analyst)",         "sentiment_report"),
    "news":        ("📰 뉴스 분석 (News Analyst)",              "news_report"),
    "fundamentals":("📈 펀더멘털 (Fundamentals Analyst)",       "fundamentals_report"),
    "debate":      ("🗣️  Bull vs Bear 토론",                    "investment_debate_state"),
    "trader":      ("💼 트레이더 1차 결정",                     "trader_investment_decision"),
    "risk":        ("⚖️  리스크 토론 (Aggressive/Conservative/Neutral)", "risk_debate_state"),
    "plan":        ("📋 리서치 매니저 투자 계획",               "investment_plan"),
    "final":       ("✅ 포트폴리오 매니저 최종 결정",            "final_trade_decision"),
}

def print_section(title, content):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if isinstance(content, str):
        print(content)
    elif isinstance(content, dict):
        for k, v in content.items():
            if v:
                print(f"\n--- {k} ---")
                print(v)

if SECTION == "all":
    keys_to_show = list(SECTIONS.keys())
else:
    keys_to_show = [SECTION]

for key in keys_to_show:
    if key not in SECTIONS:
        print(f"알 수 없는 섹션: {key}. 사용 가능: {list(SECTIONS.keys())}")
        sys.exit(1)
    title, json_key = SECTIONS[key]
    print_section(title, d.get(json_key, "(없음)"))
