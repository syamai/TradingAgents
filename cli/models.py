from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    # Wire value stays "social" for saved-config and string-keyed-caller
    # back-compat; the user-facing label is "Sentiment Analyst".
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    # KIS 수급분석 — 한국 종목 전용. 비한국 ticker는 분석가 노드가 N/A 한 줄 반환.
    SUPPLY_DEMAND = "supply_demand"


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
