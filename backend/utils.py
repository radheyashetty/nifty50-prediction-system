"""
Utilities Module
Common functions for data processing, caching, and logging
"""

import json
import pandas as pd
import numpy as np
import logging
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, pd.Series):
            return o.to_dict()
        elif isinstance(o, pd.DataFrame):
            return o.to_dict("records")
        return super().default(o)


def get_nse_sector_map() -> Dict[str, list[str]]:
    """Return sector-to-ticker map used by screener and sector analysis."""
    return {
        "Information Technology": [
            "TCS.NS",
            "INFY.NS",
            "WIPRO.NS",
            "HCLTECH.NS",
            "TECHM.NS",
        ],
        "Banking & Financial Services": [
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "SBIN.NS",
            "KOTAKBANK.NS",
            "AXISBANK.NS",
        ],
        "Insurance & NBFC": [
            "HDFCLIFE.NS",
            "SBILIFE.NS",
            "BAJFINANCE.NS",
            "BAJAJFINSV.NS",
        ],
        "Pharmaceuticals": [
            "SUNPHARMA.NS",
            "CIPLA.NS",
            "DIVISLAB.NS",
            "DRREDDY.NS",
        ],
        "Consumer Goods (FMCG)": [
            "HINDUNILVR.NS",
            "ITC.NS",
            "NESTLEIND.NS",
            "BRITANNIA.NS",
            "TATACONSUM.NS",
        ],
        "Automobile": [
            "MARUTI.NS",
            "TATAMOTORS.NS",
            "EICHERMOT.NS",
            "HEROMOTOCO.NS",
            "BAJAJ-AUTO.NS",
            "M&M.NS",
        ],
        "Energy & Oil/Gas": [
            "RELIANCE.NS",
            "ONGC.NS",
            "BPCL.NS",
        ],
        "Metals & Mining": [
            "TATASTEEL.NS",
            "JSWSTEEL.NS",
            "HINDALCO.NS",
            "COALINDIA.NS",
        ],
        "Infrastructure & Cement": [
            "ULTRACEMCO.NS",
            "GRASIM.NS",
            "ADANIPORTS.NS",
            "ADANIENT.NS",
            "LT.NS",
        ],
        "Power & Utilities": [
            "NTPC.NS",
            "POWERGRID.NS",
        ],
        "Telecom": [
            "BHARTIARTL.NS",
            "INDUSINDBK.NS",
        ],
        "Healthcare & Hospitals": [
            "APOLLOHOSP.NS",
        ],
        "Consumer & Retail": [
            "TITAN.NS",
            "ASIANPAINT.NS",
            "UPL.NS",
        ],
    }


def get_ticker_sector(ticker: str) -> str:
    """Return sector name for a ticker or 'Unknown'."""
    symbol = str(ticker or "").strip().upper()
    base_symbol = symbol.split(".")[0]
    for sector, tickers in get_nse_sector_map().items():
        for candidate in tickers:
            cand = str(candidate).strip().upper()
            if symbol == cand or base_symbol == cand.split(".")[0]:
                return sector
    return "Unknown"


if __name__ == "__main__":
    print("✓ Utilities module loaded")
