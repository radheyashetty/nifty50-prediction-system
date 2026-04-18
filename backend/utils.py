"""
Utilities Module
Common functions for data processing, caching, and logging
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Any, Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataCache:
    """Simple caching for data"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def save(self, key: str, data: Any):
        """Save data to cache"""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Data cached: {key}")
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")

    def load(self, key: str) -> Any:
        """Load data from cache"""
        filepath = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                logger.info(f"Data loaded from cache: {key}")
                return data
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
        return None


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


def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize features to 0-1 range

    Args:
        X: Feature array

    Returns:
        Normalized array, normalization parameters
    """
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    X_normalized = (X - min_vals) / (max_vals - min_vals + 1e-5)

    params = {"min_vals": min_vals, "max_vals": max_vals}

    return X_normalized, params


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Standardize features to mean=0, std=1

    Args:
        X: Feature array

    Returns:
        Standardized array, standardization parameters
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X_standardized = (X - mean) / (std + 1e-5)

    params = {"mean": mean, "std": std}

    return X_standardized, params


def create_report(title: str, sections: Dict[str, str]) -> str:
    """
    Create formatted report

    Args:
        title: Report title
        sections: Dict of section_name -> section_content

    Returns:
        Formatted report string
    """
    report = f"\n{'='*60}\n{title}\n{'='*60}\n"

    for section_name, section_content in sections.items():
        report += f"\n{section_name}\n{'-'*len(section_name)}\n{section_content}\n"

    report += f"\n{'='*60}\nReport generated: {datetime.now().isoformat()}\n"

    return report


def get_nifty50_stocks() -> Dict[str, str]:
    """Return NIFTY 50 stock mapping"""
    return {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys",
        "WIPRO.NS": "Wipro",
        "HDFC.NS": "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "AXISBANK.NS": "Axis Bank",
        "MARUTI.NS": "Maruti Suzuki",
        "BAJAJFINSV.NS": "Bajaj Finserv",
        "LT.NS": "Larsen & Toubro",
    }


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


def get_all_nse_tickers() -> list[str]:
    """Return all unique tickers from the configured NSE sector map."""
    all_tickers = []
    for tickers in get_nse_sector_map().values():
        all_tickers.extend([str(t).strip().upper() for t in tickers])
    return sorted(set([t for t in all_tickers if t]))


def validate_prediction(prediction: Dict) -> bool:
    """Validate prediction object"""
    required_keys = ["ticker", "predictions", "regime_analysis"]
    return all(key in prediction for key in required_keys)


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage"""
    return f"{value*100:.{decimals}f}%"


def get_confidence_emoji(confidence: float) -> str:
    """Get emoji based on confidence level"""
    if confidence > 0.75:
        return "🟢"
    elif confidence > 0.60:
        return "🟡"
    else:
        return "🔴"


if __name__ == "__main__":
    print("✓ Utilities module loaded")
