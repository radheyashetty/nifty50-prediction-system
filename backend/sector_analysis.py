"""
Improved Sector Analyzer
- Uses ONE screener run (fast)
- Better metrics
- Sector scoring
"""

from typing import Any, Dict, List
from .screener import StockScreener
from .utils import get_nse_sector_map


class SectorAnalyzer:
    def __init__(self, screener: StockScreener):
        self.screener = screener
        self.sector_map = get_nse_sector_map()

    def _momentum_direction(self, avg_return_5d: float) -> str:
        if avg_return_5d > 0.02:
            return "strong uptrend"
        elif avg_return_5d > 0:
            return "mild uptrend"
        elif avg_return_5d < -0.02:
            return "strong downtrend"
        elif avg_return_5d < 0:
            return "mild downtrend"
        return "sideways"

    def analyze_all_sectors(self) -> Dict[str, Any]:
        scan = self.screener.run_screener(
            min_confidence=0.0, use_cache=False, top_n=None
        )

        all_rows = (scan.get("bullish") or []) + (scan.get("bearish") or [])
        sector_results: List[Dict[str, Any]] = []

        for sector_name in self.sector_map.keys():
            rows = [r for r in all_rows if r["sector"] == sector_name]

            if not rows:
                continue

            bullish_count = sum(1 for r in rows if r["signal"].upper() == "BULLISH")
            bearish_count = len(rows) - bullish_count
            avg_conf = sum(r["confidence"] for r in rows) / len(rows)
            avg_rsi = sum(r["rsi_14"] for r in rows) / len(rows)
            avg_ret = sum(r["return_5d"] for r in rows) / len(rows)
            avg_vol = sum(r.get("volatility_30d", 0.0) or 0.0 for r in rows) / len(rows)

            bullish_rows = [r for r in rows if r["signal"].upper() == "BULLISH"]
            top_stock = None
            if bullish_rows:
                top_stock = sorted(
                    bullish_rows, key=lambda r: r["confidence"], reverse=True
                )[0]["ticker"]

            sector_score = (
                0.5 * (bullish_count / len(rows)) * 100
                + 0.3 * avg_conf * 100
                + 0.2 * avg_ret * 100
            )

            sector_results.append(
                {
                    "sector_name": sector_name,
                    "bullish_pct": (bullish_count / len(rows)) * 100,
                    "bullish_count": bullish_count,
                    "bearish_count": bearish_count,
                    "avg_confidence": avg_conf,
                    "avg_rsi": avg_rsi,
                    "avg_return_5d": avg_ret,
                    "avg_volatility": avg_vol,
                    "stock_count": len(rows),
                    "sector_score": sector_score,
                    "momentum_direction": self._momentum_direction(avg_ret),
                    "top_stock": top_stock,
                }
            )

        sector_results = sorted(
            sector_results, key=lambda x: x["sector_score"], reverse=True
        )

        insights = []
        if sector_results:
            top_sec = sector_results[0]["sector_name"]
            insights.append(
                f"{top_sec} is currently leading with highest combined conviction."
            )

        bottom_secs = [
            s
            for s in sector_results
            if s["momentum_direction"] in ("strong downtrend", "mild downtrend")
        ]
        if bottom_secs:
            down_names = ", ".join([s["sector_name"] for s in bottom_secs[:2]])
            insights.append(f"Weakness observed in {down_names}.")

        return {
            "sectors": sector_results,
            "top_sector": sector_results[0] if sector_results else None,
            "rotation_insights": insights,
        }
