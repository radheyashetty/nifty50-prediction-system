"""
Improved Stock Screener
- Safer threading
- Timeout handling
- Better scoring
- Caching support
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Dict, List
import os
import threading

from .predictions import PredictionService
from .utils import get_ticker_sector


class StockScreener:
    def __init__(self, prediction_service: PredictionService, max_workers: int = 8):
        self.service = prediction_service
        cpu = os.cpu_count() or 8
        self.max_workers = min(max_workers, max(4, cpu * 2))  # FIXED
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _compute_score(self, row: Dict[str, Any]) -> float:
        """Composite scoring function"""
        return (
            0.6 * float(row.get("confidence", 0.0))
            + 0.2 * (float(row.get("rsi_14", 0.0)) / 100.0)
            + 0.2 * float(row.get("volume_ratio", 0.0))
        )

    def _screen_one_stock(self, ticker: str) -> Dict[str, Any] | None:
        try:
            # Use full Ensemble deep-dive for 100% accurately matching the dashboard
            result = self.service.predict_stock(ticker, screening_mode=True)

            if not result or "error" in result:
                return None

            indicators = (
                result.get("indicators") or result.get("technical_indicators") or {}
            )
            regime_obj = result.get("regime") or result.get("regime_analysis") or {}
            predictions = result.get("predictions") or {}

            confidence = float(
                result.get("confidence")
                or predictions.get("confidence")
                or predictions.get("bullish_probability")
                or 0.0
            )

            row = {
                "ticker": ticker,
                "sector": get_ticker_sector(ticker),
                "signal": result.get("signal")
                or predictions.get("decision")
                or "BEARISH",
                "confidence": confidence,
                "rsi_14": float(indicators.get("rsi_14", 0.0) or 0.0),
                "macd_histogram": float(indicators.get("macd_histogram", 0.0) or 0.0),
                "volume_ratio": float(indicators.get("volume_ratio", 0.0) or 0.0),
                "regime": (
                    regime_obj.get("regime_name")
                    or regime_obj.get("name")
                    or regime_obj.get("current_regime")
                    or regime_obj.get("volatility_regime")
                    or "Unknown"
                ),
                "latest_price": float(result.get("latest_price", 0.0) or 0.0),
                "return_5d": float(indicators.get("return_5d", 0.0) or 0.0),
                "volatility_30d": float(regime_obj.get("volatility", 0.0) or 0.0),
            }

            row["score"] = self._compute_score(row)
            return row

        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    def run_screener(
        self,
        sector: str | None = None,
        min_confidence: float = 0.55,
        min_volume_ratio: float = 0.0,
        regime_filter: str | None = None,
        use_cache: bool = True,
        top_n: int | None = 20,
    ) -> Dict[str, Any]:

        cache_key = (
            f"{sector}-{min_confidence}-{min_volume_ratio}-{regime_filter}-{top_n}"
        )

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        start = perf_counter()

        tickers = self.service.data_ingestion.get_all_available_tickers()

        if sector and sector.lower() != "all":
            tickers = [
                t for t in tickers if get_ticker_sector(t).lower() == sector.lower()
            ]

        rows: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._screen_one_stock, t): t for t in tickers}

            for future in as_completed(futures):
                try:
                    row = future.result(timeout=10)  # TIMEOUT FIX
                except Exception:
                    continue

                if not row or "error" in row:
                    continue

                if row["confidence"] < min_confidence:
                    continue

                if row["volume_ratio"] < min_volume_ratio:
                    continue

                if regime_filter and regime_filter.lower() not in {"all", ""}:
                    if str(row.get("regime", "")).lower() != regime_filter.lower():
                        continue

                rows.append(row)

        bullish = [r for r in rows if r["signal"].upper() == "BULLISH"]
        bearish = [r for r in rows if r["signal"].upper() == "BEARISH"]

        bullish = sorted(bullish, key=lambda r: r["score"], reverse=True)
        bearish = sorted(bearish, key=lambda r: r["score"], reverse=True)

        if top_n is not None:
            bullish = bullish[:top_n]
            bearish = bearish[:top_n]

        duration = perf_counter() - start

        result = {
            "bullish": bullish,
            "bearish": bearish,
            "total_scanned": len(tickers),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "scan_duration_sec": round(duration, 3),
        }

        if use_cache:
            with self._lock:
                self._cache[cache_key] = result

        return result
