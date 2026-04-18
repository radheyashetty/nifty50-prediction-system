from backend.predictions import PredictionService
from backend.screener import StockScreener
from backend.sector_analysis import SectorAnalyzer
import pandas as pd
import numpy as np


def verify_all_tabs():
    service = PredictionService()
    screener = StockScreener(service)
    analyzer = SectorAnalyzer(screener)

    ticker = "RELIANCE.NS"
    print(f"\n--- [1] DASHBOARD CHECK ({ticker}) ---")
    dashboard_res = service.predict_stock(ticker, analysis_mode="cache")
    if "error" in dashboard_res:
        print(f"Error in Dashboard: {dashboard_res['error']}")
    else:
        db_signal = dashboard_res["signal"]
        db_conf = dashboard_res["confidence"]
        db_rsi = dashboard_res["indicators"]["rsi_14"]
        print(f"Signal: {db_signal} | Conf: {db_conf:.4f} | RSI: {db_rsi:.2f}")

    print(f"\n--- [2] SCREENER CHECK ({ticker}) ---")
    screener_res = screener.run_screener(min_confidence=0, use_cache=False)
    all_stocks = screener_res["bullish"] + screener_res["bearish"]
    match = next((s for s in all_stocks if s["ticker"] == ticker), None)

    if match:
        sc_signal = match["signal"]
        sc_conf = match["confidence"]
        sc_rsi = match["rsi_14"]
        print(f"Signal: {sc_signal} | Conf: {sc_conf:.4f} | RSI: {sc_rsi:.2f}")

        # Verification
        if db_signal == sc_signal and abs(db_conf - sc_conf) < 1e-4:
            print("✅ Dashboard and Screener values MATCH.")
        else:
            print("❌ Dashboard and Screener values MISMATCH!")
            print(f"Diff Conf: {abs(db_conf - sc_conf)}")
    else:
        print(f"Stock {ticker} not found in screener results.")

    print("\n--- [3] SECTOR CHECK ---")
    sector_res = analyzer.analyze_all_sectors()
    it_sector = next(
        (
            s
            for s in sector_res["sectors"]
            if s["sector_name"] == "Information Technology"
        ),
        None,
    )
    if it_sector:
        print(
            f"IT Sector: {it_sector['bullish_pct']:.1f}% Bullish | Count: {it_sector['bullish_count']}/{it_sector['stock_count']}"
        )
        print(f"Avg Volatility: {it_sector['avg_volatility']:.4f}")
    else:
        print("Sector 'Information Technology' not found.")

    print("\n--- [4] COMPARE CHECK ---")
    tickers = ["RELIANCE.NS", "TCS.NS"]
    compare_res = []
    for t in tickers:
        compare_res.append(service.predict_stock(t, screening_mode=True))

    for i, t in enumerate(tickers):
        print(
            f"Compare {t}: Signal={compare_res[i]['signal']} | Conf={compare_res[i]['confidence']:.4f}"
        )


if __name__ == "__main__":
    verify_all_tabs()
