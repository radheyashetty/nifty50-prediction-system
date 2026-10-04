import time
import os
from playwright.sync_api import sync_playwright

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUTPUT_DIR = r"docs\screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=EDGE_PATH,
            headless=True,
            args=["--disable-web-security", "--no-sandbox"]
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 960},
            device_scale_factor=2 # High-DPI crisp screenshots
        )
        page = context.new_page()
        
        # Collect console logs
        page.on("console", lambda msg: print(f"Browser console [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Browser page error: {err}"))

        print("Navigating to http://127.0.0.1:8501...")
        page.goto("http://127.0.0.1:8501", wait_until="networkidle")
        time.sleep(1)

        # 1. Select RELIANCE.NS, Cache mode, click Analyze
        print("Running Analysis on RELIANCE.NS...")
        page.select_option("#ticker", "RELIANCE.NS")
        page.select_option("#analysisMode", "cache")
        page.click("#analyzeBtn")

        # Wait for status pill to show ready / analyzed and charts to load
        page.wait_for_selector("#gaugeSignalText:not(:has-text('—'))", timeout=30000)
        time.sleep(2) # Let animations finish

        dash_path = os.path.join(OUTPUT_DIR, "01_dashboard_overview.png")
        page.screenshot(path=dash_path, full_page=False)
        print(f"Saved: {dash_path}")

        # 2. Deep Dive Tab
        print("Capturing Deep Dive tab...")
        page.click("button[data-bs-target='#analysis']")
        time.sleep(1.5)
        deep_dive_path = os.path.join(OUTPUT_DIR, "02_deep_dive_analysis.png")
        page.screenshot(path=deep_dive_path, full_page=False)
        print(f"Saved: {deep_dive_path}")

        # 3. Screener Tab
        print("Capturing Screener tab...")
        page.click("button[data-bs-target='#screener']")
        time.sleep(0.5)
        page.click("#runScreenerBtn")
        # Wait for screener summary or table
        page.wait_for_selector("#screenerBullishTable tr", timeout=60000)
        time.sleep(2)
        screener_path = os.path.join(OUTPUT_DIR, "03_stock_screener.png")
        page.screenshot(path=screener_path, full_page=False)
        print(f"Saved: {screener_path}")

        # 4. Sectors Tab
        print("Capturing Sectors tab...")
        page.click("button[data-bs-target='#sectorView']")
        time.sleep(0.5)
        page.click("#refreshSectorBtn")
        page.wait_for_selector("#sectorHeatmap .sector-tile, #sectorHeatmap div", timeout=60000)
        time.sleep(2)
        sector_path = os.path.join(OUTPUT_DIR, "04_sector_heatmap.png")
        page.screenshot(path=sector_path, full_page=False)
        print(f"Saved: {sector_path}")

        # 5. Compare Tab
        print("Capturing Compare tab...")
        page.click("button[data-bs-target='#compare']")
        time.sleep(0.5)
        page.select_option("#compareTicker1", "RELIANCE.NS")
        page.select_option("#compareTicker2", "TCS.NS")
        page.select_option("#compareTicker3", "INFY.NS")
        page.click("#runCompareBtn")
        page.wait_for_selector("#compareSignalCards .card, #compareTable tr", timeout=60000)
        time.sleep(2)
        compare_path = os.path.join(OUTPUT_DIR, "05_stock_comparison.png")
        page.screenshot(path=compare_path, full_page=False)
        print(f"Saved: {compare_path}")

        # 6. Backtest Tab
        print("Capturing Backtest tab...")
        page.click("button[data-bs-target='#backtesting']")
        time.sleep(1)
        backtest_path = os.path.join(OUTPUT_DIR, "06_strategy_backtesting.png")
        page.screenshot(path=backtest_path, full_page=False)
        print(f"Saved: {backtest_path}")

        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    run()
