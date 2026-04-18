from backend.predictions import PredictionService
import sys

try:
    s = PredictionService()
    print("Initializing PredictionService...")
    res = s.predict_stock("RELIANCE.NS", screening_mode=True)
    if "error" in res:
        print(f"Error: {res['error']}")
        sys.exit(1)
    print(f"Success! Signal: {res.get('signal')} | Conf: {res.get('confidence')}")
except Exception as e:
    print(f"Exception: {e}")
    sys.exit(1)
