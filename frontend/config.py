"""
API Configuration
"""

# API endpoints configuration
API_HOST = "127.0.0.1"
API_PORT = 8000

# Data settings
DATA_CACHE_DIR = "./cache"
MODEL_CACHE_DIR = "./models/trained_models"

# Model settings
LOOKBACK_DAYS = 365
SEQUENCE_LENGTH = 60
TEST_SIZE = 0.2

# Prediction settings
CONFIDENCE_THRESHOLD = 0.55
PREDICTION_HORIZON_DAYS = 3

# Backtesting settings
INITIAL_CAPITAL = 100000
TRANSACTION_COST_PCT = 0.001

# Risk-free rate for Sharpe calculations
RISK_FREE_RATE = 0.05

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/app.log"
