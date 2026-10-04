# 📦 Installation Guide — NIFTY 50 Stock Lab

This guide walks you through setting up and running the NIFTY 50 Stock Prediction & Analysis System on Windows, macOS, and Linux.

---

## 📋 Prerequisites

- **Python 3.9+** (Tested on Python 3.9, 3.10, 3.11, 3.12, 3.13, 3.14)
- **Git**
- **4GB+ RAM** (8GB recommended for full NIFTY 50 parallel screening)
- **Operating Systems**: Windows 10/11, macOS (Intel/Apple Silicon), Ubuntu/Debian/Fedora Linux

---

## 🚀 Quick Automated Setup

### Windows
Double-click `run.bat` or run from PowerShell / Command Prompt:
```powershell
cd nifty50-prediction-system
.\run.bat
```
`run.bat` will:
1. Verify Python installation
2. Create a virtual environment (`venv`) if not present
3. Install production dependencies from `requirements.txt`
4. Launch the web application on `http://localhost:8501`

### macOS / Linux
Run the bash script:
```bash
cd nifty50-prediction-system
chmod +x run.sh
./run.sh
```

---

## 🛠️ Manual Installation Step-by-Step

If you prefer to configure your environment manually:

### 1. Clone the Repository
```bash
git clone https://github.com/radheyashetty/nifty50-prediction-system.git
cd nifty50-prediction-system
```

### 2. Create and Activate Virtual Environment
**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation & Pre-trained Models
Run the test suite to ensure all modules and models operate cleanly:
```bash
pytest
```

### 5. Launch the Web Application
```bash
uvicorn frontend.web_app:app --host 0.0.0.0 --port 8501 --reload
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## ⚙️ Environment Variables (Optional)

Create a `.env` file in the root directory for custom configurations:
```ini
# Server configuration
API_HOST=0.0.0.0
API_PORT=8501

# Model execution
ML_USE_GPU=auto
ML_RANDOM_SEED=42

# Historical lookback default (days)
DATA_LOOKBACK_DAYS=365
```

---

## 🧪 Testing

To execute all unit and integration tests:
```bash
pytest -v
```
All 25 tests verify data ingestion, feature calculation, model prediction, screener filtering, and API endpoints.
