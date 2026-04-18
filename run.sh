#!/bin/bash

echo ""
echo "====================================================="
echo "  NIFTY 50 Stock Prediction System - Setup & Run"
echo "====================================================="
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "[OK] Python found"
$PYTHON --version
echo ""

SETUP_ONLY=0
if [ "$1" = "--setup-only" ]; then
    SETUP_ONLY=1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    $PYTHON -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

echo ""
echo "[*] Activating virtual environment..."
source venv/bin/activate

# Install dependencies only when requirements change
REQ_HASH_FILE=".requirements.sha256"
if command -v sha256sum &> /dev/null; then
    CURRENT_REQ_HASH=$(sha256sum requirements.txt | awk '{print $1}')
else
    CURRENT_REQ_HASH=$(shasum -a 256 requirements.txt | awk '{print $1}')
fi

NEED_INSTALL=1
if [ -f "$REQ_HASH_FILE" ]; then
    SAVED_REQ_HASH=$(cat "$REQ_HASH_FILE")
    if [ "$SAVED_REQ_HASH" = "$CURRENT_REQ_HASH" ]; then
        NEED_INSTALL=0
    fi
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo ""
    echo "[*] Installing dependencies (first run or requirements changed)..."
    python -m pip install --no-cache-dir -q -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
    echo "$CURRENT_REQ_HASH" > "$REQ_HASH_FILE"
    echo "[OK] Dependencies installed"
else
    echo "[OK] Dependencies already up-to-date (skipping install)"
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
    echo "[OK] Setup completed. Exiting because --setup-only was passed."
    exit 0
fi

# Start the application
echo ""
echo "====================================================="
echo "  Starting NIFTY 50 Stock Lab..."
echo "====================================================="
echo ""
echo "[+] Application will be available at: http://localhost:8501"
echo "[+] Press Ctrl+C to stop the server"
echo ""

python -m uvicorn frontend.web_app:app --host 0.0.0.0 --port 8501 --reload

