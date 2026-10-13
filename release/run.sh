#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Name Generator - LSTM Model"
echo "  Release build (generation only)"
echo "============================================"
echo

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python3 not found."
    exit 1
fi

echo "[1/2] Installing dependencies..."
pip3 install -r requirements.txt --quiet

echo "[2/2] Launching Streamlit on port 8502..."
echo
streamlit run app.py --server.port 8502
