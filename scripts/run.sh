#!/bin/bash
# ===========================================
# Product Name Generator - Launch Script
# ===========================================

set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
STREAMLIT=".venv/bin/streamlit"

if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Virtual environment not found. Run: python -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

show_menu() {
    clear
    echo "==========================================="
    echo "  Product Name Generator"
    echo "==========================================="
    echo
    echo "  [1] Web UI (Streamlit)"
    echo "  [2] Train model"
    echo "  [3] Generate names"
    echo "  [4] Evaluate model"
    echo "  [5] t-SNE visualization"
    echo "  [6] Generate corpus (LM Studio)"
    echo "  [7] Install dependencies"
    echo "  [0] Exit"
    echo
}

read_choice() {
    read -p "Select option: " CHOICE
}

webui() {
    echo "Starting Streamlit UI on port 8501..."
    $STREAMLIT run app/streamlit_app.py --server.port 8501
}

train_model() {
    read -p "Epochs [40]: " EPOCHS
    EPOCHS=${EPOCHS:-40}
    $PYTHON main.py train --epochs "$EPOCHS"
    read -p "Press Enter to continue..."
}

generate_names() {
    read -p "Categories (2+, comma separated) [cosmetics,food]: " CAT
    CAT=${CAT:-cosmetics,food}
    read -p "Start token [empty]: " STARTTOK
    read -p "Count [5]: " N
    N=${N:-5}

    echo "Choose method:"
    echo "  [1] Beam search"
    echo "  [2] Temperature"
    read -p "Method: " METHOD
    EXTRA=()
    if [ -n "$STARTTOK" ]; then EXTRA+=(--start-token "$STARTTOK"); fi
    if [ "$METHOD" = "1" ]; then
        read -p "Beam width [5]: " BEAM
        BEAM=${BEAM:-5}
        $PYTHON main.py generate --categories "$CAT" "${EXTRA[@]}" --beam "$BEAM" --n "$N"
    else
        read -p "Temperature [0.9]: " TEMP
        TEMP=${TEMP:-0.9}
        $PYTHON main.py generate --categories "$CAT" "${EXTRA[@]}" --temperature "$TEMP" --n "$N"
    fi
    read -p "Press Enter to continue..."
}

evaluate_model() {
    $PYTHON main.py eval
    read -p "Press Enter to continue..."
}

tsne_plot() {
    $PYTHON main.py tsne
    read -p "Press Enter to continue..."
}

generate_corpus() {
    echo "NOTE: LM Studio must be running on 127.0.0.1:8888 with unsloth/gemma-4-12B-it-qat-GGUF loaded"
    $PYTHON data/lmstudio_generator.py
    read -p "Press Enter to continue..."
}

install_deps() {
    echo "Installing dependencies..."
    $PYTHON -m pip install -r requirements.txt
    echo
    echo "NOTE: For GPU support, install PyTorch with CUDA:"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu128"
    read -p "Press Enter to continue..."
}

while true; do
    show_menu
    read_choice
    case $CHOICE in
        1) webui ;;
        2) train_model ;;
        3) generate_names ;;
        4) evaluate_model ;;
        5) tsne_plot ;;
        6) generate_corpus ;;
        7) install_deps ;;
        0) exit 0 ;;
        *) echo "Invalid option" && sleep 1 ;;
    esac
done
