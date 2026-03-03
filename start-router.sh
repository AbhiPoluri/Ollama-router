#!/bin/bash
# Start the Ollama Model Router
# Usage: bash start-router.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies if needed
pip install fastapi uvicorn httpx --quiet

echo "Starting Ollama Model Router..."
python router.py
