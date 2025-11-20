#!/bin/bash
# Helper script to activate venv and run training

cd "$(dirname "$0")"
source .venv/bin/activate

python3 scripts/train_lora.py "$@"

