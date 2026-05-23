#!/usr/bin/env bash
# One-command setup + training on a fresh cloud GPU pod.
# Usage:  bash scripts/setup_cloud.sh          # quick 1-yr run (~30-60 min on 4090)
#         bash scripts/setup_cloud.sh --full   # full 5-yr / 5-fold run (~2-4 hr)

set -euo pipefail

echo "=========================================="
echo " ALTUS cloud setup"
echo "=========================================="

echo
echo "[1/3] Installing Python deps..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  done."

echo
echo "[2/3] CUDA sanity check..."
python3 -c "
import torch
print(f'  torch         : {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device        : {torch.cuda.get_device_name(0)}')
    print(f'  vram (GB)     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')
else:
    print('  WARNING: CUDA not detected. Training will be very slow.')
"

echo
echo "[3/3] Launching training..."
echo "      Output streams below. Artifacts will land in artifacts/cloud_*."
echo
exec python3 -u scripts/train_cloud.py "$@"
