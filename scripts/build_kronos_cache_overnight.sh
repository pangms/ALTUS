#!/usr/bin/env bash
# Build the Kronos features cache overnight on the pod.
#
# Sequence:
#   1. Clone Kronos repo if not present
#   2. Install its requirements
#   3. Run the cache build script
#   4. If successful, the architecture sweep can include +Kronos variants.
#
# Usage:
#   cd /workspace/ALTUS
#   nohup bash scripts/build_kronos_cache_overnight.sh > artifacts/kronos_build.log 2>&1 &
#   disown
#
# After this finishes (~3-4 hours on 5090), you can run sweep_full_arch.sh
# which will automatically detect the cache and include Kronos variants.

set -uo pipefail

cd /workspace

if [ ! -d "Kronos" ]; then
    echo "[$(date)] Cloning Kronos repo..."
    git clone https://github.com/shiyu-coder/Kronos.git || { echo "git clone failed"; exit 1; }
fi

cd Kronos
echo "[$(date)] Installing Kronos requirements..."
pip install -q -r requirements.txt 2>&1 | tail -5 || echo "pip install had warnings"

cd /workspace/ALTUS
echo "[$(date)] Starting Kronos cache build..."
echo "         (this is the long-running step, ~3-4 hours on 5090)"
python3 -u scripts/build_kronos_cache.py 2>&1

if [ -f artifacts/kronos_features.parquet ]; then
    echo "[$(date)] SUCCESS: cache built at artifacts/kronos_features.parquet"
    echo "          You can now run scripts/sweep_full_arch.sh — it will detect"
    echo "          the cache and include the Kronos variants."
else
    echo "[$(date)] FAILURE: cache file not produced. Inspect output above."
    exit 1
fi
