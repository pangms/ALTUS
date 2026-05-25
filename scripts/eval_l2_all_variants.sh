#!/usr/bin/env bash
# Layer 2 cascade evaluation on the 4 TCN runs (no GPU needed).
#
# For each variant:
#   1. Extract fusion embeddings from .pt checkpoints (overwrites val_preds.npz
#      with embedding-augmented versions)
#   2. Train Layer 2 meta-labeler + measure cascade WR @ top-K%
#   3. Capture results
#
# Outputs a comparison table at the end showing L1-alone vs L1+L2 cascade
# performance across all 4 variants. Estimated time: ~1hr total on M-series Mac
# (CPU/MPS).
#
# Usage:
#   cd /Users/michaelpang/ALTUS
#   bash scripts/eval_l2_all_variants.sh 2>&1 | tee artifacts/l2_eval.log

set -uo pipefail

ARTIFACTS_DIR="artifacts/tcn_runs"
SUMMARY_FILE="artifacts/l2_eval_summary_$(date +%Y%m%d_%H%M%S).txt"

# Per-variant config: (dir_name, label)
# Families derived automatically from dir name.
VARIANTS=(
    "cloud_full_vol+trend+anomaly_20260524_105013:01_baseline"
    "cloud_full_vol+trend+anomaly+bocpd_20260524_114401:03_phaseF"
    "cloud_full_vol+trend+anomaly+round+mtf+absorp+pvd+extension+vreg+sanat+creg+lasym+rhythm+facc+surprise_20260524_111257:02_phaseE"
    "cloud_full_vol+trend+anomaly+round+mtf+absorp+pvd+extension+vreg+sanat+creg+lasym+rhythm+facc+surprise+bocpd_20260524_123154:04_phaseEF"
)

echo "============================================================" | tee "$SUMMARY_FILE"
echo " Layer 2 cascade evaluation — $(date)"                       | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"

# Detect device (mps if Apple Silicon, else cpu)
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE="mps"
else
    DEVICE="cpu"
fi
echo "Device: $DEVICE" | tee -a "$SUMMARY_FILE"

for spec in "${VARIANTS[@]}"; do
    dir_name="${spec%%:*}"
    label="${spec##*:}"
    full_path="$ARTIFACTS_DIR/$dir_name"

    if [ ! -d "$full_path" ]; then
        echo "" | tee -a "$SUMMARY_FILE"
        echo "  SKIP $label: directory not found — $full_path" | tee -a "$SUMMARY_FILE"
        continue
    fi

    # Parse families from dir name: strip "cloud_full_" prefix and "_YYYYMMDD_HHMMSS" suffix
    fam_tag=$(echo "$dir_name" | sed -E 's/^cloud_full_//; s/_[0-9]{8}_[0-9]{6}$//')
    families=$(echo "$fam_tag" | tr '+' ',')

    echo "" | tee -a "$SUMMARY_FILE"
    echo "============================================================" | tee -a "$SUMMARY_FILE"
    echo " VARIANT: $label" | tee -a "$SUMMARY_FILE"
    echo " families: $families" | tee -a "$SUMMARY_FILE"
    echo " dir:      $dir_name" | tee -a "$SUMMARY_FILE"
    echo "============================================================" | tee -a "$SUMMARY_FILE"

    # Step 1: Check if embeddings are already extracted; if not, extract them
    needs_extract=0
    if [ -f "$full_path/tcn_fold0_val_preds.npz" ]; then
        has_emb=$(python3 -c "import numpy as np; p=np.load('$full_path/tcn_fold0_val_preds.npz'); print('val_preds_fusion_embedding' in p.files)" 2>/dev/null)
        if [ "$has_emb" != "True" ]; then
            needs_extract=1
        fi
    else
        echo "  ERROR: no tcn_fold0_val_preds.npz in $full_path" | tee -a "$SUMMARY_FILE"
        continue
    fi

    if [ "$needs_extract" = "1" ]; then
        echo "" | tee -a "$SUMMARY_FILE"
        echo "[$label] Step 1: extracting fusion embeddings from .pt checkpoints..." | tee -a "$SUMMARY_FILE"
        EXTRACT_LOG="artifacts/l2_eval_${label}_extract.log"
        python3 scripts/extract_layer1_val_preds.py \
            --run-dir "$full_path" \
            --families "$families" \
            --device "$DEVICE" \
            --save-embeddings 2>&1 | tee "$EXTRACT_LOG"
    else
        echo "[$label] Step 1: embeddings already present, skipping extract" | tee -a "$SUMMARY_FILE"
    fi

    # Step 2: Train L2 + cascade evaluation
    echo "" | tee -a "$SUMMARY_FILE"
    echo "[$label] Step 2: training Layer 2 + cascade eval..." | tee -a "$SUMMARY_FILE"
    L2_LOG="artifacts/l2_eval_${label}_train.log"
    python3 scripts/train_layer2.py \
        --run-dir "$full_path" \
        --device "$DEVICE" 2>&1 | tee "$L2_LOG"

    # Capture the cascade comparison section into the summary
    echo "" | tee -a "$SUMMARY_FILE"
    echo "--- L2 cascade results: $label ---" | tee -a "$SUMMARY_FILE"
    sed -n '/Cascade/,/^$/p' "$L2_LOG" | head -40 | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"
echo " ALL DONE — $(date)" | tee -a "$SUMMARY_FILE"
echo " Master summary: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "============================================================" | tee -a "$SUMMARY_FILE"
