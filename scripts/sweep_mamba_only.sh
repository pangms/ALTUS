#!/usr/bin/env bash
# Re-run the 2 Mamba variants that were skipped by the bug in sweep_full_arch.sh.
#
# Bug context: sweep_full_arch.sh's resume check matched TCN and Mamba runs by
# families alone, but the artifact dir naming doesn't distinguish variant — so
# 05_mamba (mamba+baseline) saw 01_baseline (tcn+baseline) and falsely thought
# "already done — skip." Same for 06.
#
# This script forces both Mamba runs to run fresh, into uniquely-named artifact
# dirs (Mamba- prefix) so we keep both the existing TCN results and the new
# Mamba results side by side for disagreement computation later.
#
# Estimated runtime: 2 × ~50min on 5090 ≈ 1.5-2 hours.
# Cost: ~$2 at 5090 rates.
#
# Usage (tmux + nohup belt-and-suspenders):
#   cd /workspace/ALTUS
#   git pull origin main
#   tmux new-session -d -s mamba "bash scripts/sweep_mamba_only.sh > artifacts/sweep_mamba.log 2>&1"
#   sleep 60 && tail -20 artifacts/sweep_mamba.log

set -uo pipefail

BASE="vol,trend,anomaly"
PHASE_E="round,mtf,absorp,pvd,extension,vreg,sanat,creg,lasym,rhythm,facc,surprise"
PHASE_F="bocpd"
PHASE_EF="${PHASE_E},${PHASE_F}"

ARTIFACTS_DIR="/workspace/ALTUS/artifacts"
SUMMARY_FILE="$ARTIFACTS_DIR/sweep_mamba_summary_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$ARTIFACTS_DIR"

echo "==========================================" | tee "$SUMMARY_FILE"
echo " MAMBA-ONLY SWEEP — $(date)"                | tee -a "$SUMMARY_FILE"
echo " Re-running variants 05 and 06 that were skipped by the previous sweep bug" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

# Parallel arrays: (label, families)
LABELS=("05_mamba" "06_mamba_EF")
FAMILIES_LIST=("$BASE" "$BASE,$PHASE_EF")

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    families="${FAMILIES_LIST[$i]}"

    echo "" | tee -a "$SUMMARY_FILE"
    echo "############################################################" | tee -a "$SUMMARY_FILE"
    echo "# RUN $label: variants=mamba families=$families   $(date +%H:%M:%S)" | tee -a "$SUMMARY_FILE"
    echo "############################################################" | tee -a "$SUMMARY_FILE"

    LOG_FILE="$ARTIFACTS_DIR/sweep_mamba_${label}_$(date +%Y%m%d_%H%M%S).log"
    python3 -u scripts/train_cloud.py --full --families "$families" --variants "mamba" 2>&1 | tee "$LOG_FILE"

    echo "" | tee -a "$SUMMARY_FILE"
    echo "--- FINAL SUMMARY: $label (mamba + $families) ---" | tee -a "$SUMMARY_FILE"
    sed -n '/^FINAL SUMMARY$/,/^DONE in/p' "$LOG_FILE" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"
echo " MAMBA SWEEP COMPLETE — $(date)"             | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"

# Compute disagreement: pair each Mamba run with its TCN counterpart
echo "" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo " Phase L: computing inter-encoder disagreement" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

# Find TCN and Mamba runs by checking which val_preds.npz files they contain.
# Dir naming doesn't include variant, so we identify by contents.
# For each fold, find the FIRST tcn_fold*_val_preds.npz and the FIRST
# mamba_fold*_val_preds.npz under any matching artifact dir.

find_preds() {
    local fam_tag="$1"
    local variant="$2"
    local fold="$3"
    # Most recent matching dir that contains the right prediction file
    for d in $(ls -dt "$ARTIFACTS_DIR"/cloud_full_${fam_tag}_* 2>/dev/null); do
        local f="$d/${variant}_fold${fold}_val_preds.npz"
        if [ -f "$f" ]; then
            echo "$f"
            return
        fi
    done
}

for fam_pair in "${BASE//,/+}:baseline" "${BASE//,/+}+${PHASE_EF//,/+}:EF"; do
    fam_tag="${fam_pair%%:*}"
    label="${fam_pair##*:}"
    echo "" | tee -a "$SUMMARY_FILE"
    echo " Disagreement for: $label (fam_tag=$fam_tag)" | tee -a "$SUMMARY_FILE"
    for fold in 0 1 2; do
        TCN_PREDS=$(find_preds "$fam_tag" "tcn" "$fold")
        MAMBA_PREDS=$(find_preds "$fam_tag" "mamba" "$fold")
        if [ -n "$TCN_PREDS" ] && [ -n "$MAMBA_PREDS" ]; then
            OUT="$ARTIFACTS_DIR/disagreement_${label}_fold${fold}.npz"
            python3 scripts/compute_disagreement.py --preds-a "$TCN_PREDS" --preds-b "$MAMBA_PREDS" --output "$OUT" 2>&1 | tee -a "$SUMMARY_FILE"
        else
            echo "  fold $fold: missing TCN=[$TCN_PREDS] or MAMBA=[$MAMBA_PREDS]" | tee -a "$SUMMARY_FILE"
        fi
    done
done

echo ""
echo "#################################################################"
echo "#   ATTEMPTING TO STOP THE POD AUTOMATICALLY...                 #"
echo "#################################################################"

if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "Stopping pod $RUNPOD_POD_ID via runpodctl in 30 seconds (Ctrl+C to abort)..."
    sleep 30
    runpodctl stop pod "$RUNPOD_POD_ID" && exit 0
fi

echo ""
echo "#   !!! STOP THE POD NOW IN THE RUNPOD DASHBOARD !!!            #"
