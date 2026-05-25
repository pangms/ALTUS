#!/usr/bin/env bash
# Comprehensive L1 baseline ablation — runs every variant we can test
# meaningfully right now. Establishes the TRUE baseline for the multi-encoder
# architecture (sans Mamba CUDA kernels, sans Kronos cache).
#
# 6-variant sweep:
#   01. baseline           TCN + Phase A features only (reference point)
#   02. +PhaseE            +12 trader-frame families (Q4/5/8/11/13/23-25/28/29/32/33)
#   03. +PhaseF            +BOCPD multi-TF regime (Q19/Q20)
#   04. +SimMTM            +Self-supervised embeddings (Q27)
#   05. +PhaseE+F          features + regime combined
#   06. +PhaseE+F+SimMTM   features + regime + SSL — full architecture (sans Mamba)
#
# Each variant: 3yr × 3-fold purged walk-forward, 4mo OOS lockbox.
# Inflection head is enabled by default (Phase H) in all variants.
#
# Estimated runtime on 4090: ~17 min/variant × 6 = ~2 hours.
# Mamba excluded from this sweep — pure-PyTorch fallback is too slow without
# mamba-ssm CUDA kernels. Mamba comparison is its own dedicated session later.
#
# Usage:
#   cd /workspace/ALTUS
#   git pull origin main
#   tmux new-session -d -s sweep "bash scripts/sweep_baseline.sh > artifacts/sweep_baseline.log 2>&1"
#   sleep 60 && tail -30 artifacts/sweep_baseline.log

set -uo pipefail

BASE="vol,trend,anomaly"
# Phase E PRUNED (post-audit 2026-05-25 per Agent C feature MI analysis):
#   Kept (high MI):   round, mtf, vreg, sanat, creg, surprise, pvd
#   Dropped (~0 MI):  absorp, extension, lasym, rhythm, facc
#   Originally:       round,mtf,absorp,pvd,extension,vreg,sanat,creg,lasym,rhythm,facc,surprise
PHASE_E="round,mtf,pvd,vreg,sanat,creg,surprise"

ARTIFACTS_DIR="/workspace/ALTUS/artifacts"
SUMMARY_FILE="$ARTIFACTS_DIR/sweep_baseline_summary_$(date +%Y%m%d_%H%M%S).txt"
SIMMTM_CACHE="$ARTIFACTS_DIR/simmtm_embeddings.parquet"

mkdir -p "$ARTIFACTS_DIR"

echo "==========================================" | tee "$SUMMARY_FILE"
echo " BASELINE ARCHITECTURE SWEEP — $(date)"     | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

if [ ! -f "$SIMMTM_CACHE" ]; then
    echo " WARN: SimMTM cache not found at $SIMMTM_CACHE" | tee -a "$SUMMARY_FILE"
    echo " Variants 04, 06, 07 will fail. Run pretrain_simmtm.py + build_simmtm_cache.py first." | tee -a "$SUMMARY_FILE"
else
    echo " SimMTM cache: FOUND ($SIMMTM_CACHE)" | tee -a "$SUMMARY_FILE"
fi

# Parallel arrays: (label, families)
# Post-audit slim sweep (2026-05-25): the 6-variant comprehensive ablation made
# sense when we were chasing tiny AUC deltas in a coin-flip model. With the
# architecture-pivot fixes (3-class direction softmax, vol-scaled barriers,
# shrunk model, RevIN on), we're testing a different question now: does the
# combined fix package work, and does each remaining big component (Phase E,
# BOCPD, SimMTM) earn its place on top of the new baseline?
LABELS=(
    "01_baseline"     # rolled-up baseline — vol+trend+anomaly under the new architecture
    "02_phaseE"       # +Phase E (pruned trader-frame families)
    "03_phaseF"       # +BOCPD regime (Phase F)
    "04_full_eFsim"   # full integrated stack — Phase E + F + SimMTM
)
FAMILIES_LIST=(
    "$BASE"
    "$BASE,$PHASE_E"
    "$BASE,bocpd"
    "$BASE,$PHASE_E,bocpd,simmtm"
)

echo " Specs scheduled: ${#LABELS[@]} runs" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    families="${FAMILIES_LIST[$i]}"
    fam_tag="${families//,/+}"

    # Resume check
    existing_metrics=""
    for d in "$ARTIFACTS_DIR"/cloud_full_${fam_tag}_*; do
        if [ -f "$d/metrics.json" ]; then
            existing_metrics="$d/metrics.json"
            break
        fi
    done

    if [ -n "$existing_metrics" ]; then
        echo "" | tee -a "$SUMMARY_FILE"
        echo "# SKIP: $label  — already done: $existing_metrics" | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo "" | tee -a "$SUMMARY_FILE"
    echo "############################################################" | tee -a "$SUMMARY_FILE"
    echo "# RUN $label: families=$families   $(date +%H:%M:%S)"         | tee -a "$SUMMARY_FILE"
    echo "############################################################" | tee -a "$SUMMARY_FILE"

    LOG_FILE="$ARTIFACTS_DIR/sweep_baseline_${label}_$(date +%Y%m%d_%H%M%S).log"
    python3 -u scripts/train_cloud.py --full --families "$families" --variants "tcn" 2>&1 | tee "$LOG_FILE"

    echo "" | tee -a "$SUMMARY_FILE"
    echo "--- FINAL SUMMARY: $label ---" | tee -a "$SUMMARY_FILE"
    sed -n '/^FINAL SUMMARY$/,/^DONE in/p' "$LOG_FILE" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"
echo " SWEEP COMPLETE — $(date)"                    | tee -a "$SUMMARY_FILE"
echo " Master summary: $SUMMARY_FILE"               | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"

echo ""
echo "#################################################################"
echo "#   ATTEMPTING TO STOP THE POD AUTOMATICALLY...                 #"
echo "#################################################################"

if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "Stopping pod $RUNPOD_POD_ID via runpodctl in 30 seconds (Ctrl+C to abort)..."
    sleep 30
    runpodctl stop pod "$RUNPOD_POD_ID" && exit 0
fi

echo "#   !!! STOP THE POD NOW IN THE RUNPOD DASHBOARD !!!            #"
