#!/usr/bin/env bash
# Master ablation sweep for the FULL ALTUS Layer 1 architecture sprint.
#
# Tests each architectural component's contribution to OOS metrics in
# isolation. Resumable, auto-stops pod on completion.
#
# Components exercised:
#   • Baseline features:  vol+trend+anomaly
#   • Phase E features:   12 trader-frame families (round/mtf/absorp/pvd/extension/
#                         vreg/sanat/creg/lasym/rhythm/facc/surprise)
#   • Phase F features:   bocpd (multi-TF regime SSM)
#   • Mamba peer encoder: TCN+Mamba hybrid (selective state-space, stateful memory)
#   • Kronos features:    pre-trained foundation model embeddings (cache required)
#
# Sweep matrix (7-8 runs depending on Kronos cache availability):
#   01. baseline           vol+trend+anomaly, TCN-only
#   02. +PhaseE            baseline + 12 Phase E families, TCN-only
#   03. +PhaseF            baseline + BOCPD, TCN-only
#   04. +PhaseE+F          baseline + Phase E + Phase F, TCN-only
#   05. +Mamba             baseline, TCN+Mamba
#   06. +Mamba+E+F         baseline + Phase E + Phase F, TCN+Mamba
#   07. +Kronos            baseline + Kronos, TCN-only             (only if cache present)
#   08. MEGA               baseline + Phase E + F + Kronos, TCN+Mamba (only if cache present)
#
# Estimated runtime on 5090:
#   • TCN-only runs:  ~17 min each × 5 = ~85 min
#   • Mamba runs:     ~30-60 min each × 2 = ~60-120 min (pure-PyTorch sequential scan)
#   • Sweep total:    ~3-4 hours
#
# To include Kronos runs, build the cache first (3-4 hours on 5090):
#   bash scripts/build_kronos_cache_overnight.sh
# OR pre-existing cache at artifacts/kronos_features.parquet.
#
# Usage (run with nohup so terminal disconnect doesn't kill the sweep):
#   cd /workspace/ALTUS
#   git pull origin main
#   pip install -r requirements.txt
#   python3 tests/test_causal_invariance.py     # verify clean before launching
#   nohup bash scripts/sweep_full_arch.sh > artifacts/sweep_full.log 2>&1 &
#   disown
#
# Monitor: tail -f artifacts/sweep_full.log
# Completion banner: "FULL SWEEP COMPLETE"

set -uo pipefail   # no -e — one variant's failure should not abort the rest

BASE="vol,trend,anomaly"
PHASE_E="round,mtf,absorp,pvd,extension,vreg,sanat,creg,lasym,rhythm,facc,surprise"
PHASE_F="bocpd"
PHASE_EF="${PHASE_E},${PHASE_F}"
ALL_FEATS="${PHASE_EF},kronos"

ARTIFACTS_DIR="/workspace/ALTUS/artifacts"
SUMMARY_FILE="$ARTIFACTS_DIR/sweep_full_summary_$(date +%Y%m%d_%H%M%S).txt"
KRONOS_CACHE="$ARTIFACTS_DIR/kronos_features.parquet"

mkdir -p "$ARTIFACTS_DIR"

echo "==========================================" | tee "$SUMMARY_FILE"
echo " FULL ARCHITECTURE SWEEP — $(date)"         | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

if [ -f "$KRONOS_CACHE" ]; then
    KRONOS_AVAILABLE=1
    echo " Kronos cache: FOUND ($KRONOS_CACHE) — full 8-variant sweep" | tee -a "$SUMMARY_FILE"
else
    KRONOS_AVAILABLE=0
    echo " Kronos cache: NOT FOUND — running 6-variant sweep (Kronos variants skipped)" | tee -a "$SUMMARY_FILE"
    echo " To include Kronos: bash scripts/build_kronos_cache.py first, then re-run this sweep." | tee -a "$SUMMARY_FILE"
fi

# Specs: (label, families, variant)
# `variant` is the --variants flag — "tcn" or "mamba"
SPECS_LABEL=()
SPECS_FAMILIES=()
SPECS_VARIANT=()

add_spec() {
    SPECS_LABEL+=("$1")
    SPECS_FAMILIES+=("$2")
    SPECS_VARIANT+=("$3")
}

add_spec "01_baseline"        "$BASE"      "tcn"
add_spec "02_phaseE"          "$BASE,$PHASE_E"  "tcn"
add_spec "03_phaseF"          "$BASE,$PHASE_F"  "tcn"
add_spec "04_phaseEF"         "$BASE,$PHASE_EF" "tcn"
add_spec "05_mamba"           "$BASE"      "mamba"
add_spec "06_mamba_EF"        "$BASE,$PHASE_EF" "mamba"
if [ "$KRONOS_AVAILABLE" = "1" ]; then
    add_spec "07_kronos"          "$BASE,kronos"           "tcn"
    add_spec "08_mega"            "$BASE,$ALL_FEATS"       "mamba"
fi

echo " Specs scheduled: ${#SPECS_LABEL[@]} runs"            | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

for i in "${!SPECS_LABEL[@]}"; do
    label="${SPECS_LABEL[$i]}"
    families="${SPECS_FAMILIES[$i]}"
    variant="${SPECS_VARIANT[$i]}"
    fam_tag="${families//,/+}"

    # ---- Resume check: skip if this exact spec produced metrics.json before
    existing_metrics=""
    for d in "$ARTIFACTS_DIR"/cloud_full_${fam_tag}_*; do
        if [ -f "$d/metrics.json" ]; then
            existing_metrics="$d/metrics.json"
            break
        fi
    done

    if [ -n "$existing_metrics" ]; then
        echo "" | tee -a "$SUMMARY_FILE"
        echo "############################################################" | tee -a "$SUMMARY_FILE"
        echo "# SKIP: $label ($variant + $families)  — $existing_metrics" | tee -a "$SUMMARY_FILE"
        echo "############################################################" | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo "" | tee -a "$SUMMARY_FILE"
    echo "############################################################" | tee -a "$SUMMARY_FILE"
    echo "# RUN $label: variants=$variant families=$families   $(date +%H:%M:%S)" | tee -a "$SUMMARY_FILE"
    echo "############################################################" | tee -a "$SUMMARY_FILE"

    LOG_FILE="$ARTIFACTS_DIR/sweep_full_${label}_$(date +%Y%m%d_%H%M%S).log"
    python3 -u scripts/train_cloud.py --full --families "$families" --variants "$variant" 2>&1 | tee "$LOG_FILE"

    echo "" | tee -a "$SUMMARY_FILE"
    echo "--- FINAL SUMMARY: $label ($variant + $families) ---" | tee -a "$SUMMARY_FILE"
    sed -n '/^FINAL SUMMARY$/,/^DONE in/p' "$LOG_FILE" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"
echo " FULL SWEEP COMPLETE — $(date)"                 | tee -a "$SUMMARY_FILE"
echo " Master summary: $SUMMARY_FILE"                 | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"

# Phase L: compute per-fold disagreement between TCN and Mamba variants
# (if both completed) — provides the multi-encoder agreement signal for L2.
echo "" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"
echo " Phase L: computing inter-encoder disagreement" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

TCN_DIR=$(ls -d "$ARTIFACTS_DIR"/cloud_full_${BASE//,/+}_* 2>/dev/null | head -1)
MAMBA_DIR=$(ls -d "$ARTIFACTS_DIR"/cloud_full_${BASE//,/+}_*_$(ls "$ARTIFACTS_DIR" | grep -E "^cloud_full_${BASE//,/+}_mamba" | head -1 || echo "X") 2>/dev/null | head -1)

if [ -d "$TCN_DIR" ] && [ -d "$MAMBA_DIR" ] && [ "$TCN_DIR" != "$MAMBA_DIR" ]; then
    for fold in 0 1 2; do
        TCN_PREDS="$TCN_DIR/tcn_fold${fold}_val_preds.npz"
        MAMBA_PREDS="$MAMBA_DIR/mamba_fold${fold}_val_preds.npz"
        if [ -f "$TCN_PREDS" ] && [ -f "$MAMBA_PREDS" ]; then
            OUT="$ARTIFACTS_DIR/disagreement_fold${fold}.npz"
            python3 scripts/compute_disagreement.py --preds-a "$TCN_PREDS" --preds-b "$MAMBA_PREDS" --output "$OUT" 2>&1 | tee -a "$SUMMARY_FILE"
        fi
    done
else
    echo " Skipping disagreement: need both TCN-only and Mamba runs" | tee -a "$SUMMARY_FILE"
fi

echo ""
echo "#################################################################"
echo "#   ATTEMPTING TO STOP THE POD AUTOMATICALLY...                 #"
echo "#################################################################"

if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "Stopping pod $RUNPOD_POD_ID via runpodctl in 30 seconds (Ctrl+C to abort)..."
    sleep 30
    runpodctl stop pod "$RUNPOD_POD_ID" && {
        echo "Pod stop command sent. Billing should halt shortly."
        exit 0
    } || {
        echo "runpodctl stop failed. PLEASE STOP THE POD MANUALLY."
    }
else
    echo "runpodctl not available or RUNPOD_POD_ID unset."
fi

echo ""
echo "#################################################################"
echo "#   !!! STOP THE POD NOW IN THE RUNPOD DASHBOARD !!!            #"
echo "#################################################################"
