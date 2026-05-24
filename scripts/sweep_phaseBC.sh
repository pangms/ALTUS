#!/usr/bin/env bash
# Phase B/C ablation sweep — does each new family earn its place?
#
# Baseline = Phase A survivors (vol, trend, anomaly). For each new family
# (cross, levels, liquidity, sweep, profile, flow) we run baseline+family
# and compare val AUC vs baseline alone. Final run enables all 6 together
# to check the "full vision" matches/beats the best single-family addition.
#
# 8 runs × 3yr/3fold ≈ 17min/run ≈ 2.3hr on a 4090 (~$1.20).
#
# Resumable: skips any run whose metrics.json already exists.
# Auto-stops the pod via runpodctl at the end.
#
# Usage (always run with nohup so a terminal drop doesn't kill the sweep):
#   cd /workspace/ALTUS
#   git pull origin main
#   nohup bash scripts/sweep_phaseBC.sh > artifacts/sweep_bc.log 2>&1 &
#   echo "Sweep started, PID: $!"
#
# Monitor: tail -f artifacts/sweep_bc.log
# Completion banner: "SWEEP COMPLETE"

set -uo pipefail   # no -e — one run's failure should not abort the sweep

BASELINE="vol,trend,anomaly"

# Parallel arrays: SPECS[i] is what gets passed to --families; LABELS[i] is the
# short tag we use for log/banner readability. The artifact dir uses the spec
# with commas replaced by + (see train_cloud.py line 183).
SPECS=(
    "$BASELINE"
    "$BASELINE,cross"
    "$BASELINE,levels"
    "$BASELINE,liquidity"
    "$BASELINE,sweep"
    "$BASELINE,profile"
    "$BASELINE,flow"
    "$BASELINE,cross,levels,liquidity,sweep,profile,flow"
)
LABELS=(
    "baseline"
    "+cross"
    "+levels"
    "+liquidity"
    "+sweep"
    "+profile"
    "+flow"
    "+all6"
)

ARTIFACTS_DIR="/workspace/ALTUS/artifacts"
SUMMARY_FILE="$ARTIFACTS_DIR/sweep_bc_summary_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$ARTIFACTS_DIR"

echo "==========================================" | tee "$SUMMARY_FILE"
echo " Phase B/C A/B sweep — $(date)"             | tee -a "$SUMMARY_FILE"
echo " Baseline: $BASELINE"                       | tee -a "$SUMMARY_FILE"
echo " Variants: ${LABELS[*]}"                    | tee -a "$SUMMARY_FILE"
echo " Resumable: yes"                            | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

for i in "${!SPECS[@]}"; do
    spec="${SPECS[$i]}"
    label="${LABELS[$i]}"
    fam_tag="${spec//,/+}"  # mirror train_cloud.py's artifact dir tag

    # ---- Resume check: skip if this variant's metrics.json already exists ----
    existing_metrics=""
    for d in "$ARTIFACTS_DIR"/cloud_full_${fam_tag}_*; do
        if [ -f "$d/metrics.json" ]; then
            existing_metrics="$d/metrics.json"
            break
        fi
    done

    if [ -n "$existing_metrics" ]; then
        echo ""                                                                | tee -a "$SUMMARY_FILE"
        echo "##########################################################"     | tee -a "$SUMMARY_FILE"
        echo "# SKIP: $label ($spec)  — already done: $existing_metrics"       | tee -a "$SUMMARY_FILE"
        echo "##########################################################"     | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo ""                                                                    | tee -a "$SUMMARY_FILE"
    echo "##########################################################"         | tee -a "$SUMMARY_FILE"
    echo "# RUN: $label   ($spec)   started $(date +%H:%M:%S)"                 | tee -a "$SUMMARY_FILE"
    echo "##########################################################"         | tee -a "$SUMMARY_FILE"

    LOG_FILE="$ARTIFACTS_DIR/sweep_bc_${fam_tag}_$(date +%Y%m%d_%H%M%S).log"
    python3 -u scripts/train_cloud.py --full --families "$spec" 2>&1 | tee "$LOG_FILE"

    echo "" | tee -a "$SUMMARY_FILE"
    echo "--- FINAL SUMMARY: $label ($spec) ---" | tee -a "$SUMMARY_FILE"
    sed -n '/^FINAL SUMMARY$/,/^DONE in/p' "$LOG_FILE" | tee -a "$SUMMARY_FILE"
done

# ---- Final banner ----
echo ""                                              | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"
echo " SWEEP COMPLETE — $(date)"                     | tee -a "$SUMMARY_FILE"
echo " Master summary: $SUMMARY_FILE"                | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"
echo ""
echo "#################################################################"
echo "#                                                               #"
echo "#   ATTEMPTING TO STOP THE POD AUTOMATICALLY...                 #"
echo "#                                                               #"
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
echo "#   You are being billed every minute it stays running.         #"
echo "#################################################################"
