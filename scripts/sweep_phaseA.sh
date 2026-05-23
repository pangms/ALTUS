#!/usr/bin/env bash
# Phase A family-level A/B sweep — resumable + auto-stop.
#
# Runs the full 5yr training 6 times — once per family + once with all families —
# so we can measure each family's contribution to OOS metrics in isolation.
#
# Resumable: if a family's metrics.json already exists from a prior run, that
# family is skipped. So re-running this script after an interrupted sweep picks
# up exactly where it left off.
#
# Auto-stop: at the end, attempts to halt the pod via runpodctl so you don't
# get charged for idle time after the sweep finishes.
#
# Usage (always run with nohup so terminal disconnect doesn't kill the sweep):
#   cd /workspace/ALTUS
#   git pull origin main
#   nohup bash scripts/sweep_phaseA.sh > artifacts/sweep_full.log 2>&1 &
#   echo "Sweep started, PID: $!"
#
# To monitor: tail -f artifacts/sweep_full.log
# To check completion: look for "SWEEP COMPLETE" line near the end.

set -uo pipefail   # NOTE: no -e — we don't want one family's failure to abort the sweep

FAMILIES=("session" "trend" "vol" "exhaust" "anomaly" "all")
ARTIFACTS_DIR="/workspace/ALTUS/artifacts"
SUMMARY_FILE="$ARTIFACTS_DIR/sweep_summary_$(date +%Y%m%d_%H%M%S).txt"

mkdir -p "$ARTIFACTS_DIR"

echo "==========================================" | tee "$SUMMARY_FILE"
echo " Phase A A/B sweep — $(date)"               | tee -a "$SUMMARY_FILE"
echo " Families: ${FAMILIES[*]}"                  | tee -a "$SUMMARY_FILE"
echo " Resumable: yes (skips families with existing artifacts)" | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

for fam in "${FAMILIES[@]}"; do
    # ---- Resume check: skip if this family's metrics.json already exists ----
    # Match any prior run's artifact dir; presence of metrics.json means complete.
    existing_metrics=""
    for d in "$ARTIFACTS_DIR"/cloud_full_${fam}_*; do
        if [ -f "$d/metrics.json" ]; then
            existing_metrics="$d/metrics.json"
            break
        fi
    done

    if [ -n "$existing_metrics" ]; then
        echo ""                                                              | tee -a "$SUMMARY_FILE"
        echo "##########################################################"   | tee -a "$SUMMARY_FILE"
        echo "# SKIP: --families $fam  (already done: $existing_metrics)"   | tee -a "$SUMMARY_FILE"
        echo "##########################################################"   | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo ""                                                                  | tee -a "$SUMMARY_FILE"
    echo "##########################################################"       | tee -a "$SUMMARY_FILE"
    echo "# RUN: --families $fam   (started $(date +%H:%M:%S))"              | tee -a "$SUMMARY_FILE"
    echo "##########################################################"       | tee -a "$SUMMARY_FILE"

    LOG_FILE="$ARTIFACTS_DIR/sweep_${fam}_$(date +%Y%m%d_%H%M%S).log"
    python3 -u scripts/train_cloud.py --full --families "$fam" 2>&1 | tee "$LOG_FILE"

    # Extract FINAL SUMMARY into master summary
    echo "" | tee -a "$SUMMARY_FILE"
    echo "--- FINAL SUMMARY: --families $fam ---" | tee -a "$SUMMARY_FILE"
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

# ---- Auto-stop the pod ----
# RunPod ships runpodctl in their images and sets RUNPOD_POD_ID env var.
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

# Backup banner if auto-stop didn't succeed
echo ""
echo "#################################################################"
echo "#                                                               #"
echo "#   !!! STOP THE POD NOW IN THE RUNPOD DASHBOARD !!!            #"
echo "#   You are being billed every minute it stays running.         #"
echo "#                                                               #"
echo "#################################################################"
