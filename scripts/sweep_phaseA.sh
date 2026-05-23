#!/usr/bin/env bash
# Phase A family-level A/B sweep.
# Runs the full 5yr training 6 times — once per family + once with all families —
# so we can measure each family's contribution to OOS metrics in isolation.
#
# Each run takes ~25 min on RTX 4090, total ~2.5 hours / ~$3 cloud cost.
# Artifacts land in separate directories (cloud_full_<family>_<timestamp>).
#
# Usage:  bash scripts/sweep_phaseA.sh
#
# Output: a sweep_summary.txt at the end summarizing where each family's
# best-AUC ended up so you can paste it back to me for analysis.

set -euo pipefail

FAMILIES=("session" "trend" "vol" "exhaust" "anomaly" "all")
SUMMARY_FILE="/workspace/ALTUS/artifacts/sweep_summary_$(date +%Y%m%d_%H%M%S).txt"

echo "==========================================" | tee "$SUMMARY_FILE"
echo " Phase A A/B sweep — $(date)"            | tee -a "$SUMMARY_FILE"
echo " Families: ${FAMILIES[*]}"                | tee -a "$SUMMARY_FILE"
echo " Each run: ~25 min, ~\$0.20-0.50"         | tee -a "$SUMMARY_FILE"
echo " Total: ~2.5 hours, ~\$2-3"               | tee -a "$SUMMARY_FILE"
echo "==========================================" | tee -a "$SUMMARY_FILE"

for fam in "${FAMILIES[@]}"; do
    echo ""                                                                      | tee -a "$SUMMARY_FILE"
    echo "##########################################################"            | tee -a "$SUMMARY_FILE"
    echo "# RUN: --families $fam   (started $(date +%H:%M:%S))"                  | tee -a "$SUMMARY_FILE"
    echo "##########################################################"            | tee -a "$SUMMARY_FILE"

    # Per-run output captured to its own log file in artifacts/
    LOG_FILE="/workspace/ALTUS/artifacts/sweep_${fam}_$(date +%Y%m%d_%H%M%S).log"
    python3 -u scripts/train_cloud.py --full --families "$fam" 2>&1 | tee "$LOG_FILE"

    # Extract the FINAL SUMMARY block from this run's log into the master summary
    echo "" | tee -a "$SUMMARY_FILE"
    echo "--- FINAL SUMMARY: --families $fam ---" | tee -a "$SUMMARY_FILE"
    sed -n '/^FINAL SUMMARY$/,/^DONE in/p' "$LOG_FILE" | tee -a "$SUMMARY_FILE"
done

echo ""                                              | tee -a "$SUMMARY_FILE"
echo "=========================================="    | tee -a "$SUMMARY_FILE"
echo " SWEEP COMPLETE — $(date)"                     | tee -a "$SUMMARY_FILE"
echo " Master summary: $SUMMARY_FILE"                | tee -a "$SUMMARY_FILE"
echo "=========================================="    | tee -a "$SUMMARY_FILE"
echo ""
echo "REMEMBER TO STOP THE POD."
