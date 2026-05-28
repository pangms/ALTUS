"""Setup × Higher-Timeframe agreement — the surfer bridge.

THE PROBLEM THIS SOLVES (2026-05-27 audit, the core "surfer" gap):
  The 8 setup families emit {active, strength, direction, +2 local state} —
  all SHORT-timeframe. The higher-timeframe context (mtf_alignment, BOCPD,
  trend_hurst) lives in SEPARATE parallel feature blocks. Nothing carries the
  *interaction* — "ORB just broke long, BUT the 60m/4h trend just rolled over."
  That setup×HTF judgment is the literal definition of the surfer principle
  (catch this wave only if the bigger swell supports it), yet it was left for a
  10k-param L2 MLP to discover from raw co-occurrence plus a thin ±3pp router
  nudge from a single scalar.

  This family engineers that bridge into the signal: for every setup, it emits
  a signed agreement between the setup's proposed direction and the multi-TF
  trend — so the setup that reaches L2 reads "ORB-long, strength 0.7, 4h-aligned
  +0.6" instead of timeframe-naked.

FEATURES (8 per-setup + 4 summary = 12 total):
  {setup}_htf_agree   signed [-1,+1] — setup_direction × mtf_alignment_score
                      (0 when the setup is inactive). +1 = setup fully aligned
                      with the multi-TF swell; -1 = fully counter-trend.
  shtf_primary_agree  the htf_agree of the highest-priority active setup
  shtf_primary_60m    primary setup's agreement vs the 60m component ONLY
                      (closes the mid-TF seam the audit flagged — 60m is the
                      most important intraday horizon and gets buried in the
                      mtf mean otherwise)
  shtf_aligned_count  # active setups whose direction agrees with the swell
  shtf_opposed_count  # active setups fighting the swell (counter-trend surfs)

CAUSALITY: aggregates over already-causal pass-1 outputs (setup directions +
mtf signs). No additional shift; the orchestrator's terminal shift covers it.

SURFER PRINCIPLE NOTE: opposed setups are NOT vetoed here — a counter-trend
setup with a strong local edge is exactly the wave the surfer wants to catch.
This family only makes the tension VISIBLE to the decision layer; L2 learns the
(setup × agreement) win-rate interaction itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import clip_clamp


SETUP_PREFIXES = ("sfs", "sfa", "sld", "orb", "svwap", "spb", "scomp", "seod")

# Priority order for "primary" selection — mirrors l2_router SETUP_PRIORITY
# (highest baseline WR first). Used to pick which active setup's agreement
# becomes the shtf_primary_* summary.
SETUP_PRIORITY = ("sfs", "sfa", "sld", "orb", "svwap", "spb", "seod", "scomp")


# Marker for the two-pass build_structural_features mechanism: this family
# reads setup_X_direction + mtf_* columns from pass-1 outputs.
IS_AGGREGATOR = True


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reads per-setup direction/active + mtf_* columns from the enriched
    pass-1 frame. Degrades gracefully (zeros) if those columns are absent."""
    n = len(df_primary)
    idx = df_primary.index

    # Multi-TF swell signals. mtf_alignment_score is the full composite
    # (mean of 5m/15m/60m/240m signs); mtf_trend_sign_60m is the mid-TF seam.
    if "mtf_alignment_score" in df_primary.columns:
        mtf_score = df_primary["mtf_alignment_score"].to_numpy(dtype=np.float32)
    else:
        mtf_score = np.zeros(n, dtype=np.float32)
    if "mtf_trend_sign_60m" in df_primary.columns:
        mtf_60m = df_primary["mtf_trend_sign_60m"].to_numpy(dtype=np.float32)
    else:
        mtf_60m = np.zeros(n, dtype=np.float32)

    out: dict[str, np.ndarray] = {}
    per_setup_agree: dict[str, np.ndarray] = {}
    per_setup_active: dict[str, np.ndarray] = {}
    per_setup_dir: dict[str, np.ndarray] = {}

    aligned_count = np.zeros(n, dtype=np.float32)
    opposed_count = np.zeros(n, dtype=np.float32)

    for prefix in SETUP_PREFIXES:
        active_col = f"{prefix}_active"
        dir_col = f"{prefix}_direction"
        if active_col in df_primary.columns and dir_col in df_primary.columns:
            active = df_primary[active_col].to_numpy(dtype=np.float32)
            direction = df_primary[dir_col].to_numpy(dtype=np.float32)
        else:
            active = np.zeros(n, dtype=np.float32)
            direction = np.zeros(n, dtype=np.float32)

        is_active = active >= 0.5
        # Signed agreement: direction × swell, zeroed when inactive.
        agree = np.where(is_active, direction * mtf_score, 0.0).astype(np.float32)
        agree = clip_clamp(agree, -1.0, 1.0)
        out[f"{prefix}_htf_agree"] = agree

        per_setup_agree[prefix] = agree
        per_setup_active[prefix] = is_active
        per_setup_dir[prefix] = direction

        # Alignment counts (sign match between setup direction and swell)
        swell_sign = np.sign(mtf_score)
        dir_sign = np.sign(direction)
        aligned = is_active & (dir_sign != 0) & (dir_sign == swell_sign)
        opposed = is_active & (dir_sign != 0) & (swell_sign != 0) & (dir_sign != swell_sign)
        aligned_count += aligned.astype(np.float32)
        opposed_count += opposed.astype(np.float32)

    # Primary setup (highest-priority active) summary — agreement + 60m seam.
    primary_agree = np.zeros(n, dtype=np.float32)
    primary_60m = np.zeros(n, dtype=np.float32)
    # Build a per-bar primary selection by priority order.
    assigned = np.zeros(n, dtype=bool)
    for prefix in SETUP_PRIORITY:
        if prefix not in per_setup_active:
            continue
        take = per_setup_active[prefix] & (~assigned)
        primary_agree = np.where(take, per_setup_agree[prefix], primary_agree)
        primary_60m = np.where(
            take,
            clip_clamp(per_setup_dir[prefix] * mtf_60m, -1.0, 1.0),
            primary_60m,
        )
        assigned = assigned | take

    out["shtf_primary_agree"] = primary_agree.astype(np.float32)
    out["shtf_primary_60m"] = primary_60m.astype(np.float32)
    out["shtf_aligned_count"] = clip_clamp(aligned_count, 0.0, 8.0)
    out["shtf_opposed_count"] = clip_clamp(opposed_count, 0.0, 8.0)

    return pd.DataFrame(out, index=idx)


FEATURE_COLUMNS = (
    "sfs_htf_agree",
    "sfa_htf_agree",
    "sld_htf_agree",
    "orb_htf_agree",
    "svwap_htf_agree",
    "spb_htf_agree",
    "scomp_htf_agree",
    "seod_htf_agree",
    "shtf_primary_agree",
    "shtf_primary_60m",
    "shtf_aligned_count",
    "shtf_opposed_count",
)
