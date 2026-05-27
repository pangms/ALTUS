"""Setup Confluence — multi-setup alignment counter.

When multiple setups fire in the same direction simultaneously, conviction
should be higher than any single setup alone. This family computes the
aggregate: count of active setups by direction + the consensus score.

Distinct from L2's setup_arbitration (which picks ONE primary setup per bar
for execution): confluence gives the model awareness of stacked confirmation
as a separate signal.

Features (4 total):
  scf_long_count           # of active LONG setups (0-8)
  scf_short_count          # of active SHORT setups (0-8)
  scf_consensus_score      signed [-1, +1] — net directional consensus across active setups
  scf_total_active         total active setups (informational; modulates strength of consensus)

CAUSALITY: aggregates over already-causal setup family outputs. No additional
shift needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import clip_clamp


SETUP_PREFIXES = ("sfs", "sfa", "sld", "orb", "svwap", "spb", "scomp", "seod")


# Marker for build_structural_features two-pass mechanism: confluence
# reads setup_X_active/direction columns from pass-1 family outputs.
IS_AGGREGATOR = True


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregates over setup features. Reads setup_X_active and setup_X_direction
    from the input DataFrame — these need to be already computed upstream by
    the individual setup families. If not present, defaults to zeros.
    """
    n = len(df_primary)
    idx = df_primary.index

    long_count = np.zeros(n, dtype=np.float32)
    short_count = np.zeros(n, dtype=np.float32)
    total_active = np.zeros(n, dtype=np.float32)

    for prefix in SETUP_PREFIXES:
        active_col = f"{prefix}_active"
        dir_col = f"{prefix}_direction"
        if active_col not in df_primary.columns or dir_col not in df_primary.columns:
            continue
        active = df_primary[active_col].to_numpy(dtype=np.float32)
        direction = df_primary[dir_col].to_numpy(dtype=np.float32)
        is_long = (active >= 0.5) & (direction > 0.5)
        is_short = (active >= 0.5) & (direction < -0.5)
        long_count += is_long.astype(np.float32)
        short_count += is_short.astype(np.float32)
        total_active += (active >= 0.5).astype(np.float32)

    # Consensus: (long - short) / total — bounded [-1, +1]
    total_safe = np.maximum(total_active, 1.0)
    consensus = (long_count - short_count) / total_safe
    consensus = clip_clamp(consensus, -1.0, 1.0)

    return pd.DataFrame({
        "scf_long_count": clip_clamp(long_count, 0.0, 8.0),
        "scf_short_count": clip_clamp(short_count, 0.0, 8.0),
        "scf_consensus_score": consensus,
        "scf_total_active": clip_clamp(total_active, 0.0, 8.0),
    }, index=idx)


FEATURE_COLUMNS = (
    "scf_long_count",
    "scf_short_count",
    "scf_consensus_score",
    "scf_total_active",
)
