"""SimMTM cached embeddings feature family (Phase K).

Answers Q27 (pattern similarity to historical setups) — gives L1 access to
the self-supervised embeddings that capture non-Markovian similarity
structure not visible to supervised triple-barrier training.

This is a CACHE-ONLY family — embeddings are computed offline by
scripts/build_simmtm_cache.py and saved as artifacts/simmtm_embeddings.parquet.
At training time we load aligned to the 1m grid. Same pattern as the kronos
family.

If the cache doesn't exist, this module raises a clear error explaining how
to build it. To train without SimMTM features, omit 'simmtm' from --families.

CAUSALITY: The pretrained encoder was trained on a separate dataset (the
random-window masked-reconstruction objective) — it produces embeddings that
summarize the local window context. No future-data leakage because the
encoder operates on the past-N-bar window ending at each bar.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    from altus.config import ARTIFACT_DIR
    cache_path = ARTIFACT_DIR / "simmtm_embeddings.parquet"
    if not cache_path.exists():
        raise RuntimeError(
            f"SimMTM embeddings cache not found at {cache_path}.\n"
            f"SimMTM features must be pre-computed before training:\n"
            f"  1. Pretrain encoder (~30-90min on 5090):\n"
            f"       python3 scripts/pretrain_simmtm.py\n"
            f"  2. Build embeddings cache (~20-40min on 5090):\n"
            f"       python3 scripts/build_simmtm_cache.py\n"
            f"To train WITHOUT SimMTM features for now, omit 'simmtm' from --families."
        )
    log.info(f"Loading SimMTM embeddings from cache: {cache_path}")
    cached = pd.read_parquet(cache_path)
    # Align to df_1m.index; fill any missing with 0.0 (neutral)
    aligned = cached.reindex(df_1m.index).fillna(0.0)
    return aligned.astype(np.float32)


# Feature column count is determined at cache-build time (d_model from encoder
# config — defaults to 96). We can't know it statically. We populate this
# tuple lazily by inspecting the cache if needed (most callers use
# DataFrame.columns directly anyway).
FEATURE_COLUMNS: tuple[str, ...] = tuple(f"simmtm_{i:03d}" for i in range(96))
