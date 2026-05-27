"""Cross-Asset Setup Confirmation.

When NQ shows a setup, does ES show the same pattern simultaneously?
Strong leading-confluence signal — when both major US equity index futures
align on a directional pattern, conviction is materially higher than NQ
alone. When they diverge, it's a warning.

This family does a LIGHTWEIGHT version of setup detection on ES bars,
then compares to NQ's setup state. We don't run all 8 setup families on
ES (expensive); we use cheap proxies that capture the same directional
signal.

Features (4 total):
  cac_es_direction_proxy        ES recent direction proxy (-1, 0, +1)
  cac_nq_es_aligned             NQ and ES directions both same way (boolean)
  cac_lead_lag_signed           NQ vs ES recent return delta (in normalized units)
  cac_divergence_active         abs(lead_lag) > threshold (alert signal)

CAUSALITY: ES data loaded via altus.data.load_cross_asset (causal — past
bars only). All comparisons use shifted/rolling windows.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from altus.features.families.setup_utils import EPS, atr_safe, clip_clamp


log = logging.getLogger(__name__)


NEEDS_RAW_1M = True


def _load_es_bars(df_1m: pd.DataFrame) -> pd.DataFrame | None:
    """Try to load ES bars aligned to df_1m's date range. Returns None if
    unavailable (so the family degrades gracefully)."""
    try:
        from altus.data.loader import load_cross_asset
        start = df_1m.index[0].strftime("%Y-%m-%d")
        end = df_1m.index[-1].strftime("%Y-%m-%d")
        es = load_cross_asset("es", start=start, end=end)
        return es
    except Exception as e:
        log.warning(f"cross_asset_setup_confirm: ES data unavailable ({e}); emitting zero features")
        return None


def compute(df_primary: pd.DataFrame, df_1m: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_1m is None:
        df_1m = df_primary
    n = len(df_1m)
    idx = df_1m.index

    # Try to load ES. If unavailable, return zero features.
    es = _load_es_bars(df_1m)
    if es is None or len(es) < 60:
        zeros = np.zeros(n, dtype=np.float32)
        return pd.DataFrame({
            "cac_es_direction_proxy": zeros,
            "cac_nq_es_aligned": zeros,
            "cac_lead_lag_signed": zeros,
            "cac_divergence_active": zeros,
        }, index=idx)

    # Align ES to df_1m's index (forward-fill missing)
    es_aligned = es.reindex(idx, method="ffill")
    nq_closes = df_1m["close"].to_numpy(dtype=np.float64)
    es_closes = es_aligned["close"].to_numpy(dtype=np.float64)

    nq_atr = atr_safe(df_1m, n=14)
    # ES atr (compute from its own bars)
    es_atr = atr_safe(es_aligned.dropna(subset=["high", "low", "close"]), n=14)
    # Re-align (in case shape differs)
    es_atr_aligned = pd.Series(es_atr, index=es_aligned.dropna(subset=["high", "low", "close"]).index)
    es_atr_aligned = es_atr_aligned.reindex(idx, method="ffill").fillna(1.0).to_numpy()

    # Recent direction proxy: sign of 30-bar return normalized by ATR.
    es_ret_30 = (
        pd.Series(es_closes, index=idx) - pd.Series(es_closes, index=idx).shift(30)
    ).fillna(0.0).to_numpy()
    es_ret_30_atr = es_ret_30 / np.maximum(es_atr_aligned * 30.0, EPS)
    es_direction = np.where(es_ret_30_atr > 0.05, 1.0,
                             np.where(es_ret_30_atr < -0.05, -1.0, 0.0)).astype(np.float32)

    # NQ direction: same logic
    nq_ret_30 = (
        df_1m["close"] - df_1m["close"].shift(30)
    ).fillna(0.0).to_numpy()
    nq_ret_30_atr = nq_ret_30 / np.maximum(nq_atr * 30.0, EPS)
    nq_direction = np.where(nq_ret_30_atr > 0.05, 1.0,
                             np.where(nq_ret_30_atr < -0.05, -1.0, 0.0)).astype(np.float32)

    # Alignment: both pointing same way
    aligned = ((nq_direction != 0) & (nq_direction == es_direction)).astype(np.float32)

    # Lead-lag delta: (NQ return - ES return) in normalized terms. Positive =
    # NQ ahead of ES; could indicate NQ is leading the move OR ES is lagging
    # in catch-up territory.
    lead_lag = (nq_ret_30_atr - es_ret_30_atr).astype(np.float32)
    lead_lag = clip_clamp(lead_lag, -2.0, 2.0)

    # Divergence active: |lead-lag| > 0.5 (one is clearly ahead of the other)
    divergence = (np.abs(lead_lag) > 0.5).astype(np.float32)

    return pd.DataFrame({
        "cac_es_direction_proxy": es_direction,
        "cac_nq_es_aligned": aligned,
        "cac_lead_lag_signed": lead_lag,
        "cac_divergence_active": divergence,
    }, index=idx)


FEATURE_COLUMNS = (
    "cac_es_direction_proxy",
    "cac_nq_es_aligned",
    "cac_lead_lag_signed",
    "cac_divergence_active",
)
