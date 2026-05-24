"""Family 7: Cross-asset features (NQ, ES, ZB).

Why this matters: MNQ trades in a world. NQ (the parent contract) leads MNQ on
the millisecond. ES (broader equity) shows risk-on/risk-off divergence. ZB
(30yr bonds) shows the rate signal that drives tech valuations. The Phase A
families only see MNQ's own price action — they're blind to these cross-asset
confluences.

We use 3 cross-assets:
  • NQ  — full E-mini Nasdaq, MNQ's parent. Higher liquidity, leads MNQ in
          microstructure. Useful for confirmation and detecting MNQ-specific
          microstructure quirks.
  • ES  — S&P 500 e-mini. Broad equity. Divergence between ES and NQ/MNQ
          (sector rotation, tech vs broad market) is real signal.
  • ZB  — 30-year Treasury. Inverse rate signal. When ZB drops sharply,
          tech valuations get hit — predictive for short MNQ setups.

Per cross-asset we compute 4 features:
  1. ret_5m       — recent 5-min log return of the cross asset
  2. ret_30m      — recent 30-min log return
  3. corr_30m_mnq — rolling 30-bar correlation of returns with MNQ
  4. diverge_mnq  — binary sign disagreement on 5m return

Plus 2 aggregates across all 3 assets:
  5. cross_asset_alignment   — mean(sign agreement with MNQ direction)
  6. cross_asset_data_avail  — 1.0 if all 3 cross-assets have valid data here

Total: 14 features.

CAUSALITY: cross-asset features at bar T use cross-asset data through bar T-1
only. This is enforced by the master pipeline's .shift(1) at the end, plus our
own use of rolling windows that close at the same bar.

DATA AVAILABILITY: Our cross-asset parquets cover ~2024-04 → 2026-04, while
MNQ covers 2021-2026. For MNQ bars before 2024-04 the cross-asset features are
filled with 0 (neutral) and `cross_asset_data_avail` flags 0.0 so the model
can learn 'don't trust these features here.'
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9
CROSS_ASSETS: tuple[str, ...] = ("nq", "es", "zb")


def _load_aligned_cross_assets(df_mnq: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Load NQ, ES, ZB and align each to the MNQ 1m index.

    Returns dict {symbol: aligned_df}. Forward-fills price columns up to 5 bars
    (handles small data gaps); volume gets 0 in true gaps. Outside the cross
    asset's data range, both are NaN (and we handle that downstream).
    """
    from altus.data.loader import load_cross_asset

    aligned: dict[str, pd.DataFrame] = {}
    mnq_index = df_mnq.index
    start = mnq_index.min().isoformat()
    end = mnq_index.max().isoformat()
    for sym in CROSS_ASSETS:
        try:
            df = load_cross_asset(sym, start=start, end=end)
        except FileNotFoundError:
            # No parquet for this symbol — silently skip; features stay neutral
            aligned[sym] = pd.DataFrame(index=mnq_index, columns=["close", "volume"], dtype=float)
            continue
        df2 = df.reindex(mnq_index)
        df2[["open", "high", "low", "close"]] = df2[["open", "high", "low", "close"]].ffill(limit=5)
        df2["volume"] = df2["volume"].fillna(0)
        aligned[sym] = df2
    return aligned


def _log_returns(close: pd.Series, window_bars: int) -> pd.Series:
    """log(close_t / close_{t-window}). Returns NaN at warmup."""
    return np.log(close / close.shift(window_bars))


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-asset features aligned to df_1m.index. Returns 14 columns."""
    mnq_close = df_1m["close"]
    mnq_ret_5m = _log_returns(mnq_close, 5)
    mnq_ret_1m = _log_returns(mnq_close, 1)

    cross = _load_aligned_cross_assets(df_1m)

    out_blocks: list[pd.Series] = []
    data_avail = pd.Series(1.0, index=df_1m.index, dtype=np.float32)

    # Per-asset features
    sign_agreements: list[pd.Series] = []
    for sym in CROSS_ASSETS:
        close = cross[sym]["close"]
        # Track availability — drop avail to 0 when this asset is NaN
        sym_avail = (~close.isna()).astype(np.float32)
        data_avail *= sym_avail

        ret_5m = _log_returns(close, 5)
        ret_30m = _log_returns(close, 30)
        # Rolling correlation of 1m returns over 30 bars
        sym_ret_1m = _log_returns(close, 1)
        corr_30m = sym_ret_1m.rolling(30, min_periods=10).corr(mnq_ret_1m)
        diverge = (np.sign(ret_5m) != np.sign(mnq_ret_5m)).astype(np.float32)

        # Fill missing with neutral values
        ret_5m_filled = ret_5m.fillna(0).astype(np.float32)
        ret_30m_filled = ret_30m.fillna(0).astype(np.float32)
        corr_filled = corr_30m.fillna(0).astype(np.float32)
        diverge_filled = diverge.where(~ret_5m.isna(), 0.0).astype(np.float32)

        out_blocks.append(ret_5m_filled.rename(f"{sym}_ret_5m"))
        out_blocks.append(ret_30m_filled.rename(f"{sym}_ret_30m"))
        out_blocks.append(corr_filled.rename(f"{sym}_corr_30m_mnq"))
        out_blocks.append(diverge_filled.rename(f"{sym}_diverge_mnq"))

        # For alignment score: 1 if signs agree (sign of asset's 5m ret == sign of MNQ's)
        agreement = (np.sign(ret_5m) == np.sign(mnq_ret_5m)).astype(np.float32)
        agreement = agreement.where(~ret_5m.isna(), 0.5).astype(np.float32)
        sign_agreements.append(agreement)

    # Aggregate alignment: mean across 3 assets (0=all disagree, 1=all agree, 0.5=mixed)
    if sign_agreements:
        alignment = sum(sign_agreements) / len(sign_agreements)
    else:
        alignment = pd.Series(0.5, index=df_1m.index, dtype=np.float32)
    out_blocks.append(alignment.rename("cross_asset_alignment").astype(np.float32))
    out_blocks.append(data_avail.rename("cross_asset_data_avail"))

    feats = pd.concat(out_blocks, axis=1)
    return feats


FEATURE_COLUMNS = (
    "nq_ret_5m", "nq_ret_30m", "nq_corr_30m_mnq", "nq_diverge_mnq",
    "es_ret_5m", "es_ret_30m", "es_corr_30m_mnq", "es_diverge_mnq",
    "zb_ret_5m", "zb_ret_30m", "zb_corr_30m_mnq", "zb_diverge_mnq",
    "cross_asset_alignment",
    "cross_asset_data_avail",
)
