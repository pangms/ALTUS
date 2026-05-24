"""Family E8 (Phase E): Cross-asset correlation regime. Answers Q25 (corr breakdown/tight).

Why this matters: cross-asset correlations don't stay constant. Risk-on regimes
have tight NQ-ES correlation and inverse NQ-ZB correlation. Risk-off regimes
break that structure as flight-to-quality kicks in. Correlation breakdown is
one of the cleanest "regime shift coming" signals in macro futures, and it's
distinct from any single-asset signal.

Features (4 total):
  • cr_nq_es_corr_60       short-term (60-bar) corr of NQ/ES returns
  • cr_nq_zb_corr_60       short-term corr of NQ/ZB returns (typically negative)
  • cr_corr_regime_z       Z-score of current NQ-ES corr vs 1440-bar baseline
                            (large negative = breakdown happening)
  • cr_avail               1.0 if cross-asset data available, else 0.0 (model can
                            learn to ignore the corr features when this flags 0)

CAUSALITY: returns computed from close[T] / close[T-1]; rolling corr over past
window. Orchestrator shift handles the final feature-at-T → data-≤-T-1 step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from altus.features.families.cross_asset import _load_aligned_cross_assets


def compute(df_1m: pd.DataFrame) -> pd.DataFrame:
    cross = _load_aligned_cross_assets(df_1m)
    mnq_ret = np.log(df_1m["close"] / df_1m["close"].shift(1)).fillna(0.0)

    have_nq = ("nq" in cross) and not cross["nq"]["close"].isna().all()
    have_es = ("es" in cross) and not cross["es"]["close"].isna().all()
    have_zb = ("zb" in cross) and not cross["zb"]["close"].isna().all()

    if have_nq:
        nq_ret = np.log(cross["nq"]["close"] / cross["nq"]["close"].shift(1))
    else:
        nq_ret = pd.Series(0.0, index=df_1m.index)
    if have_es:
        es_ret = np.log(cross["es"]["close"] / cross["es"]["close"].shift(1))
    else:
        es_ret = pd.Series(0.0, index=df_1m.index)
    if have_zb:
        zb_ret = np.log(cross["zb"]["close"] / cross["zb"]["close"].shift(1))
    else:
        zb_ret = pd.Series(0.0, index=df_1m.index)

    nq_es_60 = nq_ret.rolling(60, min_periods=20).corr(es_ret).fillna(0.0)
    nq_zb_60 = nq_ret.rolling(60, min_periods=20).corr(zb_ret).fillna(0.0)

    nq_es_base_mean = nq_es_60.rolling(1440, min_periods=200).mean()
    nq_es_base_std = nq_es_60.rolling(1440, min_periods=200).std().replace(0, np.nan)
    corr_z = ((nq_es_60 - nq_es_base_mean) / nq_es_base_std).fillna(0.0).clip(-5, 5)

    avail = pd.Series(1.0 if (have_nq and have_es) else 0.0, index=df_1m.index, dtype=np.float32)

    return pd.DataFrame({
        "cr_nq_es_corr_60": nq_es_60.astype(np.float32),
        "cr_nq_zb_corr_60": nq_zb_60.astype(np.float32),
        "cr_corr_regime_z": corr_z.astype(np.float32),
        "cr_avail": avail,
    }, index=df_1m.index)


FEATURE_COLUMNS = (
    "cr_nq_es_corr_60",
    "cr_nq_zb_corr_60",
    "cr_corr_regime_z",
    "cr_avail",
)
