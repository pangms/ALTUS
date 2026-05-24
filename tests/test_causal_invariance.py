"""Causal invariance test — insurance against feature lookahead bugs.

For every feature family, we verify the following contract:

    A feature at row T must depend only on input data through row T.

We test this empirically with truncation equivalence:

    1. Build features on data[:N] (truncated)
    2. Build features on data[:N+offset] (full)
    3. For rows 0..N-1, features must be IDENTICAL between the two runs.

If any feature at row M (M < N) differs between the two runs, that feature
used input data from rows ≥ N — i.e., it looked into the future. Test fails
loudly with the family name + offending columns.

This is the kind of bug that's silent in backtest (everything looks fine) and
catastrophic in production (live features can't see the future, so the model
sees data it was never trained on). One unit test catches the whole class
forever.

Run:
    python3 tests/test_causal_invariance.py

Each family runs in isolation so a single bad family doesn't mask others.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from altus.data.loader import load_mnq
from altus.features.families import (
    absorption,
    anomaly,
    bocpd_regime,
    corr_regime,
    cross_asset,
    exhaustion,
    expectation_surprise,
    extension,
    flow,
    flow_acceleration,
    key_levels,
    liquidity_asymmetry,
    liquidity_zones,
    mtf_alignment,
    pv_divergence,
    round_levels,
    session_anatomy,
    session_time,
    sweep_detection,
    tape_rhythm,
    trend_hurst,
    vol_regime,
    volatility,
    volume_profile,
)
from altus.features.structural import StructuralSpec, build_structural_features


# Slice size: 10k bars from a window where cross-asset data exists (post-2024-05).
# N = 8000 truncate point; offset = 100 future bars that should not affect rows 0..7999.
SLICE_START = "2024-06-01"
SLICE_BARS = 10_000
TRUNCATE_AT = 8_000
OFFSET = 100  # rows N..N+OFFSET-1 are the "future" data that must not leak backward

FAMILIES = {
    "session_time":         session_time,
    "trend_hurst":          trend_hurst,
    "volatility":           volatility,
    "exhaustion":           exhaustion,
    "anomaly":              anomaly,
    "cross_asset":          cross_asset,
    "key_levels":           key_levels,
    "liquidity_zones":      liquidity_zones,
    "sweep_detection":      sweep_detection,
    "volume_profile":       volume_profile,
    "flow":                 flow,
    # Phase E additions:
    "round_levels":         round_levels,
    "mtf_alignment":        mtf_alignment,
    "absorption":           absorption,
    "pv_divergence":        pv_divergence,
    "extension":            extension,
    "vol_regime":           vol_regime,
    "session_anatomy":      session_anatomy,
    "corr_regime":          corr_regime,
    "liquidity_asymmetry":  liquidity_asymmetry,
    "tape_rhythm":          tape_rhythm,
    "flow_acceleration":    flow_acceleration,
    "expectation_surprise": expectation_surprise,
    # Phase F:
    "bocpd_regime":         bocpd_regime,
    # kronos deliberately omitted: cache lookup, no causality risk.
}


def _load_slice() -> pd.DataFrame:
    """Load a fixed slice of MNQ for testing."""
    df_full = load_mnq(start=SLICE_START)
    if len(df_full) < SLICE_BARS:
        raise RuntimeError(f"need at least {SLICE_BARS} bars from {SLICE_START}, got {len(df_full)}")
    return df_full.iloc[:SLICE_BARS].copy()


def _compare_prefix(
    truncated_out: pd.DataFrame,
    full_out: pd.DataFrame,
    n_rows: int,
    family_name: str,
) -> tuple[bool, list[str]]:
    """Compare the first n_rows of two feature DataFrames; return (passed, leak_cols)."""
    if list(truncated_out.columns) != list(full_out.columns):
        print(f"  [{family_name}] FAIL: column mismatch")
        print(f"    truncated cols: {list(truncated_out.columns)}")
        print(f"    full cols:      {list(full_out.columns)}")
        return False, list(truncated_out.columns)

    full_prefix = full_out.iloc[:n_rows]
    trunc = truncated_out.iloc[:n_rows]

    leak_cols: list[str] = []
    for col in trunc.columns:
        a = trunc[col].to_numpy()
        b = full_prefix[col].to_numpy()
        # NaN == NaN treated equal; tolerance ~1e-9 for float drift
        both_nan = np.isnan(a) & np.isnan(b) if a.dtype.kind == "f" else np.zeros_like(a, dtype=bool)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=False) if a.dtype.kind == "f" else (a == b)
        ok = close | both_nan
        if not ok.all():
            n_diff = int((~ok).sum())
            first_diff = int(np.argmax(~ok))
            leak_cols.append(f"{col} ({n_diff} rows differ, first at row {first_diff})")
    return (len(leak_cols) == 0), leak_cols


def test_family(name: str, module, df: pd.DataFrame) -> bool:
    """Test one family's causality. Returns True if passed."""
    try:
        truncated_out = module.compute(df.iloc[:TRUNCATE_AT])
        full_out = module.compute(df)
    except Exception as e:
        print(f"  [{name}] ERROR computing: {type(e).__name__}: {e}")
        return False

    passed, leak_cols = _compare_prefix(truncated_out, full_out, TRUNCATE_AT, name)
    if passed:
        print(f"  [{name}] PASS  ({len(truncated_out.columns)} cols, {TRUNCATE_AT} rows checked)")
    else:
        print(f"  [{name}] FAIL  — lookahead leak in: ")
        for c in leak_cols[:10]:
            print(f"    - {c}")
        if len(leak_cols) > 10:
            print(f"    ... and {len(leak_cols) - 10} more columns")
    return passed


def test_structural_orchestrator(df: pd.DataFrame) -> bool:
    """End-to-end test on build_structural_features with causal_shift=False.

    The orchestrator's shift(1) is itself a causal safeguard, but we want to
    verify the *underlying* family outputs are already causal before the shift.
    """
    print("\n  Orchestrator end-to-end (causal_shift=False):")
    # Use the registry's own keys (session, trend, vol, etc. — NOT the python
    # module names). Exclude kronos: it's a static cache lookup, not a temporal
    # computation, so no causality risk; unavailable in test env.
    spec = StructuralSpec.from_string(
        "session,trend,vol,exhaust,anomaly,cross,levels,liquidity,sweep,profile,flow,"
        "round,mtf,absorp,pvd,extension,vreg,sanat,creg,lasym,rhythm,facc,surprise,bocpd"
    )
    spec.causal_shift = False
    try:
        truncated_out = build_structural_features(df.iloc[:TRUNCATE_AT], spec=spec)
        full_out = build_structural_features(df, spec=spec)
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    passed, leak_cols = _compare_prefix(truncated_out, full_out, TRUNCATE_AT, "orchestrator")
    if passed:
        print(f"    PASS  ({len(truncated_out.columns)} cols total)")
    else:
        print(f"    FAIL  — first 10 leak columns:")
        for c in leak_cols[:10]:
            print(f"      - {c}")
    return passed


def main() -> int:
    print(f"Causal-invariance test")
    print(f"  slice: {SLICE_BARS} bars from {SLICE_START}")
    print(f"  truncate at row {TRUNCATE_AT}, future offset {OFFSET}")
    print(f"  contract: features[0:{TRUNCATE_AT}] must be identical when computed")
    print(f"            from data[:{TRUNCATE_AT}] vs data[:{TRUNCATE_AT + OFFSET}]\n")

    print("Loading data...")
    df = _load_slice()
    print(f"  loaded {len(df):,} bars: {df.index.min()} → {df.index.max()}\n")

    print("Per-family causality:")
    results: dict[str, bool] = {}
    for name, module in FAMILIES.items():
        results[name] = test_family(name, module, df)

    orchestrator_ok = test_structural_orchestrator(df)

    print("\n" + "=" * 60)
    n_pass = sum(results.values())
    n_fail = len(results) - n_pass
    print(f"  Families: {n_pass}/{len(results)} passed, {n_fail} failed")
    print(f"  Orchestrator: {'PASS' if orchestrator_ok else 'FAIL'}")
    print("=" * 60)

    if n_fail > 0 or not orchestrator_ok:
        print("\nLOOKAHEAD LEAK DETECTED — do not train until fixed.")
        return 1
    print("\nAll causality contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
