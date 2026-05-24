"""Phase A structural features orchestrator.

Combines the 5 foundation families into a single feature matrix, applies the
same causal shift our existing price-action pipeline uses, and returns a
DataFrame aligned to the 1m grid.

Per-family enable/disable flag is the key to the A/B testing protocol: we
toggle one family on at a time and re-train, isolating each family's contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from altus.features.families import (
    anomaly,
    cross_asset,
    exhaustion,
    kronos,
    session_time,
    trend_hurst,
    volatility,
)


_FAMILY_REGISTRY = {
    "session":   session_time,
    "trend":     trend_hurst,
    "vol":       volatility,
    "exhaust":   exhaustion,
    "anomaly":   anomaly,
    # Cross-asset: NQ + ES + ZB features. Uses already-downloaded cross-asset
    # parquets. Outside cross-asset data range, features default to neutral.
    "cross":     cross_asset,
    # Kronos: cache-only at training time. Requires running
    # scripts/build_kronos_cache.py once before enabling.
    "kronos":    kronos,
}


@dataclass
class StructuralSpec:
    """Per-family enable flags for A/B testing."""
    enabled: frozenset[str] = field(default_factory=lambda: frozenset(_FAMILY_REGISTRY.keys()))
    causal_shift: bool = True

    @classmethod
    def from_string(cls, families_str: str) -> "StructuralSpec":
        """Parse a comma-separated string like 'session,vol' into enabled families.

        Special values:
          'all'  → all families
          'none' → no families (baseline)
        """
        s = families_str.strip().lower()
        if s in ("all", ""):
            return cls(enabled=frozenset(_FAMILY_REGISTRY.keys()))
        if s == "none":
            return cls(enabled=frozenset())
        names = {n.strip() for n in s.split(",") if n.strip()}
        unknown = names - set(_FAMILY_REGISTRY.keys())
        if unknown:
            raise ValueError(f"unknown families: {unknown}. valid: {sorted(_FAMILY_REGISTRY.keys())}")
        return cls(enabled=frozenset(names))

    def summary(self) -> str:
        if not self.enabled:
            return "structural: none"
        return f"structural: {', '.join(sorted(self.enabled))}"


def build_structural_features(df_1m: pd.DataFrame, spec: StructuralSpec | None = None) -> pd.DataFrame:
    """Build the enabled structural feature families and return a single DataFrame.

    Returns an EMPTY DataFrame (indexed on df_1m.index) when no families are
    enabled, which lets pipeline.build_features keep the existing price-only path.
    """
    spec = spec or StructuralSpec()
    if not spec.enabled:
        return pd.DataFrame(index=df_1m.index)

    blocks: list[pd.DataFrame] = []
    for name, module in _FAMILY_REGISTRY.items():
        if name not in spec.enabled:
            continue
        blocks.append(module.compute(df_1m))

    X = pd.concat(blocks, axis=1)

    if spec.causal_shift:
        # Same convention as altus/features/pipeline.py: features at row T must
        # NOT include information from bar T itself, because at the moment we
        # trade at the open of bar T, bar T hasn't happened yet.
        X = X.shift(1)
    return X


def feature_column_count(spec: StructuralSpec | None = None) -> int:
    """Sum of feature counts across enabled families."""
    spec = spec or StructuralSpec()
    total = 0
    for name, module in _FAMILY_REGISTRY.items():
        if name in spec.enabled:
            total += len(module.FEATURE_COLUMNS)
    return total
