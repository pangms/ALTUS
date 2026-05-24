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
    kronos,
    liquidity_asymmetry,
    liquidity_zones,
    mtf_alignment,
    pv_divergence,
    round_levels,
    session_anatomy,
    session_time,
    simmtm,
    sweep_detection,
    tape_rhythm,
    trend_hurst,
    vol_regime,
    volatility,
    volume_profile,
)


_FAMILY_REGISTRY = {
    "session":   session_time,
    "trend":     trend_hurst,
    "vol":       volatility,
    "exhaust":   exhaustion,
    "anomaly":   anomaly,
    # Cross-asset: NQ + ES + ZB features.
    "cross":     cross_asset,
    # Phase B: market structure.
    "levels":    key_levels,         # KDE on swing points + distance/proximity features
    "liquidity": liquidity_zones,    # Untouched HTF swing extremes (stop magnets)
    "sweep":     sweep_detection,    # Sweep + failed-breakout events (trap detection)
    "profile":   volume_profile,     # Kernel-smoothed volume profile (POC, VA, LVN)
    # Phase C: order flow + cross-asset causal/lead-lag.
    "flow":      flow,               # VPIN institutional flow + cross-asset lead-lag corrs
    # Phase E: trader-frame features — each addresses a specific philosophical
    # question from the 34-question framework.
    "round":     round_levels,         # Q11 — round-number proximity (4 feats)
    "mtf":       mtf_alignment,        # Q8  — multi-timeframe trend alignment (5 feats)
    "absorp":    absorption,           # Q5  — vol-normalized move size (3 feats)
    "pvd":       pv_divergence,        # Q4  — price-volume sign correlation (3 feats)
    "extension": extension,            # Q13 — distance from last swing (3 feats)
    "vreg":      vol_regime,           # Q23 — volatility regime expansion/contraction (4 feats)
    "sanat":     session_anatomy,      # Q24 — where in the session's natural arc (5 feats)
    "creg":      corr_regime,          # Q25 — cross-asset correlation regime (4 feats)
    "lasym":     liquidity_asymmetry,  # Q28 — above-vs-below liquidity asymmetry (3 feats)
    "rhythm":    tape_rhythm,          # Q29 — tape steady vs spasmodic (3 feats)
    "facc":      flow_acceleration,    # Q32 — second derivative of flow imbalance (3 feats)
    "surprise":  expectation_surprise, # Q33 — actual vs conditional-expected (4 feats)
    # Phase F: BOCPD multi-timescale regime detector — Q19/Q20.
    # Features only, never a gate, per the [[architecture-surfer-principle]].
    "bocpd":     bocpd_regime,         # 5m/60m/4h regime age + change-prob + entropy (9 feats)
    # Kronos: cache-only at training time. Requires running
    # scripts/build_kronos_cache.py once before enabling.
    "kronos":    kronos,
    # Phase K: SimMTM self-supervised embeddings — Q27 (pattern similarity).
    # Cache-only — requires pretrain_simmtm.py + build_simmtm_cache.py.
    "simmtm":    simmtm,               # 96-D SSL embedding per bar
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
