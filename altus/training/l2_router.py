"""L2 Hierarchical Router — predictive-framework Stage 1 + Stage 3.

The existing Layer2MetaLabeler (a small MLP) is one component of this router.
The router wraps it with:

  STAGE 1 (rule-based, BEFORE the MLP):
    Setup arbitration — given the 8 setup-active flags, pick the primary
    setup or abstain on conflict. Output: primary_setup_id (or -1).

  STAGE 2 (the existing L2 MLP):
    Setup-conditional WR prediction. Inputs include setup-type one-hot +
    strength + age + recent_similar_setups_wr.

  STAGE 3 (rule-based, AFTER the MLP):
    Hard vetoes (drift, catalyst proximity, cross-asset divergence).
    Soft modulators (regime contradicts setup → -3pp WR, model confidence
    low → -2pp WR, etc).

  STAGE 4 (existing conformal gate):
    Trade only if predicted_WR > 0.51 AND conformal_lower_bound > 0.51.

This file defines Stages 1 + 3 as pure functions (no model state). Stage 2 is
the existing Layer2MetaLabeler. Stage 4 is the existing ConformalGate.

The router OUTPUT for each candidate bar is:
  RouterDecision {
    trade: bool                 # final go/no-go
    setup_id: str | None        # the dominant setup, or None if no-setup-mode
    direction: int              # +1 long, -1 short
    adjusted_wr: float          # B1 WR after Stage 3 modulators
    sizing_factor: float        # in [0, 1] — scales position size at L3
    abstain_reason: str | None  # human-readable when trade=False
  }
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Priority order — higher WR setups first. Used for arbitration tie-breaking.
SETUP_PRIORITY = [
    "sfs",    # A3 Failed Sweep — 0.62 baseline WR
    "sfa",    # A6 Failed Auction — 0.62
    "sld",    # A8 Multi-Touch Level Defense — 0.60
    "orb",    # A1 ORB — 0.57
    "svwap",  # A2 VWAP — 0.57
    "spb",    # A4 Trend Pullback — 0.57
    "seod",   # A7 EOD Reversion — 0.56
    "scomp",  # A5 Compression Breakout — 0.55
]

# Baseline conditional WR estimates from SETUPS.md. Used as cold-start
# values when the online setup-performance tracker doesn't have enough data.
BASELINE_WR = {
    "sfs":   0.62,
    "sfa":   0.62,
    "sld":   0.60,
    "orb":   0.57,
    "svwap": 0.57,
    "spb":   0.57,
    "seod":  0.56,
    "scomp": 0.55,
}


@dataclass
class SetupCandidate:
    setup_id: str
    active: float
    strength: float
    direction: int


@dataclass
class RouterDecision:
    trade: bool
    setup_id: str | None
    direction: int
    adjusted_wr: float
    sizing_factor: float
    abstain_reason: str | None = None


def arbitrate_setups(
    candidates: list[SetupCandidate],
    min_strength: float = 0.5,
    wr_gap_to_pick_winner: float = 0.05,
) -> tuple[str | None, str | None]:
    """Stage 1 — given active setups, pick the primary one or abstain.

    Rules (from SETUPS.md Section 10):
      - Filter to setups with strength >= min_strength
      - If 0 setups: return (None, "no_setup_active")
      - If 1 setup: return (setup_id, None)
      - If 2+ setups SAME DIRECTION: pick highest-priority setup (highest WR),
        return (setup_id, None) — confidence-stacked
      - If 2+ setups OPPOSITE DIRECTIONS: check WR gap.
          - If gap >= wr_gap_to_pick_winner: pick winner
          - If gap < threshold: abstain (genuine ambiguity)
    """
    qualified = [c for c in candidates if c.active >= 0.5 and c.strength >= min_strength]
    if not qualified:
        return None, "no_setup_active"
    if len(qualified) == 1:
        return qualified[0].setup_id, None

    # Group by direction
    longs = [c for c in qualified if c.direction > 0]
    shorts = [c for c in qualified if c.direction < 0]

    # All same direction → pick the highest-priority (= highest baseline WR)
    if not longs:
        return _pick_by_priority(shorts), None
    if not shorts:
        return _pick_by_priority(longs), None

    # Conflict: opposite directions both firing
    best_long = _pick_by_priority(longs)
    best_short = _pick_by_priority(shorts)
    wr_long = BASELINE_WR.get(best_long, 0.5)
    wr_short = BASELINE_WR.get(best_short, 0.5)
    if abs(wr_long - wr_short) < wr_gap_to_pick_winner:
        return None, "setup_conflict_ambiguous"
    if wr_long > wr_short:
        return best_long, None
    return best_short, None


def _pick_by_priority(cands: list[SetupCandidate]) -> str:
    """Pick the setup with highest baseline WR (= earliest in SETUP_PRIORITY)."""
    cand_ids = {c.setup_id for c in cands}
    for sid in SETUP_PRIORITY:
        if sid in cand_ids:
            return sid
    # Fallback (shouldn't happen)
    return cands[0].setup_id


def apply_modulators(
    base_wr: float,
    *,
    htf_agreement: float = 0.0,           # signed [-1, +1]
    model_confidence: float = 1.0,         # [0, 1] — softmax max prob or 1 - entropy
    recent_similar_wr: float | None = None,
    drift_score: float = 0.0,              # [0, 1] — 0 means no drift
    cross_asset_divergence: bool = False,
) -> tuple[float, list[str]]:
    """Stage 3 — soft modulators that adjust the base predicted WR.

    Returns (adjusted_wr, list_of_modifiers_applied).
    """
    wr = base_wr
    applied = []

    # HTF agreement: positive = setup aligned with macro, negative = contradicts
    if htf_agreement > 0.3:
        wr += 0.02
        applied.append("htf_aligned +2pp")
    elif htf_agreement < -0.3:
        wr -= 0.03
        applied.append("htf_contradicts -3pp")

    # Model confidence: low confidence shaves WR estimate
    if model_confidence < 0.5:
        wr -= 0.02
        applied.append("low_confidence -2pp")

    # Recent similar setups: if last N had bad WR, reduce conviction
    if recent_similar_wr is not None and recent_similar_wr < 0.50:
        wr -= 0.03
        applied.append("recent_failures -3pp")

    # Cross-asset divergence — historically a warning sign
    if cross_asset_divergence:
        wr -= 0.02
        applied.append("xasset_diverge -2pp")

    # Hard veto via drift score
    if drift_score > 0.7:
        wr = min(wr, 0.40)  # force below breakeven → guaranteed abstain
        applied.append("drift_veto")

    return float(np.clip(wr, 0.0, 1.0)), applied


def gate_decision(
    base_wr: float,
    *,
    conformal_lower_bound: float = 0.0,
    setup_id: str | None,
    direction: int,
    sizing_factor: float,
    abstain_reason: str | None,
    breakeven_wr: float = 0.512,
) -> RouterDecision:
    """Stage 4 — final go/no-go decision combining base WR + conformal."""
    if abstain_reason is not None:
        return RouterDecision(
            trade=False, setup_id=setup_id, direction=direction,
            adjusted_wr=base_wr, sizing_factor=0.0,
            abstain_reason=abstain_reason,
        )

    if base_wr < breakeven_wr:
        return RouterDecision(
            trade=False, setup_id=setup_id, direction=direction,
            adjusted_wr=base_wr, sizing_factor=0.0,
            abstain_reason=f"wr_below_breakeven ({base_wr:.3f} < {breakeven_wr})",
        )

    if conformal_lower_bound < breakeven_wr - 0.01:
        return RouterDecision(
            trade=False, setup_id=setup_id, direction=direction,
            adjusted_wr=base_wr, sizing_factor=0.0,
            abstain_reason=f"conformal_uncertain (lb={conformal_lower_bound:.3f})",
        )

    return RouterDecision(
        trade=True, setup_id=setup_id, direction=direction,
        adjusted_wr=base_wr, sizing_factor=sizing_factor,
        abstain_reason=None,
    )


def route_one_bar(
    setup_candidates: list[SetupCandidate],
    base_wr_predictor,                   # callable: (setup_id, ctx) -> float
    *,
    htf_agreement: float = 0.0,
    model_confidence: float = 1.0,
    recent_similar_wr: float | None = None,
    drift_score: float = 0.0,
    cross_asset_divergence: bool = False,
    conformal_lower_bound: float = 0.0,
) -> RouterDecision:
    """End-to-end: Stage 1 → Stage 2 → Stage 3 → Stage 4 for a single bar.

    `base_wr_predictor` is a function (typically the Layer2MetaLabeler wrapped
    to return a setup-conditional WR estimate). Receives setup_id and context;
    returns float in [0, 1]. When called with setup_id=None for no-setup
    fallback mode, uses a conservative WR estimate.
    """
    primary_setup, abstain_reason = arbitrate_setups(setup_candidates)

    if primary_setup is None:
        # Either no setup active OR ambiguous conflict
        if abstain_reason == "no_setup_active":
            # No-setup fallback: only proceed if L1-only confidence is very high.
            # Conservative — keep base_wr low so the gate likely rejects.
            base_wr = base_wr_predictor(None, {})
        else:
            return gate_decision(
                base_wr=0.0, setup_id=None, direction=0,
                sizing_factor=0.0, abstain_reason=abstain_reason,
                conformal_lower_bound=0.0,
            )
    else:
        base_wr = base_wr_predictor(primary_setup, {
            "candidates": setup_candidates,
        })

    # Determine direction for the primary setup
    direction = 0
    if primary_setup is not None:
        for c in setup_candidates:
            if c.setup_id == primary_setup:
                direction = c.direction
                break

    # Stage 3 — apply modulators
    adjusted_wr, modifiers = apply_modulators(
        base_wr,
        htf_agreement=htf_agreement,
        model_confidence=model_confidence,
        recent_similar_wr=recent_similar_wr,
        drift_score=drift_score,
        cross_asset_divergence=cross_asset_divergence,
    )

    # Compute sizing factor from WR + confidence (B1 × G1)
    # sizing_factor = (adjusted_wr - breakeven) / (max_wr - breakeven), clipped
    sizing_factor = float(np.clip((adjusted_wr - 0.51) / (0.70 - 0.51), 0.0, 1.0))
    sizing_factor *= model_confidence

    return gate_decision(
        base_wr=adjusted_wr,
        setup_id=primary_setup,
        direction=direction,
        sizing_factor=sizing_factor,
        abstain_reason=None,
        conformal_lower_bound=conformal_lower_bound,
    )
