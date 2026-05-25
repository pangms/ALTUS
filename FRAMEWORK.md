# ALTUS — The Predictive Question Framework

*The architectural contract. Every component must justify itself against one or more questions on this list. Designed BACKWARD from a high-WR machine.*

**Status:** Canonical as of 2026-05-25 (post-architectural-audit pivot).
**Supersedes:** The original 34-question framework (now in history; preserved at `philosophical_questions_framework.md` memory file).
**Companion:** SETUPS.md (specifies the 8 setups that anchor the predictive layer).

---

## 0. Why This Document Exists

The original 34-question framework was built backward from the experience of an in-the-zone discretionary trader. It was a sound philosophical starting point but, on rigorous classification, only **4 of 34 questions** asked truly predictive questions ("what is likely to happen next?"). The other 30 asked descriptive questions ("where are we now?"). The model was expected to bridge state → direction through learned co-occurrence — a combinatorial task too large for the available training data.

**This redesigned framework inverts that ratio: ~25 of ~45 questions are explicitly predictive.** Each predictive question maps to an explicit component (feature family, model head, or rule engine) whose job is to produce the predictive answer directly, not leave it to the model to infer.

The descriptive primitives are not deleted — they remain as MODULATORS (Group F) whose role is to condition the interpretation of predictive answers. State STILL matters; it just no longer pretends to be a directional signal on its own.

---

## 1. The Framework Structure

```
GROUP A — Setup Detection (8 questions)
  Is a known asymmetric setup active?

GROUP B — Directional Bias Given Context (4 questions)
  Given setup + context, which way has the edge?

GROUP C — Magnitude & Path (5 questions)
  How far will price go, and what path will it take?

GROUP D — Failure Modes (4 questions)
  What invalidates this setup?

GROUP E — Confirmation Triggers (4 questions)
  Has the entry signal actually fired?

────────── Above is the PREDICTIVE LAYER ──────────

GROUP F — Modulators (15 questions, descriptive)
  Macro regime, session, vol, structure, anchors.
  Condition the interpretation of A-E answers.

GROUP G — Self-Awareness (4 questions, meta)
  How confident is the engine? Is it drifting?

GROUP H — Aspirational (3 questions, deferred)
  Positioning, news/calendar, tick microstructure.
```

---

## 2. Group A — Setup Detection (Predictive)

*"Is any of the 8 known asymmetric setups forming right now?"*

| # | Question | Component | Output |
|---|---|---|---|
| **A1** | Is an Open Range Breakout setup forming? | `setup_orb` feature family | `(active, strength, direction, time-eligibility)` |
| **A2** | Is a VWAP rejection/reclaim setup forming? | `setup_vwap` feature family | Same |
| **A3** | Is a failed-sweep / liquidity-trap setup forming? | `setup_failed_sweep` feature family | Same |
| **A4** | Is a trend-pullback continuation setup forming? | `setup_pullback` feature family | Same |
| **A5** | Is a compression-then-breakout setup forming? | `setup_compression` feature family | Same |
| **A6** | Is a failed-auction setup forming? | `setup_failed_auction` feature family | Same |
| **A7** | Is an end-of-day reversion setup forming? | `setup_eod` feature family | Same |
| **A8** | Is a multi-touch level-defense setup forming? | `setup_level_defense` feature family | Same |

**Full specification:** SETUPS.md.

**Why these 8.** Each captures a documented intraday futures edge with non-overlapping mechanism. ORB exploits stop-cluster geometry; VWAP exploits institutional benchmarking; failed sweep exploits liquidity hunting; trend pullback exploits trend persistence + mean reversion; compression breakout exploits vol-regime cycles; failed auction exploits market-profile auction failure; EOD exploits position-squaring; level defense exploits proven institutional bids/offers. No two setups share their primary mechanism.

**Coverage:** A1-A8 collectively replace 6 of the original questions (Q9 acceptance/rejection → A6; Q12 path obstacles → C4; Q14 trapped nearby → A3; Q15 wrong-sided forced flow → B3; Q18 slow squeeze → A5; plus subsume Q2 accumulation/distribution into the path-shape head C3).

---

## 3. Group B — Directional Bias Given Context (Predictive)

*"Given the active setup AND current context, which direction has the historical-conditional edge?"*

| # | Question | Component | Output |
|---|---|---|---|
| **B1** | What's the historical-conditional WR for this setup × regime? | Setup-conditional WR feature (lookup table populated by online learning) | Float in [0, 1] |
| **B2** | Does the HTF regime confirm or contradict the setup direction? | Existing `bocpd_regime` + `mtf_alignment` × setup direction interaction | Signed agreement score |
| **B3** | What's the liquidity-asymmetry gravitational pull? | Existing `liquidity_asymmetry` (Q28) + new explicit gravity score from `prior_day_anchors` + `liquidity_zones` | Signed magnitude |
| **B4** | Is the next move more likely continuation or reversion? | New L1 head: path-shape softmax (continuation/revert/chop/reverse) | 4-class probabilities |

**Why these matter together.** B1 gives the prior. B2 modulates by regime. B3 adds liquidity-geometry conviction. B4 is the model's own forecast of how price will move. All four feed L2's hierarchical router for the final go/no-go decision.

---

## 4. Group C — Magnitude & Path (Predictive)

*"How far will price go, and what shape will the move take?"*

| # | Question | Component | Output |
|---|---|---|---|
| **C1** | What's the expected return at H+15 bars? | New L1 head: short-horizon return regression | Signed magnitude in ATR units |
| **C2** | What's the expected return at H+60 bars? | New L1 head: medium-horizon return regression | Signed magnitude in ATR units |
| **C3** | What's the most likely path shape? | New L1 head: 4-class softmax (continuation, revert, chop, reverse) | Probability vector |
| **C4** | What's the probability of clearing the nearest major level (PDH/PDL/VWAP±σ)? | New L1 head: level-clearance probability regression | Float in [0, 1] |
| **C5** | What's the expected time-to-target for this setup? | Setup-conditional lookup from `setup_X` family | Bars (int) |

**Why these matter together.** C1+C2 quantify magnitude at two horizons (setups have different time profiles). C3 gives the model's own forecast of move character — critical for sizing and stop placement. C4 lets the decision layer reason about path obstacles (replaces the old descriptive Q12). C5 caps hold time per setup type.

**Crucial insight:** Forward-projection was the biggest gap in the descriptive framework. The model was asked "did barrier hit?" with no incentive to learn "what will price do?" The C-group fixes that.

---

## 5. Group D — Failure Modes (Predictive)

*"What would tell us this setup just broke?"*

| # | Question | Component | Output |
|---|---|---|---|
| **D1** | What price would invalidate this setup? | Setup-conditional invalidation (per-setup rule in `setup_X` family) | Price level |
| **D2** | What time would invalidate this setup if no progress? | Setup-conditional time cap (per-setup rule) | Bars remaining |
| **D3** | What cross-asset behavior would invalidate? | New `cross_asset_divergence` family | Boolean + magnitude |
| **D4** | Is the current setup similar to a recently-failed setup? | Online setup-performance tracker | Recent N-trade WR |

**Why these matter together.** Failure modes let L3 size DOWN or ABSTAIN when setups are fragile — the discretionary trader's "this one doesn't feel right." Without D-group, all setups get equal treatment regardless of context fragility, which destroys WR via over-trading marginal setups.

---

## 6. Group E — Confirmation Triggers (Predictive)

*"Has the entry signal actually fired, or are we still pre-trigger?"*

| # | Question | Component | Output |
|---|---|---|---|
| **E1** | Has the setup's primary confirmation fired? | Per-setup confirmation rule (in `setup_X` family) | Boolean |
| **E2** | Has the setup's secondary confirmation fired? | Per-setup secondary rule | Boolean |
| **E3** | Is the entry trigger active this bar? | Per-setup execution trigger | Boolean |
| **E4** | Has any disqualifier appeared in last N bars? | Per-setup recent-invalidator tracker | Boolean |

**Why these matter together.** E-group is what enables CONFIRMATION ENTRIES (a documented 3-5pp WR lift). The descriptive framework had no concept of "wait for the entry trigger" — every signal was treated as actionable immediately, which both increased false-positive rate and degraded execution price.

**Implementation note:** Each A-family must define its own E1-E4 rules; the rules are setup-specific because what confirms an ORB (break-and-retest) differs from what confirms a failed sweep (rejection wick on return).

---

## 7. Group F — Modulators (Descriptive — Retained)

*"What's the current macro/regime/time/structure context?"*

These are the descriptive primitives from the original 34-question framework. They are not predictive on their own, but they CONDITION the interpretation of A-E predictive answers.

| # | Question | Component | Old Q# |
|---|---|---|---|
| **F1** | What macro regime are we in? | `bocpd_regime` (5m/60m/4h) | Q19, Q20 |
| **F2** | What session phase (open/mid/close)? | `session_anatomy` + `session_time` | Q24 |
| **F3** | What's the cross-asset alignment state? | `corr_regime` + `cross_asset` | Q25 |
| **F4** | What's the volatility regime? | `vol_regime` + `volatility` + `bocpd` | Q23 |
| **F5** | What's the structural state (trend/range/transition)? | `trend_structure` + `trend_hurst` + `mtf_alignment` | Q7, Q8 |
| **F6** | Where in market structure (relative to value, levels, anchors)? | `key_levels` + `liquidity_zones` + `prior_day_anchors` + `vwap_anchors` + `volume_profile` | Q6, Q10, Q11 |
| **F7** | Where is price extended from its mean? | `extension` + `vwap_anchors` (band position) | Q13 |
| **F8** | What's the recent flow character? | `pv_divergence` + `absorption` | Q4, Q5 |
| **F9** | Is there an inflection signal? | Inflection auxiliary head | Q26 |
| **F10** | What patterns is current state similar to? | `simmtm` SSL embedding | Q27 |
| **F11** | What is the recent surprise/expectation gap? | `expectation_surprise` | Q33 |
| **F12** | Is a known catalyst window approaching? | *(Deferred to H-tier — needs alt-data)* | Q34 |

**Why F-tier matters.** Predictive answers (A-E) gain or lose conviction based on F-tier context. The same A3 failed-sweep signal means something different in a stable-regime bull session vs a high-vol regime-change-imminent moment. The MODEL learns these interactions; the FRAMEWORK ensures the relevant modulators are surfaced.

---

## 8. Group G — Self-Awareness (Meta — Retained)

*"How confident is the engine, and against what reference?"*

| # | Question | Component | Old Q# |
|---|---|---|---|
| **G1** | How confident is the model in this signal? | Direction-softmax entropy + conformal interval width | Q30 |
| **G2** | What's the historical WR of THIS setup × regime over last 30 occurrences? | Online setup-performance tracker | (NEW) |
| **G3** | Do multi-encoder predictions agree? | Multi-encoder disagreement signal | Q30 partial |
| **G4** | Is the model in a known-bad regime (drift)? | Drift detection on prediction distribution | (NEW) |

**Why G-tier matters.** Self-awareness lets the engine SIZE DOWN or ABSTAIN when its own confidence is low. Without G-tier, the engine trades the same way whether it's certain or guessing — destroying WR via false-positive trades during model-drift periods.

---

## 9. Group H — Aspirational (Deferred — Alt-Data Needed)

These questions are accepted gaps in v1. They will close when alt-data infrastructure ships.

| # | Question | Why deferred | Required data |
|---|---|---|---|
| **H1** | What's the positioning state? (crowded, vulnerable, balanced) | No alt-data | COT, options OI, sentiment |
| **H2** | What scheduled events shape today? | No calendar integration | Economic calendar + news feed |
| **H3** | What's happening at tick/order-book level? | No tick data | CME L2 feed, T&S |

**Coverage target after v1 + H-tier:** ~45 of ~45 questions answered well.

---

## 10. Component → Question Coverage Map

Reverse mapping — every component justifies itself against ≥ 1 question.

| Component | Questions answered |
|---|---|
| `setup_orb` family | A1, partial C1-C5/D1-D4/E1-E4 (ORB-specific) |
| `setup_vwap` family | A2, partial C/D/E |
| `setup_failed_sweep` family | A3, partial C/D/E |
| `setup_pullback` family | A4, partial C/D/E |
| `setup_compression` family | A5, partial C/D/E |
| `setup_failed_auction` family | A6, partial C/D/E |
| `setup_eod` family | A7, partial C/D/E |
| `setup_level_defense` family | A8, partial C/D/E |
| Path-shape softmax head (NEW) | B4, C3 |
| Return regression heads H+15, H+60 (NEW) | C1, C2 |
| Level-clearance probability head (NEW) | C4 |
| Setup-conditional WR tracker (NEW) | B1, D4, G2 |
| Cross-asset divergence (NEW) | D3 |
| Multi-encoder disagreement signal | G1, G3 |
| Drift detector (NEW) | G4 |
| `bocpd_regime` | F1, B2 |
| `session_anatomy` + `session_time` | F2 |
| `corr_regime` + `cross_asset` | F3 |
| `vol_regime` + `volatility` | F4 |
| `trend_structure` + `trend_hurst` + `mtf_alignment` | F5, B2 |
| `key_levels` + `liquidity_zones` + `prior_day_anchors` + `vwap_anchors` + `volume_profile` | F6 |
| `extension` | F7 |
| `pv_divergence` + `absorption` | F8 |
| Inflection auxiliary head | F9 |
| `simmtm` | F10 |
| `expectation_surprise` | F11 |
| `liquidity_asymmetry` | B3 |

**Architectural minimalism check:** Every component has a unique angle. Overlapping components (e.g., multiple market-structure families) are kept because their non-overlapping parts add genuine new angle AND distribute cognitive load. Per [[feedback-architectural-minimalism]] (refined principle: unique-job justification, not aggressive cutting).

**Dropped from the v1 architecture (not in the new map):**
- `tape_rhythm` (was Q29 — low MI, weak predictive bridge per architectural audit Agent C)
- `flow_acceleration` (was Q32 — same)
- `flow.py` VPIN-style features (was Q1 — ceiling-limited without tick data; explicitly deferred to H3)

These survived multiple sweeps without earning OOS lift; dropping per [[feedback-empirical-verification]].

---

## 11. Information Flow Through the Layers

How predictive answers propagate from L1 → L2 → L3 as a unified **PredictionPacket**:

```
L1 emits per bar:
  ┌────────────────────────────────────────────────────────┐
  │ direction_softmax: [P(long), P(short), P(neither)]     │ ← was the only L1 output pre-pivot
  │ mfe/mae regression (4 outputs)                         │
  │ inflection_prob                                        │
  │ fusion_embedding (96-D)                                │
  │ ──────────── NEW predictive heads ──────────────       │
  │ path_shape_softmax: [revert, continue, chop, reverse]  │ ← C3, B4
  │ return_H15, return_H60 (signed ATR)                    │ ← C1, C2
  │ level_clearance_prob                                   │ ← C4
  └────────────────────────────────────────────────────────┘
                          ↓
  L1 features include:
  ┌────────────────────────────────────────────────────────┐
  │ setup_X_active/strength/direction (× 8 setups)         │ ← A1-A8
  │ F-tier modulators (regime, session, vol, structure)    │
  │ G-tier meta (entropy, drift)                           │
  └────────────────────────────────────────────────────────┘

L2 hierarchical router consumes the full packet:
  Stage 1: Setup arbitration → primary setup or abstain
  Stage 2: Setup-conditional WR prediction (B1 estimate)
  Stage 3: Hard-veto + soft-modulator gating
  Stage 4: Conformal gate

L3 receives full packet + L2's go/no-go + which setup won:
  Setup-conditional stop (D1)
  Setup-conditional target (max of C1/C2 and C4 level)
  Setup-conditional hold time (C5)
  Setup-conditional entry trigger (wait for E3)
  Setup-conditional sizing (B1 × G1)
```

**No information loss between layers.** The full predictive packet is preserved end to end. L3 makes execution decisions with the same information L1 generated, not a compressed summary.

---

## 12. The Empirical-Verification Gate

Per [[feedback-empirical-verification]], every component must demonstrate OOS lift before staying.

**Tiered evaluation plan** (each tier must clear its bar):

| Tier | Components added | Pass criterion |
|---|---|---|
| **Tier 0** | Baseline (current architecture pivot) | Reference point |
| **Tier A** | + 8 setup-detection families | Top-decile WR > 0.53 (+2pp over baseline) |
| **Tier A+C** | + path-shape, return regression, level-clearance heads | Top-decile WR > 0.55 |
| **Tier A+C+B** | + hierarchical L2 router with setup-conditional WR | Top-decile WR > 0.56; positive L3.1 PnL |
| **Tier A+C+B+D+E** | + failure-mode features + confirmation-entry execution | Top-decile WR > 0.58; L3.1 Sharpe > 0.5 |

If any tier fails to clear its bar, the response is NOT to add more components — it's to investigate whether the failing tier's components are correctly implemented, or whether the data/granularity is the binding constraint.

---

## 13. What Changed from the Old 34-Q Framework

| Aspect | Old (descriptive) | New (predictive) |
|---|---|---|
| Total questions | 34 | ~45 |
| Truly predictive | 4 (12%) | 25 (55%) |
| Pure descriptive | 13 (38%) | 0 in predictive layer; 15 in F-modulator |
| Setup-explicit | None | 8 setups (A1-A8) |
| Forecasting | Indirect (model has to infer) | Explicit (4 new L1 heads) |
| Failure modes | Not surfaced | 4 explicit questions (D1-D4) |
| Confirmation entries | Not modeled | 4 explicit questions (E1-E4) |
| Cross-question composition | Implicit (model learns interactions) | Explicit (L2 hierarchical router) |

The framework is more complex but FAR more actionable. Every question has a clear answer-producing component. The model no longer carries the burden of inventing the predictive bridge — the bridge is in the architecture.

---

## 14. What's Locked

Per the [[philosophical-questions-framework]] discipline:

**LOCKED until empirical results say otherwise:**
- The 8 setups (definitions in SETUPS.md)
- The A/B/C/D/E predictive question groups
- F-modulators and G-meta groupings
- The L1 → L2 → L3 information flow architecture

**Open for tuning without re-approval:**
- Detection thresholds within each setup
- Setup-conditional WR baselines (recalibrated empirically)
- Confirmation strictness levels
- Strength scoring weights
- L2 router stage parameters

**Re-approval required for:**
- Adding a 9th setup
- Removing one of the current 8
- Adding a new predictive question group beyond A-E

---

## 15. Pointers

- Setup specifications: SETUPS.md
- Architecture document: ARCHITECTURE.md (Section 3 references this document)
- Empirical-verification rule: memory/feedback_empirical_verification.md
- Architectural minimalism: memory/feedback_architectural_minimalism.md
- Surfer principle: memory/architecture_surfer_principle.md

*This framework, with SETUPS.md, is the new architectural contract. Build to it; A/B everything not contractual; revise the framework when the data demands it, not before.*
