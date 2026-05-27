# ALTUS — Architecture & Strategy

*A discretionary-trader-grade ML system for MNQ futures, built backward from the questions a serious trader actually asks.*

**Status as of 2026-05-27**: L1 + L2 + L3 functionally complete after the predictive-framework pivot. Comprehensive baseline sweep launching on RunPod (4090). Canonical predictive framework: [FRAMEWORK.md](FRAMEWORK.md). Setup library specs: [SETUPS.md](SETUPS.md).

---

## 1. What ALTUS Is

ALTUS is a Python ML trading system for MNQ (Micro E-mini Nasdaq-100) futures, deployed on TopStep prop accounts. It is a **layered architecture** with five intended layers:

| Layer | Job | Status |
|---|---|---|
| **L1 — Perception** | Multi-encoder neural model that reads the market and produces calibrated probabilities + predictive heads | ✅ Built |
| **L2 — Meta-Labeling** | 4-stage router (arbitrate setups → base WR → modulators → conformal gate) outputting trade decisions | ✅ Built |
| **L3 — Risk Engine** | Deterministic rules: grade sizing, no-overlap, EoD flatten, pre-release embargo, daily loss safety, vol circuit breaker | ✅ Built |
| **L4 — Execution** | Broker integration, realistic fill simulation, latency budget enforcement | ⏸ Planned |
| **L5 — Monitoring** | Drift detection, performance alerts, retraining triggers | ⏸ Planned |

**Aspirational targets** (whole-bot, not per-component gates):
- ≥70% win rate on confirmed signals
- **3-8 actually-tradeable signals per day** after router arbitration + conformal gate (matches the setup library spec in SETUPS.md §1 — patience is the alpha, not frequency). Earlier drafts targeted "≥15 trades/day" but that requires firing low-conviction signals which defeats the purpose of the high-WR library.
- Volatility-scaled barriers (TP = SL = 1.5 × ATR_60) replace the legacy fixed 30/30 design — labels and execution now adapt to regime

**Operating constraints**: 1-min polling cadence, TopStep account rules (daily loss limit, trailing drawdown, end-of-day flat), MNQ as initial single instrument. Trading style is "quick trader, not scalper" — the surfer model below.

---

## 2. Design Philosophy

Four principles drive every architectural decision. Each was learned the hard way from prior failed iterations or from this build's empirical results.

### 2.1 Predictive-first, question-driven architecture (work backward)

We don't ask "what models are available?" We ask **"what predictive questions does an in-the-zone trader ask, and which components are needed to answer them?"** The ~45 questions in [FRAMEWORK.md](FRAMEWORK.md) are the design contract. Every component must justify its existence against ≥1 question.

**The 2026-05-25 pivot reframed this rule.** The original 34-question framework was mostly descriptive ("where are we now?"); rigorous classification showed only 12% were truly predictive. The new framework inverts the ratio: predictive questions are the primary axis, descriptive primitives are demoted to F-tier modulators that condition the interpretation of predictive answers.

### 2.2 The Surfer Principle

The engine should behave like a surfer reading individual waves — not a trader who decides "I'm only long today because the higher timeframe is bullish." Intraday markets always have counter-trend opportunities (pullbacks, MTF reversals, intraday squeezes), and the engine must remain capable of riding them.

**Non-negotiable consequences:**
1. **Multi-timescale regime, never single-timescale.** Always 3 scales (5m / 60m / 4h). The *intersection* state is the engine's actual context.
2. **Regime signals are FEATURES, never gates.** They feed L1 and L2 as inputs; never as hard directional vetoes. The model learns the (setup × regime) interaction itself.
3. **Surfer test (acceptance criterion).** Post-deployment cascade evaluation must show counter-HTF-regime trades being taken *and* having positive expectancy. If the engine refuses counter-regime trades, the design failed.

The only place deterministic regime gating is allowed is L3's vol circuit breaker — and even there, only as a position-sizing modulator, never as a directional veto.

### 2.3 Empirical verification gate

Every feature, every component, every architectural addition must demonstrate OOS lift before staying in the system. A/B test every addition; drop what doesn't clear the bar. No exceptions for "it sounds good."

This is what the comprehensive ablation sweep is for. Components that don't earn their place via OOS metrics get dropped, regardless of architectural elegance. The 2026-05-26 build added **5 predictive-vs-pacing diagnostics** (§5.7) specifically to detect when a component appears to help PnL but is actually just sophisticated volatility detection.

### 2.4 Architectural minimalism — refined

The principle is **"every component must have a unique angle,"** not "default to drop when uncertain." Overlapping components are fine if the non-overlapping parts add genuine value AND distribute cognitive load. The discipline is **unique-job justification**, not aggressive cutting.

When components overlap and agree, that's confidence amplification. When they disagree, that's an uncertainty signal — also useful.

---

## 3. The Predictive Question Framework

**Canonical framework: see [FRAMEWORK.md](FRAMEWORK.md) and [SETUPS.md](SETUPS.md).** This section is a brief summary.

The system is designed BACKWARD from a "high-WR machine" — every component justifies itself against a predictive question. After the 2026-05-25 audit, the framework was restructured from the original 34 descriptive questions to ~45 questions in 8 groups (A-H), of which ~25 are explicitly predictive.

### The 8 groups

| Group | Purpose | Status |
|---|---|---|
| **A** | Setup detection (8 questions) | ✅ Built — 8 setup families implemented |
| **B** | Directional bias given context (4) | ✅ Bootstrapped in L2 router base_wr_predictor |
| **C** | Magnitude & path (5) | ✅ Built — return_H15, return_H60, path_shape, clears_level heads |
| **D** | Failure modes (4) | 🟡 Partial — sweep_detection + invalidation distance, full set deferred to post-sweep |
| **E** | Confirmation triggers (4) | ⏸ Post-sweep — Tier-3 additions |
| **F** | Modulators (15, the legacy descriptive layer) | ✅ Built — feed L1 as features, condition L2's setup-conditional WR |
| **G** | Self-awareness (4) | 🟡 Partial — derived_entropy + conformal gate present; tracker WR + drift score deferred |
| **H** | Aspirational (3) | ⏸ Deferred — needs alt-data (COT, news, tick microstructure) |

**Coverage target with v1 + H-tier alt-data:** ~45 of ~45 questions well-answered.

### Setup library (Group A — anchors the whole framework)

Fully specified in [SETUPS.md](SETUPS.md):

| ID | Setup | Family | Est WR | Per-setup execution |
|---|---|---|---|---|
| A1 | Open Range Breakout | `setup_orb` | 0.55-0.60 | 1.3 ATR / 1.0 ATR / 30 bars |
| A2 | VWAP Rejection/Reclaim | `setup_vwap` | 0.55-0.60 | 1.5 ATR / 1.0 ATR / 45 bars |
| A3 | Failed Sweep / Liquidity Trap | `setup_failed_sweep` | 0.60-0.65 | 1.5 ATR / 0.8 ATR / 30 bars |
| A4 | Trend Pullback | `setup_pullback` | 0.55-0.60 | 2.0 ATR / 1.0 ATR / 60 bars |
| A5 | Compression Breakout | `setup_compression` | 0.52-0.57 | 2.0 ATR / 1.0 ATR / 60 bars |
| A6 | Failed Auction | `setup_failed_auction` | 0.60-0.65 | 1.5 ATR / 1.0 ATR / 45 bars |
| A7 | End-of-Day Reversion | `setup_eod` | 0.55-0.58 | 1.0 ATR / 0.8 ATR / 20 bars |
| A8 | Multi-Touch Level Defense | `setup_level_defense` | 0.58-0.62 | 1.5 ATR / 1.0 ATR / 60 bars |

WR estimates are upper-bound expectations; the sweep refines them via the `SetupPerformanceTracker` (§7.4).

### L2 confidence modulators (6 families, Tier-2)

Built 2026-05-26. These are **bidirectional boosters** — they ADD evidence to setup signals, never veto:

| Family | Job |
|---|---|
| `path_clearance` | Clearance + obstacle strength per side (Q12) |
| `stop_pool` | Stop-pool size + trigger proximity (Q11/Q14 mash-up) |
| `setup_confluence` | Multi-setup direction counts (cross-setup agreement) |
| `cross_asset_setup_confirm` | NQ/ES alignment (Q33 cross-asset variant) |
| `vol_sweet_spot` | Per-setup vol-regime fitness (Q23 conditioned on setup) |
| `time_of_day_fitness` | Per-setup time-window fitness (Q24 conditioned on setup) |

### Honest gaps (deferred to H-tier alt-data phase)

- **H1 — Positioning**: Requires COT + options OI + sentiment
- **H2 — News/calendar**: Requires economic calendar + news feed
- **H3 — Tick microstructure**: Requires CME L2 + T&S

These remain accepted v1 gaps.

---

## 4. Question → Component Mapping (legacy 34-question — preserved for archaeology)

This mapping pre-dates the 2026-05-25 predictive pivot. It is kept as a useful descriptive-primitive index but is **superseded** as the design contract by [FRAMEWORK.md](FRAMEWORK.md). New components should be justified against the FRAMEWORK.md ~45 questions, not the table below.

The original framework's categories (Order Flow, Auction State, Liquidity & Path, Forced Flow, Regime, Temporal, External Context) are still useful for thinking about which DESCRIPTIVE primitives matter — they're preserved as F-tier modulators in the new framework. They just no longer drive directional decisions on their own.

| # | Question (legacy) | Primary mechanism | Secondary |
|---|---|---|---|
| Q1 | Large-player sponsorship | VPIN feature | Kronos |
| Q2 | Accumulation vs distribution | Kronos transfer | TCN + volume profile |
| Q3 | Transactional vs directional flow | TCN + VPIN | Mamba |
| Q4 | Volume confirming/diverging | `pv_divergence` | TCN |
| Q5 | Absorption vs conviction | `absorption` | TCN, Mamba |
| Q6 | Inside/edge/discovery of value | Volume profile + `vwap_anchors` | TCN |
| Q7 | Imbalanced vs balanced | BOCPD + Hurst | TCN |
| Q8 | Timeframe in control | `mtf_alignment` | TCN |
| Q9 | Acceptance/rejection | Kronos + volume profile | TCN |
| Q11 | Resting liquidity | `round_levels` + `prior_day_anchors` + `key_levels` | `liquidity_zones` |
| Q12 | Path obstacles | `path_clearance` (L2 modulator) | levels family |
| Q13 | Already-extended | `extension` | TCN |
| Q14 | Trapped participants | **Mamba state** | `sweep_detection`, `stop_pool` |
| Q15 | Wrong-sided forced flow | **Mamba state** | `sweep_detection` |
| Q18 | Slow squeeze | **Mamba state** + `setup_compression` | trend + Hurst |
| Q19 | Stale regime classification | **BOCPD multi-TF** + `trend_structure` | SimMTM |
| Q20 | Regime confirmed today | **BOCPD multi-TF** | Conformal gate |
| Q23 | Vol regime | `vol_regime` + `vol_sweet_spot` | BOCPD |
| Q24 | Session anatomy | `session_anatomy` + `time_of_day_fitness` | — |
| Q25 | Correlation regime | `corr_regime` + `cross_asset_setup_confirm` | — |
| Q26 | Inflection vs continuation | **Inflection head** + `path_shape` head | TCN, Mamba |
| Q27 | Pattern similarity to history | **SimMTM embedding** | — |
| Q28 | Liquidity asymmetry | `liquidity_asymmetry` | — |
| Q29 | Tape rhythm | `tape_rhythm` | VPIN |
| Q30 | Model self-confidence | **Multi-encoder disagreement** + derived_entropy | Conformal gate |
| Q31 | Move lifecycle phase | **Mamba state** | BOCPD age |
| Q32 | Flow acceleration | `flow_acceleration` | — |
| Q33 | Expected vs actual | `expectation_surprise` | — |

Q17 / Q21 / Q22 / Q34 remain alt-data gaps. The full predictive-question mapping is in [FRAMEWORK.md](FRAMEWORK.md).

---

## 5. Layer 1 — Perception Architecture

L1 is the engine's reader of the market. It produces calibrated per-bar probabilities + predictive auxiliary outputs that downstream layers consume.

### 5.1 The multi-encoder design

Three perception encoders run in parallel, each with a fundamentally different inductive bias:

**ModernTCN** (Liu et al. 2024) — *convolutional pattern recognizer*
- Hierarchical local patterns, translation-invariant, fully parallel
- Job: pattern-shape questions (Q3-Q10, Q13, Q16)
- ✅ Working, primary encoder

**Mamba-2** (Gu & Dao 2023) — *stateful selective state-space*
- Dynamic memory updates, carries state forward indefinitely
- Job: state-tracking questions TCN structurally can't answer (Q14, Q15, Q18, Q31)
- ✅ Code complete with auto-detect CUDA fast path. Pure-PyTorch fallback works on any device; `mamba-ssm` Triton kernel activates on CUDA when installed (currently disabled in baseline sweep for stability — its own dedicated session later).

**Kronos** (open-source foundation model) — *transfer learning*
- Pre-trained on more market data than we have
- Job: Q2, Q9, Q11 at population scale
- 🟡 Family + cache loader implemented; cache build needs debug. Cache-only architecture means inference is instant once cache is built.

**SimMTM** (Liu et al. 2023) — *self-supervised similarity*
- Learns continuous embeddings via masked-bar reconstruction
- Job: Q27 (pattern similarity to history) — unique angle
- ✅ Encoder + masking + pretraining + cache builder + feature family all built. Cache produced; feeds L1 as a 96-D embedding per bar.

### 5.2 Rolling-3m primary candle + multi-TF stack

L1's primary candle is **rolling 3-minute** (polled every 1m), with multi-TF context at (1, 3, 5, 15, 30, 60) min. This gives the model noise-reduced primary perception while preserving full 1m granularity for setup detection.

### 5.3 BOCPD — parallel regime tracker

Bayesian Online Change-Point Detection runs **parallel to** the encoders.

- Applied at three timescales (5m / 60m / 4h equivalents)
- 9 features: regime age, change-point probability, run-length entropy × 3 scales
- Fed into L1 as features AND into L2 as gating context
- **Strictly never used as a directional gate** (surfer principle)

### 5.4 Feature library (~40 families)

L1 ingests ~40 feature families (~250 numeric features total), grouped by phase:

| Phase | Families | Count | Purpose |
|---|---|---|---|
| A | `vol, trend, anomaly, session, exhaust, cross` | 6 | Baseline (Phase A) |
| B | `levels, liquidity, sweep, profile` | 4 | Market structure |
| C | `flow` | 1 | Order flow + cross-asset lead-lag |
| E (Phase E, pruned) | `round, mtf, pvd, vreg, sanat, creg, surprise` | 7 | Trader-frame additions, post-MI-audit prune |
| F | `bocpd` | 1 | Multi-TF regime |
| Tier-2 anchors | `pda, vwap, tstruct` | 3 | Discretionary-trader horizontal references |
| Setup library | `sfs, sfa, sld, orb, svwap, spb, scomp, seod` | 8 | Predictive A-tier setups (each emits 5 cols: active/strength/direction + 2 setup-specific) |
| L2 modulators | `pclear, spool, scnf, cac, vss, tof` | 6 | Bidirectional boosters (Tier-2; F-tier in FRAMEWORK.md) |
| Cache | `kronos, simmtm` | 2 | Foundation-model + SSL embeddings |

All families pass an explicit **causal-invariance test**: features at row T computed from data[:N] vs data[:N+offset] must be identical for rows < N. This catches the entire class of lookahead bugs.

**Two-pass family compute**: `setup_confluence` is an `IS_AGGREGATOR` family — it reads pass-1 setup outputs (`sfs_active`, etc.) and emits aggregate consensus features. The final causal shift covers both passes.

### 5.5 Output heads (post-2026-05-26 predictive pivot)

Layer 1 now produces **3 directional / 4 regression / 3 predictive auxiliary** outputs per bar:

**Direction (3-class softmax)** — replaced the legacy dual-BCE design that collapsed to a volatility detector:
- `direction_logits` over `{long_wins, short_wins, neither}` — mutually exclusive forward outcomes
- L2 derives `long_tp_prob`, `short_tp_prob` from these slices

**Excursion regression** (Huber loss, vol-scaled units):
- `mfe_long`, `mae_long`, `mfe_short`, `mae_short`

**Predictive heads** (NEW — 2026-05-26 pivot):
- `path_shape_logits` (3-class softmax: continuation / revert / chop) — Q26
- `return_H15` regression — expected log-return over next 15 minutes
- `return_H60` regression — expected log-return over next 60 minutes
- `clears_level_logit` (sigmoid) — P(price clears ≥1 ATR forward) — Q12 / C4

**Auxiliary**:
- `inflection_logit` (sigmoid) — P(price resolves AGAINST recent direction), Q26 short-horizon
- 192-D fusion embedding — exposed to L2

**Labels (vol-scaled triple-barrier)**:
- TP and SL distances are computed per-bar as `1.5 × ATR_60`
- Replaces the legacy fixed 30/30 design
- Forces the model out of "find any 30-point move regardless of volatility regime" into "find moves proportional to current vol"
- Auxiliary truths attached: `return_H15`, `return_H60`, `path_shape_class`, `clears_1atr`, `inflection_label`

### 5.6 Training discipline

- **Purged walk-forward CV** with 60-bar embargo
- **3-fold cross-validation** on 3 years of MNQ data
- **OOS lockbox**: last 4 months held out, never touched in training
- **Isotonic calibration** post-training on held-out calibration slice
- **Conformal prediction wrapper** for distribution-free abstention

### 5.7 Predictive-vs-pacing diagnostics (2026-05-26)

Five diagnostics computed in `evaluate_predictions` and surfaced in the FINAL SUMMARY of every sweep. They separate "real forward prediction" from "calibrated volatility detection":

| Diagnostic | Predictive looks like | Pacing looks like |
|---|---|---|
| `corr(P_long, P_short)` | ≈ -1 (mutually exclusive) | ≈ +1 (co-move on vol) |
| Spearman IC on `return_H15` | > 0.03 OOS | ≈ 0 |
| Spearman IC on `return_H60` | > 0.03 OOS | ≈ 0 |
| `path_shape_accuracy` | > 0.38 (vs 0.33 chance) | ≈ 0.33 |
| `clears_level_AUC` | > 0.53 | ≈ 0.50 |

`MetricsBundle.predictive_diag_verdict()` aggregates these into a coarse PREDICTIVE / WEAK / PACING-LIKE label, printed per fold + OOS + FINAL SUMMARY. If verdicts come back PACING-LIKE across variants, the architecture needs to be re-thought, not iterated on.

---

## 6. Layer 2 — Hierarchical Router Architecture

L2's job: given L1's predictions + setup detections, decide whether to actually trade and at what size.

### 6.1 Inputs to L2 (118 features)

Per candidate signal, L2 sees:

- **6 L1 raw outputs**: `long_tp_prob, short_tp_prob, mfe_long, mae_long, mfe_short, mae_short`
- **5 derived signals**: `direction, strength, margin, entropy` (3-class), `expected_r`
- **4 time features**: hour/dow sin/cos
- **Vol regime block**: 8 features (realized vol at 5m/30m/4h/1d, vol-of-vol, Hurst, percentile, regime score)
- **Trend block**: 8 features (4h/1d/1w slope + Hurst, alignment, strength)
- **Anomaly**: 1 feature (Mahalanobis)
- **BOCPD block**: 9 features (age + cp_prob + entropy × 3 timescales)
- **L2 modulators block**: 31 features across 6 families (path_clearance, stop_pool, setup_confluence, cross_asset_setup_confirm, vol_sweet_spot, time_of_day_fitness)
- **Per-setup detector outputs**: 40 features (8 setups × 5 cols each = active/strength/direction/state_a/state_b) — **2026-05-26 fix**: without these L2 could not see *which setup is firing*, making setup-conditional WR mathematically impossible.

Total: 118 features.

### 6.2 Model — small MLP meta-labeler

~10k params. Binary classification: "good trade" vs "skip." Tabular; no sequence, no attention.

### 6.3 The 4-stage router (`l2_router.route_one_bar`)

Built 2026-05-26 to replace the previous single-stage gate. Each bar with active setup features flows through:

**Stage 1 — `arbitrate_setups`**: pick a primary winner among active setups
- 0 qualified → abstain (`no_setup_active`)
- 1 qualified → take it
- 2+ same direction → highest-priority (highest baseline WR)
- 2+ opposite directions → if WR gap < 5pp, abstain (`setup_conflict_ambiguous`)

**Stage 2 — base WR predictor**: closure that returns L2's calibrated prob for the setup's direction at this bar, or `BASELINE_WR[setup_id]` prior if the bar isn't an L2 candidate. Bootstrapped per-cell WR is fed in via `SetupPerformanceTracker` (§7.4).

**Stage 3 — `apply_modulators`**: bidirectional WR adjusters (NEVER vetoes)
- HTF agreement (from `trend_alignment`): +2pp aligned / -3pp contradicts
- Model confidence (3-class softmax entropy proxy): -2pp if low
- Recent similar WR: -3pp if last-N failed
- Cross-asset divergence (from `cac_divergence_active`): -2pp
- Drift score > 0.7: hard veto floor at WR=0.40

**Stage 4 — `gate_decision`**: final go/no-go
- Abstain reason from stage 1 → trade=False
- Adjusted WR < 0.512 breakeven → abstain (`wr_below_breakeven`)
- Conformal lower bound < breakeven - 0.01 → abstain (`conformal_uncertain`)
- Else trade=True, sizing_factor scales with (adjusted_wr - 0.51) / (0.70 - 0.51) × model_confidence

Abstain-reason counts are surfaced in the sweep output for diagnostics.

### 6.4 Setup-conditional WR diagnostic

Printed before the router decisions every sweep. For each of the 8 setups:

```
sfs:  nL=  214 wr=0.612 lift= +9.2pp  |  nS=  198 wr=0.594 lift= +8.4pp  LONG✓ SHORT✓
orb:  nL=  445 wr=0.521 lift= -0.1pp  |  nS=  431 wr=0.518 lift= -0.4pp
```

`lift > 3pp` with n ≥ 30 → ✓ (predictive). `lift < -1pp` → ✗ (anti-signal). If most setups show < 1pp lift, the setup library is decoration — it identifies configurations the market doesn't actually resolve asymmetrically.

### 6.5 Cascade evaluation

Final trade decision is the cascade: L1 ranks bars → setup detectors fire → L2 router runs the 4 stages → trade taken at router sizing. Measured by:
- Top-K% selectivity (rank L2 outputs)
- Threshold mode (absolute calibrated probability)
- Conformal-gated mode (lower-bound of prediction interval)
- **Setup-aware mode** (per-setup TP/SL/hold via `compute_setup_aware_barriers`)

---

## 7. Layer 3 — Risk Engine Architecture

L3 sits between L2's trade decisions and the broker. It enforces operational discipline that the model is not allowed to override.

### 7.1 Grade-based sizing

L2's `sizing_factor` is converted to a discrete contract grade:

| Grade | Trigger | MNQ contracts |
|---|---|---|
| B | sizing_factor ≥ 0.20 | 1 |
| A | sizing_factor ≥ 0.50 | 2 |
| A+ | sizing_factor ≥ 0.75 | 3 |
| A++ | sizing_factor ≥ 0.90 + special | 3 (pyramidable) |

**A++ pyramiding rule**: A++ signals can stack (up to MAX_STACK_DEPTH = 2-3) IFF ≥5 min elapsed since previous entry AND signal is in top 0.5%. All other grades enforce **no-overlap** (one position at a time).

### 7.2 Setup-aware execution

`compute_setup_aware_barriers` reads the primary setup_id from L2 and applies per-setup TP/SL/hold parameters (see §3 setup library table). E.g., `failed_sweep` uses 1.5×ATR target / 0.8×ATR stop / 30-bar hold; `pullback` uses 2.0×ATR / 1.0×ATR / 60 bars. Falls back to default vol-scaled barriers when no setup is primary.

### 7.3 Hard rules (operational safety, never veto signal quality)

These live at L3, not L2 — they protect the account, not the prediction:

| Rule | What it does |
|---|---|
| **Pre-release embargo** | No new entries within ±30 min of 08:30/10:00/14:00 ET (CPI/NFP/FOMC windows). Existing positions are managed normally. |
| **Daily loss safety net** | If realized daily P&L ≤ -80% of TopStep daily loss limit, no new entries today. Existing positions held. |
| **Vol circuit breaker** | If 5-min realized vol > 99th-percentile of trailing 30 days, no new entries until vol drops. |
| **EoD flatten** | All positions closed by 15:55 ET regardless of P&L. |
| **Consecutive-loss cooldown** | After N losses in a row (configurable), 30-min trading pause. |

### 7.4 `SetupPerformanceTracker` (online (setup × regime) WR)

Bootstrapped from training data at L2 train time; persists to `artifacts/setup_performance_tracker.json`. For each (setup_id × regime_bucket) cell, accumulates:
- `n_trades`, `n_wins`, `wr`, `r_mean`
- Bootstrap-confidence interval via Beta posterior

The router's Stage 2 base WR predictor reads from this table when available, giving B1 ("conditional WR given setup × regime") true predictive content rather than a fixed prior.

**Note on current state**: as of 2026-05-27, the tracker writes are wired in `train_layer2.py` but no feature family READS the JSON yet — the table feeds the router only via in-memory `BASELINE_WR`. Wiring a `tracker_lookup` feature family is a post-sweep task.

### 7.5 Production sim (`production_sim.simulate_l3`)

The honest backtest. Replays L1+L2 cascade through L3 rules with realistic costs (slippage + commission, both in `SLIPPAGE_RT_POINTS` and `COMMISSION_RT_USD`). Reports:
- `total_pnl_usd`, `n_trades`, `win_rate`, `sharpe`, `max_drawdown_pct`
- `n_trades_by_grade` (B/A/A+/A++)
- `n_pyramid_entries`, `max_concurrent_contracts`
- `worst_day_pnl_usd`, `n_days_would_trip_trailing_dd`, `max_consecutive_losses`
- Per-rule abstain counters

This is the headline number — what actually lands in PnL under full operational discipline.

---

## 8. Architectural Decisions Made (and Why)

This section documents non-obvious choices for future-us reference.

### 8.1 Dropped: xLSTM
Originally planned as an alternative recurrent peer to Mamba. Dropped during architectural review — overlaps too heavily with Mamba's stateful-recurrence inductive bias.

### 8.2 Kept BOTH BOCPD and SimMTM (despite both addressing "regime")
Complementary, not competing. BOCPD = discrete regime posterior + change-point; SimMTM = continuous similarity-to-history embedding. Both earn the unique-angle test.

### 8.3 Multi-encoder vs single-encoder
TCN (convolutional) + Mamba (recurrent state) + Kronos (transfer) + SimMTM (SSL) each have a unique inductive bias. When they agree → confidence amplification. When they disagree → uncertainty signal.

### 8.4 Vol-scaled barriers replaced fixed 30/30 TP/SL (2026-05-25 pivot)
Previous sweep results showed asymmetric R:R didn't rescue the system — the model found small-magnitude winners that barely scraped +30pt regardless of regime. Root cause: the label was a fixed point distance, indifferent to current vol. The fix: `TP = SL = k × ATR_60` with k=1.5, computed per-bar. Now the model has to find moves *proportional* to current vol, which forces it out of "always look for any 30-point move" and into "find regime-appropriate moves."

### 8.5 3-class direction softmax replaced dual-BCE (2026-05-25 pivot)
Dual independent BCE on `long_tp` and `short_tp` allowed the model to collapse to a volatility detector: predict "both sides have ~50% chance" whenever vol was high. The 3-class softmax over `{long_wins, short_wins, neither}` forces mutually-exclusive forward outcomes, breaking the pacing-mode failure.

### 8.6 L2 router replaced single-gate L2 (2026-05-26 fix)
Previously L2 was a single MLP + conformal gate, with arbitrate_setups called separately and unused. The 4-stage router unifies arbitration, base WR prediction, modulator application, and conformal gating — and crucially makes the 6 modulator families and the conformal coverage actually load-bearing in the trade decision. Before the fix, modulators were trained but their evidence wasn't entering the trade gate.

### 8.7 4090 over 5090 (for now)
5090 (Blackwell, sm_120) is bleeding-edge — most ML packages don't have prebuilt wheels yet. 4090 (Ada Lovelace, sm_89) is mature — all packages install cleanly. Revisit when ecosystem catches up.

---

## 9. Current Build Status

### 9.1 What's done and verified

- ✅ All ~40 feature families pass causal-invariance test
- ✅ ModernTCN encoder + Mamba peer + SimMTM SSL all built
- ✅ BOCPD regime SSM at 3 timescales (causal verified)
- ✅ 8 setup detection families (sfs/sfa/sld/orb/svwap/spb/scomp/seod) with per-setup execution params
- ✅ 6 L2 confidence modulators (path_clearance, stop_pool, setup_confluence, cross_asset_setup_confirm, vol_sweet_spot, time_of_day_fitness)
- ✅ 3-class direction softmax + 4 new predictive heads (path_shape, return_H15, return_H60, clears_level)
- ✅ Vol-scaled triple-barrier labels (k=1.5 × ATR_60)
- ✅ L2 4-stage router (arbitrate → base_wr → modulators → conformal gate)
- ✅ 40 setup detector columns wired into L2 input (118 total features)
- ✅ L3 grade sizing + no-overlap + A++ pyramiding + EoD flatten + consecutive-loss cooldown
- ✅ L3 hard rules: pre-release embargo, daily loss safety net, vol circuit breaker
- ✅ SetupPerformanceTracker bootstrap from training data
- ✅ 5 predictive-vs-pacing diagnostics + verdict aggregator
- ✅ Setup-conditional WR diagnostic in L2 cascade eval
- ✅ Forward-leak audit passed across all 41 feature families, labels, splits, SimMTM cache, conformal calibration

### 9.2 What's running / not yet validated empirically

- 🟢 RunPod 4090 sweep launching (2026-05-27): comprehensive 6-variant ablation
- ⏸ Mamba CUDA kernel — its own dedicated session later
- ⏸ Kronos cache build — needs debug of failed build script
- 🟡 SetupPerformanceTracker readback as a feature — post-sweep task

### 9.3 The current sweep

**6-variant ablation** answering "does each big component earn its place on top of the new baseline?":

```
01. baseline               TCN + Phase A only (Tier-0 reference)
02. descriptive_full       +Phase E + BOCPD + anchors (descriptive layer complete)
03. setups_only            baseline + 8 setups (predictive A-tier alone)
04. predictive_full        anchors + setups + descriptive (Tier A all stages)
05. with_modulators        04 + L2 confidence modulators (full stack ex-SimMTM)
06. full_with_simmtm       05 + SimMTM — THE COMPREHENSIVE TEST
```

3-yr × 3-fold purged walk-forward, 4mo OOS lockbox. Each variant ends with a predictive-vs-pacing verdict before its PnL numbers are trusted.

---

## 10. Lessons Learned (so far)

### 10.1 Empirical verification works
The causal-invariance test caught a real VPIN lookahead bug, a `.bfill()` leak in `bocpd_regime.py`, and a SimMTM-pretrain OOS contamination — three real bugs that would have silently corrupted runs.

### 10.2 Build before testing partial systems
The "test each component in isolation" approach repeatedly produced discouraging results that didn't reflect what the full architecture can do. The correct discipline: build the whole thing, test once at the end.

### 10.3 Predictive targets ≠ predictive features ≠ predictive PnL
A model can have predictive targets, predictive features, and still produce non-predictive PnL because the model failed to combine the features into a signal that beats marginal returns. The 5 diagnostics in §5.7 separate these failure modes.

### 10.4 Disconnections kill silently
The 2026-05-26 audit found 3 critical disconnections in what looked like a finished architecture: predictive heads trained but never extracted, setup features missing from L2 input, full router stages 2-4 dead code. None of these were caught by syntax checks or smoke tests — only by reading the data flow end-to-end. Lesson: every output of every component must be traced to a downstream consumer before claiming "done."

### 10.5 Infrastructure friction is the silent productivity killer
RunPod migrations, GPU availability, ML package version conflicts, terminal disconnects — these consume more time than the actual ML work. Use tmux + nohup for resilience; fresh-clone when state is corrupted; mature GPU choice over bleeding-edge.

### 10.6 Asymmetric R:R isn't the answer; vol-scaling is
A 50% WR system isn't rescued by tightening or widening fixed-distance TP — the model's signal magnitude is the binding constraint. Vol-scaled barriers force the model out of "always find a 30-point move" and into "find a move appropriate to current vol."

### 10.7 L2 needs setup identity to do setup-conditional anything
A meta-labeler claiming "setup-conditional WR" must see which setup is firing. Adding the 40 setup detector columns to L2 input was a structural unlock, not an incremental tweak.

---

## 11. Roadmap Beyond L1+L2+L3

### Phase: Baseline sweep verdict (this week)
1. Sweep completes → 5 predictive diagnostics + setup-conditional WR table per variant
2. If verdicts pass → lock the load-bearing component set, move to L4
3. If verdicts fail → architectural re-think, not iteration

### Layer 4 — Execution (5-7 days)
- TopStep broker integration (Project X API or Rithmic)
- Realistic fill simulator for backtest validation
- Latency budget enforcement (L1 → L2 → L3 → order send within 30s of bar close)
- Slippage modeling matching broker's actual fills
- Adaptive trade management (partial TPs, runners, breakeven moves)

### Layer 5 — Monitoring (2-3 days)
- Live vs expected performance alerting
- Model drift detection (D-tier in FRAMEWORK.md)
- Telemetry dashboard
- Retraining triggers

### Pre-live validation
- Paper trade 2-4 weeks
- Validate live numbers match backtest
- Then very small live positions
- Scale only after extended live profitability

### Post-v1 (alt-data unlocks)
- E1-E4 confirmation triggers (FRAMEWORK.md Group E)
- D1 explicit invalidation prices, C5 time-to-target (FRAMEWORK.md)
- Tick-level microstructure features (closes H3)
- COT + options OI + sentiment (closes H1)
- Economic calendar + news (closes H2)
- Multi-instrument portfolio (MES, M6E, etc.)
- `SetupPerformanceTracker` readback as a B1-conditional-WR feature family

---

## Appendix A — Key file paths

```
altus/
├── config.py                          # central config (ModelConfig, TrainConfig, Layer2Config, L3Config)
├── data/loader.py                     # MNQ + cross-asset parquet loading
├── features/
│   ├── structural.py                  # family registry + two-pass build_structural_features
│   ├── pipeline.py                    # full feature pipeline (price + structural)
│   └── families/
│       ├── (Phase A) session_time.py, trend_hurst.py, volatility.py, exhaustion.py, anomaly.py, cross_asset.py
│       ├── (Phase B) key_levels.py, liquidity_zones.py, sweep_detection.py, volume_profile.py
│       ├── (Phase C) flow.py
│       ├── (Phase E) round_levels.py, mtf_alignment.py, pv_divergence.py, vol_regime.py,
│       │            session_anatomy.py, corr_regime.py, expectation_surprise.py
│       ├── (Phase F) bocpd_regime.py
│       ├── (Tier-2 anchors) prior_day_anchors.py, vwap_anchors.py, trend_structure.py
│       ├── (Setup library) setup_failed_sweep.py, setup_failed_auction.py, setup_level_defense.py,
│       │                   setup_orb.py, setup_vwap.py, setup_pullback.py, setup_compression.py, setup_eod.py
│       ├── (L2 modulators) path_clearance.py, stop_pool.py, setup_confluence.py,
│       │                   cross_asset_setup_confirm.py, vol_sweet_spot.py, time_of_day_fitness.py
│       └── (Cache) kronos.py, simmtm.py
├── labels/triple_barrier.py           # vol-scaled labels + path_shape + return_H15/H60 + clears_1atr
├── models/
│   ├── modern_tcn.py                  # primary encoder
│   ├── mamba.py                       # peer encoder (auto-CUDA fast path)
│   ├── simmtm.py                      # SSL encoder
│   ├── revin.py                       # reversible input normalization
│   ├── hybrid.py                      # L1 multi-encoder fusion + 3-class direction + predictive heads
│   └── layer2.py                      # L2 meta-labeler (118 input features)
├── training/
│   ├── dataset.py                     # ALTUSDataset
│   ├── train.py                       # L1 training pipeline + _predict (extracts predictive heads)
│   ├── calibration.py, conformal.py   # post-training calibration
│   ├── layer2_train.py                # L2 training + cascade eval
│   ├── l2_router.py                   # 4-stage router: arbitrate → base_wr → modulators → gate
│   ├── production_sim.py              # L3 production sim + setup-aware barriers + hard rules
│   ├── setup_performance.py           # SetupPerformanceTracker (online setup × regime WR)
│   ├── metrics.py                     # MetricsBundle + predictive_diag_verdict
│   └── sim_pnl.py                     # baseline sim (L1+L2 cascade, no L3)
└── splits.py                          # purged walk-forward CV

scripts/
├── train_cloud.py                     # main L1 training entry point (per-variant)
├── train_layer2.py                    # L2 training + setup-conditional WR diagnostic + router cascade
├── pretrain_simmtm.py                 # SSL pretraining (OOS-safe cutoff)
├── build_simmtm_cache.py              # generate SimMTM embeddings cache
├── build_kronos_cache.py              # generate Kronos embeddings cache
├── audit_features.py                  # feature MI / signal-quality audit
├── compute_disagreement.py            # multi-encoder disagreement signal
├── sweep_baseline.sh                  # 6-variant baseline ablation
└── eval_l2_all_variants.sh            # post-sweep L2 cascade eval

tests/
├── test_causal_invariance.py          # leakage guardrail (all families pass)
└── test_splits.py                     # purged walk-forward correctness

FRAMEWORK.md                           # ~45 predictive questions in 8 groups (CANONICAL)
SETUPS.md                              # 8 setup library specs (CANONICAL)
ARCHITECTURE.md                        # this document
```

---

## Appendix B — Configuration cheatsheet

| Knob | Where | Default | Purpose |
|---|---|---|---|
| `--families` | `train_cloud.py` | none | Comma list of feature families per variant |
| `--variants` | `train_cloud.py` | tcn | `tcn` / `mamba` — which L1 long-context branch |
| `ModelConfig.use_inflection` | `config.py` | True | Phase H auxiliary head |
| `TrainConfig.path_shape_w` / `return_h_w` / `clears_level_w` | `config.py` | 0.15 each | Predictive head loss weights |
| `TrainConfig.inflection_loss_weight` | `config.py` | 0.15 | Auxiliary head weight |
| `ModelConfig.use_revin` | `config.py` | True | Reversible Instance Normalization |
| `LABEL_VOL_SCALE_K` | `config.py` | 1.5 | Vol-scaled barrier multiplier (TP = SL = k × ATR_60) |
| `LABEL_HORIZON_BARS` | `config.py` | 60 | Triple-barrier max horizon |
| `LAYER1_V2_STRUCTURAL_FAMILIES` | `config.py` | (varies) | Which feature families during training |
| `L3Config.tp_atr` / `sl_atr` / `hold_bars` | `config.py` | per-setup table | Setup-aware execution params |
| `L3Config.daily_loss_safety_pct` | `config.py` | 0.80 | Fraction of TopStep daily limit before block |
| `L3Config.vol_breaker_pct` | `config.py` | 0.99 | Percentile of trailing 30d vol for circuit break |

---

## Final Note

This document is a snapshot as of 2026-05-27, post predictive-framework pivot. The build is functionally complete through L3; the sweep currently running on 4090 is the empirical-verification gate that decides whether the predictive pivot delivered. The 5 diagnostics in §5.7 are the discriminator: PnL alone can't separate predictive signal from sophisticated volatility detection.

The methodology is the asset. The architecture decisions are traceable to the ~45 questions in FRAMEWORK.md. The empirical-verification gate is in place. The surfer principle is non-negotiable. When this document's snapshot becomes stale, update it — but keep the principles.
