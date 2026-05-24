# ALTUS — Architecture & Strategy

*A discretionary-trader-grade ML system for MNQ futures, built backward from the questions a serious trader actually asks.*

**Status as of 2026-05-24**: Layer 1 + Layer 2 functionally complete; comprehensive baseline test pending on cloud GPU.

---

## 1. What ALTUS Is

ALTUS is a Python ML trading system for MNQ (Micro E-mini Nasdaq-100) futures, deployed on TopStep prop accounts. It is a **layered architecture** with five intended layers:

| Layer | Job | Status |
|---|---|---|
| **L1 — Perception** | Multi-encoder neural model that reads the market and produces calibrated probabilities | ✅ Built |
| **L2 — Meta-Labeling** | Filters L1's candidate trades; outputs binary "trade or skip" decision | ✅ Built (needs refinement) |
| **L3 — Risk Engine** | Deterministic rules: position sizing, drawdown limits, event embargoes, TopStep compliance | ⏸ Next |
| **L4 — Execution** | Broker integration, realistic fill simulation, latency budget enforcement | ⏸ Planned |
| **L5 — Monitoring** | Drift detection, performance alerts, retraining triggers | ⏸ Planned |

**Aspirational targets** (whole-bot, not per-component gates):
- ≥70% win rate
- ≥15 trades per day
- Initial design uses 30pt TP / 30pt SL (1:1 R:R); migration to volatility-based sizing + adaptive trade management is a v2 concern

**Operating constraints**: 1-min polling cadence, TopStep account rules (daily loss limit, trailing drawdown, end-of-day flat), MNQ as initial single instrument. The trading style is "quick trader, not scalper" — the surfer model below.

---

## 2. Design Philosophy

Four principles drive every architectural decision. Each was learned the hard way from prior failed iterations or from this build's empirical results.

### 2.1 Question-driven architecture (work backward)

We don't ask "what models are available?" We ask **"what questions does an in-the-zone trader ask, and which components are needed to answer them?"** The 34 philosophical questions (Section 3) are the design contract. Every component must justify its existence against ≥1 question on that list.

This inverts the usual ML-bot mistake of starting from "what ML models look cool" and trying to retrofit them into a trading system. The questions come first; the architecture serves them.

### 2.2 The Surfer Principle

The engine should behave like a surfer reading individual waves — not a trader who decides "I'm only long today because the higher timeframe is bullish." Intraday markets always have counter-trend opportunities (pullbacks, MTF reversals, intraday squeezes), and the engine must remain capable of riding them.

**Non-negotiable consequences:**
1. **Multi-timescale regime, never single-timescale.** Always 3 scales (5m / 60m / 4h). The *intersection* state (e.g., "HTF=bull, MTF=pullback, LTF=bearish-momentum") is the engine's actual context.
2. **Regime signals are FEATURES, never gates.** They feed L1 and L2 as inputs; never as hard directional vetoes. The model learns the (setup × regime) interaction itself.
3. **Surfer test (acceptance criterion).** Post-deployment cascade evaluation must show counter-HTF-regime trades being taken *and* having positive expectancy. If the engine refuses counter-regime trades, the design failed and we re-architect.

The only place deterministic regime gating is allowed is Layer 3 — and even there, only as a position-sizing modulator (counter-HTF trades get smaller size for risk management), never as a directional veto.

### 2.3 Empirical verification gate

Every feature, every component, every architectural addition must demonstrate OOS lift before staying in the system. A/B test every addition; drop what doesn't clear the bar. No exceptions for "it sounds good" or "the paper says so."

This is what the comprehensive ablation sweep is for. Components that don't earn their place via OOS metrics get dropped, regardless of architectural elegance.

### 2.4 Architectural minimalism — refined

The principle is **"every component must have a unique angle,"** not "default to drop when uncertain." Overlapping components are fine if the non-overlapping parts add genuine value AND distribute cognitive load. The discipline is **unique-job justification**, not aggressive cutting.

When components overlap and agree, that's confidence amplification. When they disagree, that's an uncertainty signal — also useful (Phase L below).

---

## 3. The Philosophical Framework — 34 Questions

The system is designed to answer 34 questions an experienced discretionary trader asks. They were developed from a 22-question source document plus 12 additions made during architectural review.

The full text of all 34 questions lives in `~/.claude/projects/-Users-michaelpang-ALTUS/memory/philosophical_questions_framework.md`. Summarized by category:

### I — Order Flow & Sponsorship (Q1-Q5, Q29, Q32)
Who is driving this move? Is order flow accumulating or distributing? Is the flow transactional or directional? Is volume confirming the move or diverging? Absorption vs conviction? Tape rhythm (steady vs spasmodic)? Flow acceleration (deceleration as reversal signal)?

### II — Auction State & Value (Q6-Q10)
Are we inside accepted value, at the edge, or in price discovery? Imbalanced (trending) vs balanced (ranging)? Which timeframe is in control? Have prices been accepted or rejected? Where are the HVNs and LVNs?

### III — Liquidity, Obstacles & Path (Q11-Q14, Q28)
Where is resting liquidity? What obstacles between current price and target? Has the move already extended? Trapped participants nearby? Liquidity asymmetry (above vs below) → directional gravity?

### IV — Forced Flow & Trapped Participants (Q15-Q18)
Who is wrong-sided and at what price are they forced to act? Stop-run fuel vs genuine new participation? Crowded positioning? Slow squeeze building?

### V — Regime, Bias & Self-Awareness (Q19-Q23, Q25, Q27, Q30)
Stale regime classification? Regime confirmed today? Is consensus about to be punished? Vol regime expansion/contraction? Correlation regime breakdown? Pattern similarity to historical setups? Model self-confidence + component agreement?

### VI — Temporal & Move Dynamics (Q24, Q26, Q31, Q33)
Where in the session's natural arc are we? Inflection vs continuation likelihood? Move lifecycle phase (initiation/acceleration/exhaustion/termination)? Expected vs actual surprise?

### VII — External Context (Q34) — *aspirational*
What scheduled events shape today's character?

**Honest gaps** (explicitly accepted, will remain partial for v1):
- **Q1** (large-lot prints): Ceiling-limited by 1m OHLCV data; tick T&S would close
- **Q17, Q21** (positioning): Requires COT + options OI + sentiment data (alt-data Phase, future)
- **Q22** (post-entry thesis): Requires Layer 3/4 execution logic
- **Q34** (external context): Requires economic calendar + news ingestion (alt-data Phase, future)

**Coverage target**: 30 of 34 questions well-answered by the planned L1+L2 architecture.

---

## 4. Question → Component Mapping

Working backward from the 34 questions to the architecture they demand:

| # | Question | Primary mechanism | Secondary |
|---|---|---|---|
| Q1 | Large-player sponsorship | VPIN feature | Kronos |
| Q2 | Accumulation vs distribution | Kronos transfer | TCN + volume profile |
| Q3 | Transactional vs directional flow | TCN + VPIN | Mamba |
| Q4 | Volume confirming/diverging | `pv_divergence` feature | TCN |
| Q5 | Absorption vs conviction | `absorption` feature | TCN, Mamba |
| Q6 | Inside/edge/discovery of value | Volume profile feats | TCN |
| Q7 | Imbalanced vs balanced | BOCPD + Hurst | TCN |
| Q8 | Timeframe in control | `mtf_alignment` feature | TCN |
| Q9 | Acceptance/rejection | Kronos + volume profile | TCN |
| Q10 | HVN/LVN | Volume profile feats | — |
| Q11 | Resting liquidity | `round_levels` + `key_levels` | `liquidity_zones`, Kronos |
| Q12 | Path obstacles | Level features | (potential: path-density feat) |
| Q13 | Already-extended | `extension` feature | TCN |
| Q14 | Trapped participants nearby | **Mamba state** | `sweep_detection` |
| Q15 | Wrong-sided forced flow | **Mamba state** | `sweep_detection` |
| Q16 | Stop-run fuel vs new | VPIN multi-horizon | TCN |
| Q17 | Crowded positioning | *(data gap — alt-data phase)* | — |
| Q18 | Slow squeeze | **Mamba state** | trend + Hurst |
| Q19 | Stale regime classification | **BOCPD multi-TF** | SimMTM |
| Q20 | Regime confirmed today | **BOCPD multi-TF** | Conformal gate |
| Q21 | Punishing consensus | *(data gap — alt-data phase)* | BOCPD partial |
| Q22 | Post-entry thesis | *(Layer 3/4 work)* | Mamba would feed it |
| Q23 | Vol regime | `vol_regime` feature | BOCPD |
| Q24 | Session anatomy | `session_anatomy` feature | — |
| Q25 | Correlation regime | `corr_regime` feature | — |
| Q26 | Inflection vs continuation | **Inflection head** | TCN, Mamba |
| Q27 | Pattern similarity to history | **SimMTM embedding** | — |
| Q28 | Liquidity asymmetry | `liquidity_asymmetry` feature | — |
| Q29 | Tape rhythm | `tape_rhythm` feature | VPIN |
| Q30 | Model self-confidence | **Multi-encoder disagreement** | Conformal gate |
| Q31 | Move lifecycle phase | **Mamba state** | BOCPD age |
| Q32 | Flow acceleration | `flow_acceleration` feature | — |
| Q33 | Expected vs actual | `expectation_surprise` feature | — |
| Q34 | External context | *(data gap — alt-data phase)* | — |

**Reading this table**: every component is *demanded by the framework*. Nothing is in the architecture opportunistically.

---

## 5. Layer 1 — Perception Architecture

L1 is the engine's reader of the market. It produces calibrated per-bar probabilities + auxiliary outputs that downstream layers consume.

### 5.1 The multi-encoder design

Three perception encoders run in parallel, each with a fundamentally different inductive bias:

**ModernTCN** (Liu et al. 2024) — *convolutional pattern recognizer*
- **Strength**: hierarchical local patterns, translation-invariant, fully parallel
- **Job**: Q3, Q4, Q5, Q6-Q10, Q13, Q16 — pattern-shape questions
- **Status**: ✅ Working, primary encoder

**Mamba-2** (Gu & Dao 2023) — *stateful selective state-space*
- **Strength**: dynamic memory updates, carries state forward indefinitely, categorically different from convolution
- **Job**: Q14, Q15, Q18, Q31 — state-tracking questions that TCN structurally can't answer
- **Status**: ✅ Code complete with auto-detect CUDA fast path. Pure-PyTorch fallback works on any device; official `mamba-ssm` Triton kernel activates on CUDA when installed.

**Kronos** (Open-source foundation model for finance) — *transfer learning*
- **Strength**: pre-trained on much more market data than we have; brings in patterns we'd never see in our local dataset
- **Job**: Q2, Q9, Q11 — well-studied patterns at population scale; "informed prior"
- **Status**: 🟡 Family + cache loader implemented. Cache build script exists but failed once and needs debug. Cache-only architecture means inference is instant once cache is built.

**SimMTM** (Liu et al. 2023) — *self-supervised similarity*
- **Strength**: learns continuous embeddings via masked-bar reconstruction; supports "what historical state is the current state most similar to" lookups
- **Job**: Q27 (pattern similarity to history) — unique angle; no other component does this
- **Status**: ✅ Encoder + masking + pretraining script + cache builder + feature family all built. Needs cloud GPU run to pretrain + build cache.

### 5.2 BOCPD — parallel regime tracker

Bayesian Online Change-Point Detection (Adams & MacKay 2007) runs **parallel to** the encoders, not as part of L1's main forward pass.

- Applied at three timescales (5m / 60m / 4h equivalents)
- Outputs 9 features: regime age, change-point probability, run-length entropy × 3 scales
- Fed into L1 as features AND into L2 as gating context
- **Strictly never used as a directional gate** (surfer principle)

Why a separate component instead of letting TCN learn regime implicitly: TCN is a pattern recognizer with no native ability to produce calibrated discrete-state posteriors. BOCPD does this cleanly with ~200 LOC and minimal compute.

### 5.3 Inflection auxiliary head

A small 2-layer MLP on the fusion embedding that predicts P(price resolves AGAINST recent direction) — the wave-about-to-break signal from Q26.

- Trained as auxiliary objective alongside the main L1 heads (weight 0.15)
- Acts as a regularizer for the shared encoder
- Output is also available to L2 as a candidate-quality signal
- Toggleable via `ModelConfig.use_inflection` for ablation

### 5.4 Feature library

L1 ingests 26 feature families (~150 numeric features total), grouped by phase:

| Phase | Families | Count | Purpose |
|---|---|---|---|
| A | `vol, trend, anomaly, session, exhaust` | 5 | Phase-A baseline (Phase A complete) |
| B | `levels, liquidity, sweep, profile` | 4 | Market structure |
| C | `flow, cross` | 2 | Order flow + cross-asset |
| E | `round, mtf, absorp, pvd, extension, vreg, sanat, creg, lasym, rhythm, facc, surprise` | 12 | Trader-frame additions, each addressing a specific philosophical question |
| F | `bocpd` | 1 | Multi-TF regime |
| (cache) | `kronos, simmtm` | 2 | Foundation-model + SSL embeddings |

All families pass an explicit **causal-invariance test**: features at row T computed from data[:N] vs data[:N+offset] must be identical for rows < N. This catches the entire class of lookahead bugs that has historically been the silent killer of ML trading bots. The test caught one real bug during development (VPIN bucket-size leak in `flow.py`).

### 5.5 Output heads

Layer 1 produces 7+ outputs per bar:
- **2 binary classification**: `long_tp_prob`, `short_tp_prob` (sigmoid + isotonic calibration)
- **4 regression**: `mfe_long`, `mae_long`, `mfe_short`, `mae_short` (Huber loss, points)
- **1 auxiliary**: `inflection_prob` (Q26)
- **192-D fusion embedding**: the compressed representation right before the heads; exposed to L2

Labels come from **triple-barrier** (López de Prado): TP (+30pt) / SL (-30pt) / timeout (60 bars), with worst-case-within-bar tie-breaking (SL wins ties).

### 5.6 Training discipline

- **Purged walk-forward CV** with 60-bar embargo (prevents label-overlap leakage — Layer 1's most important guardrail)
- **3-fold cross-validation** on 3 years of MNQ data
- **OOS lockbox**: last 4 months held out, never touched in training
- **Isotonic calibration** post-training on a held-out calibration set
- **Conformal prediction wrapper** for distribution-free abstention

---

## 6. Layer 2 — Meta-Labeling Architecture

L2's job: given L1 flagged this bar as a candidate, should we actually trade it?

### 6.1 Inputs to L2

- 32 hand-crafted features summarizing the moment (price action, regime context, recent volatility)
- The 192-D L1 fusion embedding, projected down to 16-D via a learned linear layer
- (planned) Multi-encoder disagreement signals from Phase L
- (planned) Per-bar BOCPD regime signals
- (planned) Per-bar inflection probability

### 6.2 Model

Small MLP (~10k params). Binary classification: "good trade" vs "skip."

### 6.3 Conformal gate

Wraps L2's calibrated probability. Provides distribution-free coverage guarantees on abstention — when L2 isn't confident enough (residual quantile is too wide), the trade is rejected.

### 6.4 Cascade evaluation

Final trade decision is the cascade: L1 ranks every bar → L2 filters candidates → conformal gate makes final go/no-go call. Measured by:
- Top-K% selectivity (rank L2 outputs, take top K%)
- Threshold mode (absolute probability cutoff)
- Conformal-gated mode (lower-bound of prediction interval ≥ threshold)

### 6.5 Honest current weakness

The L2 cascade as currently designed gives **only +0.02 pp WR** over L1 alone on the partial TCN baseline (run 2026-05-24). Root causes identified:
1. **Long-side bias in candidate selection** — `train_layer2.py` picks top 20% by max(long_prob, short_prob), which currently selects only long candidates. Needs per-side balanced selection.
2. **L2 calibrated probabilities crushed into a narrow range** [0.426, 0.509] — the meta-labeler isn't producing confident discriminations.
3. **TCN-only L1 doesn't give L2 enough signal diversity** — the multi-encoder additions (Mamba, SimMTM, Kronos) + disagreement signal should give L2 more to filter on.

**The L2 redesign waits until after the full L1 ablation baseline.** Fixing L2 in the current partial architecture would be debugging the wrong thing.

---

## 7. Architectural Decisions Made (and Why)

This section documents non-obvious choices for future-us reference.

### 7.1 Dropped: xLSTM
Originally planned as an alternative recurrent peer to Mamba. Dropped during architectural review — overlaps too heavily with Mamba's stateful-recurrence inductive bias. Picked one (Mamba) for being more modern, better scaling.

### 7.2 Kept BOTH BOCPD and SimMTM (despite both addressing "regime")
Considered making them substitutes. Verdict: complementary, not competing.

| | BOCPD | SimMTM |
|---|---|---|
| State type | Discrete (run length + change-prob) | Continuous (embedding vector) |
| Output for L2 | Clean gating signal | Less interpretable |
| Output for L1 | Categorical features | Rich feature vector |
| Captures | Markovian regime transitions | Non-Markovian similarity to prior states |
| Q19 angle | "We're 73% in regime A" | "Current state similar to March 2023" |

Both earn their place under the unique-angle test.

### 7.3 Multi-encoder vs single-encoder
Recalibration during build: **"every component must have a unique angle"** is the test, NOT "default to drop." Three encoders (TCN + Mamba + Kronos) + SimMTM (frozen SSL) earn their place because:
- TCN: convolutional pattern recognition
- Mamba: stateful recurrence
- Kronos: transfer learning from external data
- SimMTM: SSL similarity

Each has a unique inductive bias. When they agree, that's confidence amplification (multi-encoder disagreement signal is low → L2 trusts the prediction). When they disagree, that's uncertainty signal → L2 should be cautious.

### 7.4 30/30 TP/SL kept (for now)
Sweep results showed asymmetric R:R doesn't rescue the system — the model finds small-magnitude winners that barely scrape +30pt. Widening TP collapses WR; tightening TP collapses PnL/trade. **This points to a label/task formulation issue** (the model is being trained to find small reversion setups), not an R:R configuration issue. Will be revisited with volatility-based sizing + adaptive trade management in v2.

### 7.5 5090 → 4090 hardware choice
5090 (Blackwell, sm_120) is bleeding-edge — most ML packages don't have prebuilt wheels yet. Installing `mamba-ssm` pulled an incompatible torch upgrade. 4090 (Ada Lovelace, sm_89) is mature — all packages install cleanly. For our toolchain today, 4090 is the right choice despite being slower. Revisit 5090 in ~6 months when ecosystem catches up.

---

## 8. Current Build Status

### 8.1 What's done and verified locally

- ✅ All 26 feature families pass causal-invariance test (24 functional families + 2 cache-loaded)
- ✅ ModernTCN encoder trained and producing OOS AUC ~0.64 on TCN-only baselines
- ✅ Mamba peer encoder with auto-CUDA fast-path (pure-PyTorch fallback verified)
- ✅ SimMTM pretraining pipeline (smoke-tested with synthetic data)
- ✅ BOCPD regime SSM at 3 timescales (causal verified)
- ✅ Inflection auxiliary head (smoke-tested end-to-end including loss + backward)
- ✅ Layer 2 meta-labeler with conformal gate
- ✅ Multi-encoder disagreement script
- ✅ Comprehensive sweep tooling

### 8.2 What's blocked / not yet validated empirically

- ⏸ Comprehensive ablation sweep with all components (waiting on 4090 setup)
- ⏸ Mamba's actual contribution to cascade quality (needs full sweep with CUDA kernel)
- ⏸ SimMTM cache (needs pretrain run + cache build on 4090)
- ⏸ Kronos cache (needs debug of failed build script)
- ⏸ L2 redesign (deferred until full L1 baseline is in)

### 8.3 What's the immediate test target

**One comprehensive ablation sweep** that exercises every component on/off:

```
01. baseline_min     TCN + Phase A features only
02. +Phase E         add 12 trader-frame features
03. +Phase F (bocpd) add regime SSM
04. +Inflection      add auxiliary head
05. +Mamba           add stateful encoder (with CUDA kernels)
06. +SimMTM          add SSL embeddings
07. +Kronos          add transfer-learning features
08. FULL             all components enabled
```

That gives us the load-bearing-component picture: how much each addition lifts AUC + top-decile WR + cascade quality, on the same 3-year × 3-fold cross-validation with OOS lockbox eval.

---

## 9. Lessons Learned (so far)

### 9.1 Empirical verification works
The causal-invariance test caught a real VPIN lookahead bug that would have silently corrupted every flow-family run. The "small fixes are insurance" discipline pays off.

### 9.2 Build before testing partial systems
The "test each component in isolation" approach repeatedly produced discouraging results that didn't reflect what the full architecture can do. The user's correction — **build the whole thing, test once at the end** — was right.

### 9.3 Infrastructure friction is the silent productivity killer
RunPod migrations, GPU availability, ML package version conflicts, web terminal disconnects — these consume more time than the actual ML work. Tooling decisions matter: tmux + nohup for resilience, fresh-clone-don't-pull when state is corrupted, mature GPU choice over bleeding-edge.

### 9.4 Asymmetric R:R isn't the answer
A 50% WR system isn't rescued by tightening or widening TP — the model's signal magnitude is the binding constraint. Real volatility-based sizing + adaptive trade management is the v2 answer, not R:R fiddling.

### 9.5 L2 needs richer L1 outputs to do its job
A meta-labeler can only filter as well as the upstream signal diversity allows. Single-encoder TCN doesn't give L2 enough to discriminate on. The multi-encoder architecture is what makes L2 meaningful.

---

## 10. Roadmap Beyond L1+L2

### Phase: Get the baseline (next 1-3 sessions)
1. Complete 4090 setup with `mamba-ssm` + SimMTM pretrain + cache build
2. Run comprehensive ablation sweep
3. Lock the load-bearing component set
4. Redesign L2 based on richer L1 outputs (fix candidate-side bias, possibly redesign hand-crafted features)
5. Final L1+L2 cascade evaluation → this is the **true baseline**

### Layer 3 — Risk Engine (3-5 days)
- TopStep rules: daily loss limit, trailing drawdown, max position, end-of-day flat
- Event embargoes: no trades during FOMC/CPI/NFP/cash open windows
- Position sizing: volatility-targeted OR Kelly fraction on conformal probability
- Counter-HTF-regime size modulation (surfer principle)
- Trade concurrency rules (no overlapping positions)

### Layer 4 — Execution (5-7 days)
- TopStep broker integration (Project X API or Rithmic)
- Realistic fill simulator for backtest validation
- Latency budget enforcement (must complete L1 → L2 → L3 → order send within 30s of bar close)
- Slippage modeling matching broker's actual fills
- Adaptive trade management (partial TPs, runners, breakeven moves) — v2

### Layer 5 — Monitoring (2-3 days)
- Live vs expected performance alerting
- Model drift detection
- Telemetry dashboard
- Retraining triggers

### Pre-live validation
- Paper trade 2-4 weeks
- Validate live numbers match backtest
- Then very small live positions
- Scale only after extended live profitability

### Future (post-v1)
- Tick-level microstructure features (closes Q1 ceiling)
- COT + options OI + sentiment alt-data (closes Q17, Q21)
- Economic calendar + news ingestion (closes Q34)
- Volatility-based sizing + adaptive trade management
- Multi-instrument portfolio (MES, M6E, etc.)

---

## Appendix A — Key file paths

```
altus/
├── config.py                          # central config (ModelConfig, TrainConfig, etc.)
├── data/loader.py                     # MNQ + cross-asset parquet loading
├── features/
│   ├── structural.py                  # family registry + build_structural_features
│   ├── pipeline.py                    # full feature pipeline (price + structural)
│   └── families/
│       ├── (Phase A) session_time.py, trend_hurst.py, volatility.py, exhaustion.py, anomaly.py
│       ├── (Phase B) key_levels.py, liquidity_zones.py, sweep_detection.py, volume_profile.py
│       ├── (Phase C) flow.py, cross_asset.py
│       ├── (Phase E) round_levels.py, mtf_alignment.py, absorption.py, pv_divergence.py,
│       │            extension.py, vol_regime.py, session_anatomy.py, corr_regime.py,
│       │            liquidity_asymmetry.py, tape_rhythm.py, flow_acceleration.py,
│       │            expectation_surprise.py
│       ├── (Phase F) bocpd_regime.py
│       └── (Cache) kronos.py, simmtm.py
├── labels/triple_barrier.py           # labels + inflection target
├── models/
│   ├── modern_tcn.py                  # primary encoder
│   ├── mamba.py                       # peer encoder (auto-CUDA fast path)
│   ├── simmtm.py                      # SSL encoder
│   ├── revin.py                       # reversible input normalization
│   ├── hybrid.py                      # L1 multi-encoder fusion + heads
│   └── layer2.py                      # L2 meta-labeler
├── training/
│   ├── dataset.py, train.py           # L1 training pipeline
│   ├── calibration.py, conformal.py   # post-training calibration
│   ├── layer2_train.py                # L2 training + cascade eval
│   ├── metrics.py, sim_pnl.py         # evaluation
│   └── dataset.py                     # ALTUSDataset
└── splits.py                          # purged walk-forward CV

scripts/
├── train_cloud.py                     # main L1 training entry point
├── pretrain_simmtm.py                 # SSL pretraining
├── build_simmtm_cache.py              # generate SimMTM embeddings cache
├── build_kronos_cache.py              # generate Kronos embeddings cache
├── install_mamba_ssm.sh               # CUDA kernel install (Mamba)
├── compute_disagreement.py            # multi-encoder disagreement signal
├── eval_rr_asymmetry.py               # asymmetric R:R sweep
├── train_layer2.py                    # L2 training entry point
└── (sweep scripts) sweep_phaseBC.sh, sweep_full_arch.sh, sweep_mamba_only.sh

tests/
├── test_causal_invariance.py          # leakage guardrail (all 24 families pass)
└── test_splits.py                     # purged walk-forward correctness

ARCHITECTURE.md                        # this document
```

---

## Appendix B — Configuration cheatsheet

The system has a few config knobs worth knowing about:

| Knob | Where | Default | Purpose |
|---|---|---|---|
| `LAYER1_V2_STRUCTURAL_FAMILIES` | `config.py` | (varies by sweep) | Which feature families to use during training |
| `ModelConfig.use_inflection` | `config.py` | True | Enable Phase H auxiliary head |
| `TrainConfig.inflection_loss_weight` | `config.py` | 0.15 | Auxiliary loss term weight |
| `ModelConfig.use_revin` | `config.py` | False | Reversible Instance Normalization |
| `--variants` | `train_cloud.py` | tcn | `tcn` / `mamba` — which L1 long-context branch |
| `--families` | `train_cloud.py` | none | Comma list of feature families |
| `LABEL_HORIZON_BARS` | `config.py` | 60 | Triple-barrier max horizon |
| `TP_POINTS`, `SL_POINTS` | `config.py` | 30, 30 | Triple-barrier targets |

---

## Final Note

This document is a snapshot of architectural intent as of 2026-05-24. The build has shipped substantially in this session — Phases E, F, H (inflection head), I (Mamba CUDA fast path), K (SimMTM pipeline) all built and committed. The remaining work to reach "full L1+L2 baseline" is:

1. Validate full architecture works empirically (one cloud sweep)
2. Refine L2 based on what the data tells us
3. Build out the operational layers (L3 + L4 + L5)
4. Paper trade
5. Go live small

The methodology is the asset. The architecture decisions are traceable to the 34 questions. The empirical-verification gate is in place. The surfer principle is non-negotiable. When this document's snapshot becomes stale, update it — but keep the principles.
