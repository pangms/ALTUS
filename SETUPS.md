# ALTUS — Predictive Setup Library

*Locked specification of the 8 asymmetric setups the engine is built to detect, score, and trade. This is the philosophical-questions exercise reapplied — designed BACKWARD from "where does the WR come from."*

**Status:** Spec — locked 2026-05-25, post-architectural-audit pivot.
**Last revision:** First draft.
**Purpose:** Define the setups precisely enough that implementation, evaluation, and tuning all reference the same source of truth.

---

## 0. Why a Setup Library

Top-decile WR converging to base rate (~0.50) on bar-by-bar classification taught us this: **the model can rank "what bars look interesting" but cannot reliably predict direction from raw state.** The fix is to stop asking the model to predict direction on every bar and instead ask: *"Is one of N known asymmetric setups forming right now? If yes, which one, and how strong?"*

A "setup" here is a multi-condition pattern with documented directional edge in intraday futures markets. Each setup answers all five predictive questions simultaneously:

- **A — Detection:** Are the conditions met?
- **B — Direction:** Which way does the setup favor?
- **C — Path / magnitude:** How far should price go?
- **D — Failure mode:** What invalidates the setup?
- **E — Confirmation:** Has the entry trigger fired?

The model's job becomes: rank candidate setups, predict their conditional WR given context, and feed the decision layer a complete picture.

---

## 1. Setup Library Overview

| ID | Setup | Frame | Est WR | Avg R:R | Hold time | Frequency |
|---|---|---|---|---|---|---|
| **A1** | Open Range Breakout | NY RTH first hour | 0.55-0.60 | 1.5:1 | 30-90 min | 0.5/day |
| **A2** | VWAP Rejection/Reclaim | Trending session | 0.55-0.60 | 1:1 - 1.5:1 | 20-60 min | 1-3/day |
| **A3** | Failed Sweep / Liquidity Trap | All sessions, anchored to PDH/PDL/ONH/ONL | 0.60-0.65 | 1.5:1 - 2:1 | 15-60 min | 0.5-2/day |
| **A4** | Trend Pullback | Confirmed trend regimes | 0.55-0.60 | 1.5:1 - 2:1 | 30-60 min | 1-3/day |
| **A5** | Compression Breakout | Low-vol regime → expansion | 0.52-0.57 | 1:1 - 1.5:1 | 15-45 min | 0.5-1/day |
| **A6** | Failed Auction | Multi-touch level rejection | 0.60-0.65 | 1.5:1 - 2:1 | 30-90 min | 0.5-1/day |
| **A7** | End-of-Day Reversion | Last 30 min NY RTH | 0.55-0.58 | 1:1 | 10-25 min | 0.5-1/day |
| **A8** | Multi-Touch Level Defense | High-touch S/R | 0.58-0.62 | 1.5:1 | 20-60 min | 0.3-0.8/day |

**Aggregate frequency:** ~5-12 setup-active bars per day, ~3-6 actually-tradeable after arbitration + confirmation.
**Daily TPD target under setup-driven trading:** 3-8 trades, well below the 10-20 free-running target. **Patience is the alpha.**

**WR estimates** are anchored to academic and practitioner literature on intraday-futures pattern profitability (Wyckoff, market profile, Lopez de Prado, plus IBKR-published institutional intraday studies). These are upper-bound expectations for clean executions in a properly-conditioned regime — real numbers will vary and must be validated empirically per [[feedback-empirical-verification]].

---

## 2. A1 — Open Range Breakout (ORB)

**Thesis.** The first 30-60 minutes of the NY RTH session establishes a reference range. Algos + traders pile stops above the OR high and below the OR low. A clean break with confirmation tends to continue because the breakout triggers stop runs in its direction, providing fuel.

**Edge source.** Stop-cluster geometry + first-hour volatility expansion + institutional opening-trade unwinds.

**Detection conditions (all must be met):**

| Condition | Threshold |
|---|---|
| Time of day | After NY RTH start + 30 min AND before NY RTH start + 3 hr |
| OR range established | High[13:30 UTC : 14:00 UTC] and Low[13:30 UTC : 14:00 UTC] both observed |
| OR size | OR range > 0.5 × ATR(60min) AND < 3 × ATR(60min) (not too tight, not blown out) |
| Breakout level | close > OR_high (long) OR close < OR_low (short) |
| Range integrity | No close beyond OR boundary in the OR window itself (clean range) |
| Session character | Total NY volume up to breakout time > 0.8× 20-day average for that time |

**Strength scoring (0-1):**
```
strength = 0.5
         + 0.2 × (volume_at_breakout / avg_volume - 1.0).clip(0, 1)
         + 0.2 × (close_distance_beyond_OR / ATR).clip(0, 1)
         - 0.2 × (mins_since_break_eligible / 90).clip(0, 1)   # freshness decay
         + 0.1 × (1 if mtf_alignment agrees with break direction)
```
Clipped to [0, 1].

**Direction:** Long if close > OR_high, Short if close < OR_low.

**B-tier (directional bias modulators):**
- **B1 setup-conditional WR:** Read from historical table (initial estimate 0.57)
- **B2 HTF agreement bonus:** +0.03 if mtf_alignment_score aligns
- **B2 HTF disagreement penalty:** −0.05 if mtf strongly disagrees (counter-trend ORB)
- **B3 liquidity gravity:** +0.02 if liquidity_asymmetry favors break direction

**C-tier (path & magnitude):**
- **C1 expected_return_H15:** +0.6 × OR_range (in points)
- **C2 expected_return_H60:** +1.0 × OR_range
- **C3 path_shape:** continuation 0.50, chop 0.30, revert 0.15, reverse 0.05
- **C4 clears_next_level:** P(clears nearest PDH/VWAP+1σ) ≈ 0.55-0.65
- **C5 time_to_target:** median 35-45 min

**D-tier (failure modes):**
- **D1 invalidation_level:** Long → OR_low (full re-entry into range = failed breakout). Short → OR_high.
- **D2 time_invalidation:** 90 min after first eligible breakout signal. If no trigger by then, setup expires.
- **D3 cross-asset invalidation:** ES correlation breakdown (|NQ-ES 5m return correlation| < 0.5 last 30min) reduces conviction
- **D4 recent failure check:** if last 3 ORBs in same direction failed, conditional WR -5pp

**E-tier (confirmation triggers):**
- **E1 primary:** Bar T closes beyond OR boundary by ≥0.15 × ATR(15min)
- **E2 secondary:** Bar T+1 closes in same direction as breakout (no immediate reversal)
- **E3 entry_trigger_now:** Bar T+1 open > OR_high + 0.1 × ATR (long) OR symmetric
- **E4 disqualifiers:** None of last 3 bars closed back inside the OR

**Why this setup matters in MNQ specifically.** NQ-100 futures have outsized opening-hour volatility (US tech-stock-driven). MNQ ORBs are a documented edge in proprietary trading literature; the IBKR public-research studies show 55-58% WR on disciplined ORB execution in volatile sessions.

---

## 3. A2 — VWAP Rejection / Reclaim

**Thesis.** Session-anchored VWAP is institutional benchmark. In trending sessions, VWAP gets **held** (price tests and bounces) → mean-revert entry in direction of trend. In ranging sessions, VWAP gets **reclaimed** from one side → trend resumption.

**Edge source.** Algo-driven execution against VWAP; institutional positioning around VWAP.

**Detection conditions:**

| Condition | Threshold |
|---|---|
| In session | NY RTH (13:30-20:00 UTC) AND past first 60 min |
| Regime | Either bull (mtf_alignment_score > 0.3) or bear (< −0.3); NOT chop |
| Price near VWAP | abs(vwap_dist_atr) < 0.3 (within 0.3 ATR of VWAP) |
| Recent behavior | At least 1 prior touch + reject of VWAP in same session (price tested and bounced) OR clean reclaim from one side after break |
| VWAP slope | Aligns with regime: bull regime → vwap_slope > 0; bear → vwap_slope < 0 |

**Two sub-types:**
- **A2a Rejection:** price approaches VWAP from above in bull regime AND vwap_recent_holds_count ≥ 2 → LONG entry at touch
- **A2b Reclaim:** price was below VWAP, closes back above with vwap_slope > 0 → LONG entry on reclaim

**Strength scoring:**
```
strength = 0.5
         + 0.15 × (vwap_recent_holds_count / 4).clip(0, 1)
         + 0.10 × abs(mtf_alignment_score)
         + 0.10 × (1 if vwap_slope aligns with regime)
         + 0.10 × (1 if vwap_dist_atr < 0.15 (very close to VWAP))
         - 0.10 × (1 if recent A2 setup in same session already triggered)
```

**Direction:** Long when reject/reclaim in bull regime; Short in bear regime.

**B-tier:**
- **B1 WR:** 0.57 baseline
- **B2 HTF agreement is mandatory** — counter-trend A2 setups historically fail; require mtf_alignment to confirm
- **B4 path_shape:** strong "continuation" prior (this is a pullback-continuation pattern)

**C-tier:**
- **C1 expected_return_H15:** +0.5 × ATR(15min)
- **C2 expected_return_H60:** +1.0 × ATR(60min)
- **C3 path_shape:** continuation 0.55, chop 0.25, revert 0.15 (after the touch), reverse 0.05
- **C4 clears_next_level:** P(reaches next +1σ band) ≈ 0.55
- **C5 time_to_target:** 25-40 min

**D-tier:**
- **D1 invalidation:** Close beyond VWAP by 0.5 × ATR in opposite direction of entry
- **D2 time invalidation:** 45 min from entry if no progress > 0.3 × ATR
- **D3 cross-asset:** Cross-asset correlation breakdown → −3pp WR
- **D4 recent failures:** Last 2 A2 setups in same session that failed → skip

**E-tier:**
- **E1 primary:** Bar at VWAP must show wick rejection (long wick on appropriate side, body small relative to range)
- **E2 secondary:** Next bar continues in entry direction (close > entry bar's close for long)
- **E3 entry_trigger:** E1 AND E2 fired AND price > entry bar's high for long (breakout of rejection candle)
- **E4 disqualifier:** Two consecutive closes beyond VWAP in opposite direction within 5 bars

---

## 4. A3 — Failed Sweep / Liquidity Trap (HIGHEST WR)

**Thesis.** A specific HTF level (PDH, PDL, ONH, ONL — or strong recent swing) gets **swept** (price extends beyond it briefly), but **fails to hold** (price returns through the level within a few bars). The traders who put stops above/below got triggered — they're now on the wrong side. The reversal that traps them tends to extend because their continued unwinding adds fuel.

**Edge source.** The most documented intraday edge in liquid futures markets. Liquidity hunting is a real institutional pattern, and failed sweeps are the counter-tell.

**Detection conditions:**

| Condition | Threshold |
|---|---|
| Reference level | PDH OR PDL OR ONH OR ONL OR strong recent swing (key_levels family, n_touches ≥ 2) |
| Sweep event | Within last N=12 bars (12 min), price.high > level + 0.1 × ATR (for sweep above) OR price.low < level − 0.1 × ATR (for sweep below) |
| Failure | After the sweep, within next M=8 bars, current bar.close is on the opposite side of the level (close < level for an above-sweep) |
| Magnitude | Sweep extension was ≤ 0.6 × ATR beyond level (not a real break — just stop hunt) |
| Recent context | Level had been respected for ≥ 60 bars BEFORE the sweep (it was a real level, not a phantom) |

**Strength scoring:**
```
strength = 0.5
         + 0.2 × (sweep_extension / (0.6 × ATR)).clip(0, 1)    # cleaner sweep = stronger
         - 0.2 × (bars_since_sweep / 8).clip(0, 1)              # freshness decay
         + 0.15 × (level_age_bars / 240).clip(0, 1)             # older level = bigger trap
         + 0.10 × (1 if rejection_wick on sweep bar)
         + 0.05 × (1 if accompanied by volume spike on sweep)
```

**Direction:** Opposite of the sweep direction. Sweep above PDH → SHORT setup. Sweep below ONL → LONG setup.

**B-tier:**
- **B1 WR:** 0.62 baseline (highest in the library)
- **B2 HTF agreement:** Not required (failed-sweep is a counter-trend reversal pattern; can fire against HTF)
- **B3 liquidity asymmetry:** Strong directional pull toward bigger remaining liquidity pool

**C-tier:**
- **C1 expected_return_H15:** +0.6 × ATR
- **C2 expected_return_H60:** +1.5 × ATR (these setups can run far)
- **C3 path_shape:** reverse 0.45, continuation 0.30 (in setup-direction terms = "continuation of the reversal"), revert 0.20, chop 0.05
- **C4 clears_next_level:** P(reaches next major anchor in setup direction) ≈ 0.55
- **C5 time_to_target:** 20-45 min

**D-tier:**
- **D1 invalidation:** Re-test of swept level + close ON SWEEP SIDE = double sweep failure
- **D2 time invalidation:** 30 min after sweep with no follow-through in setup direction
- **D3 cross-asset:** ES making same failed-sweep pattern is BONUS (+3pp WR); ES diverging is mild warning (−2pp WR)
- **D4 recent failures:** If last 2 A3 setups failed in same session, conditional WR -5pp

**E-tier:**
- **E1 primary:** Close back on opposite side of swept level
- **E2 secondary:** Volume on rejection bar > 1.5× average
- **E3 entry_trigger:** E1 fired AND next bar close continues in setup direction
- **E4 disqualifier:** Within next 15 min, price re-attempts to break the level AND closes on sweep side again (this is now a confirmed break, not a sweep)

**Why this is the prized setup.** Sweeps + failures are well-documented in tape-reading literature (Steidlmayer, Dalton on market profile). The conditional WR is highest in our library, and the failure mode (D1) is precise — letting L3 use tight stops + larger size.

---

## 5. A4 — Trend Pullback

**Thesis.** In a confirmed multi-TF trend (e.g., mtf_alignment_score > 0.5), a pullback to a key short-term mean (8-EMA, 21-EMA, or 50% retracement of recent swing) offers a continuation entry. The trend has higher base rate of continuation than reversal; the pullback gives a better entry price.

**Edge source.** Trend persistence + mean reversion combined — entering on the pullback edge of the trend's ATR envelope.

**Detection conditions:**

| Condition | Threshold |
|---|---|
| Confirmed trend | mtf_alignment_score > 0.5 (long) OR < −0.5 (short) for last ≥ 20 bars |
| In pullback | For long: close < EMA(8) by 0.3-1.5 ATR AND close > EMA(21) (haven't broken structure) |
| Not too deep | For long: retracement from recent swing high is 30-62% (Fibonacci sweet spot — beyond this is a structural break) |
| Momentum oversold-in-trend | 1m RSI(14) between 30-50 for long (or 50-70 for short) |
| Trend integrity | No new lower-low in last 30 bars (long trend) |

**Strength scoring:**
```
strength = 0.4
         + 0.20 × abs(mtf_alignment_score)
         + 0.15 × (1 - abs(close - EMA21) / (2.0 * ATR)).clip(0, 1)   # closer to EMA21 = stronger
         + 0.15 × (1 if retracement is in 40-55% sweet spot)
         + 0.10 × (1 if RSI in correct zone)
         - 0.10 × (1 if 2+ consecutive same-direction pullbacks already entered)
```

**Direction:** Same as trend direction.

**B-tier:**
- **B1 WR:** 0.57 baseline
- **B2 HTF agreement is REQUIRED** — this setup IS trend-continuation by definition; if mtf disagrees, A4 doesn't fire
- **B4 path_shape:** strong continuation prior

**C-tier:**
- **C1 expected_return_H15:** +0.5 × ATR
- **C2 expected_return_H60:** +1.2 × ATR (target: recent swing extreme)
- **C3 path_shape:** continuation 0.60, chop 0.20, revert 0.15, reverse 0.05
- **C4 clears_next_level:** P(reaches recent swing high) ≈ 0.55
- **C5 time_to_target:** 30-50 min

**D-tier:**
- **D1 invalidation:** Close below EMA(21) — that's a structural pullback failure
- **D2 time invalidation:** 45 min from entry with no progress
- **D3 cross-asset:** Trend confirmation in ES required for highest conviction
- **D4 recent failures:** Last 2 A4 in same trend that failed → conviction reduced

**E-tier:**
- **E1 primary:** Bullish reaction candle at EMA touch (long entry): bullish engulfing, hammer, or pin bar
- **E2 secondary:** Close above EMA(8) on next bar
- **E3 entry_trigger:** E1 + E2 fired AND price > entry bar's high
- **E4 disqualifier:** New lower-low below EMA(21) within next 10 bars

---

## 6. A5 — Compression Breakout

**Thesis.** Markets cycle through low-vol consolidation → vol expansion. A run of consecutive bars with decreasing range + stable BOCPD regime followed by an expansion bar (range > 2× recent average) tends to mark the start of a directional move. The compression builds energy; the break releases it.

**Edge source.** Vol regime cycles. Documented in Bollinger Band literature ("squeeze") and Wyckoff distribution/accumulation.

**Detection conditions:**

| Condition | Threshold |
|---|---|
| Compression detected | Last N=20 bars had average range < 0.7 × ATR(60min) average |
| BOCPD stable | bocpd_age_60m > 30 AND bocpd_cp_prob_60m < 0.1 (regime is settled in compression) |
| Vol declining | vol_regime_score < -0.3 (below median vol) |
| Expansion bar | Current bar range > 1.5 × ATR (clean expansion) |
| Directional close | Bar closes in top/bottom 25% of its range (directional, not doji) |

**Strength scoring:**
```
strength = 0.4
         + 0.25 × (compression_intensity).clip(0, 1)             # how compressed vs baseline
         + 0.15 × (expansion_magnitude / 2.5).clip(0, 1)         # how big the expansion bar
         + 0.10 × (1 - close_position_in_bar_centerness)         # closer to extreme = stronger
         + 0.10 × (1 if expansion bar has above-avg volume)
```

**Direction:** Direction of the expansion bar's close relative to its open.

**B-tier:**
- **B1 WR:** 0.55 baseline (modest — compressions can have false breakouts)
- **B2 HTF agreement bonus:** if mtf aligns with expansion direction, +3pp WR
- **B4 path_shape:** continuation prior — once vol cycle expands, it tends to continue

**C-tier:**
- **C1 expected_return_H15:** +0.7 × ATR (decent immediate move expected)
- **C2 expected_return_H60:** +1.5 × ATR (vol expansion has tail)
- **C3 path_shape:** continuation 0.50, chop 0.25, revert 0.20, reverse 0.05
- **C4 clears_next_level:** P(reaches next major anchor) ≈ 0.50
- **C5 time_to_target:** 25-45 min

**D-tier:**
- **D1 invalidation:** Next 2-3 bars retrace > 70% of expansion bar (compression resumes — was a fakeout)
- **D2 time invalidation:** 30 min after expansion bar with no further directional progress
- **D3 cross-asset:** If only NQ expands but ES doesn't → likely false breakout; abstain
- **D4 recent failures:** Last A5 in same compression cycle failed → skip

**E-tier:**
- **E1 primary:** Expansion bar closes near its extreme (top/bottom 25%)
- **E2 secondary:** Next bar makes a higher-high (for long expansion) or lower-low
- **E3 entry_trigger:** E1 + E2 fired AND price > expansion bar high (for long)
- **E4 disqualifier:** Close back inside compression range within 5 bars

---

## 7. A6 — Failed Auction

**Thesis.** Price tests a level multiple times within a short window, fails to extend each time → eventually reverses sharply. This is the textbook "auction failure" pattern from market profile: when the market tries to discover higher (or lower) but fails to find acceptance, it reverses to find acceptance elsewhere.

**Edge source.** Market-profile auction theory — failure to extend a value range is one of the strongest reversal patterns.

**Detection conditions:**

| Condition | Threshold |
|---|---|
| Touch level | A specific price (within tolerance 0.15 × ATR) touched ≥ 3 times within last 60 bars |
| Each touch failed | After each touch, price returned ≥ 0.3 × ATR from the test level within 5 bars |
| Recent test | Most recent touch was within last 10 bars (fresh) |
| Trend context | No clear continuation regime (avoid using A6 in obvious trends — A4 takes those) |
| Level magnitude | The test level is meaningful (near a key_levels entry, OR a previous swing extreme, OR VWAP+1σ band) |

**Strength scoring:**
```
strength = 0.5
         + 0.20 × (touch_count - 3).clip(0, 3) / 3          # more touches = stronger
         + 0.15 × (level_significance)                       # is it at a "real" level
         + 0.10 × (1 - bars_since_last_touch / 10).clip(0, 1)
         + 0.05 × (1 if most recent rejection candle is large)
```

**Direction:** Opposite of the test direction. Test of resistance level (above) → SHORT. Test of support → LONG.

**B-tier:**
- **B1 WR:** 0.62 baseline
- **B2 HTF agreement:** Not required (reversal patterns can fire against HTF)
- **B3 liquidity gravity:** Strong pull away from the rejected level

**C-tier:**
- **C1 expected_return_H15:** +0.5 × ATR
- **C2 expected_return_H60:** +1.3 × ATR (target: opposite side of recent range)
- **C3 path_shape:** reverse 0.45, continuation-of-reversal 0.30, chop 0.20, revert 0.05
- **C4 clears_next_level:** P(reaches mid-range) ≈ 0.60
- **C5 time_to_target:** 30-60 min

**D-tier:**
- **D1 invalidation:** Clean break of the test level (close beyond by ≥ 0.5 × ATR) — the auction finally succeeded
- **D2 time invalidation:** 45 min from entry trigger with no progress
- **D3 cross-asset:** ES showing same failed-auction pattern is strong confirmation
- **D4 recent failures:** Failed A6 at same level recently → conviction drops

**E-tier:**
- **E1 primary:** Rejection candle on most recent touch (long upper wick at resistance, lower wick at support)
- **E2 secondary:** Next bar's close is at least 0.3 × ATR from test level
- **E3 entry_trigger:** E1 + E2 fired
- **E4 disqualifier:** Price re-approaches test level within 5 bars (failed auction is failing)

---

## 8. A7 — End-of-Day Reversion

**Thesis.** In the last 30 minutes of NY RTH, position-squaring + index-rebalance activity tends to drag price back toward session VWAP if price has been extended outside the band. This is positioning-driven, predictable, but small-magnitude.

**Edge source.** End-of-day institutional positioning unwinds. Documented pattern in equity-index futures.

**Detection conditions:**

| Condition | Threshold |
|---|---|
| Time window | Last 30 min of NY RTH (19:30-20:00 UTC EDT, equivalent EST) |
| Extension | vwap_band_position outside ±1.5σ (price is genuinely extended) |
| Trend not strongly confirmed | Avoid A7 if mtf_alignment_score > 0.7 in extension direction (strong trend won't revert) |
| Volume context | Above-average volume in the move that extended (real positioning, not phantom) |

**Strength scoring:**
```
strength = 0.4
         + 0.20 × (abs(vwap_band_position) - 1.5).clip(0, 1)   # more extended = stronger
         + 0.15 × (1 - mins_until_close / 30)                    # closer to close = stronger
         + 0.15 × (1 - abs(mtf_alignment_score) / 1.0).clip(0, 1)  # weaker trend = better
         + 0.10 × (volume_in_extension / avg_volume).clip(0, 1)
```

**Direction:** Toward VWAP. Extension above +1.5σ → SHORT (revert down). Below −1.5σ → LONG.

**B-tier:**
- **B1 WR:** 0.56 baseline (lower than A3/A6 — small-magnitude, time-dependent)
- **B2 HTF agreement:** Counter-trend by design; require WEAK trend (not just absent)
- **B3 liquidity gravity:** Toward VWAP — confirms direction

**C-tier:**
- **C1 expected_return_H15:** +0.4 × ATR
- **C2 expected_return_H60:** ~ same (limited by session close)
- **C3 path_shape:** revert 0.55, chop 0.30, continuation 0.10, reverse 0.05
- **C4 clears_next_level:** P(reaches VWAP) ≈ 0.60
- **C5 time_to_target:** 8-20 min

**D-tier:**
- **D1 invalidation:** Price extends further by 0.5 × ATR (trend strengthening — abandon revert)
- **D2 time invalidation:** Session close (hard cap)
- **D3 cross-asset:** ES showing same extension is a positive signal
- **D4 recent failures:** Skip if 2 A7 already triggered same session

**E-tier:**
- **E1 primary:** A pullback bar toward VWAP (smaller body than recent extension bars)
- **E2 secondary:** Next bar continues toward VWAP
- **E3 entry_trigger:** E1 + E2 fired AND price moved at least 0.2 × ATR toward VWAP
- **E4 disqualifier:** Within next 5 bars, price re-extends beyond entry level

---

## 9. A8 — Multi-Touch Level Defense

**Thesis.** A specific price level that has been touched and defended multiple times within a session (or across days for HTF levels) becomes increasingly significant. When price returns to that level for the Nth time (N ≥ 3), the probability of another defense is higher than the probability of a break. This is auction-theory plus stop-cluster geometry — defenders gather size at proven levels.

**Edge source.** Auction theory (Dalton, Steidlmayer) + practical execution (institutions add liquidity at proven defends).

**Detection conditions:**

| Condition | Threshold |
|---|---|
| Defined level | A specific price (within 0.1 × ATR tolerance) touched and defended ≥ 3 times in last 240 bars |
| Each defense | Price came within 0.1 × ATR of level, then moved away by ≥ 0.4 × ATR within 5 bars |
| Current approach | Price now approaching the level from outside (within 0.3 × ATR but not yet at the level) |
| Level type | Must be a "real" level (key_levels swing, PDH/PDL, VWAP, round number) |
| No structural break | No close beyond level by > 0.3 × ATR in the defending history |

**Strength scoring:**
```
strength = 0.5
         + 0.20 × (touch_count - 3).clip(0, 3) / 3
         + 0.15 × (level_age_bars / 240).clip(0, 1)        # older = more proven
         + 0.10 × (level_significance)                      # PDH > random swing
         + 0.05 × (1 if defending side has cross-asset confirmation)
```

**Direction:** Opposite of current approach direction. Approaching resistance from below → SHORT. Approaching support from above → LONG.

**B-tier:**
- **B1 WR:** 0.60 baseline
- **B2 HTF agreement:** Bonus if HTF doesn't strongly oppose the defense direction
- **B3 liquidity gravity:** Toward the level on approach; AWAY from it on entry

**C-tier:**
- **C1 expected_return_H15:** +0.5 × ATR
- **C2 expected_return_H60:** +1.2 × ATR (target: mid-range or opposite defend level)
- **C3 path_shape:** revert 0.50, continuation-of-revert 0.25, chop 0.20, reverse 0.05
- **C4 clears_next_level:** P(reaches next anchor in revert direction) ≈ 0.55
- **C5 time_to_target:** 25-45 min

**D-tier:**
- **D1 invalidation:** Clean close beyond the defended level (≥ 0.5 × ATR) — the defense finally broke
- **D2 time invalidation:** 35 min from entry trigger with no progress
- **D3 cross-asset:** ES at same equivalent level showing same behavior is strong confirmation
- **D4 recent failures:** If 2+ A8 setups failed at this level recently → break is imminent, skip

**E-tier:**
- **E1 primary:** Approach bar at level shows rejection wick or doji (not a strong continuation through)
- **E2 secondary:** Next bar shows directional close opposite the approach direction
- **E3 entry_trigger:** E1 + E2 fired
- **E4 disqualifier:** Close beyond level by > 0.2 × ATR within 5 bars (defense breaking)

---

## 10. Cross-Setup Arbitration Rules

When multiple setups fire on the same bar, the engine must arbitrate:

### Priority order (highest WR first):
1. **A3 Failed Sweep** (0.62)
2. **A6 Failed Auction** (0.62)
3. **A8 Multi-Touch Defense** (0.60)
4. **A1 ORB** (0.57)
5. **A2 VWAP** (0.57)
6. **A4 Trend Pullback** (0.57)
7. **A7 EOD Reversion** (0.56)
8. **A5 Compression Breakout** (0.55)

### Conflict resolution:
- **Same direction:** combine into a "stacked-confidence" signal — both setups firing same way gets a +2pp WR bonus
- **Opposite direction:** if WR gap > 5pp, take the higher; if ≤ 5pp, ABSTAIN (genuine ambiguity)
- **Same WR within 2pp:** prefer the FRESHER setup (lower mins_since_setup_active)
- **No setup active:** L1-only fallback mode (smaller size, stricter confidence requirement)

### Ambiguous cases that ALWAYS abstain:
- A3 fires LONG while A4 fires SHORT (sweep failure says revert up; trend pullback says continue down)
- A1 fires SHORT while A6 fires LONG (ORB break down vs failed auction at upper level)
- Multiple setups fire but each has strength < 0.5 (weak overall context)

---

## 11. Setup Performance Tracking

Every executed trade is logged with:
- Setup ID that triggered (or "L1_fallback" if no setup)
- Setup strength at entry
- Regime context (F-tier modulators)
- Outcome (win/loss, R-multiple, time-to-resolution)

Rolling 30-trade conditional WR per (setup × regime) combination drives:
- B1 setup_conditional_wr feature (used in L2)
- D4 recent_similar_setups_wr feature (used in conviction adjustment)
- Drift detection: if any setup × regime cell drops > 10pp below baseline, flag for review

This is the **self-aware feedback loop**. Setups that lose their edge get downweighted automatically. Setups that find new edge get upweighted. The engine learns its own performance.

---

## 12. Coverage of the Original 34 Questions

The 8 setups, when fully implemented, answer the predictive questions in the framework redesign:

| Setup | Primary predictive questions answered | Old descriptive Q's used as inputs |
|---|---|---|
| A1 ORB | A1, B1, C1-C5, D1-D4, E1-E4 | Q24 (session_anatomy), Q23 (vol_regime) |
| A2 VWAP | A2, B1, C1-C5, D1-D4, E1-E4 | Q6 (value), Q9 (acceptance), Q19 (regime) |
| A3 Failed Sweep | A3, B1, B3, C1-C5, D1-D4, E1-E4 | Q11 (resting liquidity), Q14, Q15 |
| A4 Trend Pullback | A4, B1, B2, C1-C5, D1-D4, E1-E4 | Q8 (TF control), Q19, Q25 (correlation) |
| A5 Compression | A5, B1, C1-C5, D1-D4, E1-E4 | Q18, Q23, Q19 (regime stability) |
| A6 Failed Auction | A6, B1, C1-C5, D1-D4, E1-E4 | Q7 (balance), Q9, Q11 |
| A7 EOD Reversion | A7, B1, C1-C5, D1-D4, E1-E4 | Q24 (anatomy), Q4 (vol confirming) |
| A8 Level Defense | A8, B1, B3, C1-C5, D1-D4, E1-E4 | Q11, Q13 (extension), Q28 (liquidity asymm) |

**Of 25 truly-predictive questions in the new framework, all 25 are answered by the setup library + downstream heads.** No predictive question is orphaned.

---

## 13. Implementation Plan

1. **Phase 1 — feature families (1 week):** Build 8 `setup_X.py` families in `altus/features/families/`. Each emits 3-5 features per the spec above. Mark `NEEDS_RAW_1M = True` since most need access to raw 1m for level-detection.

2. **Phase 2 — L1 head additions (2 days):**
   - Path-shape softmax head (4-class: revert/continue/chop/reverse) → answers C3
   - Forward return regression heads at H+15 and H+60 → answers C1, C2
   - Level-clearance probability head → answers C4

3. **Phase 3 — L2 hierarchical router (3 days):**
   - Stage 1: setup arbitration (per Section 10)
   - Stage 2: setup-conditional WR prediction (regression on setup × context)
   - Stage 3: hard-veto + soft-modulator gating
   - Stage 4: conformal gate

4. **Phase 4 — L3 setup-aware execution (2 days):**
   - Wait-for-E3-confirmation entries
   - Setup-conditional stops (from D1)
   - Setup-conditional hold times (from C5)
   - Setup-conditional sizing (from B1 conditional WR × G model confidence)

5. **Phase 5 — Performance tracking infrastructure (1 day):**
   - Setup outcome logger
   - Conditional WR computation
   - Drift detection

6. **Phase 6 — Empirical-verification sweep (overnight):**
   - Tier A (setups only)
   - Tier A+B (setups + directional bias)
   - Tier A+B+C (setups + forecasting)
   - Tier A+B+C+D+E (full predictive stack)
   - Each tier must clear OOS lift requirement per [[feedback-empirical-verification]]

---

## 14. What This Spec Locks In

Once approved, the following are LOCKED until empirical results say otherwise:
- The 8 setups (no new ones added without explicit reapproval)
- Detection conditions per setup
- WR estimates (subject to empirical recalibration)
- Arbitration priorities

What can change without re-approval:
- Strength scoring weights (tuning)
- Confirmation strictness (A/B-able)
- R:R targets (vol-scaled execution can adapt these)

This is the [[philosophical-questions-framework]] discipline reapplied: lock the contract, build to it, A/B everything that's not contractual.

---

## 15. Acceptance Criteria

The predictive framework + setup library is empirically validated when:
- ≥ 5 of 8 setups demonstrate OOS conditional WR > 0.53 (clearing breakeven by 2pp)
- ≥ 2 of 8 setups demonstrate OOS conditional WR > 0.58 (clearing strong-system threshold)
- Aggregate cascade (L1+L2+L3 with setup library) shows top-decile WR > 0.55
- Aggregate cascade shows positive expectancy after costs
- L3.1 OOS Sharpe > 0.5
- Maximum DD < 12%

If we don't hit these, the response isn't "tune the same setups" — it's "find different setups, or look outside 1m granularity, or pursue tick data." The framework's value is that it gives us crisp answers about WHERE the edge is or isn't.

---

*This document is canonical. ARCHITECTURE.md Section 3 will reference it. The 34-question framework remains in history but is superseded by the predictive question framework defined in FRAMEWORK.md (companion document).*
