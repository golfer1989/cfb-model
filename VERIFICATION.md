# Verification record

Reproduce the automated portion with `python verify_math.py` (77 checks).

---

## 2026-08-18 fix round (external audit)

An independent audit reproduced every number in this file from the raw CSVs
(all 122 calibration values bit-for-bit; an independently written ridge
implementation matched the ratings at r = 0.9998) and found no calculation
errors -- but several defects in the harness and presentation. All fixed:

| Fix | Effect |
|---|---|
| **Walk-forward blocked by ESPN week labels**: the 92 Dec/Jan postseason games carrying `week==1` landed in the FIRST block of their season and were graded off preseason ratings (RMSE 16.5 / 52% SU on those games, and +16.8pp of fake overconfidence in their win probabilities). Blocks are now 7-day windows anchored on each season's first kickoff. | Headline RMSE 15.96 -> **15.80**, straight-up 73.7% -> **74.3%**, and the bowl calibration artifact is gone |
| **First-year-head-coach adjustment added** (-1.5 margin, applied half per side, totals untouched): the ONE situational factor of nine tested that replicated in both seasons and in both directions of a fit-one/apply-other holdout. The market prices it (t = -0.66 incremental to the close), so it improves projections, not bets. | ~-0.03 RMSE; 80 flagged teams in 2024-2025 |
| **Report showed only the simulator's conviction**: the redesigned table bolded Model % (the number this project's own first rule says not to bet) and computed-but-never-rendered the market-calibrated Adj % and the Call. Adj % is back, bolded, with its 95% lower bound; Call pills (AVOID / PLAY / LEAN) surface per-row; HOW_TO_USE.md matches the table again. | Display honesty |
| **`conservative_cover_prob` bypassed the validation gate** (read raw beta; safe only through check ordering in the caller). It now returns 0.5 whenever the gate is blocked. | Gate enforced on both probability paths |
| **Gate significance was estimator-dependent**: classical t = 1.90 vs HC1-robust 2.07 / clustered-by-week 2.09 on the same data -- the verdict flipped at the threshold with the estimator. All three are now computed and the gate judges the WEAKEST. | Estimator-proof gate |
| **Drive simulator derived TD probability at 7.0 points but paid 6 + Bernoulli(0.95)** -- a -0.31 mean bias at 45-point projections. `TD_PTS = 6.95` everywhere. | Sim mean bias within +/-0.2 across the range |
| **`target_sigma` constants were ~21% low** and only correct after the blend rescale. Refitted to the residuals directly (8.14 + 0.117x); the blend stays as a guard. | Honest standalone function |
| **FCS opponents share one pooled rating** while the market prices each individually: the model loses to the close by 2.19 RMSE on FCS-involved games vs 0.57 on FBS-vs-FBS (~-0.20 RMSE available). New: `fetch_espn_data.py --fcs` pulls full FCS-vs-FCS schedules (ESPN group 81), and `backtest.py --fcs-ab` A/B-tests training augmentation and persists the verdict; nothing enables without winning that test. Promotion detection now counts FBS-LEVEL games so a promoted team keeps its fitted FCS rating (and its AVOID flag). | Pending the data fetch + A/B on this machine |
| Stale numbers in docs and docstrings (pre-leak-fix beta = 0.188 / t = 3.29 still quoted in four places; HFA "consensus 3.5-4" contradicted by nine published estimates spanning 2.4-4.1, most 2.5-3.2; power-table entry 3,464 should be ~2,330; total RMSE 15.78 is n = 934, not 1,853). | All corrected |

**Post-fix, the strongest honest statement of the model-vs-market picture:**
the fixed harness STRENGTHENED the measured edge relationship (pooled beta
0.119 -> 0.162, t = 2.43 classical / 2.60 HC1 / 2.72 clustered -- the corrupt
bowl rows had been diluting it), 2024 and 2025 now agree in sign
(0.117 / 0.217), and the pooled check passes under every estimator. The gate
STILL blocks, correctly, because 2024 alone fits at t = 1.29: the effect has
never been confirmed on a season it was not fitted to. 2026 is the live test:
if it confirms at |t| >= 2, the gate opens on its own and the report starts
quoting real probabilities -- no code change required.

---

## THE HEADLINE CORRECTION

An adversarial audit found a **look-ahead leak that invalidated this project's
original betting result**. Everything below reflects the corrected numbers.

ESPN numbers postseason weeks from 1. The fetcher stored that number verbatim,
so **135 bowl and playoff games are filed as `week = 1`** — including the
national championship (Miami at Indiana, played 2026-01-20, stored as
`season=2025, week=1`). The backtest ordered its walk-forward by
`season × 100 + week`, so predicting **week 2 of 2025** used a game played
**five months later** as training data. Every prediction from week 2 onward,
in every season, was affected.

| | Leaked (original) | Clean (date-ordered) |
|---|---|---|
| Model margin RMSE | 15.90 | **16.02** |
| Market margin RMSE | 15.10 | 15.10 |
| ATS at edge ≥2 | 53.69% | **51.57%** |
| Incremental-information t | 3.47 | **2.11** |
| Spread shrinkage β | 0.188 (t = 3.29) | **0.115 (t = 1.96), not significant** |
| **Spread shrink factor** | 0.188 | **0.000** |

The walk-forward is now keyed on **kickoff date**, with a hard assertion that
no training game kicks off at or after the first game being predicted.

## The second correction: the totals edge was market-fade

The totals factor survived the leak fix at β = 0.300 (t = 3.34). It was not a
model edge. The calibrated total is shrunk so hard toward the league mean
(slope ≈ 0.45) that the model's contribution has sd 3.8 against a market-total
sd of 6.4, so the "edge" is mostly the negated market number:

- correlation with pure market-fade: **+0.82**
- correlation with the model's own projection: **−0.29**
- fading extreme totals with **no model input at all**: t = **+3.73**, higher
  than the "model" statistic
- both terms together: fade t = +2.71, **model t = +0.50**
- driven entirely by 2024 (t = 3.21) vs 2025 (t = 1.38) — and
  `total_calibration` was itself fitted on 2024

After orthogonalizing the edge against market-fade, the model's marginal
contribution is β = 0.078 (t = 0.50). **Totals shrink factor: 0.000.**

---

## Current measured performance

**Game prediction** (walk-forward, date-ordered blocks, 1,853 games):

| Metric | Value |
|---|---|
| Margin RMSE / MAE | 15.80 / 12.57 |
| Straight-up accuracy | **74.3%** |
| Total RMSE (calibrated, n=934 held-out 2025) | 15.75 |
| Score residual σ (out-of-sample) | 11.34 |
| Win-probability Brier / log-loss | 0.164 / 0.486 |
| Correlation with ESPN FPI | **+0.995** |

Context for the RMSE: against the market on the same games the closing line
scores 15.12, and from week 9 of a season onward the model and the closing
line are effectively tied (gap +0.00 over ~700 games). The deficit is
concentrated in weeks 1-4 (preseason information the ratings cannot see) and
FCS-involved games (see the fix round above).

**Betting: no demonstrable edge.** Both shrinkage factors are zero, so every
spread and total call reads 50% and NO BET.

---

## Automated suite — 55 checks

| Area | What is verified |
|---|---|
| Ridge solver | Matches closed form to 5.6e-17; shrinkage monotone in λ; HFA unpenalized; singular systems degrade to `lstsq` rather than crashing |
| Design matrix | `+1` own offence, `−1` opponent defence; HFA `±½`, `0` at neutral; margin identity to 7e-15 |
| Sign conventions | Favourite lays points; HFA helps home; market `−7.5` ⇒ home favoured by 7.5 |
| Simulator | Means track expectations; ties resolved; probabilities sum to 1 and rise monotonically with margin |
| Calibration | Shrink slope in (0,1); fixed point = league mean; factors survive `load_calibration()`; margin and total fitted independently |
| **Leakage** | Ordering uses **kickoff dates**; includes a regression test proving a `(season, week)` key would leak 46 future games |
| Data integrity | No duplicates or missing scores; every `team_id` maps to one canonical name |

### A test that passed while the bug was live

The original leakage check built `train = games[key < cutoff]` and then
asserted no row had `key >= cutoff` — true by construction of the filter it had
just applied. It never touched the date column. It passed throughout. The
replacement uses kickoff dates and includes an explicit regression test that
**fails if the old ordering would have leaked**.

---

## Defects found and fixed

| Defect | Impact |
|---|---|
| **Postseason games filed as week 1** | Look-ahead leak; invalidated the entire betting result |
| **Totals edge was market-fade** | Sold an unvalidated strategy under the model's name |
| Home-field from a raw mean | 8.58 pts vs 3.3 fitted — ~5-point bias on every spread |
| Simulator fed in-sample σ | Understated variance 14.8%; overconfident probabilities |
| Totals unshrunk | Worse than guessing the league average |
| `resid_sd` hardcoded 13.0 | Fitted value 15.04; inflated every probability |
| Intercept α dropped + `abs(edge)` | Away picks overstated ~3 pts; recommended below break-even |
| `load_calibration()` whitelist | Silently dropped shrinkage keys, disabling all calibration |
| **`launcher._refresh_team_stats` rebuilt `team_stats.csv` from scratch** | Deleted every season it did not refetch. Destroyed 2023 and 2024 in production on 2026-08-18: 403 rows → 136. Recovered only because `data/raw_cache/` happened to still hold the JSONs. Now merges by `(season, team_id)`, and refuses to write at all if a refresh would drop a season |
| Appalachian State rated twice | Mid-window rename split it into two half-sampled entities |
| North Dakota State pooled as FCS | ~15-point error across 12 games |
| FCS games flagged PLAY 3× as often | Steered hardest toward what the model understands least |
| Low-score projections | Model said 0.5 pts where teams actually score 7.5 |

## Changes tested and REJECTED

- **Adaptive early-season decay** — worse on every slice, including early-season
- **Two-tier FCS pooling** — descriptively compelling (16.3-pt gap), predictively
  worthless; the split keys on a max, so the gap is selection
- **Margin recalibration** — 0.02 RMSE, inside noise
- **Style-matchup in the mean** — t = −0.39, R² = 0.0001; diagnostic only
- **AVOID above 12 points** — an invented rule; those bets went 49-43

### Rejected after full box-score, subset and line-movement studies

All three studies used the date-ordered harness and reported out-of-sample.
**Nothing from any of them was added to the model.**

- **Every box-score efficiency feature** (~50 tested on complete ESPN box scores
  for all 2,764 games). The apparent signal is one thing seen through correlated
  proxies (r = 0.65–0.98): the ratings mildly over-rate teams that were good last
  season. Last season's raw **point** margin — not a box-score feature — scores
  t = −3.55 by itself and explains the rest away; 11 of 12 features drop to
  |t| < 1.4 once it is controlled, and the survivor fails Benjamini-Hochberg
  (q = 0.11). Best honest 2-feature version: −0.067 RMSE (0.43%) but ATS gets
  **worse**, 50.94% → 49.45%. A 7-feature version is +0.093 RMSE — overfitting
  shown directly out of sample.
- **Lowering the recency decay** to capture that over-rating. Direct test through
  `backtest.walk_forward`: RMSE by decay 0.05 = 16.3305, 0.10 = 16.1246,
  **0.15 = 16.0636**, **0.20 = 16.0637 (production)**, 0.25 = 16.0950,
  0.30 = 16.1431. The model is already at the optimum; 0.15 beats production by
  0.0001 RMSE. **Decay stays at 0.20.**
- **Season-to-date efficiency** (both teams 4+ games). The most decisive zero in
  the study: every opponent-adjusted feature |t| ≤ 1.02 and every walk-forward
  delta positive (worse). Structural — the ratings are already fitted to those
  games' scores. Do not revisit.
- **Turnover and fumble margin.** Verified rather than assumed: within-season
  split-half reliability 0.213 and **−0.052**. Fumble recovery is literally
  random on this data.
- **Third-down rate.** The opposite of the folklore — it *is* stable
  (r_yoy = 0.420, reliability 0.572) and still has no incremental value
  (t = −0.60), because its stable component is just general efficiency.
- **Luck-stripped scoring.** Incremental t = +1.91 with the sign pointing the
  **wrong way**; prior *point* margin carries forward better (r = 0.536) than
  prior *yard* margin (0.453).
- **All 139 conditional bet triggers** (conference, spread band, rest/bye,
  line-movement size, season timing, …). Best raw result — Sun Belt at
  |edge| ≥ 3, 61.8% ATS, p = 0.0058 — becomes Bonferroni p = 0.81, BH q = 0.72,
  Westfall-Young family-wise **p = 0.335**. Ranking subsets on 2024 and testing
  on 2025 gave mean β = +0.198 against +0.206 unconditionally: **selection
  carried zero information**. Only 6 of 48 subsets cleared breakeven in both
  seasons where noise predicts ~12 — persistence is *worse* than chance.
- **Line movement as a feature.** Priced in by the close. Walk-forward, adding
  `move` to [close + model] changes RMSE by **0.0004** (t = +0.04). Blind steam
  following is 50.03% over 1,801 games and *degrades* as the move grows (46.6%
  at |move| ≥ 4).
- **Ungating the spread edge on CLV.** Measured CLV is +0.22 to +0.27 pts against
  the **0.90 pts** needed to break even at −110, it exists only in 2025
  (2024 t = −0.52), and betting the opening spread on model edge goes 49.72%.

Retained as **diagnostics only**: the closing line genuinely beats the opening
line (+0.124 RMSE spread, +0.106 total), and the model anticipates where the
*total* will move (β = +0.094, t ≈ +9, in both seasons, surviving fixed effects
and drift-correction). Neither is a bet — the model's own total is a **worse**
number than the opener it disagrees with (RMSE 16.62 vs 15.86).

---

## The statistical power floor

Recorded here so it stops being rediscovered. sd(actual margin − closing line)
= **15.08 points**.

| Question | Bets needed at 80% power |
|---|---|
| Detect a true **2-point** edge (= 55.28% ATS) | **705** |
| …carrying a k = 139 Bonferroni penalty | 1,733 |
| Detect a true **1-point** edge | 2,806 (6,903 corrected) |
| Confirm you beat the 52.38% breakeven | **~2,330** (corrected 2026-08; the earlier 3,464 did not reproduce under the standard one-sample power formula) |
| Detect a 2-percentage-point ATS edge (52.0%) | 4,906 |

The entire dataset supplies **936 bets** at |edge| ≥ 3. The largest subsets
supply 475–860; the interesting ones 130–400.

**Any proposed filter with fewer than ~700 qualifying games cannot be validated
on this data, no matter how good its backtest looks.** This is what makes subset
hunting futile here *in advance*, rather than something to discover after the
fact.

The same floor applies to rating features: at n = 1,571 with residual SD 15.90,
the minimum detectable effect is 1.12 points per SD, which is worth only 0.040
RMSE. Box scores are not where the remaining ~1.0 RMSE gap to the market lives.

The honest asymmetry: at n = 936 the 95% CI half-width on ATS% is ±3.2pp, so the
observed 51.4% is consistent with anything up to 53.2%. A real 1-point edge
**cannot be ruled out** — it equally cannot be detected with this much data, and
certainly cannot be located inside a subset.

---

## The standing validation gate

`evaluate_vs_market.py` no longer lets a pooled t-statistic alone open the bet
path. Post-fix (2026-08), the pooled spread β = +0.162 with t = +2.43
classical / +2.60 HC1-robust / +2.72 clustered-by-week — significance is
judged on the weakest of the three, and the pooled check now PASSES. It is
still not one confirmed effect:

| | β | t |
|---|---|---|
| 2024 | +0.117 | +1.29 |
| 2025 | +0.217 | +2.20 |
| pooled | +0.162 | +2.43 |

The prior-seasons fit (2024 alone) fails at t = 1.29, so the gate blocks and
the shrink factor stays 0. That is the design working: a pooled t is not
evidence here, because selection on 2024 was measured to carry **zero**
information about 2025. If 2026 confirms at |t| ≥ 2 against a 2024+2025 prior
fit, the gate opens on its own.

`validation_gate()` requires **all four**: pooled significant, the
**prior-seasons** fit significant, the **held-out** latest season confirming at
|t| ≥ 2 with matching sign, and the same sign in every season. Failing forces
`shrink_factor` to 0 — it can only ever fail *safe*.

Verified on synthetic data that it is a gate and not a hardcoded zero:

| Scenario | Old behaviour | Gated |
|---|---|---|
| Two flat seasons + one strong (pooled t = 3.03) | ships 0.217 | **0.000 — blocked** |
| Genuine edge in every season | ships 0.441 | 0.441 — allowed |
| No edge anywhere | 0.000 | 0.000 |

Deliberately strict: if 2026 replicates 2025, the prior fit is t = +1.96 and the
gate still blocks on a 1.96-vs-2.00 technicality. That asymmetry is intended —
a false negative costs a season of not betting, a false positive costs money
staked on an artifact.

---

## Honest limits

1. **No demonstrable betting edge.** Use this as a projection tool.
2. **n = 1,785 is roughly an order of magnitude too small** to detect an edge of
   the size being discussed. Reliable detection needs ~15,000 bets (~17 seasons).
   This is structural, not a data-collection gap.
3. The QB adjustment is a prior (−4.5 pts), not a fitted value.
4. Promoted-team prior rests on n = 3.
5. Totals carry R² ≈ 0.05 even after calibration.
