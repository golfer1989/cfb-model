# How to use this

## The short version

Double-click **`CFB_Report.exe`**. It asks how many days ahead to cover
(press Enter for 5), pulls live lines / rosters / injury reports for those
games, runs the model, and opens an HTML report in your browser.

The report is also saved to `data\report.html`.

---

## What the report shows, column by column

| Column | Meaning |
|---|---|
| **Away Pts / Home Pts** | Mean simulated score for each team |
| **Est Total** | Projected total points, shrunk toward the league mean |
| **Model Line** | The model's spread, quoted from the home side |
| **Market Line** | The live sportsbook line at the moment you ran it |
| **Edge** | Model margin minus market margin. Positive favours the home side |
| **Model %** | *If the projection is right*, how often that side covers. Tracks Edge directly |
| **Adj %** (bold, with a "lo" subline) | Model % after adjusting for how often the model's disagreements have actually been right. The subline is the 95% lower bound. **This is the only number with money implications** |
| **Spread Bet** | Which side the model prefers against the market number |
| **Market O/U / O/U Bet / Model % / Adj %** | Same two views, for the total |
| **Call** | AVOID when the projection itself is unreliable (non-FBS opponent, newly promoted team); PLAY / LEAN if the calibrated edge ever clears break-even; "ok" = NO BET, the normal state |

### Model % vs Adj % — the most important distinction in the project

They answer different questions, and the gap between them is the whole story.

**Model %** is the simulator's own view: *if this projection is correct*, how
often does this side cover? It moves with the Edge column, as it should —
a +2 edge is ~55%, +5.4 is ~63%, +10 is ~73%.

**Adj %** asks the harder question: historically, when this model disagreed
with a closing line by this much, **how often was it actually right?**

Right now Adj % reads 50% on spreads. That is not a placeholder — it is the
measured answer. After the 2026-08 harness fixes the relationship is stronger
than before (and passes pooled significance under every variance estimator),
but it has still never been confirmed on a season it was not fitted to:

```
2024:   beta = +0.117,  t = +1.29     not significant alone
2025:   beta = +0.217,  t = +2.20     significant
pooled: beta = +0.162,  t = +2.43     passes pooled -- but that is not enough
```

The validation gate requires the PRIOR seasons to carry the effect on their
own before the holdout is allowed to confirm it. 2024 alone does not clear
the bar, so the honest adjusted answer is still a coin flip. 2026 is the live
test.

**Bet off Adj %. Read Model % for the model's own conviction.**

If 2026 makes the effect repeat, the validation gate passes on its own and
Adj % starts tracking the edge — no code change required.

The gap between those two numbers is large. The raw simulator says a 5-point
disagreement is a 62% proposition. Measured against real closing lines, the
shrinkage factor on clean data is **not statistically distinguishable from
zero** — so the honest market-calibrated answer is 50%, no matter how big the
raw disagreement looks.

---

## Reading the calls honestly

- **PLAY** — the market-calibrated probability clears the 52.38% break-even at
  standard −110 juice, with margin.
- **LEAN** — clears break-even but sits inside historical noise.
- **NO BET** — below break-even. Most games land here, and that is the correct
  outcome for a model competing against an efficient market.
- **AVOID** — a team is rated from a prior rather than real results (a newly
  promoted FBS team), or the opponent is non-FBS. All 106 non-FBS teams share a
  single pooled rating, so the model has no team-specific view of them while
  the market prices each individually. Left ungated, 30% of FCS games earned
  PLAY against 10% of FBS games — the tool steering hardest toward what it
  understands least.

**On clean data both shrinkage factors are zero, so in practice every call
currently reads NO BET or AVOID.** The PLAY and LEAN tiers only activate if a
future refit finds a statistically significant edge.

**Break-even is 52.38%, not 50%.** At −110 you risk $110 to win $100, so a
coin-flip bettor loses money steadily.

---

## What the model does and does not use

**Moves the predicted number**
- Opponent-adjusted offence/defence ratings (ridge-regularised, λ and recency
  decay chosen by walk-forward cross-validation)
- Home-field advantage, fitted as a regression coefficient (~3.2–3.5 pts)
- Starting-quarterback adjustment when the expected starter is listed out

**Shown but deliberately NOT moving the number**
- Pass-offence vs pass-defence style matchup. This was tested directly against
  out-of-sample residuals: t = −0.39, R² = 0.0001. All style-interaction terms
  together changed the residual standard deviation by 0.03 points. It is
  noise, so it is reported as context only.

---

## Measured performance

Walk-forward over 1,853 games, ordered and blocked by **kickoff date**, every
prediction made using only games already played (updated 2026-08-18):

| Metric | Value |
|---|---|
| Margin RMSE / MAE | 15.80 / 12.57 |
| Straight-up winner accuracy | **74.3%** |
| Total RMSE (calibrated, held-out 2025, n=934) | 15.75 |
| Correlation with ESPN FPI | **+0.995** |

**Against the market: no demonstrable edge.** Over ~1,800 games with real
closing lines, the model's margin RMSE was 15.75 against the market's 15.12.
Both edge-shrinkage factors are gated to zero, so every spread and total
call reads NO BET. (Encouraging but not yet actionable: from week 9 onward
the model and the closing line are effectively tied, and the edge-vs-market
relationship now passes pooled significance -- see VERIFICATION.md for why
the gate still, correctly, blocks.)

An earlier version of this file claimed a 52-54% ATS edge. That was wrong.
It came from a look-ahead leak: ESPN files postseason games as `week = 1`, and
the backtest ordered by (season, week), so the January national championship
was in the training set for week 2 of its own season. With date ordering the
edge disappears (ATS 53.69% -> 51.57%, shrinkage t 3.29 -> 1.96). The totals
edge that survived turned out to be market-fade, not model signal. See
VERIFICATION.md.

## What this tool is good for

It is a **projection tool**, not a betting system:

- credible score, spread and total projections for every game
- a side-by-side view of where the model and the market disagree, and by how
  much, with the disagreement honestly discounted
- opponent-adjusted offence/defence ratings, strength of schedule and
  conference strength that track ESPN's FPI at +0.99
- injury and starting-QB context pulled live before kickoff

What it will not do is tell you which side to bet, because on clean data it
has not been shown to beat a closing line.

## Running it more often than weekly

The `.exe` only refreshes the schedule weeks that overlap your chosen window,
so running it daily is cheap. Run it close to kickoff to get the sharpest
line and a final injury report.

For a specific week rather than a date window:

```bash
python run_report.py --week 3 --open
```

---

## Rebuilding the data

Occasionally you will want to refresh the underlying model:

```bash
python fetch_espn_data.py          # results + schedule
python fetch_team_stats.py         # per-team season splits
python backtest.py                 # revalidate + recalibrate the simulator
python fetch_odds.py               # historical closing lines
python evaluate_vs_market.py       # refit the edge-shrinkage factor
```

The last two matter most: without them the report has no way to know how much
of its disagreement with the market is real, and every call reads NO BET.

---


## New in the 2026-08-18 build

- **First-year-head-coach adjustment** (-1.5 margin, both backtested and live;
  needs `data/cfbd_coaches_<year>.csv` from `fetch_cfbd.py`; silently off
  without them). The market already prices this -- it sharpens projections,
  not bets.
- **FCS opponent ratings (optional, two commands):**
  `python fetch_espn_data.py --fcs` downloads full FCS-vs-FCS schedules, then
  `python backtest.py --fcs-ab` measures whether training on them helps and
  turns it on only if it wins. Motivation: with one pooled FCS rating the
  model loses to the market by 2.19 RMSE on those games vs 0.57 elsewhere.
- **After changing data or code, rebuild the exe:** `python build_exe.py`
  (the shipped CFB_Report.exe/CFB_Results.exe otherwise keep running the old
  model with the old lambda and the old odds file).

## Honest limitations

1. **The model does not beat the closing line.** On the fixed harness the
   spread shrinkage is beta = 0.162 (t = 2.43 pooled, but 2024 alone fits at
   t = 1.29, so the validation gate blocks) and the totals factor, once
   market-fade is removed, is beta = 0.09 (t = 0.58). Both are gated to zero.
1b. **n = ~1,800 is about an order of magnitude too small** to detect an edge of
   the size in question. Reliable detection needs roughly 15,000 bets, about 17
   seasons. This is a structural limit, not something another season fixes.
2. **The QB adjustment is a prior, not a fitted value.** −4.5 points is a
   documented middle estimate; calibrating it properly needs in-season data.
   It is labelled low-confidence wherever it appears.
3. **Newly promoted FBS teams are rated from n=3.** North Dakota State for
   2026 is the live case. Flagged AVOID.
4. **Injury data is only as good as ESPN's feed**, which is sparse in the
   preseason and improves once games begin.
5. **Totals carry modest signal** — R² ≈ 0.05 after calibration. They beat
   guessing the average, but not by much.
