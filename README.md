# College Football Simulator

A small, dependency-light (numpy + pandas) engine that rates every team on
separate **offense** and **defense** scales, folds in **strength of
schedule** and **conference strength**, and then Monte Carlo simulates
individual games or a full upcoming schedule.

## Quick start

```bash
pip install -r requirements.txt

# (optional) regenerate the bundled synthetic demo data
python generate_sample_data.py

# full power ratings, sorted best to worst
python cfb_sim.py ratings

# conference strength ratings
python cfb_sim.py conferences

# simulate one game (20,000 Monte Carlo trials by default)
python cfb_sim.py predict "Georgia" "Alabama" --site home_a

# simulate an upcoming week/schedule -> projected win totals
python cfb_sim.py season --schedule data/sample_schedule.csv
```

Run `python cfb_sim.py -h` (or `<command> -h`) for all flags.

**⚠️ The bundled `data/sample_games.csv` is synthetic.** Team and conference
names are real, but the scores were generated from made-up "true skill"
numbers plus random noise (see `generate_sample_data.py`), purely so the
tool works out of the box and you can see the engine recover sensible
rankings. Swap in real results before trusting any prediction — see
**Using your own data** below.

## How the model works

### 1. Separate offense/defense ratings, opponent-adjusted (this *is* the SOS correction)

Every game contributes two rows of data: how many points the home team
scored, and how many the away team scored. The engine fits, across **all**
games simultaneously via alternating least squares:

```
points_scored(team) ≈ league_avg + OFF[team] − DEF[opponent] + home_field_adj
```

`OFF[team]` and `DEF[team]` are solved iteratively (same family as
Massey/SRS-style rating systems): hold defense ratings fixed and solve for
the offense rating that best explains each team's scoring output, then hold
offense fixed and solve for defense, repeat until convergence (default 60
passes, converges in practice within ~20-30).

Because a team's rating is fit *simultaneously* with every opponent's
rating, strength of schedule isn't a separate bolt-on stat — it's baked
into the fitting process. A team that scores 40 a game against bad defenses
gets a lower OFF rating than a team that scores 40 a game against good
ones; a defense that allows 17 a game against explosive offenses gets more
credit than one that allows 17 against three-and-out offenses.

### 2. Power, SOS, and conference strength are derived outputs

- **POWER[team] = OFF[team] + DEF[team]** — expected scoring margin against
  a league-average team on a neutral field.
- **SOS[team]** — the average POWER rating of the opponents a team actually
  played. Purely a reporting stat (the adjustment already happened in step
  1); useful for explaining *why* a team's rating looks the way it does.
- **Conference strength** — the average POWER rating of a conference's
  member teams.

### 3. Game prediction + Monte Carlo simulation

Expected score for a game:

```
expected_home = league_avg + OFF[home] − DEF[away] + HFA/2
expected_away = league_avg + OFF[away] − DEF[home] − HFA/2
```

`HFA` (home field advantage, in points) is estimated straight from your
data as the average home-minus-away margin in non-neutral games (override
with `--hfa`).

Each game is then simulated thousands of times (`--sims`, default 20,000)
by drawing both teams' scores from a normal distribution centered on the
expected score, with a standard deviation equal to the model's own
historical residual error (i.e. "how wrong was this model on the games it
was trained on" — a realistic proxy for game-to-game variance). Win
probability, mean/median score, projected margin, and an 80% margin range
are all computed from the simulated distribution, not just the point
estimate — that's the difference between "who's favored" and "by how much
and how confident are we."

`season` runs the same simulation across a whole list of games, thousands
of full-slate trials at once, and reports each team's projected win
distribution (mean/median/10th/90th percentile) — the same style of output
as an FPI/SP+ style "projected standings."

## Using your own data

Replace `data/sample_games.csv` with real results. Required columns:

| column | meaning |
|---|---|
| `date` | any string, not used in the math (kept for your reference) |
| `home_team` | home team name |
| `home_conf` | home team's conference |
| `away_team` | away team name |
| `away_conf` | away team's conference |
| `home_score` | final home score |
| `away_score` | final away score |
| `neutral_site` | `True`/`False` — was this at a neutral site? |

Good sources for real data: ESPN game logs, [sports-reference.com/cfb](https://www.sports-reference.com/cfb/),
or your conference/athletic department's own stats exports. More games
(especially more cross-conference games) → tighter, more reliable ratings.

For `season`, build a schedule CSV with `home_team, away_team, neutral_site`
for the games you want projected (`data/sample_schedule.csv` is an example).

## Tuning

- `--iterations N` — rating solver passes (default 60; the RMSE printed
  with `--verbose` should flatten out — if it's still dropping, use more).
- `--hfa N` — override the estimated home-field-advantage points.
- `--sims N` — Monte Carlo trial count (more = smoother probabilities,
  slower).

## Known simplifications

- Scores are simulated independently per team (no explicit correlation
  between the two teams' variance) — a common simplification for this
  class of model.
- No injury/weather/travel/momentum adjustments — this is a pure
  points-based statistical model, not a substitute for scouting.
- Ties are impossible in real college football (OT); the rare simulated
  tie is split 50/50 as a stand-in for overtime rather than modeled
  explicitly.
