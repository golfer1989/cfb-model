#!/usr/bin/env python3
"""
College Football Simulator
===========================
Rates every team on separate OFFENSE and DEFENSE efficiency scales,
derives strength-of-schedule and conference strength as outputs of
that rating system, then Monte Carlo simulates games / schedules.

Methodology (see README.md for full detail):
  1. Every game contributes two rows: (team, opponent, points_scored, site).
  2. Solve, via alternating least squares over all games simultaneously:
         points_scored(team) ~= league_avg + OFF[team] - DEF[opponent] + home_field_adj
     This is the same family of model used by Massey/SRS-style rating
     systems: it separates a team's scoring output (OFF) from what its
     opponent's defense allows (DEF), so a gaudy offensive number racked
     up against bad defenses gets discounted automatically, and a stingy
     defensive number gets rewarded more if it happened against good
     offenses. That opponent-adjustment IS the strength-of-schedule
     correction -- it happens inside the rating, not as a bolt-on.
  3. POWER = OFF + DEF (net points vs. a league-average team on a neutral
     field). SOS[team] = average POWER of teams actually played.
     Conference strength = average POWER of a conference's members.
  4. Game prediction = league_avg + OFF[home] - DEF[away] + HFA (and
     mirrored for the away score). Games are simulated thousands of times
     by drawing from a normal distribution centered on that prediction,
     with a standard deviation estimated from the model's own historical
     residuals -- so the noise level reflects how volatile real college
     football scoring actually is.
"""

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_ITERATIONS = 60
DEFAULT_SIMS = 20000
DEFAULT_HFA_FALLBACK = 2.5  # points, used only if it can't be estimated from data
DEFAULT_SEASON_DECAY = 0.55  # per-year-back weight decay; roster turnover makes older seasons less predictive
RNG_SEED = None  # set an int for reproducible simulations


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_GAME_COLS = [
    "date", "home_team", "home_conf", "away_team", "away_conf",
    "home_score", "away_score", "neutral_site",
]


def load_games(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_GAME_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"games file '{path}' is missing required columns: {missing}\n"
            f"Required columns: {REQUIRED_GAME_COLS}"
        )
    df["neutral_site"] = df["neutral_site"].astype(bool)
    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)
    return df


def team_conference_map(games):
    m = {}
    for _, r in games.iterrows():
        m[r["home_team"]] = r["home_conf"]
        m[r["away_team"]] = r["away_conf"]
    return m


def season_weights(games, current_season=None, decay=DEFAULT_SEASON_DECAY):
    """Weight each game by recency: current season = 1.0, each prior season
    decays by `decay`. Roster turnover means a 3-year-old result is a much
    weaker signal than one from this season. Returns an all-ones Series if
    the games file has no `season` column (e.g. hand-built data)."""
    if "season" not in games.columns:
        return pd.Series(1.0, index=games.index)
    cur = current_season if current_season is not None else int(games["season"].max())
    years_back = (cur - games["season"]).clip(lower=0)
    return decay ** years_back


def expand_rows(games, weights=None):
    """One row per team-performance (each game -> 2 rows)."""
    if weights is None:
        weights = pd.Series(1.0, index=games.index)
    home = pd.DataFrame({
        "team": games["home_team"],
        "opp": games["away_team"],
        "points_for": games["home_score"],
        "points_against": games["away_score"],
        "hfa_sign": np.where(games["neutral_site"], 0.0, 1.0),
        "weight": weights.values,
    })
    away = pd.DataFrame({
        "team": games["away_team"],
        "opp": games["home_team"],
        "points_for": games["away_score"],
        "points_against": games["home_score"],
        "hfa_sign": np.where(games["neutral_site"], 0.0, -1.0),
        "weight": weights.values,
    })
    return pd.concat([home, away], ignore_index=True)


def _weighted_groupby_mean(values, weights, keys, all_keys):
    """Weighted mean of `values` grouped by `keys`, filled with 0 for any
    key in all_keys with no rows (keeps every team present every pass)."""
    wsum = pd.Series(values * weights, index=keys).groupby(level=0).sum()
    wtot = pd.Series(weights, index=keys).groupby(level=0).sum()
    means = (wsum / wtot).reindex(all_keys)
    return means.fillna(0.0)


# ---------------------------------------------------------------------------
# Rating engine
# ---------------------------------------------------------------------------

@dataclass
class RatingResult:
    off: dict = field(default_factory=dict)
    defn: dict = field(default_factory=dict)
    power: dict = field(default_factory=dict)
    sos: dict = field(default_factory=dict)
    conf_strength: dict = field(default_factory=dict)
    league_avg: float = 0.0
    hfa: float = DEFAULT_HFA_FALLBACK
    resid_std: float = 14.0
    teams: list = field(default_factory=list)


def estimate_hfa(games, weights=None):
    non_neutral = games[~games["neutral_site"]]
    if len(non_neutral) < 5:
        return DEFAULT_HFA_FALLBACK
    margin = non_neutral["home_score"] - non_neutral["away_score"]
    if weights is None:
        return float(margin.mean())
    w = weights.loc[non_neutral.index]
    return float((margin * w).sum() / w.sum())


def fit_ratings(games, iterations=DEFAULT_ITERATIONS, hfa=None, verbose=False,
                 current_season=None, decay=DEFAULT_SEASON_DECAY):
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    weights = season_weights(games, current_season=current_season, decay=decay)
    rows = expand_rows(games, weights=weights)
    league_avg = float(pd.concat([games["home_score"], games["away_score"]]).mean())
    hfa = estimate_hfa(games, weights=weights) if hfa is None else hfa

    off = {t: 0.0 for t in teams}
    defn = {t: 0.0 for t in teams}
    hfa_term = rows["hfa_sign"].values * (hfa / 2.0)
    w = rows["weight"].values

    for it in range(iterations):
        # --- update OFF holding DEF fixed (recency-weighted) ---
        opp_def = rows["opp"].map(defn).values
        target_off = rows["points_for"].values - league_avg + opp_def - hfa_term
        off_series = _weighted_groupby_mean(target_off, w, rows["team"].values, teams)
        off = off_series.to_dict()

        # --- update DEF holding OFF fixed (recency-weighted) ---
        team_off = rows["team"].map(off).values
        target_def = league_avg + team_off + hfa_term - rows["points_for"].values
        def_series = _weighted_groupby_mean(target_def, w, rows["opp"].values, teams)
        defn = def_series.to_dict()

        # re-center both scales to remove the shift ambiguity (off+c, def+c
        # leaves off-def invariant, so we anchor both to mean zero each pass)
        off_mean = np.mean(list(off.values()))
        def_mean = np.mean(list(defn.values()))
        off = {t: v - off_mean for t, v in off.items()}
        defn = {t: v - def_mean for t, v in defn.items()}

        if verbose and (it % 10 == 0 or it == iterations - 1):
            pred = league_avg + rows["team"].map(off).values - rows["opp"].map(defn).values + hfa_term
            rmse = float(np.sqrt(np.mean((rows["points_for"].values - pred) ** 2)))
            print(f"  iter {it:3d}  rmse={rmse:6.3f}")

    power = {t: off[t] + defn[t] for t in teams}

    # residuals for simulation variance
    team_off_arr = rows["team"].map(off).values
    opp_def_arr = rows["opp"].map(defn).values
    pred = league_avg + team_off_arr - opp_def_arr + hfa_term
    resid = rows["points_for"].values - pred
    resid_std = float(np.sqrt(np.average(resid ** 2, weights=w)))

    # SOS = average POWER of actual opponents faced (recency-weighted)
    sos_series = _weighted_groupby_mean(rows["opp"].map(power).values, w, rows["team"].values, teams)
    sos = sos_series.to_dict()

    conf_map = team_conference_map(games)
    conf_strength = {}
    by_conf = {}
    for t in teams:
        by_conf.setdefault(conf_map.get(t, "UNK"), []).append(power[t])
    for c, vals in by_conf.items():
        conf_strength[c] = float(np.mean(vals))

    return RatingResult(
        off=off, defn=defn, power=power, sos=sos, conf_strength=conf_strength,
        league_avg=league_avg, hfa=hfa, resid_std=resid_std, teams=teams,
    ), conf_map


# ---------------------------------------------------------------------------
# Prediction / simulation
# ---------------------------------------------------------------------------

def predict_expected_scores(ratings, team_a, team_b, site="home_a"):
    """site: 'home_a' (A hosts B), 'home_b' (B hosts A), or 'neutral'."""
    if team_a not in ratings.off or team_b not in ratings.off:
        missing = [t for t in (team_a, team_b) if t not in ratings.off]
        raise KeyError(f"Unknown team(s) not found in ratings: {missing}")

    if site == "neutral":
        hfa_a, hfa_b = 0.0, 0.0
    elif site == "home_a":
        hfa_a, hfa_b = ratings.hfa / 2.0, -ratings.hfa / 2.0
    elif site == "home_b":
        hfa_a, hfa_b = -ratings.hfa / 2.0, ratings.hfa / 2.0
    else:
        raise ValueError("site must be 'home_a', 'home_b', or 'neutral'")

    exp_a = ratings.league_avg + ratings.off[team_a] - ratings.defn[team_b] + hfa_a
    exp_b = ratings.league_avg + ratings.off[team_b] - ratings.defn[team_a] + hfa_b
    return max(exp_a, 0.0), max(exp_b, 0.0)


def simulate_game(exp_a, exp_b, std_dev, n_sims=DEFAULT_SIMS, rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    scores_a = np.clip(np.round(rng.normal(exp_a, std_dev, n_sims)), 0, None)
    scores_b = np.clip(np.round(rng.normal(exp_b, std_dev, n_sims)), 0, None)

    a_wins = scores_a > scores_b
    ties = scores_a == scores_b
    # college football has no ties (OT) -- split tied draws 50/50 as a stand-in for OT
    win_prob_a = float(a_wins.mean() + 0.5 * ties.mean())

    return {
        "win_prob_a": win_prob_a,
        "win_prob_b": 1.0 - win_prob_a,
        "mean_score_a": float(scores_a.mean()),
        "mean_score_b": float(scores_b.mean()),
        "median_score_a": float(np.median(scores_a)),
        "median_score_b": float(np.median(scores_b)),
        "mean_margin_a": float((scores_a - scores_b).mean()),
        "p10_p90_margin": (
            float(np.percentile(scores_a - scores_b, 10)),
            float(np.percentile(scores_a - scores_b, 90)),
        ),
        "mean_total": float((scores_a + scores_b).mean()),
    }


def simulate_season(ratings, schedule, n_sims=DEFAULT_SIMS, rng=None):
    """schedule: DataFrame[home_team, away_team, neutral_site]."""
    rng = rng or np.random.default_rng(RNG_SEED)
    teams = sorted(set(schedule["home_team"]) | set(schedule["away_team"]))
    win_counts = {t: np.zeros(n_sims) for t in teams}

    for _, g in schedule.iterrows():
        site = "neutral" if g.get("neutral_site", False) else "home_a"
        exp_a, exp_b = predict_expected_scores(ratings, g["home_team"], g["away_team"], site=site)
        scores_a = np.clip(np.round(rng.normal(exp_a, ratings.resid_std, n_sims)), 0, None)
        scores_b = np.clip(np.round(rng.normal(exp_b, ratings.resid_std, n_sims)), 0, None)
        a_win = (scores_a > scores_b).astype(float) + 0.5 * (scores_a == scores_b)
        win_counts[g["home_team"]] += a_win
        win_counts[g["away_team"]] += 1.0 - a_win

    summary = []
    for t in teams:
        w = win_counts[t]
        summary.append({
            "team": t,
            "proj_wins_mean": float(w.mean()),
            "proj_wins_median": float(np.median(w)),
            "proj_wins_p10": float(np.percentile(w, 10)),
            "proj_wins_p90": float(np.percentile(w, 90)),
        })
    return pd.DataFrame(summary).sort_values("proj_wins_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def ratings_table(ratings, conf_map):
    rows = []
    for t in ratings.teams:
        rows.append({
            "team": t,
            "conference": conf_map.get(t, "UNK"),
            "off_rating": round(ratings.off[t], 2),
            "def_rating": round(ratings.defn[t], 2),
            "power": round(ratings.power[t], 2),
            "sos": round(ratings.sos[t], 2),
        })
    df = pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df


def conference_table(ratings):
    rows = [{"conference": c, "avg_power": round(v, 2)} for c, v in ratings.conf_strength.items()]
    df = pd.DataFrame(rows).sort_values("avg_power", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df


def spread_line(team_a, team_b, mean_margin_a):
    """Vegas-style spread string: favorite shown negative, e.g. 'Georgia -4.5'."""
    margin = round(mean_margin_a * 2) / 2.0  # nearest half-point, like a real line
    if margin == 0:
        return f"{team_a} / {team_b} PICK'EM"
    if margin > 0:
        return f"{team_a} -{margin:g} ({team_b} +{margin:g})"
    return f"{team_b} -{-margin:g} ({team_a} +{-margin:g})"


def print_game_prediction(ratings, team_a, team_b, site, n_sims):
    exp_a, exp_b = predict_expected_scores(ratings, team_a, team_b, site=site)
    result = simulate_game(exp_a, exp_b, ratings.resid_std, n_sims=n_sims)

    site_label = {
        "home_a": f"{team_a} (home) vs {team_b} (away)",
        "home_b": f"{team_b} (home) vs {team_a} (away)",
        "neutral": f"{team_a} vs {team_b} (neutral site)",
    }[site]

    print(f"\n{site_label}")
    print(f"  Model expected score:  {team_a} {exp_a:.1f} - {team_b} {exp_b:.1f}")
    print(f"  Point spread: {spread_line(team_a, team_b, result['mean_margin_a'])}")
    print(f"  Monte Carlo ({n_sims:,} sims, resid. std={ratings.resid_std:.1f} pts):")
    print(f"    {team_a} win prob: {result['win_prob_a']*100:5.1f}%   "
          f"mean score {result['mean_score_a']:.1f} (median {result['median_score_a']:.0f})")
    print(f"    {team_b} win prob: {result['win_prob_b']*100:5.1f}%   "
          f"mean score {result['mean_score_b']:.1f} (median {result['median_score_b']:.0f})")
    lo, hi = result["p10_p90_margin"]
    print(f"    Projected margin ({team_a} - {team_b}): {result['mean_margin_a']:+.1f}  "
          f"(80% range: {lo:+.1f} to {hi:+.1f})")
    print(f"    Projected total points: {result['mean_total']:.1f}")


def predict_slate(ratings, schedule, n_sims=DEFAULT_SIMS, rng=None):
    """schedule: DataFrame[home_team, away_team, neutral_site, ...passthrough cols].
    Returns one row per game with expected score, spread, win prob, total."""
    rng = rng or np.random.default_rng(RNG_SEED)
    rows = []
    for _, g in schedule.iterrows():
        home, away = g["home_team"], g["away_team"]
        if home not in ratings.off or away not in ratings.off:
            continue
        neutral = bool(g.get("neutral_site", False))
        site = "neutral" if neutral else "home_a"
        exp_home, exp_away = predict_expected_scores(ratings, home, away, site=site)
        result = simulate_game(exp_home, exp_away, ratings.resid_std, n_sims=n_sims, rng=rng)
        row = {
            "week": g.get("week"),
            "date": g.get("date"),
            "home_team": home,
            "away_team": away,
            "neutral_site": neutral,
            "exp_home_score": round(exp_home, 1),
            "exp_away_score": round(exp_away, 1),
            "spread": spread_line(home, away, result["mean_margin_a"]),
            "home_win_prob": round(result["win_prob_a"], 3),
            "away_win_prob": round(result["win_prob_b"], 3),
            "projected_total": round(result["mean_total"], 1),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("week", "date") if c in df.columns and df[c].notna().any()]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--games", default="data/sample_games.csv",
                         help="CSV of completed games used to fit ratings (default: data/sample_games.csv)")
    common.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                         help="Rating solver iterations (default: %(default)s)")
    common.add_argument("--hfa", type=float, default=None,
                         help="Override home-field-advantage points (default: estimated from data)")
    common.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                         help="Monte Carlo trials (default: %(default)s)")
    common.add_argument("--decay", type=float, default=DEFAULT_SEASON_DECAY,
                         help="Per-year-back recency weight for older seasons, "
                              "e.g. 0.55 means last season counts 55%%, two seasons "
                              "back 30%% (default: %(default)s). Only applies if the "
                              "games file has a 'season' column.")
    common.add_argument("--current-season", type=int, default=None,
                         help="Season treated as full weight 1.0 (default: max season in data)")
    common.add_argument("--verbose", action="store_true", help="Print solver convergence info")

    p = argparse.ArgumentParser(
        description="College football rating & Monte Carlo game/season simulator.",
        parents=[common],
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("ratings", help="Print full team power ratings (off/def/SOS)", parents=[common])
    sub.add_parser("conferences", help="Print conference strength ratings", parents=[common])

    pr = sub.add_parser("predict", help="Simulate a single game", parents=[common])
    pr.add_argument("team_a")
    pr.add_argument("team_b")
    pr.add_argument("--site", choices=["home_a", "home_b", "neutral"], default="neutral",
                     help="home_a = team_a hosts, home_b = team_b hosts, neutral = no HFA")

    se = sub.add_parser("season", help="Simulate an upcoming schedule (Monte Carlo projected records)", parents=[common])
    se.add_argument("--schedule", required=True,
                     help="CSV with columns: home_team, away_team, neutral_site")
    se.add_argument("--out", default=None, help="Optional path to write results CSV")

    sp = sub.add_parser("spreads", help="Predicted point spread + win prob for every game in a schedule", parents=[common])
    sp.add_argument("--schedule", required=True,
                     help="CSV with columns: home_team, away_team, neutral_site[, week, date, status]")
    sp.add_argument("--week", type=int, default=None, help="Only predict games in this week (if the schedule has a 'week' column)")
    sp.add_argument("--only-unplayed", action="store_true",
                     help="Skip games already marked STATUS_FINAL in a 'status' column")
    sp.add_argument("--out", default=None, help="Optional path to write results CSV")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    games = load_games(args.games)
    if args.verbose:
        print(f"Loaded {len(games)} games covering {len(set(games['home_team']) | set(games['away_team']))} teams.")
        print("Fitting ratings...")
    ratings, conf_map = fit_ratings(
        games, iterations=args.iterations, hfa=args.hfa, verbose=args.verbose,
        current_season=args.current_season, decay=args.decay,
    )
    if args.verbose:
        print(f"League avg points/team/game: {ratings.league_avg:.2f}")
        print(f"Home field advantage: {ratings.hfa:+.2f} pts")
        print(f"Residual std dev (sim noise): {ratings.resid_std:.2f} pts\n")

    if args.command == "ratings":
        with pd.option_context("display.width", 120, "display.max_rows", None):
            print(ratings_table(ratings, conf_map).to_string())

    elif args.command == "conferences":
        with pd.option_context("display.width", 120):
            print(conference_table(ratings).to_string())

    elif args.command == "predict":
        try:
            print_game_prediction(ratings, args.team_a, args.team_b, args.site, args.sims)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "season":
        schedule = pd.read_csv(args.schedule)
        if "neutral_site" not in schedule.columns:
            schedule["neutral_site"] = False
        schedule["neutral_site"] = schedule["neutral_site"].astype(bool)
        result = simulate_season(ratings, schedule, n_sims=args.sims)
        with pd.option_context("display.width", 120, "display.max_rows", None):
            print(result.to_string(index=False))
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"\nWrote {args.out}")

    elif args.command == "spreads":
        schedule = pd.read_csv(args.schedule)
        if "neutral_site" not in schedule.columns:
            schedule["neutral_site"] = False
        schedule["neutral_site"] = schedule["neutral_site"].astype(bool)
        if args.week is not None and "week" in schedule.columns:
            schedule = schedule[schedule["week"] == args.week]
        if args.only_unplayed and "status" in schedule.columns:
            schedule = schedule[schedule["status"] != "STATUS_FINAL"]
        skipped = sorted(
            (set(schedule["home_team"]) | set(schedule["away_team"]))
            - set(ratings.teams)
        )
        result = predict_slate(ratings, schedule, n_sims=args.sims)
        with pd.option_context("display.width", 160, "display.max_rows", None):
            print(result.to_string(index=False))
        if skipped:
            print(f"\n(skipped {len(skipped)} team(s) with no rating -- not in --games data: {', '.join(skipped)})",
                  file=sys.stderr)
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
