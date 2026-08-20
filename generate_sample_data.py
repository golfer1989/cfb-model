#!/usr/bin/env python3
"""
Generates synthetic demo data so cfb_sim.py is runnable out of the box.

IMPORTANT: The team/conference NAMES below are real, but the SCORES are
entirely synthetic (drawn from made-up "true skill" values + random noise
with a fixed seed) -- they are NOT real game results. This exists purely
to demonstrate that the rating engine recovers sensible power rankings
and produces reasonable win probabilities. Replace data/*.csv with real
results (e.g. exported from ESPN, cfbstats.com, sports-reference.com)
to get real predictions.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# 16 teams / 2 synthetic conferences with an assumed "true" power level
# (arbitrary units, just needs to create realistic separation).
TEAMS = {
    # Conference A (stronger, on average)
    "Georgia":        ("Conference A", 9.0),
    "Alabama":        ("Conference A", 8.0),
    "Texas":          ("Conference A", 7.0),
    "LSU":            ("Conference A", 5.0),
    "Tennessee":      ("Conference A", 4.5),
    "Ole Miss":       ("Conference A", 3.0),
    "Auburn":         ("Conference A", 1.0),
    "Missouri":       ("Conference A", 0.5),
    # Conference B (weaker, on average)
    "Iowa":           ("Conference B", 3.5),
    "Wisconsin":      ("Conference B", 2.5),
    "Michigan St":    ("Conference B", 1.5),
    "Illinois":       ("Conference B", 1.0),
    "Purdue":         ("Conference B", -2.0),
    "Northwestern":   ("Conference B", -3.0),
    "Rutgers":        ("Conference B", -3.5),
    "Indiana":        ("Conference B", -4.5),
}

LEAGUE_AVG_PTS = 27.0
GAME_NOISE_STD = 13.0
HFA_TRUE = 2.5


def true_score(off_team, def_team, is_home, is_neutral):
    off_skill = TEAMS[off_team][1]
    def_skill = TEAMS[def_team][1]
    hfa = 0.0 if is_neutral else (HFA_TRUE / 2.0 if is_home else -HFA_TRUE / 2.0)
    mean = LEAGUE_AVG_PTS + off_skill - def_skill * 0.4 + hfa
    return max(0, round(RNG.normal(mean, GAME_NOISE_STD)))


def build_schedule():
    """Round-robin-ish within conference + a handful of cross-conference games."""
    names = list(TEAMS.keys())
    conf_a = [t for t in names if TEAMS[t][0] == "Conference A"]
    conf_b = [t for t in names if TEAMS[t][0] == "Conference B"]

    games = []
    week = 1

    def add_game(home, away, neutral=False):
        nonlocal week
        games.append((week, home, away, neutral))

    # within-conference: each team plays 5 conference games
    for conf in (conf_a, conf_b):
        n = len(conf)
        for i in range(n):
            for j in range(1, 6):
                opp = conf[(i + j) % n]
                if opp <= conf[i]:
                    continue
                home_first = RNG.random() < 0.5
                add_game(conf[i], opp) if home_first else add_game(opp, conf[i])
        week += 1

    # cross-conference "rivalry" games, one pairing per team
    RNG.shuffle(conf_a)
    RNG.shuffle(conf_b)
    for a, b in zip(conf_a, conf_b):
        if RNG.random() < 0.5:
            add_game(a, b)
        else:
            add_game(b, a)

    return games


def main():
    games = build_schedule()
    rows = []
    for week, home, away, neutral in games:
        hs = true_score(home, away, is_home=True, is_neutral=neutral)
        as_ = true_score(away, home, is_home=False, is_neutral=neutral)
        rows.append({
            "date": f"2025-W{week:02d}",
            "home_team": home,
            "home_conf": TEAMS[home][0],
            "away_team": away,
            "away_conf": TEAMS[away][0],
            "home_score": hs,
            "away_score": as_,
            "neutral_site": neutral,
        })

    played = pd.DataFrame(rows)
    played.to_csv("data/sample_games.csv", index=False)
    print(f"Wrote data/sample_games.csv ({len(played)} synthetic games)")

    # a future "week" of games to project with `season`
    names = list(TEAMS.keys())
    RNG.shuffle(names)
    upcoming = []
    for i in range(0, len(names) - 1, 2):
        upcoming.append({
            "home_team": names[i],
            "away_team": names[i + 1],
            "neutral_site": False,
        })
    sched = pd.DataFrame(upcoming)
    sched.to_csv("data/sample_schedule.csv", index=False)
    print(f"Wrote data/sample_schedule.csv ({len(sched)} upcoming games)")


if __name__ == "__main__":
    main()
