#!/usr/bin/env python3
"""
Weekly auto-updater.

Re-pulls the current season from ESPN's public scoreboard API, merges any
newly-completed games into data/real_games.csv (dedup'd by ESPN event id,
so re-running mid-week to catch Thursday/Friday games before the Saturday
slate is safe), refreshes data/schedule_<season>.csv with whatever's still
upcoming, refits the ratings (recency-weighted, so this season's results
now count immediately), and writes a fresh spread sheet for the next
unplayed week to data/spreads_week<N>.csv.

Run this weekly during the season:
    python update_weekly.py
Safe to run more often (e.g. daily) -- it's idempotent.
"""

import argparse
import os

import pandas as pd

import cfb_sim
from fetch_espn_data import DATA_DIR, fetch_conference_map, pull_season

REAL_GAMES_COLS = [
    "date", "home_team", "home_conf", "away_team", "away_conf",
    "home_score", "away_score", "neutral_site", "season", "week", "event_id",
]


def merge_completed(existing_path, season, new_completed):
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
    else:
        existing = pd.DataFrame(columns=REAL_GAMES_COLS)
    # drop this season's old rows before re-adding the refreshed set, so
    # corrected scores / previously-missed games are picked up cleanly
    if "season" in existing.columns:
        existing = existing[existing["season"] != season]
    combined = pd.concat([existing, new_completed], ignore_index=True)
    if "event_id" in combined.columns:
        combined = combined.drop_duplicates(subset="event_id")
    combined = combined.sort_values("date")
    combined.to_csv(existing_path, index=False)
    return combined


def main():
    ap = argparse.ArgumentParser(description="Pull latest results and refresh spreads for the current week.")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--games-file", default=os.path.join(DATA_DIR, "real_games.csv"))
    ap.add_argument("--sims", type=int, default=cfb_sim.DEFAULT_SIMS)
    ap.add_argument("--decay", type=float, default=cfb_sim.DEFAULT_SEASON_DECAY)
    args = ap.parse_args()

    conf_cache = os.path.join(DATA_DIR, "conferences.json")
    print(f"Fetching {args.season} season data from ESPN...")
    conf_map = fetch_conference_map(args.season, conf_cache)
    rows = pull_season(args.season, conf_map, [(2, range(1, 18)), (3, range(1, 7))])
    df = pd.DataFrame(rows).drop_duplicates(subset="event_id")

    completed = df[df["status"] == "STATUS_FINAL"].dropna(subset=["home_score", "away_score"]).copy()
    completed["home_score"] = completed["home_score"].astype(float)
    completed["away_score"] = completed["away_score"].astype(float)
    completed = completed[REAL_GAMES_COLS]

    combined = merge_completed(args.games_file, args.season, completed)
    print(f"\n{args.games_file}: {len(combined)} total games "
          f"({len(completed)} completed {args.season} games merged in)")

    upcoming = df[df["status"] != "STATUS_FINAL"].sort_values(["week", "date"])
    sched_path = os.path.join(DATA_DIR, f"schedule_{args.season}.csv")
    upcoming.to_csv(sched_path, index=False)
    print(f"{sched_path}: {len(upcoming)} games remaining")

    if len(upcoming) == 0:
        print("\nSeason complete -- no upcoming games to project.")
        return

    next_week = int(upcoming["week"].min())
    print(f"\nRefitting ratings ({args.decay} season decay) and projecting week {next_week}...")
    games = cfb_sim.load_games(args.games_file)
    ratings, _ = cfb_sim.fit_ratings(games, current_season=args.season, decay=args.decay)
    week_games = upcoming[upcoming["week"] == next_week]
    slate = cfb_sim.predict_slate(ratings, week_games, n_sims=args.sims)

    out_path = os.path.join(DATA_DIR, f"spreads_week{next_week}.csv")
    slate.to_csv(out_path, index=False)
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(slate.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
