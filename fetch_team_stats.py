#!/usr/bin/env python3
"""
Fetch per-team season statistics (own AND opponent splits) from ESPN.

One request per season returns all ~136 FBS teams with both their own
offensive production and what they allowed to opponents -- which is exactly
what a pass-offense-vs-pass-defense style comparison needs.

ESPN returns each category twice: splitId "0" = the team's own production,
splitId "1" = what opponents did against them (i.e. the team's defense).

Output: data/team_stats.csv, one row per (season, team) with columns like
  off_passingYards, off_rushingYards, off_yardsPerPassAttempt, ...
  def_passingYards, def_rushingYards, def_yardsPerPassAttempt, ...
plus derived style metrics:
  off_pass_rate / def_pass_rate  -- share of plays that were passes
  off_ypp / def_ypp              -- yards per play
  off_plays_pg / def_plays_pg    -- pace proxy (plays per game)
"""

import argparse
import json
import os
import time
import urllib.request

import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}  # see fetch_espn_data.py -- do not embellish
BYTEAM_URL = ("https://site.web.api.espn.com/apis/common/v3/sports/football/"
              "college-football/statistics/byteam?region=us&lang=en"
              "&contentorigin=espn&season={season}&seasontype=2&limit=200")

from paths import DATA_DIR  # frozen-aware; see paths.py
CACHE_DIR = os.path.join(DATA_DIR, "raw_cache")

# stats worth keeping (ESPN repeats some names within a category; we take the
# first occurrence of each)
KEEP = {
    "passing": ["passingYards", "passingYardsPerGame", "yardsPerPassAttempt",
                "completions", "passingAttempts", "completionPct",
                "passingTouchdowns", "interceptions", "sacks", "QBRating",
                "totalYards", "totalPoints", "totalPointsPerGame", "yardsPerGame"],
    "rushing": ["rushingYards", "rushingYardsPerGame", "yardsPerRushAttempt",
                "rushingAttempts", "rushingTouchdowns", "rushingFumbles"],
}


def _get(url, retries=4):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 -- retry on any transport error
            last = e
            time.sleep(20.0 * (a + 1))
    raise RuntimeError(f"failed {url}: {last}")


def fetch_season(season, refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"teamstats_{season}.json")
    if os.path.exists(cache) and not refresh:
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    data = _get(BYTEAM_URL.format(season=season))
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f)
    time.sleep(1.0)
    return data


def parse_season(data, season):
    names_by_cat = {c["name"]: c.get("names", []) for c in data.get("categories", [])}
    rows = []
    for entry in data.get("teams", []):
        team = entry["team"]
        rec = {
            "season": season,
            "team_id": team.get("id"),
            "team": team.get("nickname") or team.get("shortDisplayName"),
            "team_display": team.get("displayName"),
        }
        for cat in entry.get("categories", []):
            cname = cat.get("name")
            if cname not in KEEP:
                continue
            names = names_by_cat.get(cname, [])
            vals = cat.get("values", [])
            # splitId "0" = own production, "1" = allowed to opponents
            prefix = "off_" if str(cat.get("splitId")) == "0" else "def_"
            seen = set()
            for nm, v in zip(names, vals):
                if nm in KEEP[cname] and nm not in seen:
                    seen.add(nm)
                    rec[prefix + nm] = v
        rows.append(rec)
    return rows


def add_derived(df):
    """Style metrics that make pass-vs-rush comparisons meaningful."""
    for side in ("off", "def"):
        pa = df.get(f"{side}_passingAttempts")
        ra = df.get(f"{side}_rushingAttempts")
        py = df.get(f"{side}_passingYards")
        ry = df.get(f"{side}_rushingYards")
        sacks = df.get(f"{side}_sacks")
        if pa is None or ra is None:
            continue
        # sacks count as pass plays in play-calling terms
        dropbacks = pa.fillna(0) + sacks.fillna(0) if sacks is not None else pa.fillna(0)
        plays = dropbacks + ra.fillna(0)
        df[f"{side}_plays"] = plays
        df[f"{side}_pass_rate"] = dropbacks / plays.replace(0, pd.NA)
        total_yards = py.fillna(0) + ry.fillna(0)
        df[f"{side}_ypp"] = total_yards / plays.replace(0, pd.NA)
        gp = df.get(f"{side}_passingYardsPerGame")
        if gp is not None and py is not None:
            games = (py / gp.replace(0, pd.NA)).round()
            df[f"{side}_games"] = games
            df[f"{side}_plays_pg"] = plays / games.replace(0, pd.NA)
    return df


def main():
    ap = argparse.ArgumentParser(description="Fetch ESPN per-team season stats (own + opponent).")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "team_stats.csv"))
    args = ap.parse_args()

    all_rows = []
    for s in args.seasons:
        print(f"Fetching team stats for {s}...", flush=True)
        data = fetch_season(s, refresh=args.refresh)
        rows = parse_season(data, s)
        print(f"  {len(rows)} teams", flush=True)
        all_rows += rows

    df = pd.DataFrame(all_rows)
    df = add_derived(df)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: {len(df)} team-seasons, {len(df.columns)} columns")
    cols = [c for c in df.columns if c.startswith(("off_", "def_"))]
    print(f"Stat columns: {len(cols)}")
    print(df[["season", "team", "off_pass_rate", "off_ypp", "def_pass_rate", "def_ypp"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
