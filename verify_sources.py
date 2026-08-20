#!/usr/bin/env python3
"""
Cross-check this model's ratings against independent published systems.

If our ratings are sound they should correlate strongly with other credible
rating systems built by other people from other data. Where they diverge, that
divergence is either a genuine edge or a bug -- and the only way to tell is to
look at WHICH teams diverge and whether the reason is explainable.

Sources compared:
  1. ESPN FPI          -- ESPN's own power index (independent methodology)
  2. ESPN Predictor    -- ESPN's per-game win probability
  3. The betting market -- DraftKings closing/current lines, the strongest
                           benchmark available since it aggregates all public
                           information plus real money
  4. Our ridge ratings

A high correlation with FPI plus a high correlation with the market means the
model is measuring the same underlying quantity everyone else is. A LOW
correlation would mean something is broken.
"""

import argparse
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

import cfb_ratings as R

HEADERS = {"User-Agent": "Mozilla/5.0"}
from paths import DATA_DIR  # frozen-aware; see paths.py
FPI_URL = ("https://site.web.api.espn.com/apis/fitt/v3/sports/football/"
           "college-football/powerindex?region=us&lang=en&season={season}&limit=200")


def _get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    return None


def fetch_fpi(season, cache=True):
    path = os.path.join(DATA_DIR, "raw_cache", f"fpi_{season}.json")
    if cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = _get(FPI_URL.format(season=season))
        if d and cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f)
    if not d:
        return None

    names = {c.get("name"): c.get("names", []) for c in d.get("categories", [])}
    rows = []
    for t in d.get("teams", []):
        team = t["team"]
        rec = {"team_id": str(team.get("id")),
               "team": team.get("nickname") or team.get("shortDisplayName"),
               "display": team.get("displayName")}
        for cat in t.get("categories", []):
            cn = cat.get("name")
            vals = cat.get("values", [])
            for nm, v in zip(names.get(cn, []), vals):
                rec[f"{cn}_{nm}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_ratings(ratings, fpi, games):
    """Join our power ratings to FPI by team id."""
    id_of = {}
    for _, r in games.iterrows():
        id_of[r["home_team"]] = str(r["home_id"])
        id_of[r["away_team"]] = str(r["away_id"])

    rows = []
    for t in ratings.teams:
        if t.startswith("__"):
            continue
        tid = id_of.get(t)
        if not tid:
            continue
        rows.append({"team": t, "team_id": tid, "our_power": ratings.power[t],
                     "our_off": ratings.off[t], "our_def": ratings.defn[t]})
    ours = pd.DataFrame(rows)
    m = ours.merge(fpi, on="team_id", how="inner", suffixes=("", "_fpi"))
    return m


def main():
    ap = argparse.ArgumentParser(description="Cross-check ratings against independent sources.")
    ap.add_argument("--games", default=os.path.join(DATA_DIR, "real_games.csv"))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--odds", default=os.path.join(DATA_DIR, "odds.csv"))
    args = ap.parse_args()

    games = pd.read_csv(args.games)
    print(f"Fitting our ratings through {args.season}...")
    hist = games[games["season"] <= args.season]
    ratings = R.fit_ratings(hist, lam=R.DEFAULT_LAMBDA, cv=False, current_season=args.season)

    print(f"Fetching ESPN FPI for {args.season}...")
    fpi = fetch_fpi(args.season)
    if fpi is None:
        print("  FPI unavailable")
        return

    m = compare_ratings(ratings, fpi, games)
    print(f"  joined {len(m)} teams\n")

    fpi_cols = [c for c in m.columns if c.startswith(("fpi", "efficiencies", "projections"))]
    num = [c for c in fpi_cols if pd.api.types.is_numeric_dtype(m[c])]

    print("=" * 70)
    print("SOURCE 1 -- ESPN FPI (independent rating system)")
    print("=" * 70)
    best = []
    for c in num:
        sub = m[["our_power", c]].dropna()
        if len(sub) < 50:
            continue
        r = float(np.corrcoef(sub["our_power"], sub[c])[0, 1])
        best.append((abs(r), r, c, len(sub)))
    best.sort(reverse=True)
    for _, r, c, n in best[:6]:
        print(f"  corr(our_power, {c:38s}) = {r:+.4f}   n={n}")

    if best:
        _, r, c, n = best[0]
        print(f"\n  Strongest match: {c}  r={r:+.4f}")
        sub = m[["team", "our_power", c]].dropna().copy()
        sub["our_rank"] = sub["our_power"].rank(ascending=False)
        sub["fpi_rank"] = sub[c].rank(ascending=False if r > 0 else True)
        sub["rank_gap"] = sub["our_rank"] - sub["fpi_rank"]
        print("\n  Biggest disagreements (we rate HIGHER than FPI):")
        for _, x in sub.nsmallest(5, "rank_gap").iterrows():
            print(f"    {x['team']:22s} ours #{x['our_rank']:>5.0f}  FPI #{x['fpi_rank']:>5.0f}")
        print("  Biggest disagreements (we rate LOWER than FPI):")
        for _, x in sub.nlargest(5, "rank_gap").iterrows():
            print(f"    {x['team']:22s} ours #{x['our_rank']:>5.0f}  FPI #{x['fpi_rank']:>5.0f}")

    # ---- source 2/3: the market -----------------------------------------
    if os.path.exists(args.odds):
        odds = pd.read_csv(args.odds)
        g = games.merge(odds, on="event_id", how="inner")
        g = g.dropna(subset=["spread_close_home"])
        if len(g) > 100:
            print("\n" + "=" * 70)
            print("SOURCE 2 -- BETTING MARKET (closing lines)")
            print("=" * 70)
            rated = []
            for _, x in g.iterrows():
                h = R.resolve_team(ratings, x["home_team"])
                a = R.resolve_team(ratings, x["away_team"])
                if not (ratings.has(h) and ratings.has(a)):
                    continue
                s = 0.0 if x["neutral_site"] else 1.0
                pm = (ratings.off[h] - ratings.defn[a]) - (ratings.off[a] - ratings.defn[h]) + s * ratings.hfa
                rated.append((pm, -x["spread_close_home"], x["home_score"] - x["away_score"]))
            arr = np.array(rated)
            if len(arr) > 50:
                print(f"  n = {len(arr)} games with closing lines")
                print(f"  corr(our margin, market margin) = "
                      f"{np.corrcoef(arr[:,0], arr[:,1])[0,1]:+.4f}")
                print(f"  corr(our margin, actual)        = "
                      f"{np.corrcoef(arr[:,0], arr[:,2])[0,1]:+.4f}")
                print(f"  corr(market,     actual)        = "
                      f"{np.corrcoef(arr[:,1], arr[:,2])[0,1]:+.4f}")
                print(f"\n  NOTE: these ratings were fitted on these same games, so the")
                print(f"  'our margin vs actual' figure here is IN-SAMPLE and flattering.")
                print(f"  Use backtest.py / evaluate_vs_market.py for the honest number.")
    else:
        print(f"\n(no {args.odds} yet -- run fetch_odds.py for the market comparison)")


if __name__ == "__main__":
    main()
