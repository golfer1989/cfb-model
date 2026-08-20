#!/usr/bin/env python3
"""
EXPERIMENT: situational factors the model ignores entirely.

THE ONLY QUESTION THAT MATTERS

Not "does this affect the game?" -- plenty of things affect the game. The
question is whether the MARKET UNDERPRICES it. A closing line already contains
every public factor that professional bettors know about, so a factor is only
worth money if it is hard to quantify, easy to overlook, or arrives late.

That filter kills most of the popular ideas immediately. Rest and travel are
on every handicapper's checklist and in every power rating. Rivalry games are
priced. What survives is usually information that is genuinely awkward to
value: bowl opt-outs, an interim coach, a roster gutted by the portal.

FACTORS TESTED HERE, and why each might or might not be priced
  rest differential   days since each team last played -- widely known, likely priced
  bye week            extra prep -- widely known, likely priced
  travel distance     miles the away team flew, from venue coordinates
  time zones crossed  body-clock effect for a west-coast team at noon Eastern
  short week          Thursday/Friday games after a Saturday
  coaching change     a first-year or interim head coach -- harder to price
  portal turnover     players lost to the transfer portal -- hard to price
  talent gap          247/composite talent, independent of results
  late-season bowl    opt-out and motivation season, notoriously mispriced

Everything is measured against the walk-forward residual, leak-free, and
every survivor must hold in BOTH seasons.

Run: python experiments/situational_factors_test.py
"""

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_DIR  # noqa: E402


def ols(cols, y, names):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float((r ** 2).sum() / dof)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return [(nm, beta[i], se[i], beta[i] / se[i] if se[i] else np.nan)
            for i, nm in enumerate(["intercept"] + names)]


def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def norm(s):
    return (str(s).strip().replace("State", "St").replace("&", "and")
            .replace(".", "").replace("'", "").lower())


def main():
    preds = pd.read_csv(os.path.join(DATA_DIR, "backtest_preds.csv"))
    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    preds["resid"] = preds["act_margin"] - preds["pred_margin"]

    key = games[["event_id", "season", "week", "home_team", "away_team",
                 "home_id", "away_id", "kickoff_utc"]]
    d = preds.merge(key, on=["season", "week", "home_team", "away_team"], how="left")
    d["kick"] = pd.to_datetime(d["kickoff_utc"], errors="coerce", utc=True)

    # ---- rest: days since each team's previous game ----------------------
    g = games.copy()
    g["kick"] = pd.to_datetime(g["kickoff_utc"], errors="coerce", utc=True)
    g = g.sort_values("kick")
    last = {}
    rest_h, rest_a = [], []
    for _, r in g.iterrows():
        for team, store in ((r["home_team"], rest_h), (r["away_team"], rest_a)):
            prev = last.get(team)
            store.append((r["kick"] - prev).days if prev is not None else np.nan)
        last[r["home_team"]] = r["kick"]
        last[r["away_team"]] = r["kick"]
    g["rest_home"], g["rest_away"] = rest_h, rest_a
    d = d.merge(g[["event_id", "rest_home", "rest_away"]], on="event_id", how="left")
    d["rest_diff"] = d["rest_home"] - d["rest_away"]
    d["home_bye"] = (d["rest_home"] >= 12).astype(float)
    d["away_bye"] = (d["rest_away"] >= 12).astype(float)
    d["bye_diff"] = d["home_bye"] - d["away_bye"]
    d["short_week"] = ((d["rest_home"] <= 5) | (d["rest_away"] <= 5)).astype(float)

    # ---- travel: distance and time zones, from venue coordinates ---------
    vpath = os.path.join(DATA_DIR, "venues.csv")
    have_travel = False
    if os.path.exists(vpath):
        v = pd.read_csv(vpath)
        # a team's home venue = the venue it most often hosts at
        home_venue = (g.merge(v, left_on="home_team", right_on=None, how="left")
                      if False else None)
        # simpler: build team -> (lat, lon) from the venue of its home games
        vp = {}
        wpath = os.path.join(DATA_DIR, "game_weather.csv")
        if os.path.exists(wpath):
            wx = pd.read_csv(wpath)[["event_id", "venue_id"]]
            wx["event_id"] = wx["event_id"].astype(str)
            gg = g.copy()
            gg["event_id"] = gg["event_id"].astype(str)
            gg = gg.merge(wx, on="event_id", how="left").merge(
                v[["venue_id", "lat", "lon"]], on="venue_id", how="left")
            hv = (gg[~gg["neutral_site"].astype(bool)]
                  .dropna(subset=["lat"])
                  .groupby("home_team")[["lat", "lon"]].median())
            vp = {t: (r["lat"], r["lon"]) for t, r in hv.iterrows()}
            gg2 = gg[["event_id", "lat", "lon"]].rename(
                columns={"lat": "game_lat", "lon": "game_lon"})
            d["event_id"] = d["event_id"].astype(str)
            d = d.merge(gg2, on="event_id", how="left")
            d["away_travel_mi"] = [
                haversine(vp.get(a, (np.nan, np.nan))[0], vp.get(a, (np.nan, np.nan))[1],
                          la, lo)
                for a, la, lo in zip(d["away_team"], d["game_lat"], d["game_lon"])]
            d["tz_cross"] = [
                abs((vp.get(a, (np.nan, np.nan))[1] - lo) / 15.0)
                if not pd.isna(lo) and a in vp else np.nan
                for a, lo in zip(d["away_team"], d["game_lon"])]
            have_travel = d["away_travel_mi"].notna().sum() > 200

    # ---- coaching change, portal, talent ---------------------------------
    def load_year(name, yr):
        p = os.path.join(DATA_DIR, f"cfbd_{name}_{yr}.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    talent_rows = []
    for yr in (2023, 2024, 2025):
        t = load_year("talent", yr)
        if t is not None and len(t):
            t = t.rename(columns={"year": "season"})
            t["key"] = t["team"].map(norm)
            talent_rows.append(t[["season", "key", "talent"]])
    if talent_rows:
        tal = pd.concat(talent_rows)
        d["hkey"] = d["home_team"].map(norm)
        d["akey"] = d["away_team"].map(norm)
        d = d.merge(tal.rename(columns={"key": "hkey", "talent": "h_talent"}),
                    on=["season", "hkey"], how="left")
        d = d.merge(tal.rename(columns={"key": "akey", "talent": "a_talent"}),
                    on=["season", "akey"], how="left")
        d["talent_diff"] = d["h_talent"] - d["a_talent"]

    portal_rows = []
    for yr in (2023, 2024, 2025):
        p = load_year("portal", yr)
        if p is not None and len(p) and "origin" in p.columns:
            out = p.groupby("origin").size().rename("portal_out").reset_index()
            out["season"] = yr
            out["key"] = out["origin"].map(norm)
            portal_rows.append(out[["season", "key", "portal_out"]])
    if portal_rows:
        po = pd.concat(portal_rows)
        d = d.merge(po.rename(columns={"key": "hkey", "portal_out": "h_portal"}),
                    on=["season", "hkey"], how="left")
        d = d.merge(po.rename(columns={"key": "akey", "portal_out": "a_portal"}),
                    on=["season", "akey"], how="left")
        d["portal_diff"] = d["h_portal"].fillna(0) - d["a_portal"].fillna(0)

    coach_rows = []
    for yr in (2023, 2024, 2025):
        c = load_year("coaches", yr)
        if c is None or not len(c) or "seasons" not in c.columns:
            continue
        for _, r in c.iterrows():
            try:
                seasons = eval(r["seasons"]) if isinstance(r["seasons"], str) else r["seasons"]
            except Exception:  # noqa: BLE001
                continue
            for s in (seasons or []):
                if s.get("year") != yr:
                    continue
                coach_rows.append({"season": yr, "key": norm(s.get("school")),
                                   "coach": f"{r.get('firstName')} {r.get('lastName')}"})
    if coach_rows:
        cf = pd.DataFrame(coach_rows).drop_duplicates(["season", "key"])
        prev = cf.copy(); prev["season"] = prev["season"] + 1
        prev = prev.rename(columns={"coach": "prev_coach"})
        cf = cf.merge(prev, on=["season", "key"], how="left")
        cf["new_coach"] = ((cf["prev_coach"].notna()) &
                           (cf["coach"] != cf["prev_coach"])).astype(float)
        d = d.merge(cf[["season", "key", "new_coach"]].rename(
            columns={"key": "hkey", "new_coach": "h_newcoach"}),
            on=["season", "hkey"], how="left")
        d = d.merge(cf[["season", "key", "new_coach"]].rename(
            columns={"key": "akey", "new_coach": "a_newcoach"}),
            on=["season", "akey"], how="left")
        d["newcoach_diff"] = d["h_newcoach"].fillna(0) - d["a_newcoach"].fillna(0)

    d["is_bowl"] = (d["kick"].dt.month.isin([12, 1])).astype(float)

    print(f"out-of-sample games: {len(d)}")
    print(f"baseline residual sd: {d['resid'].std():.3f}\n")

    FEATURES = [
        ("rest_diff", "rest days, home minus away"),
        ("bye_diff", "bye week advantage"),
        ("short_week", "either team on a short week"),
        ("away_travel_mi", "away team travel miles"),
        ("tz_cross", "time zones crossed by away team"),
        ("talent_diff", "247 talent composite gap"),
        ("portal_diff", "portal departures, home minus away"),
        ("newcoach_diff", "new head coach, home minus away"),
        ("is_bowl", "bowl / postseason game"),
    ]

    print("=" * 76)
    print("EACH FACTOR vs the model's residual (does it explain what we miss?)")
    print("=" * 76)
    print(f"  {'factor':>34} {'n':>5} {'beta':>10} {'se':>8} {'t':>7}")
    survivors = []
    for col, label in FEATURES:
        if col not in d.columns:
            print(f"  {label:>34}     -- not available")
            continue
        sub = d.dropna(subset=[col, "resid"])
        if len(sub) < 150 or sub[col].nunique() < 2:
            print(f"  {label:>34} {len(sub):>5}   (too few / no variation)")
            continue
        rows = ols([sub[col]], sub["resid"].values, [col])
        _, b, se, t = rows[1]
        flag = "  <-- significant" if abs(t) >= 2 else ""
        print(f"  {label:>34} {len(sub):>5} {b:>10.4f} {se:>8.4f} {t:>7.2f}{flag}")
        if abs(t) >= 2:
            survivors.append(col)

    if not survivors:
        print("\n  Nothing reaches |t| = 2. No situational factor here explains")
        print("  what the model misses -- consistent with the market already")
        print("  pricing all of them.")
        return 0

    print("\n" + "=" * 76)
    print("CROSS-SEASON CHECK on anything that survived")
    print("=" * 76)
    print("  A factor present in one season only is the pattern that has")
    print("  already produced two false positives in this project.\n")
    for col in survivors:
        out = []
        for s in sorted(d["season"].dropna().unique()):
            sub = d[(d["season"] == s)].dropna(subset=[col, "resid"])
            if len(sub) < 100:
                continue
            rows = ols([sub[col]], sub["resid"].values, [col])
            out.append(f"{int(s)}: beta={rows[1][1]:+.3f} t={rows[1][3]:+.2f}")
        print(f"  {col:>20}  " + "    ".join(out))

    print("\n" + "=" * 76)
    print("MULTIPLE-TESTING CHECK")
    print("=" * 76)
    k = len([c for c, _ in FEATURES if c in d.columns])
    print(f"  {k} factors tested. Bonferroni threshold for 5% family-wise error")
    print(f"  is p = {0.05/max(k,1):.4f}, i.e. |t| >= {abs(round(2.807,2)) if k>5 else 2.4:.2f} rather than 2.0.")
    print("  A single |t| just over 2 among nine tests is expected by chance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
