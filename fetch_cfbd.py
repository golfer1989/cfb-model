#!/usr/bin/env python3
"""
CollegeFootballData.com client -- the largest open college football dataset.

REQUIRES A FREE API KEY, WHICH YOU MUST CREATE YOURSELF.

    1. go to https://collegefootballdata.com/key
    2. enter an email address; the key arrives by email
    3. then either
         set an environment variable:   CFBD_API_KEY=your_key_here
         or write it to:                data/cfbd_key.txt

I will not register the account for you. Creating accounts in someone else's
name is not something I do, even when it is free and even when it would be
convenient -- the account is yours, tied to your email, under their terms.

WHAT IT ADDS THAT ESPN DOES NOT HAVE
  SP+ ratings              Bill Connelly's opponent-adjusted efficiency
  PPA / EPA                predicted points added, per team and per play
  returning production     share of last year's production coming back --
                           the single best public preseason predictor, and
                           precisely what this model lacks in week 1
  recruiting rankings      team talent composite
  advanced box scores      success rate, explosiveness, havoc, field position,
                           finishing drives
  transfer portal          roster turnover
  pregame win probabilities

WHAT IT DOES NOT FIX
  The model's problem is not a shortage of team-strength proxies -- roughly 50
  were already tested and rejected because the ratings measure the same thing
  directly from scores. The candidates here that are genuinely NEW information
  are returning production and recruiting (preseason signal, before any games
  exist) and portal turnover. Everything else is likely another proxy.

Usage:
  python fetch_cfbd.py --check              verify the key works
  python fetch_cfbd.py --returning 2026     returning production
  python fetch_cfbd.py --sp 2025            SP+ ratings
  python fetch_cfbd.py --all 2025           everything for a season
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

from paths import DATA_DIR

BASE = "https://api.collegefootballdata.com"
KEY_FILE = os.path.join(DATA_DIR, "cfbd_key.txt")
CACHE = os.path.join(DATA_DIR, "raw_cache", "cfbd")


def get_key():
    """Environment variable first, then the key file. Returns None if absent."""
    k = os.environ.get("CFBD_API_KEY")
    if k and k.strip():
        return k.strip()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, encoding="utf-8") as f:
            k = f.read().strip()
        if k:
            return k
    return None


def _get(path, params=None, key=None, tries=3):
    key = key or get_key()
    if not key:
        raise RuntimeError(
            "No CFBD API key found.\n"
            "  Get one free at https://collegefootballdata.com/key\n"
            f"  Then set CFBD_API_KEY, or save it to {KEY_FILE}")
    qs = "&".join(f"{k}={v}" for k, v in (params or {}).items() if v is not None)
    url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            time.sleep(0.4)          # be polite to a free service
            return data
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError("CFBD rejected the key (401). Check it is "
                                   "correct and still active.") from e
            if e.code == 429:
                time.sleep(20 * (i + 1))
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"CFBD request failed: {url}: {last}")


def _cached(name, path, params, refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(cpath) and not refresh:
        try:
            with open(cpath, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    data = _get(path, params)
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


# --- the pulls actually worth having --------------------------------------

def returning_production(year, refresh=False):
    """Share of last season's production returning. This is the one CFBD
    field with a strong claim to fixing a real weakness: in week 1 the model
    has no current-season data at all and cannot know that a team lost its
    quarterback and four starting linemen."""
    d = _cached(f"returning_{year}", "/player/returning", {"year": year}, refresh)
    return pd.DataFrame(d)


def sp_ratings(year, refresh=False):
    d = _cached(f"sp_{year}", "/ratings/sp", {"year": year}, refresh)
    return pd.DataFrame(d)


def ppa_teams(year, refresh=False):
    d = _cached(f"ppa_{year}", "/ppa/teams", {"year": year}, refresh)
    return pd.DataFrame(d)


def recruiting(year, refresh=False):
    d = _cached(f"recruiting_{year}", "/recruiting/teams", {"year": year}, refresh)
    return pd.DataFrame(d)


def advanced_stats(year, refresh=False):
    d = _cached(f"advstats_{year}", "/stats/season/advanced", {"year": year}, refresh)
    return pd.DataFrame(d)


def portal(year, refresh=False):
    d = _cached(f"portal_{year}", "/player/portal", {"year": year}, refresh)
    return pd.DataFrame(d)


PULLS = {
    "returning": returning_production,
    "sp": sp_ratings,
    "ppa": ppa_teams,
    "recruiting": recruiting,
    "advstats": advanced_stats,
    "portal": portal,
}


def check_key():
    k = get_key()
    if not k:
        print("No key found.")
        print("  Get one free at https://collegefootballdata.com/key")
        print(f"  Then set CFBD_API_KEY or save it to {KEY_FILE}")
        return False
    try:
        d = _get("/teams/fbs", {"year": 2025})
        print(f"Key works. /teams/fbs returned {len(d)} FBS teams.")
        return True
    except RuntimeError as e:
        print(f"Key present but request failed:\n  {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Pull CollegeFootballData.com datasets.")
    ap.add_argument("--check", action="store_true", help="verify the key works")
    ap.add_argument("--all", type=int, metavar="YEAR", help="pull everything for a year")
    ap.add_argument("--refresh", action="store_true")
    for name in PULLS:
        ap.add_argument(f"--{name}", type=int, metavar="YEAR")
    args = ap.parse_args()

    if args.check or not any([args.all] + [getattr(args, n) for n in PULLS]):
        ok = check_key()
        return 0 if ok else 1

    years = {}
    if args.all:
        years = {n: args.all for n in PULLS}
    for n in PULLS:
        y = getattr(args, n)
        if y:
            years[n] = y

    for name, year in years.items():
        try:
            df = PULLS[name](year, refresh=args.refresh)
            out = os.path.join(DATA_DIR, f"cfbd_{name}_{year}.csv")
            df.to_csv(out, index=False)
            print(f"  {name:12s} {year}: {len(df):>5} rows -> {os.path.basename(out)}")
        except RuntimeError as e:
            print(f"  {name:12s} {year}: FAILED -- {e}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
