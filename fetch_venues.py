#!/usr/bin/env python3
"""
Venue and weather enrichment.

WHY WEATHER IS WORTH TRYING WHEN ALMOST NOTHING ELSE WAS

This project has already tested and rejected ~50 box-score efficiency features,
season-to-date efficiency, turnover luck, style matchups, team-specific home
field, head-to-head history, and 139 conditional subsets. They all failed for
the same underlying reason: each was another proxy for team strength, which the
ratings already measure directly from scores.

Weather is different in kind. Wind and precipitation are uncorrelated with how
good a team is, so whatever they explain is information the ratings cannot
already contain. That does not make the effect large -- the honest prior is
that wind mainly suppresses TOTALS and barely moves margins -- but it is the
one candidate that is not structurally redundant.

SOURCES (both free, neither needs an API key)
  ESPN summary -> gameInfo.venue: name, city, state, zip, indoor/grass flags
  Open-Meteo   -> historical hourly weather by lat/lon, back to 1940, and
                  forecasts for upcoming games
                  https://open-meteo.com/  (free for non-commercial use)

Outputs:
  data/venues.csv       one row per venue: id, name, city, state, lat, lon, indoor
  data/game_weather.csv one row per game: temp, wind, precip at kickoff

Usage:
  python fetch_venues.py --venues          build the venue table
  python fetch_venues.py --weather         attach weather to completed games
  python fetch_venues.py --forecast        weather for upcoming scheduled games
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from paths import DATA_DIR

HEADERS = {"User-Agent": "Mozilla/5.0"}   # see fetch_espn_data.py
SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/football/"
           "college-football/summary?event={eid}")
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           "&start_date={d}&end_date={d}&hourly=temperature_2m,wind_speed_10m,"
           "precipitation,relative_humidity_2m&temperature_unit=fahrenheit"
           "&wind_speed_unit=mph&precipitation_unit=inch&timezone=UTC")
FORECAST = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&start_date={d}&end_date={d}&hourly=temperature_2m,wind_speed_10m,"
            "precipitation_probability,precipitation&temperature_unit=fahrenheit"
            "&wind_speed_unit=mph&precipitation_unit=inch&timezone=UTC")

VENUES = os.path.join(DATA_DIR, "venues.csv")
WEATHER = os.path.join(DATA_DIR, "game_weather.csv")
CACHE = os.path.join(DATA_DIR, "raw_cache", "venues")

# Domed and retractable-roof stadiums. Weather is irrelevant inside them, and
# leaving them in would inject pure noise -- a windy day outside a dome tells
# you nothing about the game.
INDOOR_HINTS = ("dome", "superdome", "lucas oil", "ford field", "at&t stadium",
                "alamodome", "carrier dome", "jma wireless", "fargodome",
                "u.s. bank", "caesars superdome", "mercedes-benz", "nrg stadium",
                "state farm stadium", "allegiant")


def _get(url, tries=3, timeout=25):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
            time.sleep(2 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    return None


def _is_indoor(name, venue):
    if venue.get("indoor") is True:
        return True
    n = (name or "").lower()
    return any(h in n for h in INDOOR_HINTS)


def build_venues(games, limit=None, verbose=True):
    """One ESPN summary call per game to learn its venue, then geocode each
    DISTINCT venue once. Venues repeat heavily, so this is far cheaper than it
    looks -- ~130 stadiums cover thousands of games."""
    os.makedirs(CACHE, exist_ok=True)
    known = {}
    if os.path.exists(VENUES):
        # .to_dict("records") NOT .iterrows(): iterrows yields Series, and
        # mixing Series with the plain dicts added below makes the final
        # pd.DataFrame(list(known.values())) raise
        #   AttributeError: 'dict' object has no attribute 'dtype'
        # after every request has already been made -- losing the whole run.
        known = {str(r["venue_id"]): r
                 for r in pd.read_csv(VENUES).to_dict("records")}

    eids = games["event_id"].astype(str).tolist()
    if limit:
        eids = eids[:limit]

    game_venue = {}
    for i, eid in enumerate(eids, 1):
        cpath = os.path.join(CACHE, f"{eid}.json")
        if os.path.exists(cpath):
            try:
                with open(cpath, encoding="utf-8") as f:
                    info = json.load(f)
            except (json.JSONDecodeError, OSError):
                info = None
        else:
            d = _get(SUMMARY.format(eid=eid))
            info = (d or {}).get("gameInfo", {}).get("venue")
            with open(cpath, "w", encoding="utf-8") as f:
                json.dump(info, f)
            time.sleep(0.35)
        if not info:
            continue
        vid = str(info.get("id"))
        game_venue[eid] = vid
        if vid not in known:
            addr = info.get("address") or {}
            known[vid] = {
                "venue_id": vid,
                "venue": info.get("fullName"),
                "city": addr.get("city"),
                "state": addr.get("state"),
                "zip": addr.get("zipCode"),
                "grass": info.get("grass"),
                "indoor": _is_indoor(info.get("fullName"), info),
                "lat": None, "lon": None,
            }
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(eids)} games, {len(known)} distinct venues", flush=True)

    # geocode any venue we do not have coordinates for
    todo = [v for v in known.values() if not v.get("lat")]
    if verbose and todo:
        print(f"  geocoding {len(todo)} venues...", flush=True)
    for v in todo:
        q = ", ".join([x for x in (v.get("city"), v.get("state")) if x])
        if not q:
            continue
        d = _get(GEOCODE.format(q=urllib.parse.quote(q)))
        res = (d or {}).get("results") or []
        if res:
            v["lat"], v["lon"] = res[0]["latitude"], res[0]["longitude"]
        time.sleep(0.3)

    vdf = pd.DataFrame(list(known.values()))
    vdf.to_csv(VENUES, index=False)
    return vdf, game_venue


RANGE_ARCHIVE = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
                 "&start_date={d0}&end_date={d1}&hourly=temperature_2m,wind_speed_10m,"
                 "precipitation,relative_humidity_2m&temperature_unit=fahrenheit"
                 "&wind_speed_unit=mph&precipitation_unit=inch&timezone=UTC")


def attach_weather_bulk(games, venues, game_venue, verbose=True):
    """Fetch weather one VENUE-SEASON at a time instead of one game at a time.

    Open-Meteo accepts a date range, and ~130 stadiums host thousands of games,
    so grouping by (venue, season) turns 2,764 sequential requests into a few
    hundred. Same data, roughly a seventh of the wall-clock and a seventh of
    the load on a free service.
    """
    vmap = {str(r["venue_id"]): r for _, r in venues.iterrows()}
    g = games.copy()
    g["event_id"] = g["event_id"].astype(str)
    g["venue_id"] = g["event_id"].map(game_venue)
    g["kick"] = pd.to_datetime(g.get("kickoff_utc", g["date"]), errors="coerce", utc=True)
    g = g.dropna(subset=["venue_id", "kick"])

    have = set()
    rows = []
    if os.path.exists(WEATHER):
        w = pd.read_csv(WEATHER)
        rows = w.to_dict("records")
        have = {str(e) for e in w["event_id"]}
    g = g[~g["event_id"].isin(have)]
    if not len(g):
        if verbose:
            print("  weather already complete")
        return pd.DataFrame(rows)

    groups = list(g.groupby([g["venue_id"], g["kick"].dt.year]))
    if verbose:
        print(f"  {len(g)} games across {len(groups)} venue-year groups", flush=True)

    for n, ((vid, _yr), sub) in enumerate(groups, 1):
        v = vmap.get(str(vid))
        if v is None:
            continue
        base = {"venue_id": vid, "venue": v.get("venue"),
                "indoor": bool(v.get("indoor")), "grass": v.get("grass")}
        if v.get("indoor") or pd.isna(v.get("lat")):
            for _, x in sub.iterrows():
                rows.append({**base, "event_id": x["event_id"], "temp_f": None,
                             "wind_mph": None, "precip_in": None, "humidity": None})
            continue

        d0 = sub["kick"].min().strftime("%Y-%m-%d")
        d1 = sub["kick"].max().strftime("%Y-%m-%d")
        d = _get(RANGE_ARCHIVE.format(lat=v["lat"], lon=v["lon"], d0=d0, d1=d1))
        h = (d or {}).get("hourly") or {}
        if not h.get("time"):
            continue
        idx = {t: i for i, t in enumerate(h["time"])}
        for _, x in sub.iterrows():
            key = x["kick"].strftime("%Y-%m-%dT%H:00")
            i = idx.get(key)
            if i is None:                      # fall back to nearest listed hour
                key2 = x["kick"].strftime("%Y-%m-%dT") + "18:00"
                i = idx.get(key2)
            if i is None:
                continue
            rows.append({
                **base, "event_id": x["event_id"],
                "temp_f": h["temperature_2m"][i],
                "wind_mph": h["wind_speed_10m"][i],
                "precip_in": h["precipitation"][i],
                "humidity": (h.get("relative_humidity_2m") or [None] * (i + 1))[i],
            })
        time.sleep(0.3)
        if verbose and n % 25 == 0:
            print(f"  {n}/{len(groups)} groups, {len(rows)} rows", flush=True)

    wdf = pd.DataFrame(rows).drop_duplicates("event_id")
    wdf.to_csv(WEATHER, index=False)
    return wdf


def attach_weather(games, venues, game_venue, forecast=False, verbose=True):
    """Kickoff-hour weather per game. Indoor venues are recorded as indoor and
    given no weather at all rather than a misleading outdoor reading."""
    vmap = {str(r["venue_id"]): r for _, r in venues.iterrows()}
    have = {}
    if os.path.exists(WEATHER):
        w = pd.read_csv(WEATHER)
        have = {str(e) for e in w["event_id"]}
        rows = w.to_dict("records")
    else:
        rows = []

    todo = games[~games["event_id"].astype(str).isin(have)]
    if verbose:
        print(f"  {len(todo)} game(s) need weather", flush=True)

    for i, (_, g) in enumerate(todo.iterrows(), 1):
        eid = str(g["event_id"])
        vid = game_venue.get(eid)
        v = vmap.get(str(vid)) if vid else None
        if v is None or pd.isna(v.get("lat")):
            continue

        rec = {"event_id": eid, "venue_id": vid, "venue": v.get("venue"),
               "indoor": bool(v.get("indoor")), "grass": v.get("grass")}
        if v.get("indoor"):
            # no weather inside a dome; leaving these blank keeps them out of
            # any weather regression instead of feeding it irrelevant numbers
            rows.append({**rec, "temp_f": None, "wind_mph": None,
                         "precip_in": None, "humidity": None})
        else:
            kick = pd.to_datetime(g.get("kickoff_utc") or g.get("date"),
                                  errors="coerce", utc=True)
            if pd.isna(kick):
                continue
            url = (FORECAST if forecast else ARCHIVE).format(
                lat=v["lat"], lon=v["lon"], d=kick.strftime("%Y-%m-%d"))
            d = _get(url)
            h = (d or {}).get("hourly") or {}
            if not h.get("time"):
                continue
            hour = min(int(kick.hour), len(h["time"]) - 1)
            rows.append({
                **rec,
                "temp_f": h.get("temperature_2m", [None])[hour],
                "wind_mph": h.get("wind_speed_10m", [None])[hour],
                "precip_in": h.get("precipitation", [None])[hour],
                "humidity": (h.get("relative_humidity_2m") or [None] * (hour + 1))[hour],
            })
            time.sleep(0.25)
        if verbose and i % 100 == 0:
            print(f"  weather {i}/{len(todo)}", flush=True)

    wdf = pd.DataFrame(rows).drop_duplicates("event_id")
    wdf.to_csv(WEATHER, index=False)
    return wdf


def main():
    ap = argparse.ArgumentParser(description="Venue + weather enrichment.")
    ap.add_argument("--games", default=os.path.join(DATA_DIR, "real_games.csv"))
    ap.add_argument("--venues", action="store_true", help="build the venue table")
    ap.add_argument("--weather", action="store_true", help="historical weather")
    ap.add_argument("--forecast", action="store_true", help="forecast for upcoming games")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    games = pd.read_csv(args.games)
    if args.limit:
        games = games.head(args.limit)

    print(f"Building venue table from {len(games)} games...")
    vdf, gv = build_venues(games, limit=args.limit)
    print(f"  {len(vdf)} distinct venues, "
          f"{int(vdf['lat'].notna().sum())} geocoded, "
          f"{int(vdf['indoor'].sum())} indoor")

    if args.weather or args.forecast:
        print("\nAttaching weather...")
        wdf = (attach_weather(games, vdf, gv, forecast=True) if args.forecast
               else attach_weather_bulk(games, vdf, gv))
        out = wdf[wdf["indoor"] == False]  # noqa: E712
        print(f"  {len(wdf)} games, {len(out)} outdoor with readings")
        if len(out):
            print(f"  wind  mean {out['wind_mph'].mean():.1f} mph, "
                  f"max {out['wind_mph'].max():.1f}")
            print(f"  temp  mean {out['temp_f'].mean():.1f} F, "
                  f"min {out['temp_f'].min():.1f}")
            print(f"  games with measurable precip: "
                  f"{int((out['precip_in'] > 0.01).sum())}")


if __name__ == "__main__":
    main()
