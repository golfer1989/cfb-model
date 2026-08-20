#!/usr/bin/env python3
"""
Pulls real FBS college football data from ESPN's public (unauthenticated)
JSON endpoints.

Design notes -- ESPN's edge rate-limits sustained bursts with HTTP 403.
This fetcher is built to survive that:
  * every raw week response is cached to data/raw_cache/ as JSON, so a
    re-run skips everything already retrieved (fully resumable)
  * a 403 triggers a long cooldown (not a fast retry storm, which just
    extends the block)
  * requests are spaced conservatively

Outputs:
  data/real_games.csv        -- completed games for HIST_YEARS
  data/schedule_<year>.csv   -- SCHEDULE_YEAR full schedule
  data/conferences.json      -- conference id -> name map (cached)
  data/fcs_games.csv         -- with --fcs: FCS-vs-FCS results (training-only
                                augmentation; see cfb_ratings.load_fcs_games)

WHY --fcs EXISTS. real_games.csv only contains an FCS team when it plays an
FBS school -- 1-3 games each -- so all 100+ FCS opponents share one pooled
rating, and on FCS-involved games the model loses to the closing line by
2.19 RMSE versus 0.57 on FBS-vs-FBS (the market rates each FCS team
individually). Group 81 is ESPN's FCS scoreboard: fetching it gives every
FCS team a full schedule of real results, connected to the FBS graph through
the money games. Whether that actually improves prediction is decided by
`python backtest.py --fcs-ab`, never assumed.
"""

import argparse
import json
import os
import time
import urllib.request
import urllib.error

import pandas as pd

# NOTE: ESPN's edge returns 403 for most descriptive/bot-ish User-Agent
# strings (and, oddly, for full browser UA strings sent without the rest of a
# browser's header set). The bare token below is what it accepts. Do not
# "improve" this string -- it will start 403ing on every request.
HEADERS = {"User-Agent": "Mozilla/5.0"}
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
GROUPS_URL = ("http://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
              "/seasons/{year}/types/2/groups/{group}/children?lang=en&region=us&limit=50")

REQUEST_DELAY = 1.0      # polite spacing between successful requests
COOLDOWN_403 = 75.0      # long pause after a rate-limit block
MAX_ATTEMPTS = 4

from paths import DATA_DIR  # frozen-aware; see paths.py
RAW_CACHE = os.path.join(DATA_DIR, "raw_cache")

HIST_YEARS = [2023, 2024, 2025]
SCHEDULE_YEAR = 2026
REG_WEEKS = range(1, 18)
POST_WEEKS = range(1, 6)


def _get_json(url, label="", attempts=MAX_ATTEMPTS):
    last_err = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            time.sleep(REQUEST_DELAY)
            return data
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (403, 429):
                wait = COOLDOWN_403 * (attempt + 1)
                print(f"    [rate-limited{' ' + label if label else ''}] "
                      f"cooling down {wait:.0f}s (attempt {attempt+1}/{attempts})", flush=True)
                time.sleep(wait)
            else:
                time.sleep(5.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(5.0)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _cache_path(year, seasontype, week, groups=80):
    tag = "" if groups == 80 else f"_g{groups}"
    return os.path.join(RAW_CACHE, f"{year}{tag}_st{seasontype}_wk{week}.json")


def fetch_week_cached(year, seasontype, week, refresh=False, groups=80):
    """Fetch one week, using the on-disk raw cache unless refresh=True.

    groups: ESPN division group. 80 = FBS (the default everywhere), 81 = FCS.
    FCS weeks cache under a `_g81` suffix so they never collide with the FBS
    payloads the rest of the pipeline depends on.
    """
    path = _cache_path(year, seasontype, week, groups=groups)
    if not refresh and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f), True  # (data, from_cache)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry -> refetch

    url = (f"{SCOREBOARD_URL}?dates={year}&seasontype={seasontype}"
           f"&week={week}&groups={groups}&limit=300")
    data = _get_json(url, label=f"{year} st{seasontype} wk{week} g{groups}")
    os.makedirs(RAW_CACHE, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data, False


def fetch_conference_map(year, cache_path=None, groups=(80,)):
    cache_key = str(year) if tuple(groups) == (80,) else f"{year}_g{'+'.join(map(str, groups))}"
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cache_key in cached and cached[cache_key]:
            return {int(k): v for k, v in cached[cache_key].items()}

    conf_map = {}
    for group in groups:
        data = _get_json(GROUPS_URL.format(year=year, group=group),
                         label=f"conf map {year} g{group}")
        for item in data.get("items", []):
            d = _get_json(item["$ref"], label=f"conf {year}")
            conf_map[int(d["id"])] = d.get("shortName") or d.get("name")

    if cache_path:
        cached = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
        cached[cache_key] = conf_map
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cached, f, indent=2)
    return conf_map


def _game_date_et(iso):
    """The date the game is actually played on, in US Eastern.

    ESPN timestamps are UTC. A 7:30pm Eastern Saturday kickoff is 00:30 UTC
    Sunday, so taking the UTC date verbatim moved 628 of 2,764 games (22.7%)
    forward a day -- the file claimed 395 Sunday games, a day FBS essentially
    never plays, and dated the CFP National Championship 20 January when it
    was played on the 19th.

    Eastern is the convention every schedule and record book uses for US
    college football, so that is what `date` means here. The exact kickoff
    instant is preserved separately in `kickoff_utc`.
    """
    import datetime as _dt
    try:
        t = _dt.datetime.strptime(iso, "%Y-%m-%dT%H:%MZ").replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        try:
            t = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return iso[:10]
    # US Eastern: UTC-4 in season (EDT), UTC-5 after the first Sunday in
    # November. Bowl season straddles the change, so pick per-date rather than
    # applying one offset to everything.
    year = t.year
    nov1 = _dt.datetime(year, 11, 1, tzinfo=_dt.timezone.utc)
    dst_end = nov1 + _dt.timedelta(days=(6 - nov1.weekday()) % 7)   # 1st Sunday
    mar1 = _dt.datetime(year, 3, 1, tzinfo=_dt.timezone.utc)
    dst_start = mar1 + _dt.timedelta(days=((6 - mar1.weekday()) % 7) + 7)  # 2nd Sun
    offset = -4 if dst_start <= t < dst_end else -5
    return (t + _dt.timedelta(hours=offset)).strftime("%Y-%m-%d")


def extract_games(raw, conf_map):
    rows = []
    for ev in raw.get("events", []):
        try:
            comp = ev["competitions"][0]
            competitors = comp["competitors"]
            home = next((c for c in competitors if c["homeAway"] == "home"), None)
            away = next((c for c in competitors if c["homeAway"] == "away"), None)
            if home is None or away is None:
                continue

            def info(c):
                t = c["team"]
                cid = t.get("conferenceId")
                return (
                    t.get("location") or t.get("displayName"),
                    conf_map.get(int(cid)) if cid is not None else None,
                    t.get("id"),
                )

            h_name, h_conf, h_id = info(home)
            a_name, a_conf, a_id = info(away)

            def score(c):
                s = c.get("score")
                if isinstance(s, dict):
                    s = s.get("value")
                try:
                    return float(s)
                except (TypeError, ValueError):
                    return None

            rows.append({
                "event_id": ev["id"],
                # `date` stays a plain YYYY-MM-DD for display and for the
                # walk-forward's day-level ordering. `kickoff_utc` keeps the
                # FULL timestamp, which several things genuinely need:
                #
                #   * the lock window measures hours until kickoff. Parsing a
                #     bare date gives midnight UTC, which for a 16:00Z kickoff
                #     is 16 hours early -- a "30 hour" lock was really firing
                #     ~46 hours out, exactly the staleness it exists to prevent.
                #   * the stored date is the UTC date, so a night game rolls
                #     over: the 8 Jan 2024 title game (01:30Z on the 9th) is
                #     stored as 2024-01-09. Harmless for ordering, confusing
                #     when cross-checking against any other source.
                "date": _game_date_et(ev["date"]),
                "kickoff_utc": ev["date"],
                "season": ev["season"]["year"],
                "week": ev["week"]["number"],
                "status": ev["status"]["type"]["name"],
                "home_team": h_name,
                "home_conf": h_conf or "Other/FCS",
                "home_id": h_id,
                "away_team": a_name,
                "away_conf": a_conf or "Other/FCS",
                "away_id": a_id,
                "home_score": score(home),
                "away_score": score(away),
                "neutral_site": bool(comp.get("neutralSite", False)),
            })
        except (KeyError, IndexError, TypeError):
            continue
    return rows


def pull_season(year, conf_map, seasontypes_weeks, refresh=False, groups=80):
    all_rows, cached_n, fetched_n = [], 0, 0
    for seasontype, weeks in seasontypes_weeks:
        for week in weeks:
            try:
                raw, from_cache = fetch_week_cached(year, seasontype, week,
                                                    refresh=refresh, groups=groups)
            except RuntimeError as e:
                print(f"  ! {year} st{seasontype} wk{week}: {e}", flush=True)
                continue
            rows = extract_games(raw, conf_map)
            all_rows.extend(rows)
            cached_n += from_cache
            fetched_n += (not from_cache)
            tag = "cache" if from_cache else "live "
            print(f"  [{tag}] {year} st{seasontype} wk{week:>2}: {len(rows):>3} games "
                  f"(running total {len(all_rows)})", flush=True)
    print(f"  -> {year}: {fetched_n} weeks fetched live, {cached_n} from cache", flush=True)
    return all_rows


OUT_COLS = ["date", "kickoff_utc", "home_team", "home_conf", "away_team", "away_conf",
            "home_score", "away_score", "neutral_site", "season", "week",
            "event_id", "home_id", "away_id"]


def fetch_fcs(refresh=False):
    """Fetch FCS-vs-FCS results for HIST_YEARS into data/fcs_games.csv.

    Keeps only completed games where NEITHER side is FBS -- the cross-division
    money games are already in real_games.csv, and duplicating them would
    double their weight in any concatenated fit. Fully cached and resumable,
    same as the FBS pull.
    """
    conf_cache = os.path.join(DATA_DIR, "conferences.json")
    rows = []
    for year in HIST_YEARS:
        print(f"\n=== {year} (FCS) ===", flush=True)
        conf_map = fetch_conference_map(year, conf_cache, groups=(80, 81))
        rows += pull_season(year, conf_map, [(2, REG_WEEKS), (3, POST_WEEKS)],
                            refresh=refresh, groups=81)
    if not rows:
        print("No FCS rows fetched.")
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset="event_id")
    done = df[df["status"] == "STATUS_FINAL"].dropna(subset=["home_score", "away_score"])

    # drop anything already in real_games.csv, and any game with an FBS side
    rg_path = os.path.join(DATA_DIR, "real_games.csv")
    if os.path.exists(rg_path):
        known = set(pd.read_csv(rg_path)["event_id"].astype(str))
        done = done[~done["event_id"].astype(str).isin(known)]
    from cfb_ratings import FBS_CONFERENCES
    done = done[~done["home_conf"].isin(FBS_CONFERENCES)
                & ~done["away_conf"].isin(FBS_CONFERENCES)]

    done = done[OUT_COLS].sort_values("date")
    out = os.path.join(DATA_DIR, "fcs_games.csv")
    done.to_csv(out, index=False)
    n_teams = len(set(done["home_team"]) | set(done["away_team"]))
    print(f"\nWrote data/fcs_games.csv: {len(done)} completed FCS-vs-FCS games, "
          f"{n_teams} teams, {HIST_YEARS[0]}-{HIST_YEARS[-1]}")
    print("Next: python backtest.py --fcs-ab   (validates whether augmentation "
          "helps; nothing turns on until it wins that test)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fetch real ESPN college football data (resumable).")
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore raw cache and refetch everything")
    ap.add_argument("--refresh-current", action="store_true",
                    help="Refetch only SCHEDULE_YEAR (for weekly updates)")
    ap.add_argument("--fcs", action="store_true",
                    help="Fetch FCS-vs-FCS schedules (group 81) into "
                         "data/fcs_games.csv and exit. Validate with "
                         "`python backtest.py --fcs-ab` afterwards.")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_CACHE, exist_ok=True)
    conf_cache = os.path.join(DATA_DIR, "conferences.json")

    if args.fcs:
        fetch_fcs(refresh=args.refresh)
        return

    hist_rows = []
    for year in HIST_YEARS:
        print(f"\n=== {year} (historical) ===", flush=True)
        conf_map = fetch_conference_map(year, conf_cache)
        hist_rows += pull_season(year, conf_map,
                                 [(2, REG_WEEKS), (3, POST_WEEKS)], refresh=args.refresh)

    hist = pd.DataFrame(hist_rows).drop_duplicates(subset="event_id")
    done = hist[(hist["status"] == "STATUS_FINAL")].dropna(subset=["home_score", "away_score"])
    done = done[OUT_COLS].sort_values("date")
    done.to_csv(os.path.join(DATA_DIR, "real_games.csv"), index=False)
    print(f"\nWrote data/real_games.csv: {len(done)} completed games, "
          f"{len(set(done['home_team']) | set(done['away_team']))} teams, "
          f"{HIST_YEARS[0]}-{HIST_YEARS[-1]}", flush=True)

    print(f"\n=== {SCHEDULE_YEAR} (schedule) ===", flush=True)
    conf_map = fetch_conference_map(SCHEDULE_YEAR, conf_cache)
    sched_rows = pull_season(SCHEDULE_YEAR, conf_map, [(2, REG_WEEKS), (3, POST_WEEKS)],
                             refresh=args.refresh or args.refresh_current)
    sched = pd.DataFrame(sched_rows).drop_duplicates(subset="event_id").sort_values(["week", "date"])
    sched.to_csv(os.path.join(DATA_DIR, f"schedule_{SCHEDULE_YEAR}.csv"), index=False)
    n_final = int((sched["status"] == "STATUS_FINAL").sum())
    print(f"Wrote data/schedule_{SCHEDULE_YEAR}.csv: {len(sched)} games "
          f"({n_final} already final, {len(sched)-n_final} upcoming)", flush=True)


if __name__ == "__main__":
    main()
