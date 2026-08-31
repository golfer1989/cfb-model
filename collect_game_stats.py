#!/usr/bin/env python3
"""Collect per-game team box statistics from CFBD -- the raw material for
future totals experiments (pace/tempo above all).

WHY THIS EXISTS. The model's weakest leg is totals, and the one untested
improvement candidate is matchup pace (plays per game). Testing it properly
needs PER-GAME box stats -- which nothing in this project was storing. This
module fetches CFBD's /games/teams payload (one call per season-week) and
saves it VERBATIM under data/raw_cache/cfbd/, where the website's commit
step persists it in the repo. No schema is imposed now: raw payloads let a
future analysis derive whatever it wants (plays, drives, penalties, third
downs) without re-fetching.

COST. Backfill is a one-time ~35 calls (2024, 2025, 2026-to-date); after
that, one call per new completed week -- noise against the free-tier quota.

Runs only on website builds (needs the CFBD key); failures never break the
build -- data collection is best-effort by design.
"""

import json
import os

from paths import DATA_DIR
from fetch_cfbd import _get, get_key

OUT_DIR = os.path.join(DATA_DIR, "raw_cache", "cfbd")

BACKFILL = [(2024, range(1, 17)), (2025, range(1, 17))]


def _path(season, week, season_type="regular"):
    tag = "" if season_type == "regular" else f"_{season_type}"
    return os.path.join(OUT_DIR, f"gamestats_{season}_wk{week}{tag}.json")


def _have(season, week, season_type="regular"):
    # existence is sufficient: only non-empty payloads are ever written, so
    # a size heuristic would just invite refetch loops (and one did, in the
    # unit test, before this comment existed)
    p = _path(season, week, season_type)
    return os.path.exists(p) and os.path.getsize(p) > 2


def _fetch_week(season, week, season_type="regular"):
    """One call; saves only non-empty payloads so a not-yet-played week is
    retried next run instead of being cached as an empty file."""
    data = _get("/games/teams", {"year": int(season), "week": int(week),
                                 "seasonType": season_type})
    if data:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(_path(season, week, season_type), "w", encoding="utf-8") as f:
            json.dump(data, f)
        return len(data)
    return 0


def collect(current_season, played_weeks=None, budget=40):
    """Fill every missing week, oldest first, within a per-run call budget.

    `played_weeks`: weeks of the current season with at least one completed
    game (from the schedule); only those are worth asking for.
    """
    if get_key() is None:
        return "no CFBD key; skipped"
    todo = []
    for season, weeks in BACKFILL:
        for w in weeks:
            if not _have(season, w):
                todo.append((season, w))
    for w in sorted(played_weeks or []):
        if not _have(current_season, w):
            todo.append((current_season, w))

    fetched, games = 0, 0
    for season, w in todo[:budget]:
        try:
            n = _fetch_week(season, w)
            fetched += 1
            games += n
        except RuntimeError as e:
            return f"stopped after {fetched} week(s): {e}"
    if not todo:
        return "up to date"
    return (f"fetched {fetched} week(s), {games} game payloads"
            + (f"; {len(todo) - fetched} week(s) remaining" if len(todo) > fetched else ""))


def played_weeks_from_schedule(season):
    """Weeks with at least one STATUS_FINAL game, per the schedule file."""
    import csv
    path = os.path.join(DATA_DIR, f"schedule_{season}.csv")
    weeks = set()
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "STATUS_FINAL":
                    try:
                        weeks.add(int(float(r.get("week") or 0)))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return sorted(w for w in weeks if w > 0)


if __name__ == "__main__":
    from paths import current_cfb_season
    s = current_cfb_season()
    print(collect(s, played_weeks_from_schedule(s)))
