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

Since 2026-11-10 it also collects CFBD's play-by-play team metrics per game
(/stats/game/advanced, garbage time excluded) for the play-by-play ratings
experiment: a one-time ~62-call backfill (2023-2025 plus 2026 to date) on
its own 35-call-per-run budget, then one call per new week. Each run writes
data/raw_cache/cfbd/collect_status.json so problems are visible in the repo.

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


# ---------------------------------------------------------------------------
# Season-level tables the experiments need
# ---------------------------------------------------------------------------
# Added 2026-09-27 after the preregistered churn verdict was blocked TWICE:
# cfbd_portal_<year>.csv existed only in the owner's Google Drive folder, so
# analysis could not run whenever his machine was offline. Fetching it here
# puts it in the repo, where every future run (and every future season) finds
# it automatically. One call per season, skipped once the file exists.

SEASON_TABLES = {
    "portal": "/player/portal",
    "returning": "/player/returning",
    "talent": "/talent",
}


def _has_rows(path):
    """True when the CSV exists and holds at least one data row."""
    try:
        with open(path, encoding="utf-8") as f:
            f.readline()                      # header
            return bool(f.readline().strip())
    except OSError:
        return False


def collect_season_tables(season):
    import csv as _csv
    notes = []
    for name, path in SEASON_TABLES.items():
        out = os.path.join(DATA_DIR, f"cfbd_{name}_{season}.csv")
        if _has_rows(out):          # header-only/empty files are refetched
            continue
        try:
            rows = _get(path, {"year": int(season)}) or []
        except RuntimeError as e:
            notes.append(f"{name} FAILED ({str(e)[:40]})")
            continue
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        notes.append(f"{name} {season}: {len(rows)} rows")
    return "; ".join(notes)


def collect(current_season, played_weeks=None, budget=40):
    """Fill every missing week, oldest first, within a per-run call budget.

    `played_weeks`: weeks of the current season with at least one completed
    game (from the schedule); only those are worth asking for.
    """
    if get_key() is None:
        return "no CFBD key; skipped"
    season_note = collect_season_tables(current_season)
    todo = []
    for season, weeks in BACKFILL:
        for w in weeks:
            if not _have(season, w):
                todo.append((season, w))
    # Current season: a week saved before all its games finished must be
    # fetched AGAIN. The first version cached a week the first time ANY game
    # in it was final, so a week first fetched on a Thursday night froze with
    # 1-3 games forever (2026 weeks 2-9, found 2026-10-30).
    # The staleness test must compare GAME IDS, not counts. The 2026-10-30
    # version compared "games stored" with "schedule finals", but CFBD's
    # payload also carries FCS-vs-FCS games the schedule never lists, so a
    # Saturday-evening fetch already "exceeded" the finals count and froze:
    # weeks 9-10 held only 29/56 and 45/62 of their FBS-vs-FBS games (found
    # 2026-11-08). A week is complete once it holds >= 95% of the schedule's
    # final event ids; the 5% slack stops one game CFBD never publishes from
    # triggering a refetch (an API call) on every run forever.
    finals = final_ids_by_week_from_schedule(current_season)
    for w in sorted(played_weeks or []):
        want = finals.get(w, set())
        missing = len(want - _stored_ids(current_season, w))
        if not _have(current_season, w) or missing > 0.05 * len(want):
            todo.append((current_season, w))

    fetched, games, box_note = 0, 0, None
    for season, w in todo[:budget]:
        try:
            n = _fetch_week(season, w)
            fetched += 1
            games += n
        except RuntimeError as e:
            box_note = f"box scores stopped after {fetched} week(s): {e}"
            break
    if box_note is None:
        box_note = ("box scores up to date" if not todo else
                    f"box scores: fetched {fetched} week(s), {games} game payloads"
                    + (f"; {len(todo) - fetched} week(s) remaining"
                       if len(todo) > fetched else ""))
    # play-by-play metrics run on their OWN budget and error handling, so a
    # problem there can never starve or stop the box-score collection
    adv_note, adv_status = collect_advanced(current_season, played_weeks)
    _write_status({"utc": _now_utc(), "season_tables": season_note,
                   "box": box_note, "advgame": adv_status})
    prefix = (season_note + "; ") if season_note else ""
    return prefix + box_note + "; " + adv_note


def _stored_ids(season, week, season_type="regular"):
    """Event ids (as strings) in the saved payload for this week."""
    try:
        with open(_path(season, week, season_type), encoding="utf-8") as f:
            return {str(g.get("id")) for g in (json.load(f) or [])}
    except (OSError, ValueError, AttributeError):
        return set()


def _is_bowl(row, week):
    """Postseason games are filed as week 1 in this data (CFBD numbers the
    postseason separately). A 'week 1' game kicking off in December or
    January is a bowl, never part of the regular-season week-1 payload --
    counting it there would make week 1 look stale and refetch forever."""
    d = str(row.get("kickoff_utc") or row.get("date") or "")
    return week == 1 and d[5:7] in ("12", "01")


def final_ids_by_week_from_schedule(season, fbs_only=False):
    """{week: set of STATUS_FINAL regular-season event ids} from the
    schedule file; fbs_only keeps FBS-vs-FBS games only."""
    import csv
    path = os.path.join(DATA_DIR, f"schedule_{season}.csv")
    fbs = _fbs_conferences() if fbs_only else None
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "STATUS_FINAL":
                    try:
                        w = int(float(r.get("week") or 0))
                    except ValueError:
                        continue
                    if _is_bowl(r, w):
                        continue
                    if fbs and (r.get("home_conf") not in fbs or r.get("away_conf") not in fbs):
                        continue
                    eid = str(r.get("event_id") or "").strip()
                    if eid:
                        out.setdefault(w, set()).add(eid)
    except FileNotFoundError:
        pass
    return out


# ---------------------------------------------------------------------------
# Play-by-play team metrics per game (added 2026-11-10)
# ---------------------------------------------------------------------------
# The owner chose a play-by-play rebuild of the ratings (2026-11-09): rate
# teams on success rate, EPA per play (CFBD calls it PPA) and explosiveness
# with garbage time removed, instead of on final scores alone. CFBD computes
# these from play-by-play; /stats/game/advanced returns them per team-game,
# one call per season-week. Saved VERBATIM like the box scores, so the
# experiment can derive whatever it needs without refetching. Collected for
# 2023 (priors for testing 2024) through the current season, plus every
# completed season's postseason. NOTHING in the model reads these files
# until experiments/pbp_lab.py has passed its preregistered test.

ADV_PATH = "/stats/game/advanced"
ADV_FIRST_SEASON = 2023
ADV_WEEKS = range(1, 17)
ADV_BUDGET = 35          # calls per run, separate from the box-score budget
STATUS_PATH = os.path.join(OUT_DIR, "collect_status.json")
_FBS_FALLBACK = {"SEC", "Big Ten", "Big 12", "ACC", "American", "Sun Belt",
                 "MAC", "CUSA", "Mountain West", "Pac-12", "FBS Indep."}


def _fbs_conferences():
    try:
        import cfb_ratings as R
        return set(R.FBS_CONFERENCES)
    except Exception:  # noqa: BLE001
        return set(_FBS_FALLBACK)


def _now_utc():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(status):
    """A small JSON the website commits, so a failure is diagnosable from the
    repo itself (the Actions log is not reachable from outside GitHub)."""
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=1, default=str)
    except OSError:
        pass


def _adv_path(season, week, season_type="regular"):
    tag = "" if season_type == "regular" else f"_{season_type}"
    return os.path.join(OUT_DIR, f"advgame_{season}_wk{week}{tag}.json")


def _adv_fetch(season, week, season_type="regular", keep_empty=False):
    """One call. Completed seasons keep even an empty answer (a week with no
    games must not be re-asked every run); the current season saves only
    non-empty payloads, so a not-yet-played week is simply retried."""
    data = _get(ADV_PATH, {"year": int(season), "week": int(week),
                           "seasonType": season_type,
                           "excludeGarbageTime": "true"}) or []
    if data or keep_empty:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(_adv_path(season, week, season_type), "w", encoding="utf-8") as f:
            json.dump(data, f)
    return len(data)


def _adv_ids(season, week, season_type="regular"):
    """Game ids present in a saved advanced-stats payload (team-game rows)."""
    try:
        with open(_adv_path(season, week, season_type), encoding="utf-8") as f:
            rows = json.load(f) or []
        return {str(r.get("gameId", r.get("game_id", r.get("id")))) for r in rows}
    except (OSError, ValueError, AttributeError):
        return set()


def collect_advanced(current_season, played_weeks=None, budget=ADV_BUDGET):
    """Returns (one-line note, status dict). Never raises."""
    todo = []
    for season in range(ADV_FIRST_SEASON, int(current_season)):
        for w in ADV_WEEKS:
            if not os.path.exists(_adv_path(season, w)):
                todo.append((season, w, "regular", True))
        if not os.path.exists(_adv_path(season, 1, "postseason")):
            todo.append((season, 1, "postseason", True))
    # current season: same game-ID staleness rule as the box scores, judged
    # on FBS-vs-FBS finals only (those always have play-by-play; some FCS
    # opponents do not, and must not cause endless refetching)
    finals = final_ids_by_week_from_schedule(current_season, fbs_only=True)
    for w in sorted(played_weeks or []):
        want = finals.get(w, set())
        have = _adv_ids(current_season, w)
        if not os.path.exists(_adv_path(current_season, w)) or len(want - have) > 0.05 * len(want):
            todo.append((int(current_season), w, "regular", False))

    status = {"endpoint": ADV_PATH, "fetched": [], "error": None,
              "remaining": max(len(todo) - budget, 0)}
    for season, w, st, keep in todo[:budget]:
        try:
            n = _adv_fetch(season, w, st, keep_empty=keep)
            status["fetched"].append([season, w, st, n])
        except Exception as e:  # noqa: BLE001 -- best-effort, never break a build
            status["error"] = f"{season} wk{w} {st}: {type(e).__name__}: {str(e)[:200]}"
            status["remaining"] = len(todo) - len(status["fetched"])
            break
    if not todo:
        note = "play-by-play stats up to date"
    elif status["error"]:
        note = f"play-by-play stats stopped after {len(status['fetched'])} week(s): {status['error']}"
    else:
        rows = sum(x[3] for x in status["fetched"])
        note = (f"play-by-play stats: fetched {len(status['fetched'])} week(s), {rows} team-game rows"
                + (f"; {status['remaining']} remaining" if status["remaining"] else ""))
    return note, status


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
                        w = int(float(r.get("week") or 0))
                    except ValueError:
                        continue
                    if not _is_bowl(r, w):
                        weeks.add(w)
    except FileNotFoundError:
        pass
    return sorted(w for w in weeks if w > 0)


if __name__ == "__main__":
    from paths import current_cfb_season
    s = current_cfb_season()
    print(collect(s, played_weeks_from_schedule(s)))
