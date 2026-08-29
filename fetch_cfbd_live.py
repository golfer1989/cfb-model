#!/usr/bin/env python3
"""Live game/line feed from CollegeFootballData.com -- the website's data path.

WHY THIS EXISTS. The website builds run on GitHub's cloud servers, and ESPN
rate-limits requests from that datacenter IP space so hard that a ~30-request
refresh burns 3h45m in retry cooldowns (measured: runs #1 and #2 finished
within 8 seconds of each other at 3h45m -- deterministic cooldown arithmetic,
not workload). CFBD authenticates by key, does not care where requests come
from, and serves a whole season of games or lines in ONE call, so a complete
build needs ~3 requests instead of ~30-90.

SCOPE. This module is used ONLY when the environment says so:

    DATA_SOURCE=cfbd      (set by .github/workflows/daily.yml)

The desktop exe never sets it, keeps its ESPN path untouched, and continues
to work exactly as before from a home connection.

WHAT IT DELIBERATELY DOES NOT DO. No injuries or rosters -- CFBD has no
injury endpoint. The market line (which reprices injury news within minutes)
remains the model's injury channel on the website; the explicit ESPN QB check
stays in the code as a cheap opportunistic attempt (see pregame.FAST_MODE).

COMPATIBILITY CONTRACT with the ESPN pipeline this substitutes for:
  * event ids     CFBD game `id` IS the ESPN event id (same lineage). This is
                  load-bearing (the ledger joins on it), so it is not assumed:
                  selfcheck() measures the overlap against real_games.csv and
                  the module refuses to merge if the ids do not line up.
  * team names    canonicalized against the names already in real_games.csv
                  so a team never splits into two rating nodes. Unknown names
                  are kept (a newly promoted team is legitimately new) and
                  counted in the selfcheck report.
  * conferences   CFBD long names -> the ESPN short names the ratings expect
                  ("American Athletic" -> "American"). Non-FBS -> "Other/FCS".
  * dates         `date` is the US-Eastern game date via the SAME helper the
                  ESPN path uses; `kickoff_utc` keeps the full instant.
  * spread sign   home-perspective, negative = home favoured. Derived from
                  CFBD's formattedSpread ("Georgia -13.5") matched against
                  the home/away names, cross-checked against the numeric
                  `spread` field; disagreements are counted, formattedSpread
                  wins (same philosophy as fetch_odds' details cross-check).
"""

import datetime as dt
import os
import re

import pandas as pd

from paths import DATA_DIR
from fetch_cfbd import _get, get_key
from fetch_espn_data import _game_date_et

SELFCHECK_PATH = os.path.join(DATA_DIR, "cfbd_selfcheck.txt")

# ---------------------------------------------------------------------------
# switches
# ---------------------------------------------------------------------------

def enabled():
    """True only when the environment explicitly selects CFBD and a key exists."""
    if os.environ.get("DATA_SOURCE", "").strip().lower() != "cfbd":
        return False
    if get_key() is None:
        print("  DATA_SOURCE=cfbd but no CFBD key found (CFBD_API_KEY or "
              "data/cfbd_key.txt) -- falling back to the ESPN path")
        return False
    return True


# ---------------------------------------------------------------------------
# name + conference canonicalization
# ---------------------------------------------------------------------------

# CFBD long conference names -> the short names the ESPN pipeline stores and
# cfb_ratings.FBS_CONFERENCES expects. Anything not in this map and not
# classified FBS collapses to "Other/FCS", same as the ESPN path.
CONF_MAP = {
    "American Athletic": "American",
    "Conference USA": "CUSA",
    "Mid-American": "MAC",
    "FBS Independents": "FBS Indep.",
    "Mountain West": "Mountain West",
    "Pac-12": "Pac-12",
    "SEC": "SEC",
    "Big Ten": "Big Ten",
    "Big 12": "Big 12",
    "ACC": "ACC",
    "Sun Belt": "Sun Belt",
}

# CFBD school name -> the name already used in real_games.csv. Applied only
# when the CFBD name is not already present in the historical name set, so an
# alias can never override an exact match. Unmatched names are reported by
# selfcheck(); extend this map from that report, not from guesswork.
NAME_ALIASES = {
    "Connecticut": "UConn",
    "Louisiana Monroe": "UL Monroe",
    "Southern Mississippi": "Southern Miss",
    "Hawaii": "Hawai'i",
    "San Jose State": "San José State",
    "UMass": "Massachusetts",
    "Sam Houston State": "Sam Houston",
}

_known_names = None


def _known():
    """Team names the ratings were trained on, loaded once."""
    global _known_names
    if _known_names is None:
        path = os.path.join(DATA_DIR, "real_games.csv")
        try:
            g = pd.read_csv(path, usecols=["home_team", "away_team"])
            _known_names = set(g["home_team"]) | set(g["away_team"])
        except (OSError, ValueError, KeyError):
            _known_names = set()
    return _known_names


def canon_name(name):
    if name is None:
        return None
    name = str(name).strip()
    if name in _known():
        return name
    return NAME_ALIASES.get(name, name)


def _conf(cfbd_conf, classification):
    cls = str(classification or "").lower()
    if cls and cls != "fbs":
        return "Other/FCS"
    if cfbd_conf in CONF_MAP:
        return CONF_MAP[cfbd_conf]
    # FBS but an unmapped conference name: keep it visible rather than
    # silently demoting a real FBS team to the pooled FCS node.
    return cfbd_conf or "Other/FCS"


# ---------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------

_memo = {}


def _norm_kick(iso):
    """CFBD's '2026-08-29T16:00:00.000Z' -> ESPN's '2026-08-29T16:00Z'.

    Not cosmetic: the workflow's pre-wave guard parses kickoff_utc with the
    strict ESPN minute-precision format, so an unnormalized CFBD timestamp
    would silently disable every pre-kickoff build.
    """
    s = str(iso).strip()
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    except ValueError:
        return s


def season_games(season, season_type="both"):
    """All games for a season as canonical row dicts (ESPN-pipeline schema).

    One API call per season type, memoized for the life of the process so the
    schedule refresh, the results refresh and the grader share the same fetch.
    """
    types = ("regular", "postseason") if season_type == "both" else (season_type,)
    rows = []
    for st in types:
        k = (int(season), st)
        if k not in _memo:
            try:
                _memo[k] = _get("/games", {"year": int(season),
                                           "seasonType": st}) or []
            except RuntimeError as e:
                print(f"  CFBD /games {season} {st}: {e}")
                _memo[k] = []
        rows += _memo[k]

    out = []
    for g in rows:
        gid = g.get("id")
        home = canon_name(g.get("homeTeam"))
        away = canon_name(g.get("awayTeam"))
        kick = g.get("startDate")
        if gid is None or not home or not away or not kick:
            continue
        # CFBD's /games returns EVERY division -- including D2/D3 games the
        # model has no ratings for. Unfiltered, those leaked into the 2026
        # schedule and would have fired pre-kickoff builds for games like
        # Rhode Island at Merrimack. Keep only games with an FBS side, the
        # same universe the ESPN path fetched (FBS slate + FBS-vs-FCS
        # crossovers, which the pooled-FCS node exists to price).
        hc = str(g.get("homeClassification") or "").lower()
        ac = str(g.get("awayClassification") or "").lower()
        if "fbs" not in (hc, ac):
            continue
        completed = bool(g.get("completed"))
        kick = _norm_kick(kick)
        out.append({
            "event_id": str(gid),
            "date": _game_date_et(kick),
            "kickoff_utc": kick,
            "season": int(g.get("season", season)),
            "week": int(g.get("week") or 1),
            "status": "STATUS_FINAL" if completed else "STATUS_SCHEDULED",
            "home_team": home,
            "home_conf": _conf(g.get("homeConference"), g.get("homeClassification")),
            "home_id": str(g.get("homeId")) if g.get("homeId") is not None else None,
            "away_team": away,
            "away_conf": _conf(g.get("awayConference"), g.get("awayClassification")),
            "away_id": str(g.get("awayId")) if g.get("awayId") is not None else None,
            "home_score": float(g["homePoints"]) if g.get("homePoints") is not None else None,
            "away_score": float(g["awayPoints"]) if g.get("awayPoints") is not None else None,
            "neutral_site": bool(g.get("neutralSite", False)),
        })
    return out


# ---------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------

# Best book first; same reasoning as fetch_odds.PROVIDER_RANK. "consensus"
# is CFBD's own blend and a solid fallback when no single book is quoted.
PROVIDER_RANK = ["espn bet", "draftkings", "consensus", "bovada", "caesars"]

MAX_PLAUSIBLE_SPREAD = 70.0
MIN_PLAUSIBLE_TOTAL, MAX_PLAUSIBLE_TOTAL = 20.0, 110.0


def _rank(provider):
    p = str(provider or "").lower()
    for i, name in enumerate(PROVIDER_RANK):
        if name in p:
            return i
    return len(PROVIDER_RANK) + 1


def _spread_from_formatted(formatted, home, away):
    """Home-perspective spread from 'Team -7.5'. None when unparseable.

    The formatted string names the FAVOURITE laying points. If that name
    matches the home side the home spread is negative; if it matches the away
    side, positive. Matching is by containment either way because CFBD
    truncates neither name, but belt-and-braces beats optimism.
    """
    if not formatted:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", str(formatted).strip())
    if not m:
        return None
    try:
        val = -abs(float(m.group(1)))
    except ValueError:
        return None
    name = str(formatted)[: m.start()].strip()
    if not name:
        return None
    def matches(team):
        t = str(team or "")
        return bool(t) and (name.lower() in t.lower() or t.lower() in name.lower())
    if matches(home) and not matches(away):
        return val                      # home favoured, lays the points
    if matches(away) and not matches(home):
        return -val                     # away favoured -> home side positive
    return None


sign_disagreements = 0                  # read by selfcheck()


def _pick_line(lines, home, away):
    """One line dict (run_report's `line` schema) from CFBD's per-game list."""
    global sign_disagreements
    if not lines:
        return None
    lines = sorted(lines, key=lambda l: _rank(l.get("provider")))
    ln = lines[0]

    raw = ln.get("spread")
    sp = float(raw) if isinstance(raw, (int, float)) else None
    if sp is not None and abs(sp) > MAX_PLAUSIBLE_SPREAD:
        sp = None
    fmt_sp = _spread_from_formatted(ln.get("formattedSpread"), home, away)
    if fmt_sp is not None:
        if sp is not None and abs(fmt_sp + sp) < 0.6 and abs(fmt_sp - sp) > 0.6:
            sign_disagreements += 1     # numeric field disagrees on SIGN only
        sp = fmt_sp                     # the named-favourite string wins
    tot = ln.get("overUnder")
    tot = float(tot) if isinstance(tot, (int, float)) else None
    if tot is not None and not (MIN_PLAUSIBLE_TOTAL <= tot <= MAX_PLAUSIBLE_TOTAL):
        tot = None
    if sp is None and tot is None:
        return None
    return {
        "book": ln.get("provider"),
        "details": ln.get("formattedSpread"),
        "spread_home": sp,
        "total": tot,
        "ml_home": ln.get("homeMoneyline"),
        "ml_away": ln.get("awayMoneyline"),
    }


def lines_for_season(season, season_type="regular"):
    """{event_id -> line dict} for a season. ONE api call.

    For an upcoming game this is the current line (CFBD refreshes through the
    week); for a completed game it is the closing number -- which is exactly
    what the report and the grader respectively need.
    """
    k = ("lines", int(season), season_type)
    if k not in _memo:
        try:
            _memo[k] = _get("/lines", {"year": int(season),
                                       "seasonType": season_type}) or []
        except RuntimeError as e:
            print(f"  CFBD /lines {season} {season_type}: {e}")
            _memo[k] = []
    out = {}
    for g in _memo[k]:
        gid = g.get("id") if g.get("id") is not None else g.get("gameId")
        if gid is None:
            continue
        line = _pick_line(g.get("lines") or [], g.get("homeTeam"), g.get("awayTeam"))
        if line:
            out[str(gid)] = line
    return out


def lines_for_grading(season):
    """Regular + postseason closing lines in one dict (2 calls, memoized)."""
    d = dict(lines_for_season(season, "regular"))
    d.update(lines_for_season(season, "postseason"))
    return d


# ---------------------------------------------------------------------------
# self-check: measure the compatibility contract before trusting it
# ---------------------------------------------------------------------------

def selfcheck(force=False, sample_season=2025):
    """Compare CFBD's view of a finished season against the ESPN-era files.

    Runs once (the report is cached on disk and committed by the workflow so
    it can be read without the Actions logs). Costs 3 API calls. Returns True
    when the id space demonstrably matches; on failure the caller should NOT
    merge CFBD rows into the historical files.
    """
    if os.path.exists(SELFCHECK_PATH) and not force:
        with open(SELFCHECK_PATH, encoding="utf-8") as f:
            return f.readline().strip().endswith("PASS")

    lines_out = []
    say = lambda s: (print("  " + s), lines_out.append(s))

    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    ref = games[games["season"] == sample_season].copy()
    ref["event_id"] = ref["event_id"].astype(str)
    cf = pd.DataFrame(season_games(sample_season))

    if not len(cf) or not len(ref):
        say("VERDICT: FAIL (no data to compare)")
        _write_check(lines_out, ok=False)
        return False

    merged = ref.merge(cf, on="event_id", suffixes=("_espn", "_cfbd"))
    id_overlap = len(merged) / len(ref)
    name_eq = float((merged["home_team_espn"] == merged["home_team_cfbd"]).mean())
    tid_eq = float((merged["home_id_espn"].astype(str)
                    == merged["home_id_cfbd"].astype(str)).mean())
    fin = merged[merged["home_score_espn"].notna() & merged["home_score_cfbd"].notna()]
    score_eq = float((fin["home_score_espn"] == fin["home_score_cfbd"]).mean()) if len(fin) else 0.0
    conf_eq = float((merged["home_conf_espn"] == merged["home_conf_cfbd"]).mean())

    ok = id_overlap >= 0.80 and name_eq >= 0.90 and score_eq >= 0.98

    say(f"sample season {sample_season}: {len(ref)} ESPN-era games, "
        f"{len(cf)} from CFBD, {len(merged)} joined on event_id")
    say(f"event_id overlap : {id_overlap:6.1%}  (need >=80%)")
    say(f"home name match  : {name_eq:6.1%}  (need >=90%)")
    say(f"home score match : {score_eq:6.1%}  (need >=98%)")
    say(f"team id match    : {tid_eq:6.1%}  (informational)")
    say(f"conference match : {conf_eq:6.1%}  (informational)")

    bad_names = sorted(set(
        merged.loc[merged["home_team_espn"] != merged["home_team_cfbd"],
                   ["home_team_espn", "home_team_cfbd"]]
        .itertuples(index=False, name=None)))[:25]
    if bad_names:
        say("name mismatches (espn <- cfbd): " +
            "; ".join(f"{a} <- {b}" for a, b in bad_names))

    # spread sign: compare CFBD closing spreads to the ESPN-era odds file
    try:
        odds = pd.read_csv(os.path.join(DATA_DIR, "odds.csv"))
        odds["event_id"] = odds["event_id"].astype(str)
        cl = lines_for_grading(sample_season)
        j = odds[odds["event_id"].isin(cl.keys())].dropna(subset=["spread_close_home"])
        j = j.head(200)
        if len(j):
            agree = sum(
                1 for _, r in j.iterrows()
                if cl[r["event_id"]]["spread_home"] is not None
                and abs(cl[r["event_id"]]["spread_home"] - r["spread_close_home"]) <= 3.0)
            say(f"closing-spread agreement (±3 pts, n={len(j)}): {agree/len(j):6.1%}")
            say(f"formatted-vs-numeric sign disagreements seen: {sign_disagreements}")
            if agree / len(j) < 0.80:
                ok = False
                say("spread convention DISAGREES with the historical file -- failing")
    except (OSError, ValueError, KeyError) as e:
        say(f"(spread cross-check skipped: {type(e).__name__})")

    say("")
    _write_check(lines_out, ok=ok)
    return ok


def _write_check(body, ok):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(SELFCHECK_PATH, "w", encoding="utf-8") as f:
        f.write(f"CFBD compatibility selfcheck {stamp} -- "
                f"{'PASS' if ok else 'FAIL'}\n")
        f.write("\n".join(body) + "\n")
    print(f"  wrote {SELFCHECK_PATH} ({'PASS' if ok else 'FAIL'})")


if __name__ == "__main__":
    if not get_key():
        raise SystemExit("no CFBD key (CFBD_API_KEY or data/cfbd_key.txt)")
    os.environ.setdefault("DATA_SOURCE", "cfbd")
    ok = selfcheck(force=True)
    raise SystemExit(0 if ok else 1)
