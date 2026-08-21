#!/usr/bin/env python3
"""
Pregame run: pull the live market line, injury report and starting lineups for
games kicking off soon, re-run the model with that information, and print a
model-vs-market comparison.

Intended to run ~10 minutes before kickoff, which is when the market line is
effectively the CLOSING line -- the sharpest number the market produces -- and
when injury/inactive information is final.

For each game it gathers, in one pass:
  * the current sportsbook line (spread, total, moneyline) from ESPN's
    pickcenter feed, which carries a real book (DraftKings)
  * both teams' injury reports
  * both teams' quarterback depth, and who is expected to start
  * the model's own projection, adjusted if the expected starter has changed
and then reports where the model and the market disagree.

Usage:
  python pregame.py --window 120        # games kicking off within 120 minutes
  python pregame.py --event 401856766   # one specific game
  python pregame.py --week 1            # a whole week (for a dry run)
"""

import argparse
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

import cfb_ratings as R
import cfb_predict as P

HEADERS = {"User-Agent": "Mozilla/5.0"}
from paths import DATA_DIR, current_cfb_season  # frozen-aware; see paths.py

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={eid}"
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{tid}/injuries"
ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{tid}/roster"

# ---------------------------------------------------------------------------
# QB adjustment
#
# HONEST STATUS: this is a PRIOR, not a fitted value. Estimating a QB's points
# value properly requires in-season snap-level data that does not exist yet for
# 2026. Published work and betting-market behaviour put a starter->backup
# downgrade in college football in the 3-7 point range, larger when the starter
# is elite and the backup is unproven. Until there is 2026 data to calibrate
# against, the model applies a conservative middle value and LABELS every such
# adjustment as low confidence. It is deliberately not tuned to look precise.
# ---------------------------------------------------------------------------
QB_DOWNGRADE_DEFAULT = -4.5
QB_DOWNGRADE_UNKNOWN_BACKUP = -6.0
QB_ADJ_MAX = 9.0

# first-year-head-coach set for the current season; filled by main(). Empty
# set = adjustment off (e.g. when cfbd_coaches CSVs are absent).
coach_new = set()


# FAST MODE -- set by website builds (see run_report). ESPN rate-limits
# GitHub's cloud servers, so there the injury/QB lookups are strictly
# opportunistic: one attempt, short timeout, no retry sleeps, and after 8
# consecutive failures a circuit breaker stops asking entirely. Worst case
# the build spends under a minute discovering ESPN won't talk to it, then
# runs exactly as it always has when no injury data is published. On a home
# connection (desktop exe) nothing here changes.
FAST_MODE = False
_fail_streak = 0
_BREAKER_AT = 8


def set_fast_mode(on=True):
    global FAST_MODE, _fail_streak
    FAST_MODE = bool(on)
    _fail_streak = 0


def _get(url, tries=3, timeout=20):
    global _fail_streak
    if FAST_MODE:
        if _fail_streak >= _BREAKER_AT:
            return None
        tries, timeout = 1, 6
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            _fail_streak = 0
            return out
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # a real answer, not a failure
            last = e
            if not FAST_MODE:
                time.sleep(2 * (i + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            if not FAST_MODE:
                time.sleep(2 * (i + 1))
    if FAST_MODE:
        _fail_streak += 1
        if _fail_streak == _BREAKER_AT:
            print("    (ESPN is refusing this server -- skipping further "
                  "injury/QB lookups this build)", flush=True)
    return None


# ---------------------------------------------------------------------------
# Data pulls
# ---------------------------------------------------------------------------

def get_market(eid):
    """Current book line. Returns None when the game is not yet priced."""
    d = _get(SUMMARY.format(eid=eid))
    if not d:
        return None
    out = {"espn_predictor": None, "line": None}

    pred = d.get("predictor") or {}
    if pred.get("homeTeam"):
        try:
            out["espn_predictor"] = float(pred["homeTeam"]["gameProjection"])
        except (KeyError, TypeError, ValueError):
            pass

    pcs = [p for p in (d.get("pickcenter") or [])
           if "live" not in str(p.get("provider", {}).get("name", "")).lower()]
    if pcs:
        pcs.sort(key=lambda p: p.get("provider", {}).get("priority", 99))
        pc = pcs[0]
        home = pc.get("homeTeamOdds") or {}
        away = pc.get("awayTeamOdds") or {}
        out["line"] = {
            "book": pc.get("provider", {}).get("name"),
            "details": pc.get("details"),
            "spread_home": pc.get("spread"),        # negative = home favoured
            "total": pc.get("overUnder"),
            "ml_home": home.get("moneyLine"),
            "ml_away": away.get("moneyLine"),
        }
    return out


def get_injuries(tid):
    d = _get(INJURIES.format(tid=tid))
    if not d:
        return []
    items = d.get("injuries") or d.get("items") or []
    out = []
    for it in items:
        ath = it.get("athlete") or {}
        out.append({
            "name": ath.get("displayName") or it.get("displayName"),
            "position": (ath.get("position") or {}).get("abbreviation"),
            "status": it.get("status") or (it.get("type") or {}).get("description"),
            "detail": (it.get("details") or {}).get("type") or it.get("shortComment"),
        })
    return out


def get_quarterbacks(tid):
    """Roster QBs in listing order. ESPN does not expose a reliable public
    depth chart for college, so order is a weak signal only -- it is reported,
    not trusted."""
    d = _get(ROSTER.format(tid=tid))
    if not d:
        return []
    qbs = []
    for grp in d.get("athletes", []):
        for a in grp.get("items", []):
            pos = (a.get("position") or {}).get("abbreviation")
            if pos == "QB":
                qbs.append({
                    "name": a.get("displayName"),
                    "jersey": a.get("jersey"),
                    "class": (a.get("experience") or {}).get("displayValue"),
                    "id": a.get("id"),
                })
    return qbs


def qb_status(qbs, injuries):
    """Cross-reference the QB list against the injury report."""
    inj_by_name = {i["name"]: i for i in injuries if i.get("name")}
    out = []
    for q in qbs:
        inj = inj_by_name.get(q["name"])
        status = (inj or {}).get("status")
        out.append({**q, "injury_status": status,
                    "out": bool(status and str(status).lower() in
                                ("out", "doubtful", "injured reserve"))})
    return out


def qb_adjustment(qb_info, team):
    """Points adjustment if the expected starter appears unavailable.

    Returns (points, note, confidence). Zero when nothing indicates a change --
    the ratings already reflect the team playing its normal quarterback.
    """
    if not qb_info:
        return 0.0, "no QB data available", "none"
    listed = [q for q in qb_info if not q["out"]]
    out_qbs = [q for q in qb_info if q["out"]]
    if not out_qbs:
        return 0.0, "no QB listed out; ratings already reflect the usual starter", "n/a"

    # The first-listed QB being out is the case that actually matters.
    if qb_info and qb_info[0]["out"]:
        if listed:
            return (QB_DOWNGRADE_DEFAULT,
                    f"likely starter {qb_info[0]['name']} listed "
                    f"{qb_info[0]['injury_status']}; next up {listed[0]['name']}",
                    "LOW -- prior, not calibrated")
        return (QB_DOWNGRADE_UNKNOWN_BACKUP,
                f"{qb_info[0]['name']} out and no clear backup identified",
                "LOW -- prior, not calibrated")
    names = ", ".join(q["name"] for q in out_qbs)
    return 0.0, f"QB(s) out but not the likely starter: {names}", "n/a"


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(model_margin, model_total, line):
    """Model vs market. Positive edge means the model likes the HOME side."""
    if not line or line.get("spread_home") is None:
        return None
    mkt_margin = -float(line["spread_home"])
    edge = model_margin - mkt_margin
    out = {
        "market_margin": mkt_margin,
        "edge_points": edge,
        "side": "HOME" if edge > 0 else "AWAY",
    }
    if line.get("total") is not None and model_total is not None:
        out["market_total"] = float(line["total"])
        out["total_edge"] = model_total - float(line["total"])
        out["total_side"] = "OVER" if out["total_edge"] > 0 else "UNDER"
    return out


def run_game(ratings, row, calib, styles=None, n_sims=P.DEFAULT_SIMS, verbose=True):
    eid = int(row["event_id"])
    home, away = row["home_team"], row["away_team"]
    hid, aid = str(row.get("home_id")), str(row.get("away_id"))

    market = get_market(eid) or {}
    inj_h, inj_a = get_injuries(hid), get_injuries(aid)
    qb_h = qb_status(get_quarterbacks(hid), inj_h)
    qb_a = qb_status(get_quarterbacks(aid), inj_a)

    adj_h, note_h, conf_h = qb_adjustment(qb_h, home)
    adj_a, note_a, conf_a = qb_adjustment(qb_a, away)
    qb_adj = {}
    if adj_h:
        qb_adj[home] = max(-QB_ADJ_MAX, min(QB_ADJ_MAX, adj_h))
    if adj_a:
        qb_adj[away] = max(-QB_ADJ_MAX, min(QB_ADJ_MAX, adj_a))

    if not (ratings.has(home) and ratings.has(away)):
        return {"event_id": eid, "error": f"unrated team: {home} / {away}"}

    pred = P.predict_game(ratings, home, away,
                          neutral=bool(row.get("neutral_site", False)),
                          n_sims=n_sims, calib=calib, styles=styles,
                          home_id=hid, away_id=aid, qb_adj=qb_adj,
                          seed=eid,
                          margin_adj=P.coach_margin_adj(coach_new, home, away))
    cmp_ = compare(pred["mean_margin"], pred["total"], market.get("line"))

    return {
        "event_id": eid, "home": home, "away": away,
        "kickoff": row.get("date"),
        "prediction": pred,
        "market": market.get("line"),
        "espn_predictor_home_pct": market.get("espn_predictor"),
        "comparison": cmp_,
        "injuries": {home: inj_h, away: inj_a},
        "qb": {home: {"depth": qb_h, "note": note_h, "confidence": conf_h, "adj": adj_h},
               away: {"depth": qb_a, "note": note_a, "confidence": conf_a, "adj": adj_a}},
        "low_confidence_teams": [t for t in (home, away) if t in getattr(ratings, "promoted", {})],
    }


def format_result(res):
    if res.get("error"):
        return f"[skip] {res['error']}"
    p = res["prediction"]
    L = []
    L.append("=" * 74)
    L.append(f"{res['away']}  at  {res['home']}      kickoff {res.get('kickoff')}")
    L.append("=" * 74)
    L.append(f"  MODEL   spread {p['spread_str']:<22} total {p['total']:.1f}"
             f"   {res['home']} win {p['home_win_prob']*100:.1f}%")

    mk = res.get("market")
    if mk:
        sp = mk.get("spread_home")
        side = f"{res['home']} {sp:+g}" if sp is not None else "n/a"
        L.append(f"  MARKET  spread {side:<22} total "
                 f"{mk.get('total') if mk.get('total') is not None else 'n/a'}"
                 f"   [{mk.get('book')}]")
        c = res.get("comparison")
        if c:
            L.append("")
            L.append(f"  EDGE    {c['edge_points']:+.1f} pts toward {c['side']}"
                     + (f"   |   total {c['total_edge']:+.1f} -> {c['total_side']}"
                        if "total_edge" in c else ""))
            mag = abs(c["edge_points"])
            verdict = ("no meaningful disagreement" if mag < 1.5 else
                       "modest disagreement" if mag < 3 else
                       "notable disagreement -- check injuries/lineups below" if mag < 6 else
                       "LARGE disagreement -- treat with suspicion, verify inputs")
            L.append(f"          {verdict}")
    else:
        L.append("  MARKET  not yet priced (no line posted for this game)")

    for team in (res["home"], res["away"]):
        q = res["qb"][team]
        inj = res["injuries"][team]
        starter = q["depth"][0]["name"] if q["depth"] else "unknown"
        L.append("")
        L.append(f"  {team}")
        L.append(f"    QB (listed first): {starter}")
        if q["adj"]:
            L.append(f"    QB ADJUSTMENT    : {q['adj']:+.1f} pts  [{q['confidence']}]")
            L.append(f"      {q['note']}")
        out_players = [i for i in inj if str(i.get("status", "")).lower() in
                       ("out", "doubtful")]
        if out_players:
            L.append(f"    OUT/DOUBTFUL ({len(out_players)}): "
                     + ", ".join(f"{i['name']} ({i['position']})" for i in out_players[:6]))
        elif inj:
            L.append(f"    injury report: {len(inj)} listed, none out/doubtful")
        else:
            L.append("    injury report: none published")

    if res.get("low_confidence_teams"):
        L.append("")
        L.append(f"  !! LOW CONFIDENCE: {', '.join(res['low_confidence_teams'])} "
                 f"rated from a prior, not from FBS game results")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Pregame line + injury + lineup run.")
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--games", default=os.path.join(DATA_DIR, "real_games.csv"))
    ap.add_argument("--window", type=int, default=None,
                    help="include games kicking off within this many minutes")
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--event", type=int, nargs="*", default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--sims", type=int, default=P.DEFAULT_SIMS)
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()
    if args.season is None:
        args.season = current_cfb_season()
    if args.schedule is None:
        args.schedule = os.path.join(DATA_DIR, f"schedule_{args.season}.csv")

    # first-year-head-coach margin adjustment (same one the backtest applies)
    global coach_new
    coach_new = P.load_coach_changes([args.season]).get(args.season, set())
    if coach_new:
        print(f"Coach adjustment active for {len(coach_new)} teams "
              f"({P.COACH_CHANGE_MARGIN:+.1f} margin each)")

    sched = pd.read_csv(args.schedule)
    if "neutral_site" in sched.columns:
        sched["neutral_site"] = sched["neutral_site"].astype(bool)

    if args.event:
        sel = sched[sched["event_id"].isin(args.event)]
    elif args.week is not None:
        sel = sched[sched["week"] == args.week]
    elif args.window is not None:
        now = dt.datetime.now(dt.timezone.utc)
        hi = now + dt.timedelta(minutes=args.window)
        ts = pd.to_datetime(sched["date"], errors="coerce", utc=True)
        sel = sched[(ts >= now - dt.timedelta(minutes=15)) & (ts <= hi)]
    else:
        ap.error("choose one of --window, --week, or --event")

    sel = sel[sel["home_team"].notna() & (sel["home_team"] != "TBD")
              & (sel["away_team"] != "TBD")]
    if len(sel) == 0:
        print("No games match that selection.")
        return

    print(f"Loading ratings and running {len(sel)} game(s)...\n")
    games = pd.read_csv(args.games)
    calib = P.load_calibration()
    if (calib.get("fcs_augmentation") or {}).get("enabled"):
        fcs = R.load_fcs_games()
        if fcs is not None:
            games = pd.concat([games, fcs], ignore_index=True)
            print(f"FCS augmentation ON: +{len(fcs)} FCS-vs-FCS training games")
    ratings = R.fit_ratings(games, lam=R.DEFAULT_LAMBDA, cv=False,
                            current_season=args.season, schedule=sched)
    styles = P.load_team_styles()

    results = []
    for _, row in sel.iterrows():
        res = run_game(ratings, row, calib, styles=styles, n_sims=args.sims)
        results.append(res)
        print(format_result(res))
        print()

    priced = [r for r in results if r.get("comparison")]
    if priced:
        edges = sorted(priced, key=lambda r: -abs(r["comparison"]["edge_points"]))
        print("=" * 74)
        print("BIGGEST MODEL-vs-MARKET DISAGREEMENTS")
        print("=" * 74)
        for r in edges[:8]:
            c = r["comparison"]
            print(f"  {c['edge_points']:+6.1f} pts  {r['away']} at {r['home']}"
                  f"   (model {r['prediction']['spread_str']}, "
                  f"market {r['home']} {r['market']['spread_home']:+g})")
        print(f"\n  {len(results)-len(priced)} of {len(results)} games had no line posted.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
