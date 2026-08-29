#!/usr/bin/env python3
"""
CFB betting report generator.

Pulls only the games kicking off in the next N days (default 5), fetches the
CURRENT market line, roster and injury report for each, runs the Monte Carlo
model, and writes a self-contained HTML report.

Per game it reports exactly:
    * projected total points, and each team's projected points
    * the model's estimated line, next to the live market line
    * an OVER / UNDER call and a SPREAD call
    * the model's win probability for each of those two bets

Every recommendation carries an explicit confidence label derived from the
backtest, not from how large the disagreement is. That distinction matters:
measured on historical closing lines, this model's against-the-spread hit rate
does NOT improve as its disagreement with the market grows -- if anything it
falls. A large gap usually means the model is missing information the market
has, not that an edge exists.

Usage:
    python run_report.py                 # next 5 days
    python run_report.py --days 3
    python run_report.py --week 1        # a specific week instead
    python run_report.py --open          # open the report when finished
"""

import argparse
import datetime as dt
import html
import json
import os
import sys
import webbrowser

import numpy as np
import pandas as pd

import cfb_ratings as R
import cfb_predict as P
import pregame as PG
import ledger as LG
from fetch_espn_data import fetch_conference_map, extract_games, fetch_week_cached

from paths import DATA_DIR, current_cfb_season  # frozen-aware; see paths.py
BREAKEVEN_110 = 0.5238

# How close to kickoff a game must be before its prediction is LOCKED into the
# ledger. Games further out are shown as previews and recorded later.
#
# This exists because the ledger is immutable, and immutability is only a
# virtue if the locked number was made with current information. Running the
# report on Thursday with a 5-day window would otherwise freeze Saturday's
# picks using ratings that have not seen Thursday's or Friday's results, and
# using a line that will move before kickoff. The record would then be of
# predictions nobody would actually have bet.
#
# 30 hours covers the normal pattern -- open it the evening before, or the
# morning of -- while still refusing to lock a Saturday game on Thursday.
LOCK_WINDOW_HOURS = 30


# ---------------------------------------------------------------------------
# Schedule window
# ---------------------------------------------------------------------------

def refresh_schedule_window(season, days, verbose=True):
    """Fetch only the weeks that overlap the next `days` days.

    A full-season pull is ~22 requests; this normally needs 1-2. The cached
    raw payloads are reused, and the current week is always refetched so the
    report picks up reschedules and newly-final scores.
    """
    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(days=days)

    sched_path = os.path.join(DATA_DIR, f"schedule_{season}.csv")
    base = pd.read_csv(sched_path) if os.path.exists(sched_path) else None
    weeks = []
    if base is not None and "date" in base.columns:
        ts = pd.to_datetime(base["date"], errors="coerce", utc=True)
        sel = base[(ts >= now - dt.timedelta(days=1)) & (ts <= horizon)]
        weeks = sorted(sel["week"].dropna().astype(int).unique().tolist())
    if not weeks:
        weeks = [1]

    try:
        import fetch_cfbd_live as CF
    except ImportError:      # frozen build without the module -> ESPN path
        class CF:
            enabled = staticmethod(lambda: False)
    if CF.enabled():
        # Website path: the whole season in one memoized CFBD call -- ESPN
        # rate-limits GitHub's servers into hours of retry cooldowns, and a
        # schedule refresh must never be the thing that makes a pre-kickoff
        # build miss its game. This fetch is the AUTHORITATIVE full season,
        # so it REPLACES the stored schedule outright instead of merging:
        # merging preserved rows the source no longer vouches for, which is
        # how a week of unfiltered D2/D3 games would have survived their own
        # fix. The ESPN path below still merges (it fetches partial weeks).
        rows = CF.season_games(season)
        if verbose:
            print(f"Refreshing schedule from CFBD: {len(rows)} games in one "
                  f"call (window {horizon.date()})", flush=True)
        if rows:
            fresh = pd.DataFrame(rows)
            fresh["event_id"] = fresh["event_id"].astype(str)
            fresh = fresh.drop_duplicates("event_id").sort_values(["week", "date"])
            fresh.to_csv(sched_path, index=False)
            return fresh
        return base if base is not None else pd.DataFrame()
    else:
        if verbose:
            print(f"Refreshing schedule for week(s) {weeks} "
                  f"(games through {horizon.date()})...", flush=True)
        conf_map = fetch_conference_map(season, os.path.join(DATA_DIR, "conferences.json"))
        rows = []
        for wk in weeks:
            try:
                raw, _ = fetch_week_cached(season, 2, wk, refresh=True)
                rows += extract_games(raw, conf_map)
            except Exception as e:  # noqa: BLE001
                print(f"  ! week {wk}: {e}", flush=True)
    if not rows:
        return base if base is not None else pd.DataFrame()

    fresh = pd.DataFrame(rows)
    # event_id MUST be compared as a string on both sides. pandas reads it back
    # from CSV as int64 while ESPN's JSON returns it as a string, so an
    # unnormalized `.isin()` matched ZERO rows: nothing was filtered out, the
    # fresh rows were appended anyway, and every refresh silently doubled the
    # schedule. It had reached 1045 rows for a 946-game season, and every game
    # was appearing twice in the report.
    fresh["event_id"] = fresh["event_id"].astype(str)
    fresh = fresh.drop_duplicates("event_id")

    if base is not None and len(base):
        base = base.copy()
        base["event_id"] = base["event_id"].astype(str)
        keep = base[~base["event_id"].isin(fresh["event_id"])]
        merged = pd.concat([keep, fresh], ignore_index=True)
    else:
        merged = fresh

    # belt and braces: whatever happened above, one row per game leaves here
    merged = merged.drop_duplicates("event_id").sort_values(["week", "date"])
    merged.to_csv(sched_path, index=False)
    return merged


def select_games(sched, days=None, week=None):
    s = sched.copy()
    s = s[s["home_team"].notna() & (s["home_team"] != "TBD") & (s["away_team"] != "TBD")]
    if "status" in s.columns:
        s = s[s["status"] != "STATUS_FINAL"]
    if week is not None:
        return s[s["week"] == week]
    now = dt.datetime.now(dt.timezone.utc)
    hi = now + dt.timedelta(days=days)
    # Select on the KICKOFF INSTANT, not the bare date. `date` is the US
    # Eastern game day and parses to midnight UTC, which sits 12-20 hours
    # BEFORE any real kickoff on that day -- so the moment the clock passed
    # 6h after midnight, the "not more than 6h started" lower bound silently
    # dropped the entire same-day slate. (Caught by the end-to-end test on a
    # simulated game day; no real game day had occurred since the date
    # column switched to Eastern.) Rows with no kickoff timestamp fall back
    # to end-of-day so a listed game is never dropped for a missing time.
    kick = (pd.to_datetime(s["kickoff_utc"], errors="coerce", utc=True)
            if "kickoff_utc" in s.columns
            else pd.Series(pd.NaT, index=s.index))
    day = pd.to_datetime(s["date"], errors="coerce", utc=True)
    ts = kick.fillna(day + pd.Timedelta(hours=28))   # ~midnight Eastern
    return s[(ts >= now - dt.timedelta(hours=6)) & (ts <= hi)]


# ---------------------------------------------------------------------------
# Betting calls
# ---------------------------------------------------------------------------

def make_calls(pred, sim_h, sim_a, line):
    """Turn the simulated distribution into explicit bet calls with win
    probabilities computed AGAINST THE ACTUAL MARKET NUMBER."""
    out = {"spread": None, "total": None}
    if not line:
        return out

    margin = sim_h - sim_a
    total = sim_h + sim_a

    sp = line.get("spread_home")
    if sp is not None:
        # home covers when (home margin + line) > 0
        adj = margin + float(sp)
        push = float(np.mean(adj == 0))
        home_cov = float(np.mean(adj > 0))
        away_cov = 1.0 - home_cov - push
        pick_home = home_cov >= away_cov
        out["spread"] = {
            "market_line": float(sp),
            "pick": "HOME" if pick_home else "AWAY",
            "win_prob": home_cov if pick_home else away_cov,
            "push_prob": push,
            "model_line": -pred["mean_margin"],
            "edge": pred["mean_margin"] - (-float(sp)),
        }

    tot = line.get("total")
    if tot is not None:
        t = float(tot)
        push = float(np.mean(total == t))
        over = float(np.mean(total > t))
        under = 1.0 - over - push
        pick_over = over >= under
        out["total"] = {
            "market_total": t,
            "pick": "OVER" if pick_over else "UNDER",
            "win_prob": over if pick_over else under,
            "push_prob": push,
            "model_total": pred["total"],
            "edge": pred["total"] - t,
        }
    return out


def market_cover_prob(signed_edge, shrink_factor, resid_sd=None, alpha=0.0,
                      pick_home=True):
    """Probability that the model's chosen side covers the market line.

    The fitted relationship is a full line, not a slope through the origin:

        expected cover margin = alpha + beta * signed_edge

    Three things here were wrong in an earlier version and each pushed the
    number in the same direction -- toward recommending more bets:

    1. SIGN. `abs(edge)` was passed in, which treats home and away picks as
       mirror images. They are not, because alpha is nonzero.
    2. INTERCEPT. alpha = +0.92 (t = 2.59) was dropped. Home sides covered by
       that much on average regardless of the model, so discarding it
       overstated every AWAY pick by roughly 3 points of win probability. At
       an edge of 6 the truth is HOME 55.4% / AWAY 50.5%, not 53.5% for both:
       away picks in the 6-10 point band were being recommended while sitting
       BELOW the 52.38% break-even.
    3. SPREAD. resid_sd was hardcoded at 13.0 against a fitted 15.04, which
       inflates every probability and widens the PLAY band.

    ON ALPHA, AND WHY IT DEFAULTS TO ZERO HERE.
    The intercept is real and stable (+1.04 in 2024, +0.79 in 2025) -- home
    sides did cover by about a point more than the line on average. But a
    positive average cover margin is NOT the same as a profitable bet. The
    raw home-side ATS record over these games is 51.43% (CI [49.1, 53.7]),
    which sits BELOW the 52.38% break-even: the margin shows up as bigger
    home wins, not as more home covers. Feeding alpha into the probability
    puts a permanent thumb on the scale for home sides -- it would rate a
    zero-edge home bet at 52.4% and start recommending it -- on the strength
    of a bias that demonstrably does not pay at -110.

    So callers should leave alpha at 0. The model's own disagreement is what
    drives a recommendation; a market-wide home tilt is disclosed in the
    report rather than bet on. Pass alpha explicitly only for diagnostics.

    `signed_edge` keeps its sign (positive favours home). The probability
    returned is for the side actually being recommended.
    """
    from math import erf, sqrt
    sd = resid_sd if resid_sd else 15.04
    expected = alpha + shrink_factor * signed_edge
    # if we are taking the away side, the away side covers when the home cover
    # margin is negative
    z = (expected if pick_home else -expected) / (sd * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def conservative_cover_prob(signed_edge, shrink_info, resid_sd=None,
                            pick_home=True):
    """Cover probability using the 95% LOWER BOUND of the fitted shrink factor.

    beta is an estimate, not a known constant. Quoting only the point
    estimate treats an uncertain parameter as certain, which is exactly how a
    marginal edge gets talked into looking like a real one. (Current fitted
    values live in calibration.json; as of the 2026-08 refit the spread beta
    is ~0.12 with se ~0.06 and the validation gate holds the shrink factor
    at 0.)

    A bet is only labelled PLAY when it clears break-even under BOTH the point
    estimate and this bound. That makes PLAY rare and meaningful rather than
    a label on a third of the slate.

    THE GATE APPLIES HERE TOO. An earlier version read the raw fitted beta,
    so a gate-blocked edge could still produce a >52.38% "conservative"
    probability -- harmless only because of the order of checks in
    confidence_label. If the validation gate failed, the honest lower bound
    on the market-calibrated edge is zero, full stop.
    """
    if not shrink_info:
        return 0.5
    if not shrink_info.get("gate_passed", False):
        return 0.5
    b = shrink_info.get("beta", 0.0)
    se = shrink_info.get("beta_se", 0.0)
    lower = max(b - 1.96 * se, 0.0)
    return market_cover_prob(signed_edge, lower,
                             resid_sd or shrink_info.get("resid_sd"),
                             alpha=0.0, pick_home=pick_home)


def confidence_label(win_prob, edge, is_low_conf_team, calibrated=True,
                     conservative_prob=None, fcs_involved=False):
    """Turn a calibrated cover probability into a recommendation.

    `win_prob` must already be the MARKET-calibrated probability (see
    market_cover_prob), never the simulator's internal one.

    WHAT THE HISTORICAL RECORD ACTUALLY SUPPORTS, and what it does not:

    (Numbers below predate the leak fix and are kept only as the shape of the
    argument; current fitted values live in calibration.json. As of the
    2026-08 refit the spread beta is ~0.12 with t ~1.9 -- NOT significant --
    and the validation gate holds both shrink factors at 0, so every
    market-calibrated probability reads 50% and every call reads NO BET.)

    On the pre-fix data the continuous relationship looked real -- beta 0.188
    (t = 3.29) for spreads, 0.287 (t = 3.26) for totals. Most of that came
    from the look-ahead leak; it is quoted here as a warning, not a fact.

    The binary win/loss translation is far noisier still.
    Against-the-spread accuracy by edge BAND over 1,785 games:

        0-2   50.2% | 2-4   55.9% | 4-6   52.0% | 6-8   50.3%
        8-10  50.0% | 10-12 56.0% | 12-15 52.5% | 15+   53.9%

    That is not a trend, it is a zigzag -- exactly what noise looks like. No
    band is individually significant, and the pattern gives no basis for
    "bet the big disagreements". An earlier version of this function refused
    every edge above 12 points on the theory that huge gaps mean missing
    information; the data does not support that either (53.3% there, n=92),
    so the rule was removed rather than kept as a comfortable-sounding
    heuristic.

    The thresholds below therefore lean on the calibrated probability alone,
    and nothing here should be read as a validated betting system.
    """
    if not calibrated:
        return ("NO BET", "edge calibration unavailable; run evaluate_vs_market.py")
    if is_low_conf_team:
        return ("AVOID", "a team here is rated from a prior, not real results")
    if fcs_involved:
        # Every non-FBS opponent shares ONE pooled rating, so the model has no
        # team-specific view of them at all while the market prices each one
        # individually. That guarantees large disagreements, and those
        # disagreements are model ignorance rather than insight.
        #
        # Measured: FCS-involved games hit 51.35% ATS at edge>=2
        # (CI [44.2, 58.5] -- spans losing money) versus 53.69% for FBS-vs-FBS.
        # Left ungated, 30% of FCS games earned PLAY against 10% of FBS games:
        # the tool was steering hardest toward what it understands least.
        return ("AVOID", "opponent is non-FBS and shares a single pooled rating; "
                         "large edges here reflect model ignorance, not an edge")
    if win_prob < BREAKEVEN_110:
        return ("NO BET", f"below the {BREAKEVEN_110:.2%} break-even at -110")
    # PLAY requires the edge to survive the uncertainty in beta itself
    if conservative_prob is not None and conservative_prob >= BREAKEVEN_110 \
            and win_prob >= 0.545:
        return ("PLAY", "clears break-even even at the 95% lower bound of the fitted edge")
    return ("LEAN", "clears break-even on the point estimate only; "
                    "does not survive the uncertainty in that estimate")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#ffffff;--fg:#16191d;--muted:#5b636d;--line:#e3e6ea;--card:#f7f8fa;
--pos:#0a7d3f;--neg:#b3261e;--warn:#8a6100;--accent:#1a4f8a;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0f1216;--fg:#e8eaed;--muted:#9aa3ad;--line:#272c33;--card:#161a20;
--pos:#4ade80;--neg:#f87171;--warn:#fbbf24;--accent:#7fb2ee;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:24px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.note{background:var(--card);border-left:3px solid var(--warn);padding:12px 14px;
border-radius:4px;margin-bottom:20px;font-size:13px;color:var(--fg)}
.note b{color:var(--warn)}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:1180px;font-variant-numeric:tabular-nums}
th{background:var(--card);text-align:right;padding:9px 10px;font-size:11px;
text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
border-bottom:1px solid var(--line);white-space:nowrap;position:sticky;top:0}
th.l,td.l{text-align:left}
td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--card)}
.game{font-weight:600}
.dim{color:var(--muted);font-size:12px}
.pill{display:inline-block;padding:2px 7px;border-radius:10px;font-size:11px;font-weight:600}
.play{background:rgba(10,125,63,.14);color:var(--pos)}
.lean{background:rgba(138,97,0,.16);color:var(--warn)}
.nobet{background:rgba(120,120,120,.16);color:var(--muted)}
.avoid{background:rgba(179,38,30,.14);color:var(--neg)}
.sep{border-left:2px solid var(--line)}
footer{margin-top:22px;color:var(--muted);font-size:12px;line-height:1.7}
h2{font-size:16px;margin:30px 0 4px}
.rec{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:12px 14px;margin:10px 0 16px;font-size:13px;line-height:1.8;
white-space:pre-wrap;font-variant-numeric:tabular-nums}
.win{color:var(--pos);font-weight:600}
.loss{color:var(--neg);font-weight:600}
.pushr{color:var(--muted);font-weight:600}
code{background:var(--card);padding:1px 5px;border-radius:3px}
"""


def _pill(kind):
    cls = {"PLAY": "play", "LEAN": "lean", "NO BET": "nobet", "AVOID": "avoid"}[kind]
    return f'<span class="pill {cls}">{html.escape(kind)}</span>'


def _res_cell(v):
    v = str(v) if v is not None else ""
    cls = {"WIN": "win", "LOSS": "loss", "PUSH": "pushr"}.get(v, "dim")
    return f'<span class="{cls}">{html.escape(v)}</span>' if v else '<span class="dim">-</span>'


def build_results_section(graded, summary_text):
    """Completed games: what the model said, what happened, who won the bet."""
    if graded is None or not len(graded):
        return ""
    head = """<tr>
<th class="l">Game</th><th class="l">Date</th>
<th>Final</th><th>Projected</th>
<th class="sep">Close Spread</th><th>Our Pick</th><th>ATS</th>
<th class="sep">Close O/U</th><th>Our Pick</th><th>O/U</th>
<th class="sep">Winner</th>
</tr>"""
    body = []
    for _, r in graded.iterrows():
        sp = r.get("close_spread_home")
        tot = r.get("close_total")
        spick = r.get("spread_pick")
        if isinstance(spick, str):
            side = r["home_team"] if spick.upper().startswith("HOME") else r["away_team"]
            spick_s = html.escape(str(side))
        else:
            spick_s = '<span class="dim">-</span>'
        body.append(f"""<tr>
<td class="l game">{html.escape(str(r['away_team']))} at {html.escape(str(r['home_team']))}</td>
<td class="l dim">{html.escape(str(r.get('kickoff',''))[:10])}</td>
<td><b>{r['final_away']:.0f} - {r['final_home']:.0f}</b></td>
<td class="dim">{r['exp_away']:.1f} - {r['exp_home']:.1f}</td>
<td class="sep">{'' if pd.isna(sp) else f"{r['home_team']} {sp:+g}"}</td>
<td>{spick_s}</td><td>{_res_cell(r.get('spread_result'))}</td>
<td class="sep">{'' if pd.isna(tot) else f"{tot:g}"}</td>
<td>{html.escape(str(r.get('total_pick') or '-'))}</td><td>{_res_cell(r.get('total_result'))}</td>
<td>{_res_cell('WIN' if r.get('su_correct')==1 else ('LOSS' if r.get('su_correct')==0 else None))}</td>
</tr>""")
    return f"""
<h2>Completed games &mdash; how the model actually did</h2>
<div class="rec">{html.escape(summary_text)}</div>
<div class="wrap"><table>{head}{''.join(body)}</table></div>
"""


def build_html(rows, meta, results_html=""):
    # Layout per the owner's spec (2026-08-21): the page shows the market
    # line, the simulated score, the straight-up Win %, and the model's own
    # cover % for the spread and O/U sides it prefers -- and nothing else.
    # No Call column, no NO BET machinery on the page. The percentages shown
    # are the SIMULATOR'S estimates measured against the listed market
    # number; the Results tab grades every locked pick against the closing
    # line, so the season record -- not a pill -- is the running verdict on
    # how much those percentages can be trusted. The call/gate machinery
    # still runs underneath and is recorded in the ledger for analysis.
    head = """<tr>
<th class="l">Game</th><th class="l">Kickoff</th><th>Status</th>
<th>Sim Score</th><th>Win %</th>
<th class="sep">Market Line</th><th>Spread Pick</th><th>Cover %</th>
<th class="sep">Market O/U</th><th>O/U Pick</th><th>Cover %</th>
</tr>"""

    body = []
    for r in rows:
        p = r["pred"]
        calls = r["calls"]
        sp, to = calls.get("spread"), calls.get("total")
        game = f'{html.escape(r["away"])} at {html.escape(r["home"])}'
        # straight-up win probability -- the model IS differentiated here
        # (74.3% accuracy, Brier 0.164), unlike the spread where the
        # market-calibrated cover probability is a flat 50%
        wp = p["home_win_prob"]
        fav = r["home"] if wp >= 0.5 else r["away"]
        winprob = (f'{html.escape(fav)} {max(wp, 1-wp)*100:.0f}%')
        # LOCKED rows are in the immutable ledger and will be graded; PREVIEW
        # rows are informational and get recorded on a later run nearer
        # kickoff. The call machinery (AVOID/PLAY/etc) still runs and is
        # recorded in the ledger, but the page no longer renders it -- the
        # one piece kept visible is a dim data-quality marker on games where
        # the projection itself is unreliable (non-FBS opponent or newly
        # promoted team), because that is a fact about the number shown.
        _conf = (r.get("spread_conf") or ("", ""))[0]
        _tconf = (r.get("total_conf") or ("", ""))[0]
        lowflag = ('<br><span class="dim">low-data matchup</span>'
                   if "AVOID" in (_conf, _tconf) else "")

        if r.get("locked"):
            lockcell = '<span class="pill play">LOCKED</span>'
        else:
            h = r.get("hrs_to_kick")
            lockcell = ('<span class="pill nobet">preview</span>'
                        + (f'<br><span class="dim">{h:.0f}h out</span>'
                           if h is not None else ""))

        sim_score = (f'{p["exp_away"]:.1f} &ndash; {p["exp_home"]:.1f}'
                     f'<br><span class="dim">total {p["total"]:.1f}</span>')
        mk = (f'{r["home"]} {sp["market_line"]:+g}' if sp
              else '<span class="dim">no line</span>')

        if sp:
            side = r["home"] if sp["pick"] == "HOME" else r["away"]
            spread_bet = (f'{html.escape(side)} '
                          f'{sp["market_line"] if sp["pick"]=="HOME" else -sp["market_line"]:+g}')
            sp_cov = f'<b>{sp.get("model_win_prob", sp["win_prob"])*100:.1f}%</b>'
        else:
            spread_bet = sp_cov = '<span class="dim">-</span>'

        if to:
            ou_bet = f'{to["pick"]} {to["market_total"]:g}'
            ou_cov = f'<b>{to.get("model_win_prob", to["win_prob"])*100:.1f}%</b>'
            mkt_tot = f'{to["market_total"]:g}'
        else:
            ou_bet = ou_cov = mkt_tot = '<span class="dim">-</span>'

        body.append(f"""<tr>
<td class="l game">{game}{lowflag}</td><td class="l dim">{html.escape(str(r.get("kick","")))}</td>
<td>{lockcell}</td>
<td>{sim_score}</td><td>{winprob}</td>
<td class="sep">{mk}</td><td>{spread_bet}</td><td>{sp_cov}</td>
<td class="sep">{mkt_tot}</td><td>{ou_bet}</td><td>{ou_cov}</td>
</tr>""")

    m = meta
    return f"""<title>CFB Model Report</title>
<style>{CSS}</style>
{LG.site_nav_html('report')}
<h1>College Football Model Report</h1>
<div class="sub">Generated {html.escape(m['generated'])} &middot;
{m['n_games']} games &middot; {m['n_priced']} with a market line &middot;
ratings fit on {m['n_train']} games through {m['through']}</div>

{LG.record_strip_html(m.get('record'))}

<div class="note">
<b>What the numbers are.</b> <b>Sim Score</b> is the average of
{m['sims']:,} Monte Carlo simulations of this matchup. <b>Win %</b> is how
often the named team wins those simulations outright. <b>Cover %</b> is how
often the model's preferred side covers the listed market number
(spread) or lands on the listed total (O/U), pushes excluded &mdash; the
model's own estimate, measured against the real line shown.<br><br>
<b>LOCKED vs preview.</b> A <b>LOCKED</b> row is written to the permanent
record and gets graded against the closing line after the game; a
<b>preview</b> row locks at a later update nearer kickoff (within
{m['lock_hours']:g} hours), once the lines and lineups are final. The
<b>Season Results</b> tab keeps the running scorecard of every locked pick
&mdash; that graded record, not these percentages, is the proof of what the
model can actually do. For reference, the market's closing line has
historically been the more accurate predictor (margin RMSE
{m['margin_rmse']:.2f} vs the market's ~15.1 on ~1,800 graded games), and
{BREAKEVEN_110:.2%} is the win rate needed to profit at standard &minus;110
pricing.
</div>

<div class="wrap"><table>{head}{''.join(body)}</table></div>
{results_html}
<footer>
<b>Columns.</b> <code>Sim Score</code> is the mean simulated score, away &ndash; home;
its <code>total</code> is shrunk toward the league mean &mdash; unshrunk totals scored
worse than guessing the average.
<code>Market Line</code> is quoted from the home side, negative = home favoured.
<code>Cover %</code> is the Monte Carlo probability that the named pick beats the
listed market number, excluding pushes.<br>
<b>Method.</b> Ridge-regularised opponent-adjusted offence/defence ratings
(&lambda; and recency decay chosen by walk-forward cross-validation), simulated
{m['sims']:,} times per game with a drive-based multinomial that reproduces real
key-number clustering on margins of 3 and 7. Noise is calibrated from out-of-sample
backtest residuals. Ratings correlate +0.99 with ESPN FPI.<br>
<b>Not included in the line.</b> Pass-vs-pass and rush-vs-rush style matchup was tested
and had no predictive signal (t = &minus;0.39), so it never moves these numbers.
</footer>"""


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate the CFB betting report.")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--season", type=int, default=None,
                    help="defaults to the season the current date falls in")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true", help="open the report when done")
    ap.add_argument("--lock-hours", type=float, default=LOCK_WINDOW_HOURS,
                    help="lock predictions into the ledger only within this many "
                         "hours of kickoff (default: %(default)s)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="use the cached schedule instead of refetching")
    args = ap.parse_args()
    if args.season is None:
        args.season = current_cfb_season()

    print("=" * 62)
    print("  COLLEGE FOOTBALL MODEL REPORT")
    print("=" * 62)

    if args.no_refresh:
        sched = pd.read_csv(os.path.join(DATA_DIR, f"schedule_{args.season}.csv"))
    else:
        sched = refresh_schedule_window(args.season, args.days)

    sel = select_games(sched, days=args.days, week=args.week)
    if len(sel) == 0:
        print(f"\nNo upcoming games in the next {args.days} days.")
        print("Try --days 14, or --week N for a specific week.")
        return 0
    print(f"\n{len(sel)} game(s) to report on.\n")

    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    calib = P.load_calibration()

    # FCS-vs-FCS training augmentation: only when a --fcs-ab run validated it
    fit_games = games
    fcs_flag = calib.get("fcs_augmentation") or {}
    if fcs_flag.get("enabled"):
        fcs = R.load_fcs_games()
        if fcs is not None:
            fit_games = pd.concat([games, fcs], ignore_index=True)
            print(f"FCS augmentation ON: +{len(fcs)} FCS-vs-FCS training games "
                  f"(validated by backtest.py --fcs-ab)")
        else:
            print("NOTE: calibration enables FCS augmentation but "
                  "data/fcs_games.csv is missing -- fitting without it.")

    # first-year head coach margin adjustment (same one the backtest applies)
    coach_new = P.load_coach_changes([args.season]).get(args.season, set())
    if coach_new:
        print(f"Coach adjustment active: {len(coach_new)} first-year-coach teams "
              f"({P.COACH_CHANGE_MARGIN:+.1f} margin each, market-priced -- "
              f"projection accuracy only)")

    print("Fitting ratings...", flush=True)
    ratings = R.fit_ratings(fit_games, lam=R.DEFAULT_LAMBDA, cv=False,
                            current_season=args.season, schedule=sched)
    shrink_info = calib.get("edge_shrinkage") or {}
    shrink = shrink_info.get("shrink_factor")
    # totals get their OWN factor -- margins and totals have different error
    # structures and are fitted separately (see calibration.json for current values)
    tshrink_info = calib.get("total_edge_shrinkage") or {}
    tshrink = tshrink_info.get("shrink_factor", shrink)
    calibrated = shrink is not None
    if calibrated:
        print(f"Edge shrinkage -- margins {shrink:.3f}, totals "
              f"{(tshrink if tshrink is not None else 0.0):.3f} "
              f"(fitted on {shrink_info.get('n','?')} historical games)")
    else:
        print("NOTE: no edge calibration yet -- every bet call will read NO BET.")
        print("      Run fetch_odds.py then evaluate_vs_market.py to enable calls.")
        shrink = 0.0
        tshrink = 0.0

    # Website builds price every game from ONE CFBD lines call instead of an
    # ESPN summary request per game (5 requests/game -> ~0). The explicit
    # ESPN injury/QB check stays on but in fast-fail mode: from GitHub's
    # rate-limited address space it must never stall the build -- it either
    # answers quickly or the model runs without it, exactly as it already
    # does whenever ESPN publishes nothing.
    try:
        import fetch_cfbd_live as CF
    except ImportError:      # frozen build without the module -> ESPN path
        class CF:
            enabled = staticmethod(lambda: False)
    cfbd_lines = None
    if CF.enabled():
        cfbd_lines = CF.lines_for_season(args.season)
        PG.set_fast_mode(True)
        print(f"Market lines from CFBD: {len(cfbd_lines)} games priced in one call")

    rows = []
    ledger_rows = []
    preview_rows = []
    for i, (_, g) in enumerate(sel.iterrows(), 1):
        home, away = g["home_team"], g["away_team"]
        print(f"  [{i}/{len(sel)}] {away} at {home}", flush=True)
        h = R.resolve_team(ratings, home)
        a = R.resolve_team(ratings, away)
        if not (ratings.has(h) and ratings.has(a)):
            continue

        eid = int(g["event_id"])
        if cfbd_lines is not None:
            line = cfbd_lines.get(str(eid))
            market = {"espn_predictor": None, "line": line}
        else:
            market = PG.get_market(eid) or {}
            line = market.get("line")

        inj_h = PG.get_injuries(str(g.get("home_id")))
        inj_a = PG.get_injuries(str(g.get("away_id")))
        qb_h = PG.qb_status(PG.get_quarterbacks(str(g.get("home_id"))), inj_h)
        qb_a = PG.qb_status(PG.get_quarterbacks(str(g.get("away_id"))), inj_a)
        adj_h, _, _ = PG.qb_adjustment(qb_h, home)
        adj_a, _, _ = PG.qb_adjustment(qb_a, away)
        qb_adj = {}
        if adj_h:
            qb_adj[h] = adj_h
        if adj_a:
            qb_adj[a] = adj_a

        eh, ea = P.expected_scores(ratings, h, a,
                                   neutral=bool(g.get("neutral_site", False)),
                                   qb_adj=qb_adj,
                                   margin_adj=P.coach_margin_adj(coach_new, home, away))
        sim_h, sim_a = P.simulate(eh, ea, n_sims=args.sims,
                                  sigma=calib["score_resid_std"], seed=eid)
        pred = P.summarize(home, away, sim_h, sim_a, calib=calib)
        # totals are reported calibrated, so bet calls must use the same scale
        cal_total, _ = P.calibrate_total(sim_h + sim_a, calib)
        calls = make_calls(pred, sim_h, sim_a, line)
        if calls.get("total"):
            t = calls["total"]["market_total"]
            push = float(np.mean(cal_total == t))
            over = float(np.mean(cal_total > t))
            under = 1.0 - over - push
            pick_over = over >= under
            calls["total"].update({"pick": "OVER" if pick_over else "UNDER",
                                   "win_prob": over if pick_over else under,
                                   "push_prob": push,
                                   "edge": pred["total"] - t})

        low_conf = any(t in getattr(ratings, "promoted", {}) for t in (home, away))
        # non-FBS opponent => the model is using a single pooled rating node
        fcs_involved = (g.get("home_conf") not in R.FBS_CONFERENCES
                        or g.get("away_conf") not in R.FBS_CONFERENCES)

        # Replace the simulator's internal cover probability with the
        # market-calibrated one. The former answers "if the model is right,
        # how often does this cover?"; only the latter is decision-relevant.
        if calls.get("spread"):
            raw_edge = calls["spread"]["edge"]
            pick_home = calls["spread"]["pick"] == "HOME"
            calls["spread"]["model_win_prob"] = calls["spread"]["win_prob"]
            # signed edge + fitted intercept + fitted residual sd
            calls["spread"]["win_prob"] = market_cover_prob(
                raw_edge, shrink, shrink_info.get("resid_sd"),
                alpha=0.0, pick_home=pick_home)
            calls["spread"]["cons_prob"] = conservative_cover_prob(
                raw_edge, shrink_info, pick_home=pick_home)
            sp_conf = confidence_label(calls["spread"]["win_prob"], raw_edge,
                                       low_conf, calibrated,
                                       calls["spread"]["cons_prob"], fcs_involved)
        else:
            sp_conf = ("NO BET", "no line")

        if calls.get("total"):
            raw_t = calls["total"]["edge"]
            calls["total"]["model_win_prob"] = calls["total"]["win_prob"]
            _tsd = tshrink_info.get("resid_sd")
            pick_over = calls["total"]["pick"] == "OVER"
            calls["total"]["win_prob"] = market_cover_prob(
                raw_t, tshrink if tshrink is not None else 0.0, _tsd,
                alpha=0.0, pick_home=pick_over)
            calls["total"]["cons_prob"] = conservative_cover_prob(
                raw_t, tshrink_info, pick_home=pick_over)
            to_conf = confidence_label(calls["total"]["win_prob"], raw_t,
                                       low_conf, calibrated,
                                       calls["total"]["cons_prob"], fcs_involved)
        else:
            to_conf = ("NO BET", "no line")

        # LOCK only if kickoff is close enough that the inputs are current.
        # Everything else is a preview: shown in the table, not written to the
        # immutable record, and picked up on a later run nearer to kickoff.
        # prefer the full timestamp; a bare date parses to midnight UTC and
        # would make the lock window fire hours early for an afternoon kickoff
        kick_ts = pd.to_datetime(g.get("kickoff_utc") or g.get("date"),
                                 errors="coerce", utc=True)
        hrs = ((kick_ts - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600.0
               if pd.notna(kick_ts) else 0.0)
        is_locked = hrs <= args.lock_hours

        (ledger_rows if is_locked else preview_rows).append({
            "event_id": str(eid), "predicted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "season": args.season, "week": g.get("week"), "kickoff": g.get("date"),
            "home_team": home, "away_team": away,
            "neutral_site": bool(g.get("neutral_site", False)),
            "exp_home": round(pred["exp_home"], 2), "exp_away": round(pred["exp_away"], 2),
            "pred_margin": round(pred["mean_margin"], 2),
            "pred_total": round(pred["total"], 2),
            "home_win_prob": round(pred["home_win_prob"], 4),
            "line_spread_home": (line or {}).get("spread_home"),
            "line_total": (line or {}).get("total"),
            "book": (line or {}).get("book"),
            "spread_pick": (calls["spread"]["pick"] if calls.get("spread") else None),
            "spread_call": sp_conf[0],
            "spread_win_prob": round(calls["spread"]["win_prob"], 4) if calls.get("spread") else None,
            "total_pick": (calls["total"]["pick"] if calls.get("total") else None),
            "total_call": to_conf[0],
            "total_win_prob": round(calls["total"]["win_prob"], 4) if calls.get("total") else None,
            "model_version": LG.model_version(calib),
        })

        rows.append({"home": home, "away": away, "kick": str(g.get("date", ""))[:16],
                     "locked": is_locked, "hrs_to_kick": hrs,
                     "spread_conf": sp_conf, "total_conf": to_conf,
                     "pred": pred, "calls": calls,
                     "spread_conf": sp_conf, "total_conf": to_conf,
                     "injuries": {home: inj_h, away: inj_a}})

    # 1) grade anything that finished since the last run. This happens BEFORE
    #    recording, so a game can never be graded against a prediction written
    #    in the same pass.
    print("\nUpdating results for completed games...")
    try:
        LG.grade()
    except Exception as e:  # noqa: BLE001
        print(f"  (grading skipped: {type(e).__name__}: {str(e)[:70]})")

    # 2) lock in today's predictions. Existing rows are never overwritten --
    #    re-running before kickoff leaves the original call standing, which is
    #    the entire point of keeping a prospective record.
    if ledger_rows:
        n_new, n_kept = LG.record(ledger_rows)
        print(f"  ledger: {n_new} prediction(s) LOCKED "
              f"(kickoff within {args.lock_hours:g}h)"
              + (f", {n_kept} already locked from an earlier run" if n_kept else ""))
    if preview_rows:
        print(f"  {len(preview_rows)} game(s) shown as PREVIEW only -- too far out "
              f"to lock; re-run within {args.lock_hours:g}h of kickoff to record them")

    if not rows:
        print("\nNo games could be rated.")
        return 0

    oos = calib
    meta = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_games": len(rows),
        "n_priced": sum(1 for r in rows if r["calls"].get("spread")),
        "n_train": len(games),
        "through": str(games["season"].max()),
        "sims": args.sims,
        "margin_rmse": oos.get("margin_resid_std", 16.0),
        "su": oos.get("su_accuracy") or 0.737,
        "shrink": shrink,
        "calibrated": calibrated,
        "shrink_n": shrink_info.get("n"),
        "lock_hours": args.lock_hours,
        "record": LG.summarize(),
    }
    # completed games + running record, so the report becomes a season-long
    # scorecard rather than a snapshot
    # The full results table lives in CFB_Results.exe. This report stays
    # forward-looking; all it carries is a one-line pointer to the record so
    # the two tools stay obviously connected.
    results_html = ""
    try:
        led = LG._load()
        n_graded = int(led["graded_at"].notna().sum())
        if n_graded:
            s_ = LG.summarize(led)
            bits = []
            su = s_.get("straight_up")
            if su:
                bits.append(f"straight up {su['w']}-{su['n']-su['w']} ({su['pct']*100:.0f}%)")
            for k, lab in (("spread_all", "ATS"), ("total_all", "O/U")):
                r_ = s_.get(k)
                if r_:
                    bits.append(f"{lab} {r_['w']}-{r_['l']} ({r_['pct']*100:.0f}%)")
            results_html = (
                '<div class="rec"><b>Record so far:</b> '
                + html.escape(" &middot; ".join(bits)).replace("&amp;middot;", "&middot;")
                + f' &nbsp;<span class="dim">over {n_graded} completed games &mdash; '
                  'open CFB_Results.exe for the full scorecard</span></div>')
    except Exception as e:  # noqa: BLE001
        print(f"  (record line skipped: {type(e).__name__}: {str(e)[:60]})")

    out = args.out or os.path.join(DATA_DIR, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(rows, meta, results_html))

    print(f"\n{'='*62}")
    print(f"  Report written: {out}")
    print(f"{'='*62}")
    plays = [r for r in rows if r["spread_conf"][0] == "PLAY" or r["total_conf"][0] == "PLAY"]
    print(f"  {len(rows)} games | {meta['n_priced']} priced | {len(plays)} flagged PLAY")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
