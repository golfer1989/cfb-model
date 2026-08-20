#!/usr/bin/env python3
"""
Prediction ledger: an append-only record of what the model said BEFORE each
game, graded against what actually happened.

WHY THIS IS THE MOST IMPORTANT FILE IN THE PROJECT

Every accuracy number quoted so far is a backtest -- the model scored against
history it was built from. Backtests are where look-ahead leaks hide, and this
project has already had one bad enough to invent an edge that did not exist.
A prospective ledger cannot leak, because the prediction is written to disk
before the game kicks off and is never touched again.

THE ONE RULE: a prediction is IMMUTABLE once written.

It is trivially easy, and completely worthless, to "update" a stored
prediction later using ratings that have since seen the result. That is not
tracking, it is retro-fitting with extra steps. So `record()` refuses to
overwrite an existing row for an event, and grading only ever ADDS outcome
columns to a row that already exists.

Each row stores the calibration in force at the time, so a change in the model
mid-season does not silently contaminate the record -- you can always split
the ledger by model version and grade each separately.

Files:
  data/ledger.csv         one immutable row per (event_id) prediction
Usage:
  python ledger.py grade      fetch finals + closing lines, grade open rows
  python ledger.py record     show the running record
"""

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from paths import DATA_DIR

LEDGER = os.path.join(DATA_DIR, "ledger.csv")
BREAKEVEN_110 = 0.5238

COLUMNS = [
    # --- written at prediction time, NEVER modified afterwards -------------
    "event_id", "predicted_at", "season", "week", "kickoff",
    "home_team", "away_team", "neutral_site",
    "exp_home", "exp_away", "pred_margin", "pred_total", "home_win_prob",
    "line_spread_home", "line_total", "book",
    "spread_pick", "spread_call", "spread_win_prob",
    "total_pick", "total_call", "total_win_prob",
    "model_version",
    # --- filled in later by grade(), outcome only -------------------------
    "graded_at", "final_home", "final_away", "act_margin", "act_total",
    "close_spread_home", "close_total",
    "su_correct", "spread_result", "total_result",
]


# columns holding text; created as object dtype so grading can write strings
# into a previously-empty frame without pandas raising a dtype warning
TEXT_COLS = {
    "event_id", "predicted_at", "kickoff", "home_team", "away_team", "book",
    "spread_pick", "spread_call", "total_pick", "total_call", "model_version",
    "graded_at", "su_correct", "spread_result", "total_result",
}


def _load():
    if not os.path.exists(LEDGER):
        df = pd.DataFrame({c: pd.Series(dtype="object" if c in TEXT_COLS else "float64")
                           for c in COLUMNS})
        return df
    df = pd.read_csv(LEDGER)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.Series([np.nan] * len(df),
                              dtype="object" if c in TEXT_COLS else "float64")
    for c in TEXT_COLS:
        df[c] = df[c].astype("object")
    df["event_id"] = df["event_id"].astype(str)
    return df[COLUMNS]


def _save(df):
    df = df.drop_duplicates("event_id", keep="first")
    df.to_csv(LEDGER, index=False)


def model_version(calib):
    """A short fingerprint of the calibration a prediction was made under, so
    the record can be split if the model changes mid-season."""
    es = (calib or {}).get("edge_shrinkage") or {}
    ts = (calib or {}).get("total_edge_shrinkage") or {}
    return (f"sig{calib.get('score_resid_std', 0):.2f}"
            f"_es{es.get('shrink_factor', 0):.3f}"
            f"_ts{ts.get('shrink_factor', 0):.3f}")


MAX_LOCK_LEAD_HOURS = 48


def record(rows, max_lead_hours=MAX_LOCK_LEAD_HOURS):
    """Append predictions. Existing events are left EXACTLY as they were.

    A second guard sits here alongside the caller's lock window: a prediction
    made more than `max_lead_hours` before kickoff is REFUSED outright. The
    caller already filters on this, but the ledger is the thing whose
    integrity matters, so it enforces the rule itself rather than trusting
    every future caller to remember. An early build without that filter locked
    99 week-1 games between 256 and 472 hours before kickoff; they had to be
    thrown away, because a pick frozen eleven days early with stale ratings
    and a line that moved since is not a pick anyone would have made.

    Returns (n_new, n_skipped). A skip is not an error -- it means the report
    was re-run before kickoff and the original call stands, which is the whole
    point of the ledger.
    """
    df = _load()
    have = set(df["event_id"])
    now = dt.datetime.now(dt.timezone.utc)

    def _fresh_enough(r):
        k = pd.to_datetime(r.get("kickoff"), errors="coerce", utc=True)
        if pd.isna(k):
            return True                      # unknown kickoff: let it through
        return (k - now).total_seconds() / 3600.0 <= max_lead_hours

    fresh = [r for r in rows
             if str(r["event_id"]) not in have and _fresh_enough(r)]
    skipped = len(rows) - len(fresh)
    if fresh:
        add = pd.DataFrame(fresh)
        add["event_id"] = add["event_id"].astype(str)
        for c in COLUMNS:
            if c not in add.columns:
                add[c] = np.nan
        df = pd.concat([df, add[COLUMNS]], ignore_index=True)
        _save(df)
    return len(fresh), skipped


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade(verbose=True):
    """Fill in outcomes for any ungraded row whose game has finished.

    Grading uses the CLOSING line, which is what was asked for and is the
    standard benchmark. Note the honest caveat: a bet placed when the report
    ran would have been at the line AT THAT TIME, which may differ. Both
    numbers are stored so the difference is visible rather than hidden.
    """
    import fetch_odds as FO
    from fetch_espn_data import fetch_week_cached, extract_games, fetch_conference_map

    df = _load()
    if df.empty:
        return df, 0
    open_rows = df[df["graded_at"].isna()]
    if open_rows.empty:
        if verbose:
            print("  nothing new to grade")
        return df, 0

    # pull final scores for the seasons/weeks represented among ungraded rows
    finals = {}
    conf_cache = os.path.join(DATA_DIR, "conferences.json")
    for season in sorted(open_rows["season"].dropna().unique()):
        season = int(season)
        try:
            cmap = fetch_conference_map(season, conf_cache)
        except Exception:  # noqa: BLE001
            continue
        weeks = sorted(open_rows[open_rows["season"] == season]["week"].dropna().unique())
        for wk in weeks:
            for st in (2, 3):
                try:
                    raw, _ = fetch_week_cached(season, st, int(wk), refresh=True)
                except Exception:  # noqa: BLE001
                    continue
                for gm in extract_games(raw, cmap):
                    if gm["status"] == "STATUS_FINAL":
                        finals[str(gm["event_id"])] = gm

    graded = 0
    for i, row in open_rows.iterrows():
        eid = str(row["event_id"])
        gm = finals.get(eid)
        if not gm or gm["home_score"] is None:
            continue

        fh, fa = float(gm["home_score"]), float(gm["away_score"])
        margin, total = fh - fa, fh + fa

        # closing line for this event
        close_sp = close_tot = np.nan
        try:
            data, _ = FO.fetch_event_odds(int(eid), refresh=True)
            rec = FO.parse_odds(data, int(eid))
            if rec:
                close_sp = rec.get("spread_close_home")
                if close_sp is None:
                    close_sp = rec.get("spread_current")
                close_tot = rec.get("total_close")
                if close_tot is None:
                    close_tot = rec.get("total_current")
        except Exception:  # noqa: BLE001
            pass
        # fall back to the line captured at prediction time
        if close_sp is None or (isinstance(close_sp, float) and np.isnan(close_sp)):
            close_sp = row["line_spread_home"]
        if close_tot is None or (isinstance(close_tot, float) and np.isnan(close_tot)):
            close_tot = row["line_total"]

        df.at[i, "graded_at"] = dt.datetime.now().isoformat(timespec="seconds")
        df.at[i, "final_home"] = fh
        df.at[i, "final_away"] = fa
        df.at[i, "act_margin"] = margin
        df.at[i, "act_total"] = total
        df.at[i, "close_spread_home"] = close_sp
        df.at[i, "close_total"] = close_tot

        # straight up -- the model picks a winner on every game
        pred_home_win = float(row["pred_margin"]) > 0
        df.at[i, "su_correct"] = int(pred_home_win == (margin > 0))

        # spread, graded against the CLOSING number
        df.at[i, "spread_result"] = _grade_spread(row, margin, close_sp)
        df.at[i, "total_result"] = _grade_total(row, total, close_tot)
        graded += 1

    _save(df)
    if verbose:
        print(f"  graded {graded} newly-completed game(s)")
    return df, graded


def _grade_spread(row, act_margin, close_spread_home):
    """WIN / LOSS / PUSH for the side the model picked, vs the closing line."""
    pick = row.get("spread_pick")
    if not isinstance(pick, str) or not pick or pd.isna(close_spread_home):
        return "NO PICK"
    cover_margin = act_margin + float(close_spread_home)   # >0 => home covered
    if cover_margin == 0:
        return "PUSH"
    home_covered = cover_margin > 0
    took_home = pick.upper().startswith("HOME")
    return "WIN" if (took_home == home_covered) else "LOSS"


def _grade_total(row, act_total, close_total):
    pick = row.get("total_pick")
    if not isinstance(pick, str) or not pick or pd.isna(close_total):
        return "NO PICK"
    if act_total == float(close_total):
        return "PUSH"
    went_over = act_total > float(close_total)
    took_over = pick.upper().startswith("OVER")
    return "WIN" if (took_over == went_over) else "LOSS"


# ---------------------------------------------------------------------------
# Running record
# ---------------------------------------------------------------------------

def wilson(w, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(df=None, only_calls=("PLAY", "LEAN")):
    """Running record. Returns a dict of tallies with honest intervals."""
    df = _load() if df is None else df
    done = df[df["graded_at"].notna()]
    out = {"n_graded": int(len(done))}
    if done.empty:
        return out

    su = done["su_correct"].dropna()
    if len(su):
        w, n = int(su.sum()), len(su)
        lo, hi = wilson(w, n)
        out["straight_up"] = {"w": w, "n": n, "pct": w / n, "lo": lo, "hi": hi}

    for kind, res_col, call_col in [("spread", "spread_result", "spread_call"),
                                    ("total", "total_result", "total_call")]:
        for scope, sel in [("all", done),
                           ("recommended", done[done[call_col].isin(only_calls)])]:
            r = sel[res_col].dropna()
            r = r[r.isin(["WIN", "LOSS", "PUSH"])]
            wins = int((r == "WIN").sum())
            losses = int((r == "LOSS").sum())
            push = int((r == "PUSH").sum())
            n = wins + losses
            if n == 0:
                continue
            lo, hi = wilson(wins, n)
            # flat-stake ROI at -110
            roi = (wins * (100 / 110.0) - losses) / n
            out[f"{kind}_{scope}"] = {"w": wins, "l": losses, "push": push,
                                      "n": n, "pct": wins / n,
                                      "lo": lo, "hi": hi, "roi": roi}

    # model vs market on the games that have finished
    m = done.dropna(subset=["pred_margin", "act_margin"])
    if len(m) >= 5:
        out["margin_rmse"] = float(np.sqrt(np.mean((m["pred_margin"] - m["act_margin"]) ** 2)))
        mk = m.dropna(subset=["close_spread_home"])
        if len(mk) >= 5:
            out["market_rmse"] = float(np.sqrt(np.mean(
                (-mk["close_spread_home"] - mk["act_margin"]) ** 2)))
    return out


def format_summary(s):
    if not s.get("n_graded"):
        return "No completed games in the ledger yet."
    L = [f"Season-to-date record over {s['n_graded']} completed games:"]
    su = s.get("straight_up")
    if su:
        L.append(f"  Straight up : {su['w']}-{su['n']-su['w']} "
                 f"= {su['pct']*100:.1f}%  [{su['lo']*100:.1f}, {su['hi']*100:.1f}]")
    for key, label in [("spread_all", "ATS, all games"),
                       ("spread_recommended", "ATS, PLAY/LEAN only"),
                       ("total_all", "O/U, all games"),
                       ("total_recommended", "O/U, PLAY/LEAN only")]:
        r = s.get(key)
        if not r:
            continue
        flag = ""
        if r["n"] >= 100:
            flag = ("  BEATS break-even" if r["lo"] > BREAKEVEN_110 else
                    "  below break-even" if r["hi"] < BREAKEVEN_110 else
                    "  inside noise")
        L.append(f"  {label:22s}: {r['w']}-{r['l']}"
                 + (f"-{r['push']}" if r["push"] else "")
                 + f" = {r['pct']*100:.1f}%  ROI {r['roi']*100:+.1f}%"
                 + f"  [{r['lo']*100:.1f}, {r['hi']*100:.1f}]{flag}")
    if "margin_rmse" in s:
        line = f"  Margin RMSE : model {s['margin_rmse']:.2f}"
        if "market_rmse" in s:
            line += f"  vs market {s['market_rmse']:.2f}"
        L.append(line)
    n = max((s.get(k, {}).get("n", 0) for k in ("spread_all", "spread_recommended")), default=0)
    if 0 < n < 200:
        L.append(f"\n  NOTE: {n} graded bets is far too few to conclude anything.")
        L.append("  Detecting a 2-point ATS edge reliably needs ~700; confirming")
        L.append("  you beat -110 needs ~3500. Treat this as bookkeeping, not proof.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Prediction ledger: grade and report.")
    ap.add_argument("cmd", choices=["grade", "record", "show"], default="show", nargs="?")
    args = ap.parse_args()

    if args.cmd == "grade":
        print("Grading completed games...")
        grade()
    df = _load()
    print(f"\nLedger: {len(df)} predictions, "
          f"{int(df['graded_at'].notna().sum())} graded\n")
    print(format_summary(summarize(df)))


if __name__ == "__main__":
    main()
