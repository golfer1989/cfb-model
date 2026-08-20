#!/usr/bin/env python3
"""
Walk-forward backtest harness.

Two jobs:

  1. HONEST VALIDATION. Every prediction is made using only games that had
     already been played, so the reported error is what the model would
     actually have achieved, not a curve fit to data it already saw.

  2. CALIBRATION OF THE SIMULATOR'S NOISE. The residual standard deviation
     computed on the training data is biased LOW, because the ratings were
     fitted to those same games. With ~370 free parameters the in-sample
     sigma understates true error by several percent, and a simulator fed a
     too-small sigma produces win probabilities that are systematically too
     confident. This harness measures sigma out-of-sample and writes it to
     data/calibration.json for the simulator to consume.

Outputs:
  data/backtest_preds.csv  -- one row per out-of-sample game
  data/calibration.json    -- sigma, correlation and bias for the simulator

Run:  python backtest.py                 (default: validate 2024-2025)
      python backtest.py --seasons 2025
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import cfb_ratings as R
import cfb_predict as P

from paths import DATA_DIR  # frozen-aware; see paths.py


def walk_forward(games, seasons=(2024, 2025), lam=R.DEFAULT_LAMBDA, decay=R.DEFAULT_DECAY,
                 min_train=400, conf_shrink=True, k_conf=2.0, verbose=True,
                 extra_train=None, coach_changes=None):
    """Predict every game in `seasons` using only games that had ALREADY BEEN
    PLAYED at kickoff.

    Ordering MUST come from the kickoff date, never from (season, week).
    ESPN numbers postseason weeks from 1, and the fetcher stores that number
    verbatim, so every bowl and playoff game -- including the national
    championship played the following January -- lands in the data as
    `week == 1` of its season. 135 such rows exist in 2023-2025.

    Keying the walk-forward on `season*100 + week` therefore put those games
    in the TRAINING slice for every subsequent week of their own season, at
    full same-season weight. Predicting week 2 of 2025 was done with the
    January 2026 national championship already in hand. The assertion below
    exists so this cannot silently return.

    BLOCKING is also date-derived, for the same reason. An earlier version
    iterated ESPN's week labels, so the December/January postseason games
    carrying `week == 1` landed in the FIRST test block of their season and
    were graded off ratings fitted only on games before the season opener.
    That is the safe direction -- no leak -- but it charged the model for 92
    bowl games predicted with zero current-season information (RMSE 16.5,
    straight-up 52%, vs 13.9 / 62% when trained through December), which
    understated the headline by ~0.12 RMSE and, worse, poisoned the fitted
    sigma and win-probability calibration with rows no production run would
    ever produce. Blocks are now 7-day windows anchored on each season's
    first kickoff, so a January playoff game is predicted from ratings that
    know everything through the week before it -- exactly what a live run in
    January would do.

    `extra_train`: optional extra games (e.g. FCS-vs-FCS schedules) added to
    the TRAINING slice only, filtered by the same kickoff cutoff. Test games
    always come from `games`.

    `coach_changes`: optional {season: set(normalized team names)} of teams
    with a first-year head coach; applies the fitted margin adjustment the
    production predictor applies (see cfb_predict.COACH_CHANGE_MARGIN).
    """
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games["neutral_site"] = games["neutral_site"].astype(bool)
    # order on the true kickoff instant. `date` is now the US Eastern game
    # date (correct for display and cross-referencing) but two games on the
    # same local date can be hours apart, and only the timestamp orders them.
    games["_kick"] = pd.to_datetime(
        games["kickoff_utc"] if "kickoff_utc" in games.columns else games["date"],
        errors="coerce", utc=True)
    games = games.dropna(subset=["_kick"])

    ext = None
    if extra_train is not None and len(extra_train):
        ext = extra_train.dropna(subset=["home_score", "away_score"]).copy()
        ext["neutral_site"] = ext["neutral_site"].astype(bool)
        ext["_kick"] = pd.to_datetime(
            ext["kickoff_utc"] if "kickoff_utc" in ext.columns else ext["date"],
            errors="coerce", utc=True)
        ext = ext.dropna(subset=["_kick"])

    rows = []
    for season in seasons:
        sg = games[games["season"] == season]
        if len(sg) == 0:
            continue
        # 7-day blocks anchored on the season's first kickoff. NEVER ESPN's
        # week label -- postseason games carry week==1 (see docstring).
        day0 = sg["_kick"].min()
        blk = ((sg["_kick"] - day0).dt.days // 7)
        for b in sorted(blk.unique()):
            test = sg[blk == b]
            if len(test) == 0:
                continue
            cutoff = test["_kick"].min()
            train = games[games["_kick"] < cutoff]
            if ext is not None:
                train = pd.concat([train, ext[ext["_kick"] < cutoff]],
                                  ignore_index=True)
            if len(train) < min_train:
                continue
            # hard guarantee: nothing in training kicked off at or after the
            # first game being predicted
            assert train["_kick"].max() < cutoff, (
                f"look-ahead leak: training data extends to "
                f"{train['_kick'].max()} but test block starts {cutoff}")
            # current_season MUST be passed explicitly: if it defaulted to the
            # max season present in the training slice, then when predicting
            # week 1 of a new season the previous season would be treated as
            # "current" and get full weight.
            r = R.fit_ratings(train, lam=lam, cv=False, decay=decay,
                              current_season=season, conf_shrink=conf_shrink,
                              k_conf=k_conf)
            nc = (coach_changes or {}).get(season)
            for _, g in test.iterrows():
                h = R.resolve_team(r, g["home_team"])
                a = R.resolve_team(r, g["away_team"])
                if not (r.has(h) and r.has(a)):
                    continue
                s = 0.0 if g["neutral_site"] else 1.0
                eh = r.mu + r.off[h] - r.defn[a] + s * r.hfa / 2
                ea = r.mu + r.off[a] - r.defn[h] - s * r.hfa / 2
                if nc:
                    m_adj = P.coach_margin_adj(nc, g["home_team"], g["away_team"])
                    eh += m_adj / 2.0
                    ea -= m_adj / 2.0
                # same soft floor the predictor applies, so the calibration
                # below is fitted on production numbers rather than raw ones
                eh = float(P.score_floor(eh))
                ea = float(P.score_floor(ea))
                rows.append({
                    "season": season, "week": g["week"], "date": g.get("date"),
                    "home_team": g["home_team"], "away_team": g["away_team"],
                    "neutral_site": bool(g["neutral_site"]),
                    "exp_home": eh, "exp_away": ea,
                    "pred_margin": eh - ea, "pred_total": eh + ea,
                    "act_home": g["home_score"], "act_away": g["away_score"],
                    "act_margin": g["home_score"] - g["away_score"],
                    "act_total": g["home_score"] + g["away_score"],
                    "n_train": len(train),
                })
            if verbose:
                print(f"  {season} blk{b:>2}: trained on {len(train):>4} games, "
                      f"predicted {len(test):>3}", flush=True)
    return pd.DataFrame(rows)


def metrics(df):
    m = {}
    em = df["pred_margin"] - df["act_margin"]
    et = df["pred_total"] - df["act_total"]
    m["n_games"] = int(len(df))
    m["margin_rmse"] = float(np.sqrt(np.mean(em ** 2)))
    m["margin_mae"] = float(np.mean(np.abs(em)))
    m["margin_bias"] = float(np.mean(em))
    m["total_rmse"] = float(np.sqrt(np.mean(et ** 2)))
    m["total_mae"] = float(np.mean(np.abs(et)))
    m["total_bias"] = float(np.mean(et))
    m["su_accuracy"] = float(np.mean((df["pred_margin"] > 0) == (df["act_margin"] > 0)))

    # residual structure for the simulator
    rh = df["act_home"] - df["exp_home"]
    ra = df["act_away"] - df["exp_away"]
    m["score_resid_std"] = float(np.std(pd.concat([rh, ra])))
    m["resid_corr"] = float(np.corrcoef(rh, ra)[0, 1])
    m["margin_resid_std"] = float(np.std(em))
    m["total_resid_std"] = float(np.std(et))
    return m


def fit_total_calibration(df):
    """Totals need shrinking. Margins do not.

    Measured on the walk-forward predictions: the raw total model is WORSE
    than simply predicting the league-average total (RMSE 16.67 vs 16.48).
    Regressing actual on predicted gives a slope near 0.45, meaning the model
    spreads its total projections roughly twice as wide as reality warrants.

    The cause is structural. A margin is (off_h - def_a) - (off_a - def_h), so
    rating errors partially CANCEL. A total is (off_h - def_a) + (off_a -
    def_h), so the same errors ACCUMULATE. Totals are therefore intrinsically
    noisier and must be shrunk toward the mean.

    Fitting on the earlier season and applying to the later one (no leakage)
    moves held-out total RMSE from 16.43 to 15.87, which finally beats the
    predict-the-mean benchmark. Even so R^2 is only ~0.05: the honest read is
    that this model has real but modest skill on totals, and the card should
    say so rather than implying precision it does not have.
    """
    seasons = sorted(df["season"].unique())
    if len(seasons) >= 2:
        fit = df[df["season"] != seasons[-1]]
    else:
        fit = df
    if len(fit) < 100:
        return {"slope": 1.0, "intercept": 0.0, "r2": None, "n_fit": len(fit),
                "note": "insufficient data; calibration disabled"}
    slope, intercept = np.polyfit(fit["pred_total"], fit["act_total"], 1)
    r = float(np.corrcoef(fit["pred_total"], fit["act_total"])[0, 1])
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": r ** 2, "n_fit": int(len(fit)),
            "fit_seasons": [int(s) for s in seasons[:-1]] or [int(seasons[0])]}


def apply_total_calibration(pred_total, cal):
    return cal["slope"] * np.asarray(pred_total, dtype=float) + cal["intercept"]


def calibration_check(df, sigma_margin):
    """Are the model's implied win probabilities actually calibrated?

    Uses a Normal approximation on the margin purely as a calibration probe;
    the production simulator uses the drive-based model instead.
    """
    from math import erf, sqrt
    p = df["pred_margin"].values / (sigma_margin * sqrt(2.0))
    win_p = np.array([0.5 * (1 + erf(x)) for x in p])
    actual = (df["act_margin"] > 0).astype(float).values

    brier = float(np.mean((win_p - actual) ** 2))
    eps = 1e-12
    ll = float(-np.mean(actual * np.log(win_p + eps) + (1 - actual) * np.log(1 - win_p + eps)))

    bins = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    table = []
    for lo, hi in bins:
        sel = (win_p >= lo) & (win_p < hi)
        if sel.sum() >= 10:
            table.append({
                "bucket": f"{lo:.0%}-{hi:.0%}",
                "n": int(sel.sum()),
                "predicted": round(float(win_p[sel].mean()), 3),
                "actual": round(float(actual[sel].mean()), 3),
                "gap": round(float(win_p[sel].mean() - actual[sel].mean()), 3),
            })
    return {"brier": brier, "log_loss": ll, "buckets": table}


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest and simulator calibration.")
    ap.add_argument("--games", default=os.path.join(DATA_DIR, "real_games.csv"))
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--lam", type=float, default=R.DEFAULT_LAMBDA)
    ap.add_argument("--decay", type=float, default=R.DEFAULT_DECAY)
    ap.add_argument("--no-conf-shrink", action="store_true")
    ap.add_argument("--no-coach-adj", action="store_true",
                    help="Disable the first-year-head-coach margin adjustment")
    ap.add_argument("--fcs-ab", action="store_true",
                    help="Run the walk-forward WITH and WITHOUT FCS-vs-FCS "
                         "training augmentation (needs data/fcs_games.csv from "
                         "fetch_espn_data.py --fcs), print the comparison, and "
                         "persist the winner into calibration.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    games = pd.read_csv(args.games)

    # coach-change context (leak-free: season S coach lists predate season S)
    coach_changes = None
    if not args.no_coach_adj:
        coach_changes = P.load_coach_changes(args.seasons)
        n_cc = sum(len(v) for v in coach_changes.values())
        if n_cc:
            print(f"Coach adjustment active: {n_cc} first-year-coach teams "
                  f"across {args.seasons} ({P.COACH_CHANGE_MARGIN:+.1f} pts of margin each)")
        else:
            coach_changes = None

    # FCS augmentation: honor the previously validated decision unless --fcs-ab
    fcs = R.load_fcs_games()
    prior_fcs_flag = {}
    cpath = os.path.join(DATA_DIR, "calibration.json")
    if os.path.exists(cpath):
        try:
            with open(cpath, encoding="utf-8") as f:
                prior_fcs_flag = (json.load(f).get("fcs_augmentation") or {})
        except (json.JSONDecodeError, OSError):
            prior_fcs_flag = {}
    fcs_info = dict(prior_fcs_flag) if prior_fcs_flag else {"enabled": False}

    print(f"Walk-forward backtest over {args.seasons} "
          f"(lambda={args.lam}, decay={args.decay}, "
          f"conf_shrink={not args.no_conf_shrink})\n")

    if args.fcs_ab:
        if fcs is None:
            print("--fcs-ab needs data/fcs_games.csv. Run: python fetch_espn_data.py --fcs")
            return 1
        print("=== A/B: baseline (no FCS augmentation) ===")
        df_base = walk_forward(games, seasons=tuple(args.seasons), lam=args.lam,
                               decay=args.decay, conf_shrink=not args.no_conf_shrink,
                               verbose=False, coach_changes=coach_changes)
        print("=== A/B: with FCS-vs-FCS training games ===")
        df_fcs = walk_forward(games, seasons=tuple(args.seasons), lam=args.lam,
                              decay=args.decay, conf_shrink=not args.no_conf_shrink,
                              verbose=False, coach_changes=coach_changes,
                              extra_train=fcs)
        mb, mf = metrics(df_base), metrics(df_fcs)
        # FCS-involved subset: where the augmentation is supposed to help
        gk = games[["season", "week", "home_team", "away_team",
                    "home_conf", "away_conf"]]
        def _fcs_rmse(df):
            j = df.merge(gk, on=["season", "week", "home_team", "away_team"], how="left")
            m = (~j["home_conf"].isin(R.FBS_CONFERENCES)) | \
                (~j["away_conf"].isin(R.FBS_CONFERENCES))
            e = j.loc[m, "pred_margin"] - j.loc[m, "act_margin"]
            return float(np.sqrt(np.mean(e ** 2))), int(m.sum())
        fb, nb = _fcs_rmse(df_base)
        ff, nf = _fcs_rmse(df_fcs)
        print("\n===== FCS AUGMENTATION A/B (identical everything else) =====")
        print(f"  overall RMSE : base {mb['margin_rmse']:.4f}  vs  FCS {mf['margin_rmse']:.4f}"
              f"   ({mf['margin_rmse']-mb['margin_rmse']:+.4f})")
        print(f"  FCS-game RMSE: base {fb:.4f} (n={nb})  vs  FCS {ff:.4f} (n={nf})"
              f"   ({ff-fb:+.4f})")
        print(f"  straight-up  : base {mb['su_accuracy']*100:.2f}%  vs  FCS {mf['su_accuracy']*100:.2f}%")
        enabled = (mf["margin_rmse"] < mb["margin_rmse"]) and (ff < fb)
        verdict = ("IMPROVES both overall and FCS games -> ENABLED" if enabled
                   else "does not improve both -> left OFF")
        print(f"\n  VERDICT: augmentation {verdict}")
        fcs_info = {
            "enabled": bool(enabled), "n_fcs_games": int(len(fcs)),
            "ab": {"rmse_base": mb["margin_rmse"], "rmse_fcs": mf["margin_rmse"],
                   "fcs_rmse_base": fb, "fcs_rmse_fcs": ff,
                   "n_fcs_involved": nb},
        }
        df = df_fcs if enabled else df_base
    else:
        use_fcs = bool(fcs_info.get("enabled")) and fcs is not None
        if fcs_info.get("enabled") and fcs is None:
            print("NOTE: calibration says FCS augmentation is enabled but "
                  "data/fcs_games.csv is missing -- running without it.")
            use_fcs = False
        if use_fcs:
            print(f"FCS augmentation ON ({len(fcs)} FCS-vs-FCS training games; "
                  f"validated by a previous --fcs-ab run)")
        df = walk_forward(games, seasons=tuple(args.seasons), lam=args.lam,
                          decay=args.decay, conf_shrink=not args.no_conf_shrink,
                          verbose=not args.quiet, coach_changes=coach_changes,
                          extra_train=fcs if use_fcs else None)

    out = os.path.join(DATA_DIR, "backtest_preds.csv")
    df.to_csv(out, index=False)

    m = metrics(df)
    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 60)
    print(f"  games evaluated       : {m['n_games']}")
    print(f"  MARGIN  RMSE / MAE    : {m['margin_rmse']:.2f} / {m['margin_mae']:.2f}"
          f"   (bias {m['margin_bias']:+.2f})")
    print(f"  TOTAL   RMSE / MAE    : {m['total_rmse']:.2f} / {m['total_mae']:.2f}"
          f"   (bias {m['total_bias']:+.2f})")
    print(f"  straight-up accuracy  : {m['su_accuracy']*100:.1f}%")
    print(f"  score residual sigma  : {m['score_resid_std']:.2f}  "
          f"(corr between teams {m['resid_corr']:+.3f})")

    # in-sample comparison, to show the size of the bias being corrected
    full = R.fit_ratings(games, lam=args.lam, cv=False, decay=args.decay,
                         conf_shrink=not args.no_conf_shrink)
    print(f"\n  in-sample sigma would be {full.resid_std:.2f} vs out-of-sample "
          f"{m['score_resid_std']:.2f}")
    print(f"  -> using the in-sample value would understate game variance by "
          f"{(1 - full.resid_std/m['score_resid_std'])*100:.1f}%")

    cal = calibration_check(df, m["margin_resid_std"])
    print("\n" + "=" * 60)
    print("WIN-PROBABILITY CALIBRATION")
    print("=" * 60)
    print(f"  Brier score : {cal['brier']:.4f}   (lower is better; 0.25 = coin flip)")
    print(f"  log loss    : {cal['log_loss']:.4f}")
    if cal["buckets"]:
        print(f"\n  {'bucket':>10} {'n':>5} {'predicted':>10} {'actual':>8} {'gap':>7}")
        for b in cal["buckets"]:
            print(f"  {b['bucket']:>10} {b['n']:>5} {b['predicted']:>10.3f} "
                  f"{b['actual']:>8.3f} {b['gap']:>+7.3f}")
        print("\n  gap = predicted - actual; consistently positive means overconfident")

    # --- totals calibration --------------------------------------------
    tcal = fit_total_calibration(df)
    print("\n" + "=" * 60)
    print("TOTALS CALIBRATION")
    print("=" * 60)
    naive_total = float(np.sqrt(np.mean((df["act_total"].mean() - df["act_total"]) ** 2)))
    print(f"  raw total RMSE            : {m['total_rmse']:.3f}")
    print(f"  predict-the-mean RMSE     : {naive_total:.3f}"
          f"   {'<-- raw model is WORSE than this' if m['total_rmse'] > naive_total else ''}")
    if tcal.get("slope") is not None and len(df["season"].unique()) >= 2:
        last = sorted(df["season"].unique())[-1]
        te = df[df["season"] == last]
        cal_pred = apply_total_calibration(te["pred_total"], tcal)
        cal_rmse = float(np.sqrt(np.mean((cal_pred - te["act_total"]) ** 2)))
        raw_rmse = float(np.sqrt(np.mean((te["pred_total"] - te["act_total"]) ** 2)))
        print(f"\n  shrink fitted on {tcal.get('fit_seasons')}, tested on {last} (no leakage):")
        print(f"    slope {tcal['slope']:.4f}, intercept {tcal['intercept']:.2f}"
              f"  (slope 1.0 would mean no shrink needed)")
        print(f"    raw {raw_rmse:.3f}  ->  calibrated {cal_rmse:.3f}"
              f"   ({raw_rmse - cal_rmse:+.3f})")
        print(f"    R^2 = {tcal['r2']:.4f}  -- totals carry real but MODEST signal")

    calib = {
        "fitted_on": {"games_file": args.games, "seasons": args.seasons},
        "params": {"lam": args.lam, "decay": args.decay,
                   "conf_shrink": not args.no_conf_shrink},
        "out_of_sample": m,
        "total_calibration": tcal,
        "calibration": {"brier": cal["brier"], "log_loss": cal["log_loss"]},
        "coach_adjustment": {
            "applied": coach_changes is not None,
            "margin_points": P.COACH_CHANGE_MARGIN,
            "teams_flagged": (sum(len(v) for v in coach_changes.values())
                              if coach_changes else 0),
            "note": ("first-year head coach, fitted on 2024-2025 residuals "
                     "(betas -1.63/-1.43, holdout-replicated both directions); "
                     "market prices it, so it improves projections not bets"),
        },
        "fcs_augmentation": fcs_info,
        "NOTE": ("score_resid_std here is OUT-OF-SAMPLE and is the value the "
                 "simulator should use. The in-sample figure from fit_ratings "
                 "is biased low because ratings were fitted to those games."),
    }
    cpath = os.path.join(DATA_DIR, "calibration.json")
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)
    print(f"\nWrote {out}\nWrote {cpath}")


if __name__ == "__main__":
    main()
