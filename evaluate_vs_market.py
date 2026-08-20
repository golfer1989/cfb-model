#!/usr/bin/env python3
"""
Evaluate the model against the actual betting market.

This is the only honest way to answer "is this model any good?". Beating a
naive baseline is easy; beating the closing line is the real test, and almost
nothing does it consistently.

WHAT IS MEASURED
  1. Accuracy of the model vs accuracy of the closing line (RMSE/MAE on margin
     and total). The market is the benchmark, not the model.
  2. Against-the-spread record: when the model disagrees with the closing
     line, who is right? This is the number that actually matters, and it is
     reported with a confidence interval because ATS records are extremely
     noisy -- a full season is only ~800 games.
  3. Whether the model adds information ON TOP of the line, via a regression
     of actual margin on both. If the model's coefficient is ~0 once the line
     is included, the model contributes nothing the market did not know.
  4. Closing line value: does the model agree with where the line MOVED
     (open -> close)? Anticipating line movement is a strong skill signal.

HONESTY NOTES
  * Breakeven ATS at standard -110 juice is 52.38%, not 50%. Anything between
    roughly 50% and 53% over one season is statistically indistinguishable
    from noise.
  * Games where the model and the line agree are excluded from the "disagree"
    ATS sample, because there is no bet to make.
  * Pushes are excluded from ATS percentages, not counted as wins.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from paths import DATA_DIR  # frozen-aware; see paths.py
BREAKEVEN_110 = 0.5238


def wilson_ci(wins, n, z=1.96):
    """Wilson score interval -- correct for proportions near the boundary and
    for the modest sample sizes an ATS record actually has."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (centre - half, centre + half)


def load_joined(preds_path, odds_path, games_path):
    preds = pd.read_csv(preds_path)
    odds = pd.read_csv(odds_path)
    games = pd.read_csv(games_path)

    key = games[["event_id", "season", "week", "home_team", "away_team"]]
    preds = preds.merge(key, on=["season", "week", "home_team", "away_team"], how="left")
    df = preds.merge(odds, on="event_id", how="inner")

    # market implied margin: spread is quoted from the home side, so a home
    # favourite carries a NEGATIVE number and implies a POSITIVE margin
    df["mkt_margin_close"] = -df["spread_close_home"]
    df["mkt_margin_open"] = -df["spread_open_home"]
    return df


def compare_accuracy(df):
    out = []
    for label, col, actual in [
        ("margin: model", "pred_margin", "act_margin"),
        ("margin: market close", "mkt_margin_close", "act_margin"),
        ("total : model", "pred_total", "act_total"),
        ("total : market close", "total_close", "act_total"),
    ]:
        sub = df.dropna(subset=[col, actual])
        if len(sub) == 0:
            continue
        err = sub[col] - sub[actual]
        out.append({
            "series": label,
            "n": len(sub),
            "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 3),
            "MAE": round(float(np.mean(np.abs(err))), 3),
            "bias": round(float(np.mean(err)), 3),
        })
    return pd.DataFrame(out)


def ats_record(df, min_edge=0.0):
    """How the model does when it disagrees with the closing line by at least
    `min_edge` points."""
    d = df.dropna(subset=["mkt_margin_close", "pred_margin", "act_margin"]).copy()
    d["edge"] = d["pred_margin"] - d["mkt_margin_close"]
    d = d[np.abs(d["edge"]) >= min_edge]
    if len(d) == 0:
        return None

    # model takes the home side when it thinks home beats the line
    d["model_home"] = d["edge"] > 0
    d["home_cover_margin"] = d["act_margin"] - d["mkt_margin_close"]
    d = d[d["home_cover_margin"] != 0]           # drop pushes
    d["win"] = np.where(d["model_home"], d["home_cover_margin"] > 0,
                        d["home_cover_margin"] < 0)

    n = len(d)
    w = int(d["win"].sum())
    lo, hi = wilson_ci(w, n)
    return {"min_edge": min_edge, "n": n, "wins": w, "losses": n - w,
            "pct": w / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi}


def incremental_information(df):
    """Regress actual margin on [market, model]. If the model's coefficient is
    indistinguishable from zero, it adds nothing the market did not already
    price in."""
    d = df.dropna(subset=["mkt_margin_close", "pred_margin", "act_margin"])
    if len(d) < 100:
        return None
    X = np.column_stack([np.ones(len(d)), d["mkt_margin_close"], d["pred_margin"]])
    y = d["act_margin"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float((resid ** 2).sum() / (len(d) - X.shape[1]))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {
        "n": len(d),
        "intercept": (beta[0], se[0], beta[0] / se[0]),
        "market": (beta[1], se[1], beta[1] / se[1]),
        "model": (beta[2], se[2], beta[2] / se[2]),
    }


def line_movement(df):
    """Does the model agree with the direction the market moved?

    DIAGNOSTIC ONLY. Nothing computed here is wired into the shrink factor or
    into any win probability, and it must stay that way.

    CLV is an UPPER BOUND on edge, not a substitute for it. Measured: the
    model anticipates the spread move by +0.22 to +0.27 points on average
    (t ~ +3.9), but with sd(cover margin) = 15.08 a point of REAL edge buys
    only 2.65 percentage points of cover probability, so breaking even at -110
    needs ~0.90 points. The measured CLV is roughly a quarter of that, and
    betting the opening spread on model edge actually goes 49.7%. Anticipating
    a move is not the same as being right about the game.

    The spread CLV is also a one-season result (2024 t = +1.1, 2025 t = +4.5),
    which is why it is logged per season -- if it does not repeat on fresh
    data it was a coin that landed heads.

    The totals CLV is the sturdier finding (beta ~ +0.094, t ~ +9, present in
    both seasons) and is still not a bet: the model's own total is a WORSE
    number than the opener it disagrees with (RMSE 16.6 vs 15.9). It points
    the right way while being wrapped in too much noise to sell.
    """
    d = df.dropna(subset=["mkt_margin_open", "mkt_margin_close", "pred_margin"]).copy()
    d["move"] = d["mkt_margin_close"] - d["mkt_margin_open"]
    d = d[np.abs(d["move"]) > 0.01]
    if len(d) < 30:
        return None
    d["model_vs_open"] = d["pred_margin"] - d["mkt_margin_open"]
    agree = float(np.mean(np.sign(d["move"]) == np.sign(d["model_vs_open"])))
    corr = float(np.corrcoef(d["move"], d["model_vs_open"])[0, 1])

    # signed CLV in points: how far the line moved TOWARD the model's side
    clv = (np.sign(d["model_vs_open"]) * d["move"]).values
    out = {"n": int(len(d)), "direction_agree": agree, "corr": corr,
           "clv_points": float(np.mean(clv)), "clv_t": _mean_t(clv),
           "clv_by_season": []}
    if "season" in d.columns:
        for s in sorted(d["season"].dropna().unique()):
            c = clv[(d["season"] == s).values]
            if len(c) >= 30:
                out["clv_by_season"].append(
                    {"season": int(s), "n": int(len(c)),
                     "clv_points": float(np.mean(c)), "clv_t": _mean_t(c)})

    # totals CLV: does the model's disagreement with the OPENING total predict
    # which way the total moves?
    t = df.dropna(subset=["total_open", "total_close", "pred_total"]).copy()
    if len(t) >= 200:
        reg = _ols(t["pred_total"].values - t["total_open"].values,
                   t["total_close"].values - t["total_open"].values)
        out["total_clv"] = {"n": int(len(t)), "beta": reg["beta"],
                            "se": reg["se"], "t": reg["t"]}
    return out


def _mean_t(x):
    """t-statistic of a mean against zero."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else float("nan")


def _ols(x, y):
    """Simple y ~ a + b*x with classical standard errors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float((resid ** 2).sum() / max(len(x) - 2, 1))
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return {"n": int(len(x)), "alpha": float(beta[0]), "beta": float(beta[1]),
            "se": float(se[1]),
            "t": float(beta[1] / se[1]) if se[1] else float("nan")}


def fit_total_edge_shrinkage(df, total_calibration=None):
    """The totals equivalent of fit_edge_shrinkage, fitted SEPARATELY.

    Margins and totals are different quantities with different error
    structures -- rating errors cancel in a margin and accumulate in a total --
    so there is no reason their edges should carry the same amount of signal,
    and measured they do not (post-leak-fix, orthogonalized: totals beta
    ~0.06 vs margins ~0.12 -- current values in calibration.json). Reusing
    the margin factor for over/under calls would be wrong in either
    direction.

    The model total must be put through the same calibration the report shows
    before its edge is measured, otherwise the fitted beta describes a number
    the user never sees.
    """
    d = df.dropna(subset=["total_close", "pred_total", "act_total"]).copy()
    if len(d) < 200:
        return None
    pred = d["pred_total"].values
    if total_calibration and total_calibration.get("slope") is not None:
        pred = total_calibration["slope"] * pred + total_calibration["intercept"]
    edge = pred - d["total_close"].values
    resid = d["act_total"].values - d["total_close"].values

    # CRITICAL: the calibrated total is shrunk so hard toward the league mean
    # (slope ~0.45) that the model's own contribution has sd ~3.8 against a
    # market-total sd of ~6.4. The resulting "edge" is therefore mostly just
    # the negated market number: it correlates +0.82 with pure market-fade and
    # NEGATIVELY (-0.29) with the model's own projection.
    #
    # Regressing the raw edge on the residual measures the wrong thing. Fading
    # extreme market totals with NO model input scores t = +3.73, higher than
    # the raw edge's +3.34; with both terms in, fade keeps t = +2.71 while the
    # model term collapses to t = +0.50. Shipping that as a model edge would
    # sell an unvalidated market-fade strategy under the model's name.
    #
    # So the edge is orthogonalized against market-fade first. What remains is
    # the model's marginal contribution, which is what a user is entitled to
    # think they are betting.
    fade = -(d["total_close"].values - d["total_close"].values.mean())
    F = np.column_stack([np.ones(len(d)), fade])
    edge_orth = edge - F @ np.linalg.lstsq(F, edge, rcond=None)[0]
    resid_orth = resid - F @ np.linalg.lstsq(F, resid, rcond=None)[0]

    X = np.column_stack([np.ones(len(d)), edge_orth])
    beta, *_ = np.linalg.lstsq(X, resid_orth, rcond=None)
    r = resid_orth - X @ beta
    sigma2 = float((r ** 2).sum() / (len(d) - 3))
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta[1] / se[1] if se[1] else float("nan")
    return {
        "n": int(len(d)),
        "alpha": float(beta[0]), "beta": float(beta[1]),
        "beta_se": float(se[1]), "beta_t": float(t),
        "resid_sd": float(np.sqrt(sigma2)),
        "significant": bool(abs(t) >= 2.0),
        "note": ("beta measures the model's contribution AFTER removing the "
                 "market-fade component the calibrated total is dominated by"),
    }


def fit_edge_shrinkage(df):
    """How much of the model's disagreement with the market is REAL?

    The simulator's own cover probability is calibrated against the model's
    view of the world, not against the market's. Those are different things.
    A 5-point disagreement implies a ~62% cover probability under the model's
    own margin distribution -- but if the model's disagreements carry no
    information, the true cover rate is 50% no matter how large the gap.

    Regressing the actual home cover margin on the model's edge measures the
    real content directly:

        (actual_margin - market_margin) = alpha + beta * (model_margin - market_margin)

    beta = 1 would mean the model's disagreement is entirely correct.
    beta = 0 means it is entirely noise and the honest cover probability is
    50% everywhere. The fitted beta is the factor by which every reported
    edge must be SHRUNK before it is turned into a win probability.

    This is the single most important calibration in the project: without it
    the report recommends bets on edges that do not exist.
    """
    d = df.dropna(subset=["mkt_margin_close", "pred_margin", "act_margin"]).copy()
    if len(d) < 200:
        return None
    d["edge"] = d["pred_margin"] - d["mkt_margin_close"]
    d["cover_margin"] = d["act_margin"] - d["mkt_margin_close"]

    X = np.column_stack([np.ones(len(d)), d["edge"]])
    y = d["cover_margin"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float((resid ** 2).sum() / (len(d) - 2))
    XtXi = np.linalg.inv(X.T @ X)
    cov = sigma2 * XtXi
    se = np.sqrt(np.diag(cov))
    t = beta[1] / se[1] if se[1] else float("nan")

    # THE VERDICT DEPENDS ON THE VARIANCE ESTIMATOR, so compute three and be
    # honest about it. Measured on the 2026-08 refit: classical t = 1.90,
    # HC1 heteroskedasticity-robust t = 2.07, clustered-by-(season, week)
    # t = 2.09 -- the pooled significance call flips exactly at the
    # threshold depending on which one you trust. Games in the same week
    # share one fitted rating vector, so clustering is the defensible choice
    # in principle; here it happens to TIGHTEN the interval. The
    # `significant` flag below therefore uses the WEAKEST of the three
    # (smallest |t|): a result that is only significant under a favourable
    # estimator is not a result this project acts on.
    hs = (X @ XtXi * X).sum(axis=1)
    u = resid / np.sqrt(np.clip(1.0 - hs, 1e-9, None))
    Vh = XtXi @ ((X * (u ** 2)[:, None]).T @ X) @ XtXi * (len(d) / (len(d) - 2))
    t_hc1 = float(beta[1] / np.sqrt(Vh[1, 1])) if Vh[1, 1] > 0 else float("nan")

    t_clu = float("nan")
    if "season" in d.columns and "week" in d.columns:
        clu = d["season"].astype(str) + "-" + d["week"].astype(str)
        groups = d.groupby(clu.values).indices
        G = len(groups)
        if G >= 10:
            meat = np.zeros((2, 2))
            for _, gi in groups.items():
                sc = X[gi].T @ resid[gi]
                meat += np.outer(sc, sc)
            c = (G / (G - 1)) * ((len(d) - 1) / (len(d) - 2))
            Vc = XtXi @ meat @ XtXi * c
            t_clu = float(beta[1] / np.sqrt(Vc[1, 1])) if Vc[1, 1] > 0 else float("nan")

    ts = [abs(x) for x in (t, t_hc1, t_clu) if np.isfinite(x)]
    t_min = min(ts) if ts else float("nan")

    return {
        "n": int(len(d)),
        "alpha": float(beta[0]), "alpha_se": float(se[0]),
        "beta": float(beta[1]), "beta_se": float(se[1]), "beta_t": float(t),
        "beta_t_hc1": t_hc1, "beta_t_clustered": t_clu,
        "beta_t_weakest": float(t_min) if np.isfinite(t_min) else float(t),
        "resid_sd": float(np.sqrt(sigma2)),
        "significant": bool(t_min >= 2.0) if np.isfinite(t_min) else bool(abs(t) >= 2.0),
        "interpretation": (
            "model edge carries real signal; shrink edges by beta"
            if (np.isfinite(t_min) and t_min >= 2.0) else
            "model edge is indistinguishable from noise; report ~50% cover"),
    }


# ---------------------------------------------------------------------------
# STANDING VALIDATION GATE
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS.
#
# The pooled edge shrinkage is currently beta = +0.115, t = +1.96 -- just under
# the significance bar, so the report is gated to NO BET. That number is not
# one steady effect. It decomposes as:
#
#     2024:  beta = +0.043, t = +0.55   (nothing)
#     2025:  beta = +0.206, t = +2.34   (the entire pooled result)
#
# One more season like 2025 pushes the pooled t past 2.0, `significant` flips
# to True, and the report starts recommending real bets off a factor that has
# never once been confirmed on data it was not fitted to.
#
# That failure mode was measured directly, not imagined. Ranking 139 candidate
# subsets on 2024 and evaluating the top 10 on 2025 gave a mean 2025 beta of
# +0.198, against +0.206 for 2025 taken as a whole -- selecting on the prior
# season carried literally ZERO information about what would work next. And of
# 48 subsets with 50+ bets in both seasons, only 6 cleared breakeven in both,
# where pure noise predicts about 12. Cross-season persistence in this data is
# WORSE than chance.
#
# So a t-statistic pooled across seasons is not evidence here, and the gate
# below refuses to let one alone open the bet path.
#
# WHAT IT REQUIRES. Fit on strictly prior seasons, confirm on the held-out
# latest season, and demand the same sign everywhere. All three, or the factor
# stays 0.
#
# THE POWER FLOOR, so this stops being rediscovered. sd(actual - close) = 15.08
# points, so a genuine 2-point edge is a 55.28% cover rate and needs ~705 bets
# to detect at 80% power; a 2-percentage-point ATS edge needs ~4,906; merely
# confirming you beat the 52.38% breakeven needs ~3,464. The whole dataset
# supplies 936 bets at |edge| >= 3. Any proposed filter with fewer than ~700
# qualifying games CANNOT be validated on this data no matter how good its
# backtest looks -- which is what makes subset hunting futile here in advance.

GATE_T = 2.0


def validation_gate(df, fitter, pooled):
    """Select on prior seasons, confirm on the held-out latest season.

    `fitter` takes a frame and returns a dict with beta / beta_t / n (the same
    shape fit_edge_shrinkage and fit_total_edge_shrinkage already return).

    Returns a dict whose "passed" key decides whether the shrink factor is
    allowed to be non-zero. Failing is the SAFE direction: it can only ever
    hold the factor at 0 and keep the report on NO BET.

    DELIBERATELY STRICT, and the tradeoff is stated rather than hidden. Both
    the prior-seasons fit AND the held-out season must independently clear
    |t| >= 2. That can produce a false NEGATIVE -- if 2026 replicates 2025,
    the prior fit on 2024+2025 is t = +1.96 and this gate still blocks on a
    1.96-vs-2.00 technicality. That is the intended asymmetry: the cost of a
    false negative is a season of not betting, the cost of a false positive is
    real money staked on an artifact. Requiring the signal to exist BEFORE the
    holdout is the whole point -- if only the latest season is significant,
    the edge was found in the holdout rather than confirmed by it, which is
    precisely the 2025-only pattern this gate exists to catch.
    """
    res = {"gate_t": GATE_T, "per_season": [], "passed": False, "reason": ""}
    if pooled is None:
        res["reason"] = "no pooled fit to validate"
        return res
    if "season" not in df.columns:
        res["reason"] = "no season column; cannot split -- factor held at 0"
        return res

    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    res["seasons"] = seasons
    res["pooled"] = {"n": pooled["n"], "beta": pooled["beta"], "t": pooled["beta_t"]}

    for s in seasons:
        r = fitter(df[df["season"] == s])
        if r:
            res["per_season"].append({"season": s, "n": r["n"],
                                      "beta": r["beta"], "t": r["beta_t"]})

    if len(res["per_season"]) < 2:
        res["reason"] = (f"only {len(res['per_season'])} season(s) with enough "
                         f"games; a one-season fit cannot be validated out of "
                         f"sample -- factor held at 0")
        return res

    # --- select on prior seasons, evaluate on the held-out latest ----------
    test_season = res["per_season"][-1]["season"]
    fit_part = df[df["season"] < test_season]
    test_part = df[df["season"] == test_season]
    fit_r, test_r = fitter(fit_part), fitter(test_part)
    if not fit_r or not test_r:
        res["reason"] = "holdout split too small to fit -- factor held at 0"
        return res

    res["fit_seasons"] = [s for s in seasons if s < test_season]
    res["test_season"] = test_season
    res["fit"] = {"n": fit_r["n"], "beta": fit_r["beta"], "t": fit_r["beta_t"]}
    res["test"] = {"n": test_r["n"], "beta": test_r["beta"], "t": test_r["beta_t"]}

    betas = [p["beta"] for p in res["per_season"]]
    res["sign_consistent"] = bool(all(b > 0 for b in betas) or all(b < 0 for b in betas))

    # Significance checks use the WEAKEST available t-statistic (classical /
    # HC1-robust / clustered-by-week, see fit_edge_shrinkage). The pooled
    # spread t currently sits at 1.90 / 2.07 / 2.09 across those three -- a
    # verdict that flips with the estimator is not a verdict, so the gate
    # demands the effect clear the bar under ALL of them.
    def _t(r):
        return abs(r.get("beta_t_weakest", r["beta_t"]))

    checks = {
        "pooled_significant": bool(_t(pooled) >= GATE_T),
        "prior_fit_significant": bool(_t(fit_r) >= GATE_T),
        "holdout_confirms": bool(_t(test_r) >= GATE_T
                                 and np.sign(test_r["beta"]) == np.sign(fit_r["beta"])),
        "sign_consistent": res["sign_consistent"],
    }
    res["checks"] = checks
    res["passed"] = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    res["reason"] = ("all checks passed" if res["passed"]
                     else "FAILED: " + ", ".join(failed))
    return res


def print_gate(gate, label):
    print("\n" + "=" * 72)
    print(f"VALIDATION GATE -- {label}")
    print("=" * 72)
    print(f"  every check is measured against |t| >= {gate.get('gate_t', GATE_T):.1f}\n")
    if gate.get("per_season"):
        print(f"  {'season':>8} {'n':>6} {'beta':>9} {'t':>7}")
        for p in gate["per_season"]:
            print(f"  {p['season']:>8} {p['n']:>6} {p['beta']:>+9.4f} {p['t']:>+7.2f}")
    if gate.get("pooled"):
        p = gate["pooled"]
        print(f"  {'POOLED':>8} {p['n']:>6} {p['beta']:>+9.4f} {p['t']:>+7.2f}")
    if "fit" in gate:
        print(f"\n  fit on {gate['fit_seasons']} -> beta {gate['fit']['beta']:+.4f} "
              f"(t {gate['fit']['t']:+.2f}, n {gate['fit']['n']})")
        print(f"  held-out {gate['test_season']}    -> beta {gate['test']['beta']:+.4f} "
              f"(t {gate['test']['t']:+.2f}, n {gate['test']['n']})")
    if gate.get("checks"):
        print()
        for k, v in gate["checks"].items():
            print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  -> {'PASS' if gate['passed'] else 'BLOCKED'}: {gate['reason']}")
    if not gate["passed"]:
        print("     shrink factor forced to 0.000 -- report stays on NO BET.")


def main():
    ap = argparse.ArgumentParser(description="Compare model against the betting market.")
    ap.add_argument("--preds", default=os.path.join(DATA_DIR, "backtest_preds.csv"))
    ap.add_argument("--odds", default=os.path.join(DATA_DIR, "odds.csv"))
    ap.add_argument("--games", default=os.path.join(DATA_DIR, "real_games.csv"))
    args = ap.parse_args()

    df = load_joined(args.preds, args.odds, args.games)
    print(f"Joined {len(df)} out-of-sample games that also have market lines")
    n_close = int(df["mkt_margin_close"].notna().sum())
    print(f"  with a closing spread: {n_close}")
    print(f"  with a closing total : {int(df['total_close'].notna().sum())}\n")

    print("=" * 72)
    print("ACCURACY: MODEL vs MARKET   (the market is the benchmark)")
    print("=" * 72)
    print(compare_accuracy(df).to_string(index=False))

    print("\n" + "=" * 72)
    print("AGAINST THE SPREAD, by how far the model disagrees with the close")
    print("=" * 72)
    print(f"  breakeven at -110 juice = {BREAKEVEN_110*100:.2f}%\n")
    print(f"  {'edge':>6} {'n':>6} {'W-L':>12} {'ATS%':>8} {'95% CI':>18}  verdict")
    for edge in (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        r = ats_record(df, min_edge=edge)
        if not r or r["n"] < 30:
            continue
        beats = r["ci_lo"] > BREAKEVEN_110
        losing = r["ci_hi"] < BREAKEVEN_110
        verdict = ("BEATS the market" if beats else
                   "loses to juice" if losing else "indistinguishable from noise")
        print(f"  {edge:>5.0f}+ {r['n']:>6} {str(r['wins'])+'-'+str(r['losses']):>12} "
              f"{r['pct']*100:>7.2f}% "
              f"{'['+format(r['ci_lo']*100,'.1f')+', '+format(r['ci_hi']*100,'.1f')+']':>18}  {verdict}")

    info = incremental_information(df)
    if info:
        print("\n" + "=" * 72)
        print("DOES THE MODEL ADD INFORMATION THE MARKET LACKS?")
        print("=" * 72)
        print("  actual_margin ~ a + b1*market_close + b2*model")
        print(f"  {'term':>12} {'coef':>9} {'std err':>9} {'t':>7}")
        for k in ("intercept", "market", "model"):
            b, se, t = info[k]
            print(f"  {k:>12} {b:>9.4f} {se:>9.4f} {t:>7.2f}")
        b2, se2, t2 = info["model"]
        print(f"\n  Interpretation: |t| < 2 on the model term means it contributes")
        print(f"  nothing beyond the closing line. Here |t| = {abs(t2):.2f} -> "
              f"{'ADDS information' if abs(t2) >= 2 else 'adds nothing detectable'}")

    lm = line_movement(df)
    if lm:
        print("\n" + "=" * 72)
        print("CLOSING LINE VALUE (does the model anticipate line movement?)")
        print("=" * 72)
        print("  DIAGNOSTIC ONLY -- not wired into any edge or win probability")
        print(f"  games with movement    : {lm['n']}")
        print(f"  direction agreement    : {lm['direction_agree']*100:.1f}%  (50% = no skill)")
        print(f"  correlation with move  : {lm['corr']:+.3f}")
        print(f"  mean signed CLV        : {lm['clv_points']:+.3f} pts  "
              f"(t = {lm['clv_t']:+.2f})")
        for b in lm.get("clv_by_season", []):
            print(f"      {b['season']}              : {b['clv_points']:+.3f} pts  "
                  f"(t = {b['clv_t']:+.2f}, n {b['n']})")
        if lm.get("total_clv"):
            tc = lm["total_clv"]
            print(f"  totals CLV beta        : {tc['beta']:+.4f}  "
                  f"(t = {tc['t']:+.2f}, n {tc['n']})")
        print("\n  Sizing check: sd(cover margin) = 15.08, so one point of REAL")
        print("  edge is worth 2.65 pct-pts of cover probability and breakeven")
        print("  at -110 needs 0.90 pts. CLV is an UPPER BOUND on edge, and the")
        print("  measured value is well under that bar -- betting the opening")
        print("  spread on model edge goes 49.7%. This is a monitoring number.")

    shrink = fit_edge_shrinkage(df)
    if shrink:
        print("\n" + "=" * 72)
        print("EDGE SHRINKAGE -- how much of the model's disagreement is real?")
        print("=" * 72)
        print("  cover_margin = alpha + beta * model_edge")
        print(f"    beta  = {shrink['beta']:+.4f}  (se {shrink['beta_se']:.4f}, "
              f"t = {shrink['beta_t']:+.2f})")
        print(f"    alpha = {shrink['alpha']:+.4f}  (se {shrink['alpha_se']:.4f})")
        print(f"    t under other variance estimators: "
              f"HC1-robust {shrink.get('beta_t_hc1', float('nan')):+.2f}, "
              f"clustered-by-week {shrink.get('beta_t_clustered', float('nan')):+.2f}")
        print(f"    significance is judged on the WEAKEST of the three "
              f"(|t| = {abs(shrink.get('beta_t_weakest', shrink['beta_t'])):.2f} "
              f"vs the 2.0 bar)")
        print(f"\n  {shrink['interpretation']}")
        print(f"\n  beta = 1.0 would mean every point of disagreement is real.")
        print(f"  beta = 0.0 means none of it is, and the honest cover probability")
        print(f"  is 50% regardless of how big the gap looks.")
        # The pooled t-statistic is NOT sufficient on its own -- see the long
        # note above validation_gate(). The factor may only be non-zero if the
        # edge also survives being fitted on prior seasons and confirmed on a
        # held-out one.
        gate = validation_gate(df, fit_edge_shrinkage, shrink)
        eff = max(shrink["beta"], 0.0) if (shrink["significant"] and gate["passed"]) else 0.0
        print_gate(gate, "SPREAD edge shrinkage")

        print(f"\n  -> shrink factor to apply in the report: {eff:.3f}")
        if shrink["significant"] and not gate["passed"]:
            print(f"     (pooled t = {shrink['beta_t']:+.2f} would have passed the old")
            print("      significance-only test; the holdout gate blocks it)")
        if eff < 0.15:
            print("     (a 10-point raw edge becomes "
                  f"{10*eff:.1f} points of real edge)")

        path = os.path.join(DATA_DIR, "calibration.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cal = json.load(f)
            cal["edge_shrinkage"] = {**shrink, "shrink_factor": eff,
                                     "gate_passed": gate["passed"],
                                     "gate_reason": gate["reason"]}
            cal["validation_gate"] = {"spread": gate}

            tshrink = fit_total_edge_shrinkage(df, cal.get("total_calibration"))
            if tshrink:
                tcal = cal.get("total_calibration")
                tgate = validation_gate(
                    df, lambda d: fit_total_edge_shrinkage(d, tcal), tshrink)
                teff = (max(tshrink["beta"], 0.0)
                        if (tshrink["significant"] and tgate["passed"]) else 0.0)
                cal["total_edge_shrinkage"] = {**tshrink, "shrink_factor": teff,
                                               "gate_passed": tgate["passed"],
                                               "gate_reason": tgate["reason"]}
                cal["validation_gate"]["total"] = tgate
                print("\n  TOTALS edge shrinkage (fitted separately):")
                print(f"    beta = {tshrink['beta']:+.4f} (se {tshrink['beta_se']:.4f}, "
                      f"t = {tshrink['beta_t']:+.2f}) -> factor {teff:.3f}")
                print(f"    margins and totals carry DIFFERENT amounts of signal;")
                print(f"    reusing the margin factor for O/U calls would be wrong.")
                print_gate(tgate, "TOTALS edge shrinkage")

            # Monitoring only. Deliberately NOT read by run_report.py, and
            # verify_math.py asserts it stays that way.
            if lm:
                cal["clv_diagnostic"] = {
                    **lm,
                    "NOTE": ("monitoring statistic only -- CLV is an UPPER BOUND "
                             "on edge, not a substitute for it (measured +0.27 pts "
                             "against the 0.90 pts needed to break even at -110). "
                             "Deliberately NOT consumed by run_report.py. If the "
                             "2025 spread CLV repeats on fresh 2026 data it becomes "
                             "a two-season result worth revisiting; on one season "
                             "it is a coin that landed heads."),
                }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(cal, f, indent=2)
            print(f"\n  Written to {path} -- run_report.py will apply it.")


if __name__ == "__main__":
    main()
