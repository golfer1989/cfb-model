#!/usr/bin/env python3
"""
EXPERIMENT: does pass-vs-pass / rush-vs-rush style matchup actually predict
anything beyond the additive offense-minus-defense model?

This is the question behind "high pass offense vs high pass defense". The
intuition is strong; the statistical reality needs checking before we let it
move a point spread.

METHOD (leak-free by construction)
  * Team style/efficiency stats come from season S-1. Using season S stats to
    predict season S games would leak the outcome into the predictor.
  * The baseline margin prediction comes from a walk-forward ridge fit using
    only games played BEFORE the game being predicted.
  * We then ask: can style-interaction features explain any of the baseline's
    residual? If the interaction is real, it shows up here.

Tested features (all z-scored, prior season):
  pass_mismatch = home pass-offense strength  x  away pass-defense weakness
  rush_mismatch = home rush-offense strength  x  away rush-defense weakness
  (and the mirrored versions for the away team)
  pace_sum      = combined tempo, for totals

Run: python experiments/matchup_interaction_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cfb_ratings as R  # noqa: E402


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=0)


def build_style(ts):
    """Per (season, team_id) style z-scores, computed within season."""
    out = []
    for season, grp in ts.groupby("season"):
        g = grp.copy()
        # offense: higher = better/more pass-oriented
        g["z_off_pass_eff"] = zscore(g["off_yardsPerPassAttempt"])
        g["z_off_rush_eff"] = zscore(g["off_yardsPerRushAttempt"])
        g["z_off_pass_rate"] = zscore(g["off_pass_rate"])
        g["z_off_ypp"] = zscore(g["off_ypp"])
        g["z_pace"] = zscore(g["off_plays_pg"])
        # defense: ESPN "def_" = what opponents did TO them, so INVERT so
        # higher = better defense
        g["z_def_pass_eff"] = -zscore(g["def_yardsPerPassAttempt"])
        g["z_def_rush_eff"] = -zscore(g["def_yardsPerRushAttempt"])
        g["z_def_ypp"] = -zscore(g["def_ypp"])
        g["z_def_pass_rate_faced"] = zscore(g["def_pass_rate"])
        out.append(g)
    cols = ["season", "team_id"] + [c for c in out[0].columns if c.startswith("z_")]
    return pd.concat(out)[cols]


def walk_forward_residuals(games, lam=R.DEFAULT_LAMBDA):
    """Predict each 2024 and 2025 game using only strictly-earlier games."""
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games["neutral_site"] = games["neutral_site"].astype(bool)
    key = games["season"] * 100 + games["week"]
    rows = []
    for season in (2024, 2025):
        for wk in sorted(games[games["season"] == season]["week"].unique()):
            k = season * 100 + wk
            train, test = games[key < k], games[key == k]
            if len(train) < 400 or len(test) == 0:
                continue
            r = R.fit_ratings(train, lam=lam, cv=False, current_season=season)
            for _, row in test.iterrows():
                h = R.resolve_team(r, row["home_team"])
                a = R.resolve_team(r, row["away_team"])
                if not (r.has(h) and r.has(a)):
                    continue
                s = 0.0 if row["neutral_site"] else 1.0
                pm = (r.off[h] - r.defn[a]) - (r.off[a] - r.defn[h]) + s * r.hfa
                pt = 2 * r.mu + (r.off[h] - r.defn[a]) + (r.off[a] - r.defn[h])
                rows.append({
                    "season": season, "week": wk,
                    "home_id": str(row["home_id"]), "away_id": str(row["away_id"]),
                    "pred_margin": pm, "act_margin": row["home_score"] - row["away_score"],
                    "pred_total": pt, "act_total": row["home_score"] + row["away_score"],
                })
    return pd.DataFrame(rows)


def ols(X, y):
    """Least squares with intercept; returns (beta, r2, se)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    n, k = X1.shape
    sigma2 = ss_res / max(n - k, 1)
    try:
        cov = sigma2 * np.linalg.inv(X1.T @ X1)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return beta, r2, se


def main():
    games = pd.read_csv("data/real_games.csv")
    ts = pd.read_csv("data/team_stats.csv")
    ts["team_id"] = ts["team_id"].astype(str)
    style = build_style(ts)

    print("Building walk-forward baseline predictions (this takes a moment)...")
    res = walk_forward_residuals(games)
    print(f"  {len(res)} out-of-sample games\n")

    # attach PRIOR-season style for both teams (no leakage)
    style_prev = style.copy()
    style_prev["season"] = style_prev["season"] + 1  # season S-1 stats -> season S games

    h = style_prev.add_prefix("h_").rename(columns={"h_season": "season", "h_team_id": "home_id"})
    a = style_prev.add_prefix("a_").rename(columns={"a_season": "season", "a_team_id": "away_id"})
    df = res.merge(h, on=["season", "home_id"], how="inner").merge(a, on=["season", "away_id"], how="inner")
    print(f"After joining prior-season style for both teams: {len(df)} games\n")

    df["resid_margin"] = df["act_margin"] - df["pred_margin"]
    df["resid_total"] = df["act_total"] - df["pred_total"]

    # ---- interaction features -------------------------------------------
    # "home pass offense vs away pass defense": product of home pass-offense
    # z and away pass-defense WEAKNESS (negated strength)
    df["pass_mismatch_h"] = df["h_z_off_pass_eff"] * (-df["a_z_def_pass_eff"])
    df["pass_mismatch_a"] = df["a_z_off_pass_eff"] * (-df["h_z_def_pass_eff"])
    df["rush_mismatch_h"] = df["h_z_off_rush_eff"] * (-df["a_z_def_rush_eff"])
    df["rush_mismatch_a"] = df["a_z_off_rush_eff"] * (-df["h_z_def_rush_eff"])
    df["net_pass_mismatch"] = df["pass_mismatch_h"] - df["pass_mismatch_a"]
    df["net_rush_mismatch"] = df["rush_mismatch_h"] - df["rush_mismatch_a"]
    # style opposition: does a pass-heavy team meet a pass-funnel defense?
    df["style_align_h"] = df["h_z_off_pass_rate"] * (-df["a_z_def_pass_eff"])
    df["style_align_a"] = df["a_z_off_pass_rate"] * (-df["h_z_def_pass_eff"])
    df["net_style_align"] = df["style_align_h"] - df["style_align_a"]
    df["pace_sum"] = df["h_z_pace"] + df["a_z_pace"]

    print("=" * 78)
    print("TEST 1: do style interactions explain the MARGIN residual?")
    print("=" * 78)
    print(f"baseline residual std = {df['resid_margin'].std():.3f} pts\n")

    single = ["net_pass_mismatch", "net_rush_mismatch", "net_style_align"]
    print(f"{'feature':>22} {'beta (pts)':>11} {'std err':>9} {'t':>7} {'R^2':>9}")
    print("-" * 62)
    for f in single:
        X = df[[f]].values
        y = df["resid_margin"].values
        beta, r2, se = ols(X, y)
        t = beta[1] / se[1] if se[1] and not np.isnan(se[1]) else float("nan")
        print(f"{f:>22} {beta[1]:>11.4f} {se[1]:>9.4f} {t:>7.2f} {r2:>9.5f}")

    Xall = df[single].values
    beta, r2, se = ols(Xall, df["resid_margin"].values)
    print(f"\n  all three jointly: R^2 = {r2:.5f}  "
          f"(residual std {df['resid_margin'].std():.3f} -> "
          f"{df['resid_margin'].std()*np.sqrt(max(1-r2,0)):.3f})")

    print("\n" + "=" * 78)
    print("TEST 2: does combined PACE explain the TOTAL residual?")
    print("=" * 78)
    print(f"baseline total residual std = {df['resid_total'].std():.3f} pts\n")
    for f in ["pace_sum", "h_z_pace", "a_z_pace"]:
        beta, r2, se = ols(df[[f]].values, df["resid_total"].values)
        t = beta[1] / se[1] if se[1] and not np.isnan(se[1]) else float("nan")
        print(f"{f:>22} {beta[1]:>11.4f} {se[1]:>9.4f} {t:>7.2f} {r2:>9.5f}")

    print("\n" + "=" * 78)
    print("TEST 3: reference -- do PLAIN (non-interaction) efficiency edges help?")
    print("=" * 78)
    df["net_ypp_edge"] = (df["h_z_off_ypp"] + df["h_z_def_ypp"]) - (df["a_z_off_ypp"] + df["a_z_def_ypp"])
    beta, r2, se = ols(df[["net_ypp_edge"]].values, df["resid_margin"].values)
    t = beta[1] / se[1]
    print(f"{'net_ypp_edge':>22} {beta[1]:>11.4f} {se[1]:>9.4f} {t:>7.2f} {r2:>9.5f}")
    print("\n(This is the honest yardstick: if a simple prior-season efficiency")
    print(" edge beats the fancy interaction terms, the interaction is noise.)")

    print("""
INTERPRETATION GUIDE
  |t| < 2      -> indistinguishable from noise; must NOT move the spread
  R^2 < 0.005  -> explains <0.5% of residual variance; worthless for the mean
  A feature can still be worth SHOWING as a diagnostic even if it fails here.
""")


if __name__ == "__main__":
    main()
