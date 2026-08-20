#!/usr/bin/env python3
"""
EXPERIMENT: does the model need TEAM-SPECIFIC home advantage, or HEAD-TO-HEAD
history, on top of what it already does?

Two widely-believed effects, tested against out-of-sample residuals:

  1. TEAM-SPECIFIC HFA. The model fits ONE league-wide home-field coefficient
     (~3.3 pts). But some venues are supposed to be worth more -- altitude
     (Air Force, Wyoming, Utah State), hostile crowds, long travel for the
     visitor. If real, a per-team home advantage should explain residual.

  2. HEAD-TO-HEAD HISTORY. "Team A always beats Team B." Rivalry games are
     supposed to defy the ratings. If real, the margin of previous meetings
     should predict the current margin beyond what the ratings say.

Both are tested LEAK-FREE: the feature for a given game is built only from
games played strictly BEFORE it, and it is scored against the residual from
the date-ordered walk-forward.

Run: python experiments/hfa_and_h2h_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_DIR  # noqa: E402


def ols(X, y, names):
    cols = [np.asarray(c, dtype=float) for c in X]
    X1 = np.column_stack([np.ones(len(cols[0]))] + cols)
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    s2 = float((r ** 2).sum() / dof)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    except np.linalg.LinAlgError:
        se = np.full(X1.shape[1], np.nan)
    out = []
    for i, nm in enumerate(["intercept"] + names):
        out.append((nm, beta[i], se[i], beta[i] / se[i] if se[i] else np.nan))
    return out, float(np.sqrt(np.mean(r ** 2)))


def main():
    preds = pd.read_csv(os.path.join(DATA_DIR, "backtest_preds.csv"))
    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    games["_kick"] = pd.to_datetime(games["date"], errors="coerce")
    preds["_kick"] = pd.to_datetime(preds["date"], errors="coerce")
    preds["resid"] = preds["act_margin"] - preds["pred_margin"]

    print(f"Out-of-sample games available: {len(preds)}")
    print(f"Baseline residual sd: {preds['resid'].std():.3f}\n")

    # =====================================================================
    print("=" * 74)
    print("1. TEAM-SPECIFIC HOME FIELD ADVANTAGE")
    print("=" * 74)
    print("The model already fits ONE league-wide HFA. Question: do individual")
    print("home venues deviate from it in a way that repeats?\n")

    home_games = preds[~preds["neutral_site"].astype(bool)]
    print(f"non-neutral out-of-sample games: {len(home_games)}")

    # per-team mean residual when at HOME -- if a venue is worth more than the
    # league constant, its home residual should be persistently positive
    by_home = home_games.groupby("home_team")["resid"].agg(["mean", "count"])
    by_home = by_home[by_home["count"] >= 8].sort_values("mean", ascending=False)
    print(f"teams with >=8 home games out-of-sample: {len(by_home)}\n")
    print("  largest positive home residuals (model under-rates their home edge):")
    for t, r in by_home.head(5).iterrows():
        print(f"    {t:22s} {r['mean']:+6.2f} pts over {int(r['count'])} home games")
    print("  largest negative:")
    for t, r in by_home.tail(5).iterrows():
        print(f"    {t:22s} {r['mean']:+6.2f} pts over {int(r['count'])} home games")

    # THE TEST THAT MATTERS: is a team's home residual in one season
    # predictive of its home residual in the next? If not, the spread above
    # is noise being mistaken for venue strength.
    hh = home_games.copy()
    a = hh[hh["season"] == 2024].groupby("home_team")["resid"].agg(["mean", "count"])
    b = hh[hh["season"] == 2025].groupby("home_team")["resid"].agg(["mean", "count"])
    both = a.join(b, lsuffix="_24", rsuffix="_25", how="inner")
    both = both[(both["count_24"] >= 4) & (both["count_25"] >= 4)]
    if len(both) >= 25:
        r = float(np.corrcoef(both["mean_24"], both["mean_25"])[0, 1])
        print(f"\n  >> year-over-year correlation of a team's home residual:")
        print(f"     r = {r:+.4f} across {len(both)} teams")
        print(f"     {'PERSISTENT -- team-specific HFA is real' if abs(r) > 0.25 else 'NOT PERSISTENT -- the spread above is noise'}")

    # altitude check, the most-cited specific claim
    ALTITUDE = ["Air Force", "Wyoming", "Colorado State", "Utah State", "Colorado",
                "New Mexico", "BYU", "Utah", "Boise State", "Nevada"]
    alt = home_games[home_games["home_team"].isin(ALTITUDE)]
    rest = home_games[~home_games["home_team"].isin(ALTITUDE)]
    if len(alt) >= 30:
        diff = alt["resid"].mean() - rest["resid"].mean()
        se = np.sqrt(alt["resid"].var() / len(alt) + rest["resid"].var() / len(rest))
        print(f"\n  altitude/mountain-west venues: {alt['resid'].mean():+.2f} pts "
              f"vs {rest['resid'].mean():+.2f} elsewhere")
        print(f"    difference {diff:+.2f} (t = {diff/se:+.2f}) over {len(alt)} games")

    # =====================================================================
    print("\n" + "=" * 74)
    print("2. HEAD-TO-HEAD HISTORY (does the previous meeting predict this one?)")
    print("=" * 74)

    # build prior-meeting features using ONLY games kicked off earlier
    g = games.dropna(subset=["home_score", "away_score"]).copy()
    g["pair"] = g.apply(
        lambda r: tuple(sorted([str(r["home_team"]), str(r["away_team"])])), axis=1)
    g = g.sort_values("_kick")

    hist = {}
    prev_margin, prev_n, prev_days = [], [], []
    for _, r in g.iterrows():
        key = r["pair"]
        past = hist.get(key, [])
        if past:
            # margin from the perspective of THIS game's home team
            vals = [m if t == r["home_team"] else -m for (t, m, _) in past]
            prev_margin.append(float(np.mean(vals)))
            prev_n.append(len(past))
            prev_days.append((r["_kick"] - past[-1][2]).days)
        else:
            prev_margin.append(np.nan)
            prev_n.append(0)
            prev_days.append(np.nan)
        hist.setdefault(key, []).append(
            (r["home_team"], r["home_score"] - r["away_score"], r["_kick"]))
    g["prev_margin"] = prev_margin
    g["prev_n"] = prev_n
    g["days_since"] = prev_days

    m = preds.merge(
        g[["date", "home_team", "away_team", "prev_margin", "prev_n", "days_since"]],
        on=["date", "home_team", "away_team"], how="left")
    rem = m.dropna(subset=["prev_margin"])
    print(f"out-of-sample games with a prior meeting on record: {len(rem)} "
          f"of {len(m)} ({len(rem)/max(len(m),1)*100:.0f}%)")

    if len(rem) >= 100:
        res, _ = ols([rem["prev_margin"]], rem["resid"].values, ["prev_margin"])
        print(f"\n  residual ~ prior meeting margin:")
        for nm, b, se, t in res:
            print(f"    {nm:16s} {b:+9.4f}  se {se:.4f}  t = {t:+6.2f}")
        print(f"\n  |t| < 2 means the previous meeting tells you NOTHING the")
        print(f"  ratings did not already know.")

        # does it matter more for genuine rivals (frequent, recent meetings)?
        recent = rem[rem["days_since"] <= 400]
        if len(recent) >= 80:
            res2, _ = ols([recent["prev_margin"]], recent["resid"].values, ["prev_margin"])
            b, se, t = res2[1][1], res2[1][2], res2[1][3]
            print(f"\n  restricted to meetings within ~1 year (n={len(recent)}):")
            print(f"    prev_margin      {b:+9.4f}  se {se:.4f}  t = {t:+6.2f}")

        # conference rivals only
        if "prev_n" in rem.columns:
            rival = rem[rem["prev_n"] >= 2]
            if len(rival) >= 80:
                res3, _ = ols([rival["prev_margin"]], rival["resid"].values, ["prev_margin"])
                b, se, t = res3[1][1], res3[1][2], res3[1][3]
                print(f"\n  restricted to pairs met >=2 times before (n={len(rival)}):")
                print(f"    prev_margin      {b:+9.4f}  se {se:.4f}  t = {t:+6.2f}")

    print("\n" + "=" * 74)
    print("INTERPRETATION")
    print("=" * 74)
    print("""Both effects are heavily believed and both are easy to fool yourself
about, because a handful of teams will always show a large home residual and a
few rivalries will always look lopsided. The only question that matters is
whether the pattern REPEATS. That is what the year-over-year correlation and
the leak-free prior-meeting regression above measure.""")


if __name__ == "__main__":
    main()
