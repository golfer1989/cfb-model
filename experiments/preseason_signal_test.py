#!/usr/bin/env python3
"""
EXPERIMENT: do returning production and recruiting fix the model's real
weakness -- early-season games?

THE HYPOTHESIS WORTH TESTING

Nearly everything tested so far failed because it was another proxy for team
strength, which the ratings already measure directly from scores. Returning
production is different for one specific reason: in week 1 the model has NO
current-season data and is extrapolating entirely from last year's team. It
cannot know that a team lost its quarterback and four starting linemen, or
that another returns everyone. That is a real, identifiable blind spot, and
returning production is exactly the information that fills it.

Recruiting is the same shape of argument, one level weaker: talent arriving
is a leading indicator the score-based ratings have not seen yet.

LEAKAGE POSITION
Returning production and recruiting rankings for season S are published BEFORE
season S starts. Using them to predict season S games is legitimate -- it is
information that genuinely existed at kickoff. The check is that we join
season S features to season S games, never season S+1 features backwards.

Run: python experiments/preseason_signal_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_DIR  # noqa: E402


def ols(cols, y, names):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float((r ** 2).sum() / dof)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    rows = []
    for i, nm in enumerate(["intercept"] + names):
        t = beta[i] / se[i] if se[i] and not np.isnan(se[i]) else np.nan
        rows.append((nm, beta[i], se[i], t))
    return rows, float(np.sqrt(np.mean(r ** 2)))


def norm_team(s):
    """CFBD and ESPN spell some teams differently."""
    return (str(s).strip().replace("State", "St").replace("&", "and")
            .replace(".", "").replace("'", "").lower())


def load_preseason():
    """Returning production + recruiting, keyed (season, normalized team)."""
    frames = []
    for yr in (2023, 2024, 2025, 2026):
        rp = os.path.join(DATA_DIR, f"cfbd_returning_{yr}.csv")
        rc = os.path.join(DATA_DIR, f"cfbd_recruiting_{yr}.csv")
        if not os.path.exists(rp):
            continue
        r = pd.read_csv(rp)
        r["season"] = yr
        keep = ["season", "team", "percentPPA", "percentPassingPPA",
                "percentRushingPPA", "usage", "passingUsage", "rushingUsage"]
        r = r[[c for c in keep if c in r.columns]]
        if os.path.exists(rc):
            c = pd.read_csv(rc)[["team", "rank", "points"]]
            c = c.rename(columns={"rank": "recruit_rank", "points": "recruit_points"})
            r = r.merge(c, on="team", how="left")
        frames.append(r)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["team"].map(norm_team)
    return df


def main():
    preds = pd.read_csv(os.path.join(DATA_DIR, "backtest_preds.csv"))
    pre = load_preseason()
    if pre is None:
        print("No CFBD preseason files found. Run fetch_cfbd.py first.")
        return 1

    preds["resid"] = preds["act_margin"] - preds["pred_margin"]
    preds["hkey"] = preds["home_team"].map(norm_team)
    preds["akey"] = preds["away_team"].map(norm_team)

    h = pre.add_prefix("h_").rename(columns={"h_season": "season", "h_key": "hkey"})
    a = pre.add_prefix("a_").rename(columns={"a_season": "season", "a_key": "akey"})
    df = preds.merge(h, on=["season", "hkey"], how="left").merge(
        a, on=["season", "akey"], how="left")

    matched = df["h_percentPPA"].notna() & df["a_percentPPA"].notna()
    print(f"out-of-sample games: {len(df)}")
    print(f"  with preseason data for BOTH teams: {int(matched.sum())} "
          f"({matched.mean()*100:.0f}%)")
    d = df[matched].copy()
    if len(d) < 200:
        print("  too few matched games to test")
        return 1

    # differentials -- what matters is home MINUS away, same as the ratings
    d["ret_ppa_diff"] = d["h_percentPPA"] - d["a_percentPPA"]
    d["ret_pass_diff"] = d["h_percentPassingPPA"] - d["a_percentPassingPPA"]
    d["ret_usage_diff"] = d["h_usage"] - d["a_usage"]
    if "h_recruit_points" in d.columns:
        d["recruit_diff"] = (d["h_recruit_points"].fillna(0)
                             - d["a_recruit_points"].fillna(0))

    print(f"\nbaseline residual sd: {d['resid'].std():.3f}\n")
    print("=" * 74)
    print("1. FULL SEASON -- does preseason info explain residual at all?")
    print("=" * 74)
    print(f"{'feature':>20} {'beta':>10} {'se':>8} {'t':>7}")
    feats = [c for c in ("ret_ppa_diff", "ret_pass_diff", "ret_usage_diff",
                         "recruit_diff") if c in d.columns]
    for f in feats:
        sub = d.dropna(subset=[f])
        rows, _ = ols([sub[f]], sub["resid"].values, [f])
        _, b, se, t = rows[1]
        print(f"{f:>20} {b:>10.4f} {se:>8.4f} {t:>7.2f}")

    print("\n" + "=" * 74)
    print("2. BY PART OF SEASON -- the hypothesis is that it helps EARLY")
    print("=" * 74)
    print("If returning production matters, it should matter most in weeks 1-4,")
    print("when the ratings have no current-season games to work from, and fade")
    print("as real results accumulate.\n")
    print(f"{'window':>12} {'n':>5} " + " ".join(f"{f.split('_')[1][:6]:>8}" for f in feats))
    for lbl, m in [("week 1-2", d["week"] <= 2), ("week 3-4", d["week"].between(3, 4)),
                   ("week 5-8", d["week"].between(5, 8)), ("week 9+", d["week"] >= 9)]:
        sub = d[m]
        if len(sub) < 60:
            continue
        ts = []
        for f in feats:
            s2 = sub.dropna(subset=[f])
            if len(s2) < 40:
                ts.append("  --")
                continue
            rows, _ = ols([s2[f]], s2["resid"].values, [f])
            ts.append(f"{rows[1][3]:>8.2f}")
        print(f"{lbl:>12} {len(sub):>5} " + " ".join(ts))
    print("\n(values are t-statistics; |t| > 2 is the bar)")

    print("\n" + "=" * 74)
    print("3. WALK-FORWARD VALUE -- would adding it reduce error?")
    print("=" * 74)
    early = d[d["week"] <= 4].dropna(subset=feats)
    if len(early) >= 150:
        base = float(np.sqrt(np.mean(early["resid"] ** 2)))
        rows, fitted = ols([early[f] for f in feats], early["resid"].values, feats)
        print(f"  early-season games: {len(early)}")
        print(f"  residual RMSE {base:.4f} -> {fitted:.4f} "
              f"({fitted - base:+.4f}) with all {len(feats)} features")
        print("\n  CAUTION: that figure is IN-SAMPLE on the residual and is the")
        print("  best case. A real gain must survive walk-forward refitting, and")
        print("  the measured detection floor on this dataset is ~0.040 RMSE.")
        if base - fitted < 0.04:
            print("  -> below the detection floor. Not usable.")
        else:
            print("  -> above the floor; worth a proper walk-forward test.")

    print("\n" + "=" * 74)
    print("4. CROSS-SEASON CONSISTENCY")
    print("=" * 74)
    print("A feature that only works in one season is the signature that has")
    print("already fooled this project twice.\n")
    for f in feats:
        out = []
        for s in sorted(d["season"].dropna().unique()):
            sub = d[(d["season"] == s)].dropna(subset=[f])
            if len(sub) < 80:
                continue
            rows, _ = ols([sub[f]], sub["resid"].values, [f])
            out.append(f"{int(s)}: t={rows[1][3]:+.2f}")
        print(f"  {f:>20}  " + "   ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
