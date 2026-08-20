#!/usr/bin/env python3
"""
EXPERIMENT: use EVERY candidate signal, weighted by how much evidence supports
it, instead of a binary include/exclude at t = 2.

THE ARGUMENT FOR DOING THIS

"Even a 0.05% edge is worth having" is correct as stated. A hard significance
cutoff throws away a feature at t = 1.99 and keeps one at t = 2.01, which is
arbitrary -- nothing about the world changes across that line.

THE ARGUMENT FOR NOT SIMPLY ADDING THEM ALL

Measured on this project: seven efficiency features, each individually
harmless, made walk-forward RMSE 0.093 WORSE. A coefficient with t = 0.8 has a
confidence interval spanning both signs, so including it at face value is a
coin flip on whether the model improves or degrades -- and those coin flips
compound.

THE RESOLUTION

Shrink each coefficient toward zero in proportion to how unreliable it is,
then include everything. This is the James-Stein positive-part estimator:

    beta_used = beta_hat * max(0, 1 - 1/t^2)

    t = 0.8  ->  weight 0.00   (evidence too weak; contributes nothing)
    t = 1.5  ->  weight 0.56
    t = 2.0  ->  weight 0.75
    t = 3.0  ->  weight 0.89
    t = 5.0  ->  weight 0.96

A weak feature is not discarded by fiat -- it is admitted with the weight its
evidence justifies, which for very weak evidence is near zero. That is exactly
"factor it in as an advantage" done in a way that cannot blow up.

CRITICAL: the coefficients are re-fitted at every step using ONLY prior games,
so this is a genuine walk-forward test and not a curve fit.

Run: python experiments/shrinkage_ensemble_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_DIR  # noqa: E402


def norm(s):
    return (str(s).strip().replace("State", "St").replace("&", "and")
            .replace(".", "").replace("'", "").lower())


def fit_shrunk(X, y):
    """OLS, then James-Stein positive-part shrinkage on each coefficient."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    s2 = float((r ** 2).sum() / dof)
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    except np.linalg.LinAlgError:
        return np.zeros(X1.shape[1]), np.zeros(X1.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
        w = np.maximum(0.0, 1.0 - 1.0 / np.square(t))
    w[0] = 1.0                      # never shrink the intercept
    return beta * w, w


def build_features(preds, games):
    """Every candidate this project has tested, assembled in one frame.

    All are PRIOR-season quantities, so each is knowable before kickoff.
    """
    d = preds.copy()
    d["resid"] = d["act_margin"] - d["pred_margin"]
    d["hkey"] = d["home_team"].map(norm)
    d["akey"] = d["away_team"].map(norm)

    # prior-season point margin -- the benchmark everything else must beat
    pm = []
    for y in sorted(games["season"].unique()):
        g = games[games["season"] == y]
        m = {}
        for _, r in g.iterrows():
            m.setdefault(r["home_team"], []).append(r["home_score"] - r["away_score"])
            m.setdefault(r["away_team"], []).append(r["away_score"] - r["home_score"])
        for t_, v in m.items():
            pm.append({"season": y + 1, "key": norm(t_), "prior_margin": float(np.mean(v))})
    pm = pd.DataFrame(pm)
    d = d.merge(pm.rename(columns={"key": "hkey", "prior_margin": "h_pm"}),
                on=["season", "hkey"], how="left")
    d = d.merge(pm.rename(columns={"key": "akey", "prior_margin": "a_pm"}),
                on=["season", "akey"], how="left")
    d["f_prior_margin"] = d["h_pm"] - d["a_pm"]

    # CFBD opponent-adjusted advanced metrics
    adv = []
    for y in (2023, 2024, 2025):
        p = os.path.join(DATA_DIR, f"cfbd_advanced_{y}.csv")
        if os.path.exists(p):
            a = pd.read_csv(p)
            a["key"] = a["team"].map(norm)
            a["season"] = a["season"] + 1
            adv.append(a)
    if adv:
        adv = pd.concat(adv)
        d = d.merge(adv.add_prefix("h_").rename(
            columns={"h_season": "season", "h_key": "hkey"}), on=["season", "hkey"], how="left")
        d = d.merge(adv.add_prefix("a_").rename(
            columns={"a_season": "season", "a_key": "akey"}), on=["season", "akey"], how="left")
        for m in ("ppa", "successRate", "explosiveness", "pointsPerOpportunity",
                  "stuffRate", "lineYards"):
            ho, ao = f"h_off_{m}", f"a_off_{m}"
            hd, ad = f"h_def_{m}", f"a_def_{m}"
            if all(c in d.columns for c in (ho, ao, hd, ad)):
                d[f"f_{m}"] = (d[ho] - d[ad]) - (d[ao] - d[hd])

    # returning production + recruiting
    ret = []
    for y in (2023, 2024, 2025, 2026):
        p = os.path.join(DATA_DIR, f"cfbd_returning_{y}.csv")
        if os.path.exists(p):
            r_ = pd.read_csv(p)
            r_["season"] = y
            r_["key"] = r_["team"].map(norm)
            ret.append(r_[["season", "key", "percentPPA", "usage"]])
    if ret:
        ret = pd.concat(ret)
        d = d.merge(ret.rename(columns={"key": "hkey", "percentPPA": "h_ret",
                                        "usage": "h_use"}),
                    on=["season", "hkey"], how="left")
        d = d.merge(ret.rename(columns={"key": "akey", "percentPPA": "a_ret",
                                        "usage": "a_use"}),
                    on=["season", "akey"], how="left")
        d["f_returning"] = d["h_ret"] - d["a_ret"]
        d["f_usage"] = d["h_use"] - d["a_use"]

    # talent composite
    tal = []
    for y in (2023, 2024, 2025):
        p = os.path.join(DATA_DIR, f"cfbd_talent_{y}.csv")
        if os.path.exists(p):
            t_ = pd.read_csv(p)
            if len(t_):
                t_ = t_.rename(columns={"year": "season"})
                t_["key"] = t_["team"].map(norm)
                tal.append(t_[["season", "key", "talent"]])
    if tal:
        tal = pd.concat(tal).drop_duplicates(["season", "key"])
        d = d.merge(tal.rename(columns={"key": "hkey", "talent": "h_tal"}),
                    on=["season", "hkey"], how="left")
        d = d.merge(tal.rename(columns={"key": "akey", "talent": "a_tal"}),
                    on=["season", "akey"], how="left")
        d["f_talent"] = d["h_tal"] - d["a_tal"]

    # rest
    g2 = games.copy()
    g2["kick"] = pd.to_datetime(g2.get("kickoff_utc", g2["date"]), errors="coerce", utc=True)
    g2 = g2.sort_values("kick")
    last, rh, ra = {}, [], []
    for _, r in g2.iterrows():
        rh.append((r["kick"] - last[r["home_team"]]).days if r["home_team"] in last else np.nan)
        ra.append((r["kick"] - last[r["away_team"]]).days if r["away_team"] in last else np.nan)
        last[r["home_team"]] = r["kick"]
        last[r["away_team"]] = r["kick"]
    g2["rh"], g2["ra"] = rh, ra
    d = d.merge(g2[["event_id", "rh", "ra"]], on="event_id", how="left") \
        if "event_id" in d.columns else d
    if "rh" in d.columns:
        d["f_rest"] = d["rh"] - d["ra"]

    feats = [c for c in d.columns if c.startswith("f_")]
    return d, feats


def main():
    preds = pd.read_csv(os.path.join(DATA_DIR, "backtest_preds.csv"))
    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    key = games[["event_id", "season", "week", "home_team", "away_team"]]
    preds = preds.merge(key, on=["season", "week", "home_team", "away_team"], how="left")

    d, feats = build_features(preds, games)
    d["kick"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["resid", "kick"]).sort_values("kick")
    print(f"candidate features: {len(feats)}")
    print("  " + ", ".join(f.replace("f_", "") for f in feats))

    usable = d.dropna(subset=feats)
    print(f"\ngames with ALL features present: {len(usable)} of {len(d)}")
    if len(usable) < 400:
        # fall back to filling gaps with 0 (a neutral value for a difference)
        for f in feats:
            d[f] = d[f].fillna(0.0)
        usable = d
        print("  (missing values filled with 0 -- a neutral value for a difference)")

    # ---- WALK-FORWARD: refit coefficients on prior games only ------------
    usable = usable.sort_values("kick").reset_index(drop=True)
    split_dates = sorted(usable["kick"].unique())
    start = len(split_dates) // 3          # need history before predicting

    base_err, shrunk_err, raw_err = [], [], []
    weights_seen = []
    for i in range(start, len(split_dates)):
        day = split_dates[i]
        tr = usable[usable["kick"] < day]
        te = usable[usable["kick"] == day]
        if len(tr) < 300 or len(te) == 0:
            continue
        Xtr = tr[feats].values
        ytr = tr["resid"].values
        b_shrunk, w = fit_shrunk(Xtr, ytr)
        b_raw, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(ytr)), Xtr]), ytr, rcond=None)
        weights_seen.append(w[1:])

        Xte = np.column_stack([np.ones(len(te)), te[feats].values])
        adj_shrunk = Xte @ b_shrunk
        adj_raw = Xte @ b_raw
        y = te["resid"].values
        base_err.append(y)                    # model as-is
        shrunk_err.append(y - adj_shrunk)     # with shrunk adjustment
        raw_err.append(y - adj_raw)           # with unshrunk adjustment

    base = np.concatenate(base_err)
    shr = np.concatenate(shrunk_err)
    raw = np.concatenate(raw_err)
    n = len(base)

    print(f"\nwalk-forward evaluation on {n} games")
    print("=" * 66)
    print(f"  {'variant':>34} {'RMSE':>9} {'vs base':>9}")
    r0 = float(np.sqrt(np.mean(base ** 2)))
    r1 = float(np.sqrt(np.mean(shr ** 2)))
    r2 = float(np.sqrt(np.mean(raw ** 2)))
    print(f"  {'model as it ships (no features)':>34} {r0:>9.4f} {'--':>9}")
    print(f"  {'+ ALL features, James-Stein shrunk':>34} {r1:>9.4f} {r1-r0:>+9.4f}")
    print(f"  {'+ ALL features, unshrunk OLS':>34} {r2:>9.4f} {r2-r0:>+9.4f}")

    W = np.array(weights_seen)
    print(f"\n  average shrinkage weight actually applied, by feature:")
    for j, f in enumerate(feats):
        print(f"    {f.replace('f_',''):>22} {W[:, j].mean():>6.3f}")

    print("\n" + "=" * 66)
    if r1 < r0 - 0.005:
        print("  VERDICT: the shrunk ensemble helps. Worth shipping.")
    elif r1 < r0 + 0.005:
        print("  VERDICT: no measurable difference either way.")
        print("  Shrinkage did its job -- it drove the weak features to ~0, so the")
        print("  ensemble reproduces the base model instead of degrading it.")
        print("  That is the correct outcome when the features carry no signal,")
        print("  and it is exactly why unshrunk inclusion is dangerous: see the")
        print("  unshrunk row above.")
    else:
        print("  VERDICT: it makes things WORSE. Do not ship.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
