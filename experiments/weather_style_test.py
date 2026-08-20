#!/usr/bin/env python3
"""
EXPERIMENT: does WEATHER hurt PASS-HEAVY teams more than run-heavy ones?

WHY THIS HYPOTHESIS IS DIFFERENT FROM THE ONES ALREADY REJECTED

This project rejected pass-offense-vs-pass-defense matchup (t = -0.39) and a
dozen similar style interactions. They failed because both sides of the
interaction were team-strength proxies, and the ratings already measure team
strength directly from scores. Multiplying two noisy proxies produced noise.

Weather is not a proxy for anything. Wind speed on a given Saturday is
exogenous -- it has no relationship to how good either team is. So an
interaction between wind and play-style is information the ratings structurally
cannot contain, and the physical mechanism is not in dispute: wind degrades the
passing game far more than the running game.

THE PREDICTIONS, STATED BEFORE LOOKING

  1. TOTALS: high wind suppresses scoring. Strongest, most likely to be real.
  2. TOTALS x STYLE: two pass-heavy teams in high wind should fall further
     below the projected total than two run-heavy teams in the same wind.
  3. MARGIN x STYLE: if one team passes far more than the other, wind should
     hurt the pass-heavy team's margin. Weakest of the three -- margins are
     differences, and both teams play in the same weather.

Leakage: pass rate comes from the PRIOR season, so it is known before kickoff.
Weather is measured at the actual kickoff hour and is likewise knowable from a
forecast beforehand.

Run: python experiments/weather_style_test.py
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
    out = []
    for i, nm in enumerate(["intercept"] + names):
        t = beta[i] / se[i] if se[i] and not np.isnan(se[i]) else np.nan
        out.append((nm, beta[i], se[i], t))
    return out, float(np.sqrt(np.mean(r ** 2)))


def show(rows, header=None):
    if header:
        print(f"\n  {header}")
    print(f"  {'term':>22} {'beta':>10} {'se':>8} {'t':>7}")
    for nm, b, se, t in rows:
        star = "  <-- significant" if abs(t) >= 2 else ""
        print(f"  {nm:>22} {b:>10.4f} {se:>8.4f} {t:>7.2f}{star}")


def main():
    wpath = os.path.join(DATA_DIR, "game_weather.csv")
    if not os.path.exists(wpath):
        print("No data/game_weather.csv yet. Run:  python fetch_venues.py")
        return 1

    preds = pd.read_csv(os.path.join(DATA_DIR, "backtest_preds.csv"))
    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    wx = pd.read_csv(wpath)
    ts = pd.read_csv(os.path.join(DATA_DIR, "team_stats.csv"))
    ts["team_id"] = ts["team_id"].astype(str)

    key = games[["event_id", "season", "week", "home_team", "away_team",
                 "home_id", "away_id"]]
    d = preds.merge(key, on=["season", "week", "home_team", "away_team"], how="left")
    d["event_id"] = d["event_id"].astype(str)
    wx["event_id"] = wx["event_id"].astype(str)
    d = d.merge(wx, on="event_id", how="inner")

    # PRIOR-season pass rate, so nothing from the game leaks into its predictor
    prev = ts[["season", "team_id", "off_pass_rate"]].copy()
    prev["season"] = prev["season"] + 1
    d["home_id"] = d["home_id"].astype(str)
    d["away_id"] = d["away_id"].astype(str)
    d = d.merge(prev.rename(columns={"team_id": "home_id",
                                     "off_pass_rate": "h_pass_rate"}),
                on=["season", "home_id"], how="left")
    d = d.merge(prev.rename(columns={"team_id": "away_id",
                                     "off_pass_rate": "a_pass_rate"}),
                on=["season", "away_id"], how="left")

    print(f"games with weather: {len(d)}")
    out = d[(d["indoor"] == False) & d["wind_mph"].notna()].copy()  # noqa: E712
    print(f"  outdoor with readings: {len(out)}")
    out = out.dropna(subset=["h_pass_rate", "a_pass_rate"])
    print(f"  also with prior-season pass rates: {len(out)}")
    if len(out) < 200:
        print("  too few to test")
        return 1

    out["margin_resid"] = out["act_margin"] - out["pred_margin"]
    out["total_resid"] = out["act_total"] - out["pred_total"]
    out["wind"] = out["wind_mph"].astype(float)
    out["rain"] = out["precip_in"].fillna(0).astype(float)
    out["cold"] = (out["temp_f"].astype(float) < 40).astype(float)
    # centre pass rates so the interaction is interpretable
    pr_mean = pd.concat([out["h_pass_rate"], out["a_pass_rate"]]).mean()
    out["h_pass_c"] = out["h_pass_rate"] - pr_mean
    out["a_pass_c"] = out["a_pass_rate"] - pr_mean
    out["pass_sum_c"] = out["h_pass_c"] + out["a_pass_c"]     # for totals
    out["pass_diff_c"] = out["h_pass_c"] - out["a_pass_c"]    # for margin

    print(f"\n  wind: mean {out['wind'].mean():.1f} mph, "
          f"p90 {out['wind'].quantile(0.9):.1f}, max {out['wind'].max():.1f}")
    print(f"  games with rain > 0.01in: {int((out['rain'] > 0.01).sum())}")
    print(f"  games below 40F: {int(out['cold'].sum())}")
    print(f"  prior-season pass rate: mean {pr_mean:.3f}, "
          f"range {out['h_pass_rate'].min():.2f}-{out['h_pass_rate'].max():.2f}")

    print("\n" + "=" * 74)
    print("PREDICTION 1: does wind suppress TOTAL scoring? (main effect)")
    print("=" * 74)
    rows, _ = ols([out["wind"], out["rain"], out["cold"]],
                  out["total_resid"].values, ["wind_mph", "rain_in", "cold<40F"])
    show(rows)
    print("\n  negative wind coefficient = fewer points than projected")

    print("\n" + "=" * 74)
    print("PREDICTION 2: WIND x PASS-HEAVINESS on the total")
    print("=" * 74)
    print("  Two pass-heavy teams in wind should fall further below the total")
    print("  than two run-heavy teams in the same wind.")
    out["wind_x_pass_sum"] = out["wind"] * out["pass_sum_c"]
    rows, _ = ols([out["wind"], out["pass_sum_c"], out["wind_x_pass_sum"]],
                  out["total_resid"].values,
                  ["wind_mph", "pass_rate_sum", "wind X pass_sum"])
    show(rows)
    print("\n  the INTERACTION term is the hypothesis. Negative and |t|>2 means")
    print("  wind hurts pass-heavy games more, which is the claim.")

    print("\n" + "=" * 74)
    print("PREDICTION 3: WIND x STYLE MISMATCH on the MARGIN  [the key test]")
    print("=" * 74)
    print("  Wind is not just a suppressant, it is a RELATIVE advantage. If one")
    print("  team runs the ball and the other throws it, a gale should tilt the")
    print("  game toward the running team -- its offence is far less degraded.")
    print("  So this should appear in the MARGIN, not only in the total.")
    print()
    print("  pass_diff = home pass rate MINUS away pass rate (prior season).")
    print("  A NEGATIVE interaction is the hypothesis: as wind rises, the more")
    print("  pass-heavy side loses ground.")
    out["wind_x_pass_diff"] = out["wind"] * out["pass_diff_c"]
    rows, _ = ols([out["wind"], out["pass_diff_c"], out["wind_x_pass_diff"]],
                  out["margin_resid"].values,
                  ["wind_mph", "pass_rate_diff", "wind X pass_diff"])
    show(rows)

    # A cleaner, assumption-free version of the same claim: take only games
    # where the two teams have genuinely different styles, split by wind, and
    # ask directly whether the run-heavier team beat its projection.
    print("\n  DIRECT VERSION -- no interaction term, just the comparison:")
    gap = out["pass_diff_c"].abs()
    mism = out[gap >= gap.quantile(0.60)].copy()
    if len(mism) >= 120:
        # margin residual from the RUN-HEAVIER team's point of view
        mism["run_team_resid"] = np.where(
            mism["pass_diff_c"] < 0,          # home is the run-heavier side
            mism["margin_resid"],
            -mism["margin_resid"])
        print(f"    {len(mism)} games with a clear style mismatch "
              f"(top 40% by pass-rate gap)")
        print(f"    {'wind band':>16} {'n':>5} {'run-heavy team resid':>22}")
        for lo, hi, lbl in [(0, 8, "calm (<8 mph)"), (8, 13, "8-13 mph"),
                            (13, 18, "13-18 mph"), (18, 99, "18+ mph")]:
            b = mism[(mism["wind"] >= lo) & (mism["wind"] < hi)]
            if len(b) < 30:
                print(f"    {lbl:>16} {len(b):>5}   (too few)")
                continue
            m = b["run_team_resid"].mean()
            se = b["run_team_resid"].std() / np.sqrt(len(b))
            print(f"    {lbl:>16} {len(b):>5} {m:>+15.2f} pts  (t {m/se:+.2f})")
        print("\n    If the claim holds, the run-heavy team's residual should")
        print("    climb as wind increases. Flat means wind does not shift the")
        print("    balance between styles.")

    print("\n" + "=" * 74)
    print("RAIN x PASS-HEAVINESS")
    print("=" * 74)
    out["rain_x_pass_sum"] = out["rain"] * out["pass_sum_c"]
    rows, _ = ols([out["rain"], out["pass_sum_c"], out["rain_x_pass_sum"]],
                  out["total_resid"].values,
                  ["rain_in", "pass_rate_sum", "rain X pass_sum"])
    show(rows)

    print("\n" + "=" * 74)
    print("HIGH-WIND SUBSET -- where the effect should be visible if anywhere")
    print("=" * 74)
    for thr in (12, 15, 18, 20):
        hi = out[out["wind"] >= thr]
        if len(hi) < 60:
            print(f"  wind >= {thr:>2} mph: only {len(hi)} games, skipping")
            continue
        lo = out[out["wind"] < thr]
        print(f"\n  wind >= {thr} mph (n={len(hi)}) vs below (n={len(lo)}):")
        print(f"    mean total residual: {hi['total_resid'].mean():+.2f} "
              f"vs {lo['total_resid'].mean():+.2f}")
        se = np.sqrt(hi["total_resid"].var() / len(hi) + lo["total_resid"].var() / len(lo))
        diff = hi["total_resid"].mean() - lo["total_resid"].mean()
        print(f"    difference {diff:+.2f} (t = {diff/se:+.2f})")
        # within high wind, does pass-heaviness matter?
        if len(hi) >= 80:
            rows, _ = ols([hi["pass_sum_c"]], hi["total_resid"].values, ["pass_sum"])
            _, b, s, t = rows[1]
            print(f"    within high wind, pass-heaviness: beta {b:+.2f} (t {t:+.2f})")

    print("\n" + "=" * 74)
    print("CROSS-SEASON CHECK on the headline interaction")
    print("=" * 74)
    for s in sorted(out["season"].dropna().unique()):
        sub = out[out["season"] == s]
        if len(sub) < 100:
            continue
        rows, _ = ols([sub["wind"], sub["pass_sum_c"], sub["wind_x_pass_sum"]],
                      sub["total_resid"].values, ["w", "p", "wXp"])
        print(f"  {int(s)}: n={len(sub):>4}  wind t={rows[1][3]:+.2f}   "
              f"interaction t={rows[3][3]:+.2f}")
    print("\n  An effect present in only one season is the signature that has")
    print("  already fooled this project twice. It must hold in both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
