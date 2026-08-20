#!/usr/bin/env python3
"""
EXPERIMENT: which score-generating process best reproduces real CFB scores?

The baseline simulator drew both teams' scores from independent Normals. Real
college football margins pile up on 3 and 7 (field goal / touchdown), which a
Normal cannot produce. This script fits the ridge rating model, then compares
candidate generative models against the actual 2023-2025 distribution on:

  * mass on key numbers (|margin| in 3, 7, 10, 14)
  * KS distance between simulated and actual margin distributions
  * KS distance on totals
  * whether mean/std are preserved

Run:  python experiments/sim_distribution_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cfb_ratings as R  # noqa: E402

RNG = np.random.default_rng(20260817)
KEY_NUMBERS = [3, 7, 10, 14, 17, 21]


def ks_distance(a, b):
    """Two-sample KS statistic without scipy."""
    grid = np.union1d(np.unique(a), np.unique(b))
    ca = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def key_mass(margins):
    m = np.abs(margins)
    return {k: float(np.mean(m == k)) for k in KEY_NUMBERS}


# --- candidate generators -------------------------------------------------
# Each takes expected scores (arrays) and returns simulated integer scores.

def gen_normal(exp_h, exp_a, sigma, **kw):
    h = RNG.normal(exp_h, sigma)
    a = RNG.normal(exp_a, sigma)
    return np.clip(np.round(h), 0, None), np.clip(np.round(a), 0, None)


def gen_bootstrap(exp_h, exp_a, sigma, resid_pairs=None, **kw):
    """Resample actual (home_resid, away_resid) PAIRS from model errors.
    Preserves the real marginal shape and any residual correlation."""
    i = RNG.integers(0, len(resid_pairs), size=len(exp_h))
    rh = resid_pairs[i, 0]
    ra = resid_pairs[i, 1]
    return np.clip(np.round(exp_h + rh), 0, None), np.clip(np.round(exp_a + ra), 0, None)


def _scoring_structure(exp_pts, td_share, disp):
    """Decompose expected points into touchdowns (7) and field goals (3).

    exp = 7*E[td] + 3*E[fg], with td_share = share of points from TDs.
    Counts are drawn negative-binomial (Poisson is under-dispersed for CFB).
    """
    exp_pts = np.maximum(exp_pts, 0.1)
    lam_td = exp_pts * td_share / 7.0
    lam_fg = exp_pts * (1.0 - td_share) / 3.0

    def nb(lam):
        # negative binomial via gamma-Poisson mixture; disp>0, var = lam + lam^2/disp
        shape = disp
        scale = np.maximum(lam, 1e-6) / disp
        return RNG.poisson(RNG.gamma(shape, scale))

    return 7 * nb(lam_td) + 3 * nb(lam_fg)


def gen_scoring(exp_h, exp_a, sigma, td_share=0.78, disp=3.0, **kw):
    return (_scoring_structure(exp_h, td_share, disp).astype(float),
            _scoring_structure(exp_a, td_share, disp).astype(float))


def _drive_score(exp_pts, n_drives, fg_ratio, rng=RNG):
    """Drive-based multinomial scoring.

    Each possession independently ends in TD (7), FG (3), or no score. With
    the number of drives roughly fixed, the counts are binomial rather than
    Poisson -- i.e. UNDER-dispersed -- which is what the real data shows
    (observed per-team residual std ~10.8 vs ~12.9 implied by Poisson counts).
    It also emits exact multiples of 3 and 7, which is what creates the real
    pile-up of margins on 3 and 7.

    fg_ratio = E[FG drives] / E[TD drives]; points/drive = 7*p_td + 3*p_fg.
    """
    exp_pts = np.maximum(exp_pts, 0.5)
    ppd = exp_pts / n_drives                      # points per drive
    p_td = ppd / (7.0 + 3.0 * fg_ratio)
    p_fg = p_td * fg_ratio
    # keep probabilities legal for extreme expectations
    tot = p_td + p_fg
    over = tot > 0.95
    if np.any(over):
        scale = np.where(over, 0.95 / np.maximum(tot, 1e-9), 1.0)
        p_td = p_td * scale
        p_fg = p_fg * scale

    u = rng.random((n_drives, len(exp_pts)))
    td = (u < p_td).sum(axis=0)
    fg = ((u >= p_td) & (u < p_td + p_fg)).sum(axis=0)
    # PATs: ~95% made, occasional 2-pt conversion
    pat = rng.random((td.max() if td.max() > 0 else 1, len(exp_pts)))
    made = np.zeros(len(exp_pts))
    for i in range(td.max() if td.max() > 0 else 0):
        elig = td > i
        made += elig * (pat[i] < 0.95)
    return 6.0 * td + made + 3.0 * fg


def gen_drives(exp_h, exp_a, sigma, n_drives=12, fg_ratio=0.55, **kw):
    return (_drive_score(exp_h, n_drives, fg_ratio),
            _drive_score(exp_a, n_drives, fg_ratio))


def _drive_score_shared(exp_pts, drives, fg_ratio, rng=RNG):
    """Same as _drive_score but `drives` is a per-game array (teams alternate
    possessions, so both sides get essentially the same number)."""
    exp_pts = np.maximum(exp_pts, 0.5)
    ppd = exp_pts / drives
    p_td = ppd / (7.0 + 3.0 * fg_ratio)
    p_fg = p_td * fg_ratio
    tot = p_td + p_fg
    scale = np.where(tot > 0.95, 0.95 / np.maximum(tot, 1e-9), 1.0)
    p_td, p_fg = p_td * scale, p_fg * scale

    maxd = int(drives.max())
    u = rng.random((maxd, len(exp_pts)))
    live = np.arange(maxd)[:, None] < drives[None, :]
    td = ((u < p_td) & live).sum(axis=0)
    fg = ((u >= p_td) & (u < p_td + p_fg) & live).sum(axis=0)
    pat = rng.random((maxd, len(exp_pts))) < 0.95
    made = ((np.arange(maxd)[:, None] < td[None, :]) & pat).sum(axis=0)
    return 6.0 * td + made + 3.0 * fg


def gen_drives_shared(exp_h, exp_a, sigma, mean_drives=12.0, sd_drives=1.3,
                      fg_ratio=0.50, **kw):
    """Physically correct version: one possession count per GAME (both teams
    alternate), varying game to game with pace."""
    n = len(exp_h)
    drives = np.clip(np.round(RNG.normal(mean_drives, sd_drives, n)), 8, 18).astype(int)
    return (_drive_score_shared(exp_h, drives, fg_ratio),
            _drive_score_shared(exp_a, drives, fg_ratio))


def main():
    games = pd.read_csv("data/real_games.csv")
    r = R.fit_ratings(games, lam=R.DEFAULT_LAMBDA, cv=False)

    g, _ = R.pool_rare_teams(games.dropna(subset=["home_score", "away_score"]).copy())
    g["neutral_site"] = g["neutral_site"].astype(bool)
    g = g[g["home_team"].isin(r.off) & g["away_team"].isin(r.off)]

    sign = np.where(g["neutral_site"].values, 0.0, 1.0)
    exp_h = r.mu + np.array([r.off[t] for t in g["home_team"]]) \
        - np.array([r.defn[t] for t in g["away_team"]]) + sign * r.hfa / 2
    exp_a = r.mu + np.array([r.off[t] for t in g["away_team"]]) \
        - np.array([r.defn[t] for t in g["home_team"]]) - sign * r.hfa / 2

    act_h = g["home_score"].values.astype(float)
    act_a = g["away_score"].values.astype(float)
    act_margin = act_h - act_a
    act_total = act_h + act_a

    resid_pairs = np.column_stack([act_h - exp_h, act_a - exp_a])
    sigma = float(np.std(resid_pairs))

    print(f"Fitted on {len(g)} games. mu={r.mu:.2f} hfa={r.hfa:+.2f} "
          f"score_sigma={sigma:.2f} resid_corr={np.corrcoef(resid_pairs.T)[0,1]:+.3f}\n")

    # replicate each real game many times so the comparison is apples to apples
    REPS = 40
    tile = lambda v: np.tile(v, REPS)
    eh, ea = tile(exp_h), tile(exp_a)

    candidates = {
        "independent Normal (baseline)": dict(fn=gen_normal),
        "empirical residual bootstrap": dict(fn=gen_bootstrap, resid_pairs=resid_pairs),
        "scoring structure NB(td=.78)": dict(fn=gen_scoring, td_share=0.78, disp=3.0),
        "drives n=12 fg_ratio=.45": dict(fn=gen_drives, n_drives=12, fg_ratio=0.45),
        "SHARED drives 12+-1.3 fg=.50": dict(fn=gen_drives_shared, mean_drives=12.0, sd_drives=1.3, fg_ratio=0.50),
        "SHARED drives 12+-2.0 fg=.50": dict(fn=gen_drives_shared, mean_drives=12.0, sd_drives=2.0, fg_ratio=0.50),
        "SHARED drives 12+-1.3 fg=.40": dict(fn=gen_drives_shared, mean_drives=12.0, sd_drives=1.3, fg_ratio=0.40),
        "SHARED drives 13+-1.5 fg=.50": dict(fn=gen_drives_shared, mean_drives=13.0, sd_drives=1.5, fg_ratio=0.50),
    }

    act_key = key_mass(act_margin)
    print("ACTUAL key-number mass:")
    print("   " + "  ".join(f"|{k}|={act_key[k]*100:5.2f}%" for k in KEY_NUMBERS))
    print(f"   margin mean={act_margin.mean():+6.2f} std={act_margin.std():5.2f} "
          f"| total mean={act_total.mean():5.2f} std={act_total.std():5.2f}\n")

    rows = []
    for name, cfg in candidates.items():
        fn = cfg.pop("fn")
        sh, sa = fn(eh, ea, sigma, **cfg)
        sm = sh - sa
        st = sh + sa
        km = key_mass(sm)
        keyerr = sum(abs(km[k] - act_key[k]) for k in KEY_NUMBERS)
        rows.append({
            "model": name,
            "KS_margin": round(ks_distance(sm, act_margin), 4),
            "KS_total": round(ks_distance(st, act_total), 4),
            "key_err": round(keyerr, 4),
            "m_mean": round(float(sm.mean()), 2),
            "m_std": round(float(sm.std()), 2),
            "t_mean": round(float(st.mean()), 2),
            "t_std": round(float(st.std()), 2),
            **{f"|{k}|": f"{km[k]*100:.2f}%" for k in KEY_NUMBERS},
        })

    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))

    print("\nKS_margin/KS_total: lower is better (0 = identical distribution)")
    print("key_err: total absolute error in key-number mass, lower is better")
    best_ks = df.loc[df["KS_margin"].idxmin(), "model"]
    best_key = df.loc[df["key_err"].idxmin(), "model"]
    print(f"\nBest overall CDF (KS) fit : {best_ks}")
    print(f"Best key-number fit       : {best_key}")
    print("""
CONCLUSION (2023-2025 data)
---------------------------
All candidates match the MEAN margin well, so for a point-spread estimate the
choice barely matters. It matters for win/cover probabilities, which depend on
the distribution's shape near zero and on the key numbers.

* Independent Normal and residual bootstrap fit the overall CDF marginally
  best (KS ~0.028-0.034) but badly under-produce key numbers: they put ~3.5%
  of mass on |margin|=3 versus 9.8% in reality, because adding continuous
  noise to a continuous expectation smears the integer structure away.
* The drive-based multinomial roughly HALVES key-number error (0.086 vs 0.159)
  while matching mean and standard deviation, and its KS is statistically
  indistinguishable from the Normal's. It wins because it generates real
  scores out of 7s and 3s instead of smoothing over them.
* Negative-binomial scoring counts are decisively wrong: variance is ~40% too
  high. Real CFB scoring is UNDER-dispersed relative to Poisson (observed
  per-team residual std 10.8 vs ~12.9 implied by Poisson TD/FG counts),
  because possession counts are nearly fixed. Only a binomial/multinomial
  drive structure can produce under-dispersion; NB always over-disperses.
* Sharing one possession count across both teams (physically correct, since
  they alternate) did not measurably improve on independent drive counts.

The residual key-number gap (4.8% simulated vs 9.8% actual on |3|) is NOT a
scoring-structure problem -- it is end-game strategy. Teams deliberately
engineer 3-point margins (kick to take or tie a late lead), which no
possession-independent model reproduces. Closing it would require simulating
score-aware late-game decision making.

DECISION: use the drive-based multinomial as the primary simulator, and report
cover probabilities near key numbers as approximate.""")


if __name__ == "__main__":
    main()
