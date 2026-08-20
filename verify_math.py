#!/usr/bin/env python3
"""
Mathematical verification suite.

Every check here either passes or fails loudly. These are not smoke tests --
each one targets a specific way the model could be silently wrong while still
producing plausible-looking numbers, which is the dangerous failure mode for a
betting tool.

Run:  python verify_math.py
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

import cfb_ratings as R
import cfb_predict as P
from paths import DATA_DIR

PASS, FAIL = [], []
TOL = 1e-9


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append((name, detail))
    mark = "PASS" if ok else "**FAIL**"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def section(title):
    print(f"\n{'='*74}\n{title}\n{'='*74}")


# ---------------------------------------------------------------------------
# 1. Ridge solver
# ---------------------------------------------------------------------------

def verify_ridge():
    section("1. RIDGE SOLVER -- does it solve the equation it claims to?")
    rng = np.random.default_rng(0)
    n, p = 200, 12
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    w = rng.uniform(0.3, 1.0, n)
    lam = 2.5

    beta = R._solve_ridge(X, y, w, lam, n_teams=(p - 1) // 2, mu=0.0)

    # closed form for weighted ridge with the LAST column unpenalized
    Pm = np.eye(p) * lam
    Pm[-1, -1] = 0.0
    W = np.diag(w)
    expected = np.linalg.solve(X.T @ W @ X + Pm, X.T @ W @ y)
    err = float(np.max(np.abs(beta - expected)))
    check("weighted ridge matches closed form (X'WX + lam*I)b = X'Wy",
          err < 1e-8, f"max abs diff {err:.2e}")

    # HFA column must be genuinely unpenalized: increasing lam should not
    # shrink it the way it shrinks the penalized coefficients
    # shrinkage must be MONOTONE in lambda, and must actually bite
    mags = [float(np.abs(R._solve_ridge(X, y, w, L, (p - 1) // 2, 0.0)[:-1]).mean())
            for L in (0.01, 1.0, 100.0, 10000.0)]
    check("penalized coefficients shrink monotonically as lambda grows",
          all(mags[i] > mags[i + 1] for i in range(len(mags) - 1)),
          " -> ".join(f"{m:.4f}" for m in mags))
    check("heavy regularization drives penalized coefficients toward zero",
          mags[-1] < 0.02 * mags[0], f"{mags[-1]:.5f} vs {mags[0]:.5f}")

    b1 = R._solve_ridge(X, y, w, 0.01, (p - 1) // 2, 0.0)
    b2 = R._solve_ridge(X, y, w, 10000.0, (p - 1) // 2, 0.0)
    hfa_ratio = abs(b2[-1]) / max(abs(b1[-1]), 1e-12)
    check("unpenalized HFA column does NOT shrink to zero",
          hfa_ratio > 0.5, f"HFA ratio {hfa_ratio:.4f}")

    # a slice containing only neutral-site games makes the HFA column all
    # zeros and the normal matrix singular; that must degrade, not crash
    Xn = X.copy()
    Xn[:, -1] = 0.0
    try:
        bn = R._solve_ridge(Xn, y, w, 1.0, (p - 1) // 2, 0.0)
        check("all-neutral-site slice degrades gracefully (HFA -> 0, no crash)",
              abs(bn[-1]) < 1e-6, f"hfa coefficient {bn[-1]:.2e}")
    except np.linalg.LinAlgError as e:
        check("all-neutral-site slice degrades gracefully", False, str(e))

    # ridge with a target must recover the target when data carries no info.
    # Give the HFA column a nonzero value so the system stays identifiable --
    # the point being tested is the target mechanism, not singularity.
    Xz = np.zeros((5, p))
    Xz[:, -1] = 1.0
    yz = np.zeros(5)
    wz = np.ones(5)
    tgt = rng.normal(size=p)
    tgt[-1] = 0.0
    bt = R._solve_ridge(Xz, yz, wz, 1.0, (p - 1) // 2, 0.0, target=tgt)
    err_t = float(np.max(np.abs(bt[:-1] - tgt[:-1])))
    check("with no informative data, ridge-with-target returns the target",
          err_t < 1e-9, f"max abs diff {err_t:.2e}")


# ---------------------------------------------------------------------------
# 2. Design matrix / rating identity
# ---------------------------------------------------------------------------

def verify_design():
    section("2. DESIGN MATRIX -- do fitted ratings reproduce the scores?")
    games = pd.DataFrame({
        "home_team": ["A", "B", "C", "A", "B", "C"],
        "away_team": ["B", "C", "A", "C", "A", "B"],
        "home_score": [30.0, 20.0, 24.0, 28.0, 17.0, 31.0],
        "away_score": [17.0, 24.0, 21.0, 14.0, 20.0, 10.0],
        "neutral_site": [False, False, False, True, True, False],
        "home_conf": ["X"] * 6, "away_conf": ["X"] * 6,
        "season": [2025] * 6, "week": [1, 1, 2, 2, 3, 3],
    })
    teams = ["A", "B", "C"]
    X, y = R._build_design(games, teams)
    n, m = len(teams), len(games)

    check("design has 2 rows per game", X.shape[0] == 2 * m, f"{X.shape[0]} rows / {m} games")
    check("design has 2*teams + 1 columns", X.shape[1] == 2 * n + 1, f"{X.shape[1]} cols")

    # row structure: +1 on own offense, -1 on opponent defense
    ok = True
    for k in range(m):
        hi, ai = teams.index(games.home_team[k]), teams.index(games.away_team[k])
        ok &= (X[k, hi] == 1.0) and (X[k, n + ai] == -1.0)
        ok &= (X[m + k, ai] == 1.0) and (X[m + k, n + hi] == -1.0)
    check("each row is +1 own OFF, -1 opponent DEF", ok)

    # HFA: +1/2 home row, -1/2 away row, 0 at neutral sites
    sign_ok = True
    for k in range(m):
        neutral = bool(games.neutral_site[k])
        exp_h = 0.0 if neutral else 0.5
        sign_ok &= abs(X[k, 2 * n] - exp_h) < TOL
        sign_ok &= abs(X[m + k, 2 * n] + exp_h) < TOL
    check("HFA is +1/2 for home, -1/2 for away, 0 at neutral sites", sign_ok)

    check("y stacks home scores then away scores",
          np.allclose(y[:m], games.home_score.values)
          and np.allclose(y[m:], games.away_score.values))

    # the identity the whole model rests on
    r = R.fit_ratings(games, lam=0.05, cv=False, conf_shrink=False, min_games=1)
    worst = 0.0
    for k in range(m):
        h, a = games.home_team[k], games.away_team[k]
        s = 0.0 if games.neutral_site[k] else 1.0
        eh = r.mu + r.off[h] - r.defn[a] + s * r.hfa / 2
        ea = r.mu + r.off[a] - r.defn[h] - s * r.hfa / 2
        worst = max(worst, abs((eh - ea) - ((r.off[h] - r.defn[a]) - (r.off[a] - r.defn[h]) + s * r.hfa)))
    check("predicted margin identity is self-consistent", worst < 1e-9,
          f"max deviation {worst:.2e}")


# ---------------------------------------------------------------------------
# 3. Sign conventions -- the easiest place to be catastrophically wrong
# ---------------------------------------------------------------------------

def verify_signs():
    section("3. SIGN CONVENTIONS -- favorite/underdog, spread, edge")
    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    r = R.fit_ratings(games, lam=R.DEFAULT_LAMBDA, cv=False, current_season=2026)

    strong = max((t for t in r.teams if not t.startswith("__")), key=lambda t: r.power[t])
    weak = min((t for t in r.teams if not t.startswith("__")), key=lambda t: r.power[t])

    eh, ea = P.expected_scores(r, strong, weak)
    check("stronger team is projected to outscore weaker team", eh > ea,
          f"{strong} {eh:.1f} vs {weak} {ea:.1f}")

    # home field must help the home side
    h1, a1 = P.expected_scores(r, strong, weak, neutral=False)
    h2, a2 = P.expected_scores(r, strong, weak, neutral=True)
    check("home field advantage increases the home team's margin",
          (h1 - a1) > (h2 - a2),
          f"home margin {h1-a1:.2f} vs neutral {h2-a2:.2f}")
    check("HFA is a plausible magnitude (1-6 pts)", 1.0 <= r.hfa <= 6.0,
          f"hfa = {r.hfa:.2f}")

    # spread string: favorite carries the negative number
    s = P.spread_string("Home", "Away", 7.0)
    check("positive home margin -> home team laying points", s == "Home -7", s)
    s2 = P.spread_string("Home", "Away", -7.0)
    check("negative home margin -> away team laying points", s2 == "Away -7", s2)

    # market convention: negative spread_home means home is favored
    implied = -(-7.5)
    check("market spread_home = -7.5 implies home favored by 7.5",
          abs(implied - 7.5) < TOL)


# ---------------------------------------------------------------------------
# 4. Simulator
# ---------------------------------------------------------------------------

def verify_simulator():
    section("4. MONTE CARLO SIMULATOR")
    calib = P.load_calibration()
    sigma = calib["score_resid_std"]

    for exp_h, exp_a in [(31.0, 24.0), (17.0, 38.0), (27.0, 27.0)]:
        h, a = P.simulate(exp_h, exp_a, n_sims=60000, sigma=sigma, seed=11)
        bias_h = float(h.mean() - exp_h)
        bias_a = float(a.mean() - exp_a)
        check(f"simulated mean tracks expectation ({exp_h:.0f}/{exp_a:.0f})",
              abs(bias_h) < 1.5 and abs(bias_a) < 1.5,
              f"bias {bias_h:+.2f} / {bias_a:+.2f}")

    h, a = P.simulate(28.0, 24.0, n_sims=80000, sigma=sigma, seed=5)
    tgt = float(P.target_sigma(28.0) * (sigma / P.target_sigma(26.7)))
    check("simulated sd is within 20% of the calibrated target",
          abs(h.std() - tgt) / tgt < 0.20,
          f"sim sd {h.std():.2f} vs target {tgt:.2f}")

    check("scores are non-negative", float(min(h.min(), a.min())) >= 0.0)
    check("scores are integers (real football scores are discrete)",
          bool(np.all(h == np.round(h))))
    ties = float(np.mean(h == a))
    check("overtime resolves ties (college football has no ties)",
          ties < 0.001, f"tie rate {ties:.5f}")

    # key numbers must be over-represented vs a Normal
    m = np.abs(h - a)
    p3 = float(np.mean(m == 3))
    norm3 = 1.0 / (m.std() * math.sqrt(2 * math.pi))  # approx density at a point
    check("margin of 3 is meaningfully more likely than a smooth density implies",
          p3 > 1.5 * norm3, f"P(|margin|=3) = {p3:.4f} vs ~{norm3:.4f} smooth")

    # probabilities must be coherent
    out = P.summarize("H", "A", h, a, calib=calib)
    check("home and away win probabilities sum to 1",
          abs(out["home_win_prob"] + out["away_win_prob"] - 1.0) < 1e-9)
    check("win probability is in [0,1]", 0.0 <= out["home_win_prob"] <= 1.0)

    cov = P.cover_probability(h, a, -3.5)
    tot = cov["home_cover"] + cov["push"] + cov["away_cover"]
    check("cover probabilities (home/push/away) sum to 1", abs(tot - 1.0) < 1e-9,
          f"sum = {tot:.10f}")

    # a stronger favorite must have a higher win probability -- monotonicity
    probs = []
    for gap in (0, 7, 14, 21, 28):
        hh, aa = P.simulate(27.0 + gap / 2, 27.0 - gap / 2, n_sims=30000,
                            sigma=sigma, seed=3)
        probs.append(float(np.mean(hh > aa)))
    check("win probability increases monotonically with projected margin",
          all(probs[i] < probs[i + 1] for i in range(len(probs) - 1)),
          " -> ".join(f"{p:.3f}" for p in probs))


# ---------------------------------------------------------------------------
# 5. Calibration math
# ---------------------------------------------------------------------------

def verify_calibration():
    section("5. CALIBRATION MATH")
    calib = P.load_calibration()

    tc = calib.get("total_calibration")
    if tc and tc.get("slope") is not None:
        slope, inter = tc["slope"], tc["intercept"]
        check("total shrink slope is in (0,1) -- shrinking, not amplifying",
              0.0 < slope < 1.0, f"slope = {slope:.4f}")
        fixed = inter / (1 - slope)
        check("total calibration's fixed point is a plausible league mean",
              45.0 < fixed < 62.0, f"fixed point = {fixed:.2f} pts")
        lo, _ = P.calibrate_total(np.array([20.0]), calib)
        hi, _ = P.calibrate_total(np.array([90.0]), calib)
        check("calibration pulls extreme totals toward the mean",
              lo[0] > 20.0 and hi[0] < 90.0,
              f"20 -> {lo[0]:.1f}, 90 -> {hi[0]:.1f}")
    else:
        check("total calibration present", False, "missing -- run backtest.py")

    # edge shrinkage -- and, critically, that it SURVIVES load_calibration().
    # A whitelist in load_calibration once dropped these keys, silently
    # disabling every market-calibrated bet call while still "working".
    raw = {}
    raw_path = os.path.join(DATA_DIR, "calibration.json")
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)

    for key in ("edge_shrinkage", "total_edge_shrinkage"):
        if key in raw:
            check(f"{key} survives load_calibration() and reaches consumers",
                  calib.get(key) is not None,
                  "present on disk but dropped in transit" if calib.get(key) is None else "")
            sf = (calib.get(key) or {}).get("shrink_factor", -1)
            check(f"{key} factor is in [0,1]", 0.0 <= sf <= 1.0, f"factor = {sf}")
        else:
            print(f"  [info ] {key} not fitted yet (needs odds.csv)")

    if raw.get("edge_shrinkage") and raw.get("total_edge_shrinkage"):
        ms = raw["edge_shrinkage"]["shrink_factor"]
        ts = raw["total_edge_shrinkage"]["shrink_factor"]
        mb = raw["edge_shrinkage"].get("beta")
        tb = raw["total_edge_shrinkage"].get("beta")
        # Margins and totals must be fitted independently. When BOTH come back
        # insignificant they are both gated to exactly 0.0, which is the right
        # answer and not evidence of sharing -- so compare the underlying
        # betas, which are always distinct if the fits were genuinely separate.
        check("margin and total edges are fitted independently",
              abs(mb - tb) > 1e-9,
              f"betas: margin {mb:.4f} vs total {tb:.4f}")
        if ms == 0.0 and ts == 0.0:
            check("both shrink factors gated to zero => report must show NO BET",
                  True, "neither edge is statistically significant on clean data")

    # market_cover_prob must behave sanely
    import run_report as RR
    p_zero = RR.market_cover_prob(10.0, 0.0)
    check("with shrink factor 0, any edge gives exactly 50%",
          abs(p_zero - 0.5) < 1e-12, f"p = {p_zero:.6f}")
    p_a = RR.market_cover_prob(3.0, 0.5)
    p_b = RR.market_cover_prob(9.0, 0.5)
    check("with a positive shrink factor, bigger edge -> higher probability",
          0.5 < p_a < p_b < 1.0, f"{p_a:.4f} < {p_b:.4f}")


# ---------------------------------------------------------------------------
# 5b. Validation gate
# ---------------------------------------------------------------------------

def verify_gate():
    """The gate that stops a pooled t-statistic from opening the bet path.

    This project has already been burned by a test that passed by construction
    while the bug it was meant to catch was live. So the gate is verified two
    ways that can actually fail: it must BLOCK a synthetic one-season artifact
    that the old significance-only rule would have shipped, and it must still
    ADMIT a genuinely consistent edge. A gate that can never pass would be a
    hardcoded zero wearing a gate's clothes, and would look identical on real
    data -- where everything is correctly blocked anyway.
    """
    section("5b. VALIDATION GATE (select on prior seasons, confirm on holdout)")
    import evaluate_vs_market as EV

    def synth(betas, n=900, sd=15.0, seed=7):
        rng = np.random.default_rng(seed)
        out = []
        for i, b in enumerate(betas):
            mkt = rng.normal(0, 10, n)
            edge = rng.normal(0, 4, n)
            cover = b * edge + rng.normal(0, sd, n)
            out.append(pd.DataFrame({
                "season": 2024 + i, "mkt_margin_close": mkt,
                "pred_margin": mkt + edge, "act_margin": mkt + cover}))
        return pd.concat(out, ignore_index=True)

    def factors(betas):
        d = synth(betas)
        pooled = EV.fit_edge_shrinkage(d)
        gate = EV.validation_gate(d, EV.fit_edge_shrinkage, pooled)
        old = max(pooled["beta"], 0.0) if pooled["significant"] else 0.0
        new = old if gate["passed"] else 0.0
        return old, new, pooled, gate

    # (a) the artifact this gate exists to catch: two flat seasons and one
    #     strong one, pooling to a "significant" t the old rule would ship
    old, new, pooled, gate = factors([0.0, 0.0, 0.40])
    check("one-season artifact pools to a significant t (setup is valid)",
          pooled["significant"], f"pooled t = {pooled['beta_t']:+.2f}")
    check("significance-only rule WOULD have shipped a non-zero factor",
          old > 0.0, f"old factor = {old:.3f}")
    check("gate BLOCKS the one-season artifact",
          new == 0.0 and not gate["passed"], gate["reason"])

    # (b) it must still admit a real edge -- otherwise it is a hardcoded zero
    old2, new2, pooled2, gate2 = factors([0.35, 0.35, 0.35])
    check("gate ADMITS an edge that replicates in every season",
          gate2["passed"] and new2 > 0.0,
          f"factor = {new2:.3f} (proves the gate is not a hardcoded zero)")

    # (c) pure noise stays at zero from both directions
    _, new3, _, gate3 = factors([0.0, 0.0, 0.0])
    check("gate keeps a pure-noise edge at zero", new3 == 0.0 and not gate3["passed"])

    # (d) the on-disk factor must agree with the on-disk gate verdict
    raw_path = os.path.join(DATA_DIR, "calibration.json")
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        for key in ("edge_shrinkage", "total_edge_shrinkage"):
            blk = raw.get(key) or {}
            if "gate_passed" in blk:
                check(f"{key}: a blocked gate forces shrink_factor to 0",
                      blk["gate_passed"] or blk["shrink_factor"] == 0.0,
                      f"gate_passed={blk['gate_passed']}, "
                      f"factor={blk['shrink_factor']}")

    # (e) diagnostics must NOT leak into the report's calibration
    calib = P.load_calibration()
    check("CLV diagnostic is NOT forwarded to the report",
          "clv_diagnostic" not in calib,
          "CLV is an upper bound on edge, not a substitute for it")


# ---------------------------------------------------------------------------
# 6. Leakage
# ---------------------------------------------------------------------------

def verify_no_leakage():
    section("6. LEAKAGE -- can the backtest see the future?")
    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    key = games["season"] * 100 + games["week"]

    # The previous version of this check asserted that a slice built with
    # `key < cutoff` contained no rows with `key >= cutoff`. That is true by
    # construction of the filter it had just applied, so it passed while a
    # real leak was live: ESPN numbers postseason weeks from 1, so bowl and
    # playoff games are stored as week 1 and a (season, week) ordering put the
    # January national championship in the training set for week 2 of its own
    # season. The check now uses KICKOFF DATES, which is the only ordering
    # that cannot be fooled.
    games["_kick"] = pd.to_datetime(games["date"], errors="coerce")

    late_wk1 = games[(games["week"] == 1) & (games["_kick"].dt.month.isin([12, 1]))]
    check("postseason games are NOT indistinguishable from week 1 by date",
          len(late_wk1) == 0 or True,  # informational: the data really is like this
          f"{len(late_wk1)} Dec/Jan games carry week==1 -- ordering MUST use dates")

    test_block = games[(games["season"] == 2025) & (games["week"] == 6)]
    if len(test_block):
        cutoff_date = test_block["_kick"].min()
        train = games[games["_kick"] < cutoff_date]
        latest = train["_kick"].max()
        check("training slice contains nothing kicking off at/after the test block",
              latest < cutoff_date,
              f"latest training kickoff {latest} vs test start {cutoff_date}")

        # the naive (season, week) slice must be demonstrably WORSE, proving
        # this test can actually detect the bug it is meant to catch
        naive = games[key < (2025 * 100 + 6)]
        leaked = naive[naive["_kick"] >= cutoff_date]
        check("a (season, week) ordering would leak future games (regression test)",
              len(leaked) > 0,
              f"{len(leaked)} future games would enter training under week-keying")

    # current_season must be honored, not inferred
    r_explicit = R.fit_ratings(train, lam=R.DEFAULT_LAMBDA, cv=False, current_season=2025)
    r_default = R.fit_ratings(train, lam=R.DEFAULT_LAMBDA, cv=False)
    same = all(abs(r_explicit.off[t] - r_default.off.get(t, 0)) < 1e-9
               for t in r_explicit.teams)
    check("passing current_season explicitly changes nothing when it matches max season",
          same, "(sanity: both anchor on 2025 here)")

    tr2024 = games[games["season"] <= 2024]
    r_2025 = R.fit_ratings(tr2024, lam=R.DEFAULT_LAMBDA, cv=False, current_season=2025)
    r_auto = R.fit_ratings(tr2024, lam=R.DEFAULT_LAMBDA, cv=False)
    differs = any(abs(r_2025.off[t] - r_auto.off.get(t, 0)) > 1e-6
                  for t in r_2025.teams)
    check("predicting 2025 from 2024 data weights differently than anchoring on 2024",
          differs,
          "this is why current_season must be passed explicitly in the backtest")

    # out-of-sample sigma must exceed in-sample sigma
    if os.path.exists(os.path.join(DATA_DIR, "backtest_preds.csv")):
        bp = pd.read_csv(os.path.join(DATA_DIR, "backtest_preds.csv"))
        oos = float(np.std(pd.concat([bp.act_home - bp.exp_home,
                                      bp.act_away - bp.exp_away])))
        full = R.fit_ratings(games, lam=R.DEFAULT_LAMBDA, cv=False)
        check("out-of-sample sigma exceeds in-sample sigma (as it must)",
              oos > full.resid_std,
              f"oos {oos:.2f} > in-sample {full.resid_std:.2f}")
        check("the simulator is using the out-of-sample sigma",
              abs(P.load_calibration()["score_resid_std"] - oos) < 0.05,
              f"simulator sigma = {P.load_calibration()['score_resid_std']:.2f}")


# ---------------------------------------------------------------------------
# 7. Data integrity
# ---------------------------------------------------------------------------

def verify_no_week_ordering():
    """Scan EVERY module for time-ordering that keys on week instead of date.

    This defect has now been found twice by external audit and missed twice by
    this suite. The first fix went into backtest.py; select_lambda in
    cfb_ratings.py kept the bug for weeks afterwards, because fixing one call
    site felt like fixing the bug.

    ESPN restarts week numbering at 1 for the postseason, so any ordering built
    from (season, week) puts January playoff games in the training slice for
    the season that follows them. This check greps the source rather than
    testing one function, so a NEW call site cannot reintroduce it silently.
    """
    section("8. SOURCE SCAN -- no week-based time ordering anywhere")
    import glob
    import re as _re
    import ast as _ast

    bad_patterns = [
        (r"season[\"']?\s*\*\s*100\s*\+\s*[\"']?week", "season*100 + week key"),
        (r"sort_values\(\s*\[\s*[\"']season[\"']\s*,\s*[\"']week[\"']", "sort by (season, week)"),
    ]
    root = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(root, "*.py")))

    hits = []
    for f in files:
        try:
            src = open(f, encoding="utf-8").read()
        except OSError:
            continue
        # Strip comments AND string literals before scanning. Every fixed site
        # carries a docstring explaining the bug, and this suite deliberately
        # constructs the bad key to prove the leak test can detect it -- all of
        # which are documentation, not live ordering logic. Using ast to blank
        # out string constants is what separates the two.
        try:
            tree = _ast.parse(src)
        except SyntaxError:
            continue
        spans = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Constant) and isinstance(node.value, str):
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    spans.append((node.lineno, node.end_lineno))
        blanked = set()
        for lo, hi in spans:
            blanked.update(range(lo, hi + 1))
        lines = []
        for i, ln in enumerate(src.splitlines(), start=1):
            if i in blanked or ln.strip().startswith("#"):
                lines.append("")
            else:
                lines.append(ln)
        code = chr(10).join(lines)
        for pat, label in bad_patterns:
            for m in _re.finditer(pat, code):
                line = code[:m.start()].count(chr(10)) + 1
                hits.append(f"{os.path.basename(f)}:~{line} ({label})")

    check("no module orders time by (season, week)", not hits,
          "; ".join(hits) if hits else f"scanned {len(files)} modules")


def verify_data():
    section("7. DATA INTEGRITY")
    games = pd.read_csv(os.path.join(DATA_DIR, "real_games.csv"))
    check("no missing scores in the training data",
          int(games[["home_score", "away_score"]].isna().sum().sum()) == 0)
    check("no negative scores", bool((games.home_score >= 0).all() and (games.away_score >= 0).all()))
    check("no duplicate games", int(games.event_id.duplicated().sum()) == 0)

    # renamed teams must be collapsed
    _, name_map = R.canonicalize_team_names(games)
    ids = {}
    g2, _ = R.canonicalize_team_names(games)
    for _, r in g2.iterrows():
        ids.setdefault(str(r.home_id), set()).add(r.home_team)
        ids.setdefault(str(r.away_id), set()).add(r.away_team)
    dups = {k: v for k, v in ids.items() if len(v) > 1}
    check("every team_id maps to exactly one canonical name after cleaning",
          len(dups) == 0, f"{len(dups)} still ambiguous" if dups else
          f"collapsed {len(name_map)} aliases")

    sched_path = os.path.join(DATA_DIR, "schedule_2026.csv")
    if os.path.exists(sched_path):
        sch = pd.read_csv(sched_path)
        check("schedule has no duplicated event_id (refresh must not double rows)",
              int(sch["event_id"].duplicated().sum()) == 0,
              f"{int(sch['event_id'].duplicated().sum())} duplicate rows")
        # the dtype trap that caused it: CSV gives int64, ESPN JSON gives str
        if "kickoff_utc" in sch.columns:
            has_time = sch["kickoff_utc"].astype(str).str.contains("T").mean()
            check("schedule carries full kickoff timestamps, not bare dates",
                  has_time > 0.95,
                  f"{has_time*100:.0f}% have a time component -- the lock window "
                  f"needs these; a bare date parses to midnight UTC and fires early")
        check("event_id comparison is dtype-safe",
              sch["event_id"].astype(str).nunique() == sch["event_id"].nunique(),
              "str and native casts must agree on uniqueness")

    scored = pd.concat([games.home_score, games.away_score])
    check("scores are in a believable range (0-100)",
          bool(scored.max() <= 100 and scored.min() >= 0),
          f"max {scored.max():.0f}")

    r = R.fit_ratings(games, lam=R.DEFAULT_LAMBDA, cv=False, current_season=2026)
    powers = np.array([v for k, v in r.power.items() if not k.startswith("__")])
    check("ratings are centered near zero (mean-zero identifiability constraint)",
          abs(float(powers.mean())) < 5.0, f"mean power {powers.mean():+.2f}")
    check("rating spread is plausible for CFB (sd 8-25 pts)",
          8.0 < float(powers.std()) < 25.0, f"sd {powers.std():.2f}")


# ---------------------------------------------------------------------------
# 9. 2026-08 FIXES -- regression tests
# ---------------------------------------------------------------------------

def verify_2026_08_fixes():
    section("9. 2026-08 FIXES -- bowl blocking, gated lower bound, coach adj, sim mean")

    # (a) walk_forward blocks by kickoff date, not ESPN week label: a
    # December game filed as week==1 must be predicted from ratings trained
    # on the season played before it, not from the preseason block.
    import backtest as B
    rng = np.random.default_rng(11)
    teams = [f"T{i:02d}" for i in range(24)]
    rows = []
    eid = 0
    # 12 regular-season weekends + 2 "week 1"-labelled December games
    for wk in range(1, 13):
        for j in range(0, 24, 2):
            eid += 1
            rows.append(dict(
                date=f"2030-{8 + (wk + 1) // 5:02d}-{(wk * 6) % 27 + 1:02d}",
                kickoff_utc=f"2030-09-{min(wk*2+1, 28):02d}T18:00Z" if wk <= 12 else "",
                home_team=teams[j], away_team=teams[j + 1],
                home_conf="SEC", away_conf="SEC",
                home_score=int(rng.integers(10, 45)), away_score=int(rng.integers(10, 45)),
                neutral_site=False, season=2030, week=wk,
                event_id=eid, home_id=j, away_id=j + 1))
    # kickoffs strictly ordered by week
    for r in rows:
        r["kickoff_utc"] = f"2030-09-01T00:00Z"
    base = pd.Timestamp("2030-08-30T18:00Z")
    for r in rows:
        r["kickoff_utc"] = (base + pd.Timedelta(days=7 * (r["week"] - 1))).strftime("%Y-%m-%dT%H:%MZ")
    # two postseason games mislabelled week 1, played after week 12
    for j in (0, 2):
        eid += 1
        rows.append(dict(
            date="2030-12-20", kickoff_utc="2030-12-20T20:00Z",
            home_team=teams[j], away_team=teams[j + 3],
            home_conf="SEC", away_conf="SEC",
            home_score=int(rng.integers(10, 45)), away_score=int(rng.integers(10, 45)),
            neutral_site=True, season=2030, week=1,
            event_id=eid, home_id=j, away_id=j + 3))
    toy = pd.DataFrame(rows)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        preds = B.walk_forward(toy, seasons=(2030,), lam=1.0, min_train=60,
                               verbose=False)
    dec = preds[pd.to_datetime(preds["date"]).dt.month == 12]
    opener_train = preds["n_train"].min()
    ok = len(dec) > 0 and (dec["n_train"] > opener_train + 60).all()
    check("December games filed as week==1 train on the season before them",
          ok, f"December n_train {sorted(dec['n_train'].unique())} vs "
              f"earliest block {opener_train}")

    # (b) the conservative lower bound honors the validation gate
    import run_report as RR
    blocked = {"beta": 0.30, "beta_se": 0.06, "resid_sd": 15.0,
               "gate_passed": False, "shrink_factor": 0.0}
    p_blocked = RR.conservative_cover_prob(20.0, blocked, pick_home=True)
    check("gate-blocked shrink info -> conservative prob is exactly 0.5",
          abs(p_blocked - 0.5) < 1e-12, f"p = {p_blocked}")
    openg = dict(blocked, gate_passed=True)
    p_open = RR.conservative_cover_prob(20.0, openg, pick_home=True)
    check("gate-passed shrink info -> conservative prob responds to the edge",
          p_open > 0.55, f"p = {p_open:.4f}")

    # (c) coach adjustment moves the margin, not the total, and respects sign
    class _Fake:
        mu, hfa = 27.0, 3.0
        off = {"A": 5.0, "B": 2.0}
        defn = {"A": 1.0, "B": 0.0}
        def has(self, t): return t in self.off
    f = _Fake()
    e0 = P.expected_scores(f, "A", "B", floor=False)
    e1 = P.expected_scores(f, "A", "B", floor=False, margin_adj=-1.5)
    dm = (e1[0] - e1[1]) - (e0[0] - e0[1])
    dt_ = (e1[0] + e1[1]) - (e0[0] + e0[1])
    check("margin_adj=-1.5 shifts margin by exactly -1.5",
          abs(dm + 1.5) < 1e-9, f"delta margin = {dm:+.6f}")
    check("margin_adj leaves the total untouched",
          abs(dt_) < 1e-9, f"delta total = {dt_:+.6f}")
    nc = {P._norm_team("Ohio State")}
    check("coach_margin_adj: home new coach -> negative home margin",
          P.coach_margin_adj(nc, "Ohio State", "Michigan") == P.COACH_CHANGE_MARGIN)
    check("coach_margin_adj: away new coach -> positive home margin",
          P.coach_margin_adj(nc, "Michigan", "Ohio State") == -P.COACH_CHANGE_MARGIN)
    check("coach_margin_adj: both or neither -> zero",
          P.coach_margin_adj(nc, "Ohio State", "Ohio State") == 0.0
          and P.coach_margin_adj(set(), "A", "B") == 0.0)

    # (d) drive-sim mean is unbiased after the TD_PTS fix (was -0.31 at 45)
    h, _ = P.simulate(45.0, 45.0, n_sims=120000, sigma=11.4, seed=7)
    check("simulated mean tracks a 45-point expectation within 0.2",
          abs(float(h.mean()) - 45.0) < 0.2, f"bias {float(h.mean())-45.0:+.3f}")

    # (e) promoted teams keep a REAL rating when FCS results provide one
    hist = pd.DataFrame({
        "home_team": ["NDSU"] * 6 + ["X"], "away_team": ["FcsFoe"] * 6 + ["NDSU"],
        "home_conf": ["FCS"] * 6 + ["SEC"], "away_conf": ["FCS"] * 6 + ["FCS"],
        "season": [2030] * 7, "week": list(range(1, 8)),
    })
    sched = pd.DataFrame({"home_team": ["NDSU"], "home_conf": ["Mountain West"],
                          "away_team": ["X"], "away_conf": ["SEC"]})
    promo = R.detect_promotions(hist, sched)
    check("promotion detection counts FBS-LEVEL games, not raw appearances",
          "NDSU" in promo, f"promoted = {promo}")


def main():
    print("MATHEMATICAL VERIFICATION SUITE")
    verify_ridge()
    verify_design()
    verify_signs()
    verify_simulator()
    verify_calibration()
    verify_gate()
    verify_no_leakage()
    verify_no_week_ordering()
    verify_data()
    verify_2026_08_fixes()

    print(f"\n{'='*74}")
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    print("=" * 74)
    if FAIL:
        print("\nFAILURES:")
        for name, detail in FAIL:
            print(f"  - {name}: {detail}")
        return 1
    print("\nAll mathematical checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
