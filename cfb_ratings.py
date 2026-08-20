#!/usr/bin/env python3
"""
Core opponent-adjusted rating engine (ridge-regularized).

Replaces the naive alternating-least-squares approach in cfb_sim.py with a
single regularized least-squares solve, which fixes several real problems
found in the 2023-2025 ESPN data:

  * HOME FIELD BIAS. The naive "average home margin" estimate gives 8.58
    points, because home teams in non-conference play are systematically the
    stronger team (FBS teams host FCS/G5 opponents for money games). Fitting
    HFA as an unpenalized regression coefficient alongside team ratings gives
    ~3.2-3.4 points, inside the published college-football range (nine
    modern published estimates span 2.4-4.1, most sitting 2.5-3.2; HFA has
    also been declining ~0.1 pt/yr league-wide, so expect drift downward).
    Using the naive number would bias every single spread by ~5 points.

  * NEAR-DISCONNECTED CONFERENCES. Conferences play few cross-conference
    games, so the design matrix is poorly conditioned and unregularized
    ratings drift. Ridge shrinkage toward zero (= league average) stabilizes
    this. Lambda is chosen by walk-forward cross-validation, never by eye:
    lambda directly trades off against HFA (over-shrinking ratings pushes
    team strength into the HFA term), so it must be selected on held-out
    prediction error.

  * FCS / LOW-SAMPLE OPPONENTS. 60 of the 240 teams in the data appear in
    <=3 games (FBS teams' one-off FCS opponents). Giving each its own free
    parameter fits noise. They are pooled into a single replacement-level
    "__FCS__" entity by default.

Ratings are separated into OFF and DEF on a points scale:
    points_scored(team vs opp) = mu + OFF[team] - DEF[opp] + hfa_share
so OFF > 0 means "scores more than average against an average defense" and
DEF > 0 means "allows fewer points than average" (higher is better for both).
"""

from dataclasses import dataclass, field

import os

import numpy as np
import pandas as pd

FCS_LABEL = "__FCS__"          # legacy single-node label (tiered=False)
FCS_HI_LABEL = "__FCS_HI__"    # competitive FCS (mean -22.9 vs FBS)
FCS_LO_LABEL = "__FCS_LO__"    # weak FCS (mean -39.3 vs FBS)
FCS_TIER_THRESHOLD = -14       # best single result vs FBS separating the tiers
# Walk-forward CV over 2024-2025 (1853 out-of-sample games) selected these.
# decay=0.20 means last season is worth 20% of the current one -- much lower
# than intuition suggests, but roster turnover in college football is severe.
DEFAULT_DECAY = 0.20        # per-season-back recency weight (asymptotic, late season)
DECAY_EARLY = 0.85          # effective decay when no current-season games exist yet
ADAPT_K = 250               # games-scale over which decay slides early -> late
DEFAULT_MIN_GAMES = 4       # below this, a team is pooled into __FCS__
# Walk-forward CV on 2023-2025 puts the optimum at ~0.25 with a very flat
# basin from 0.1-0.8 (RMSE varies by <0.03 pts across it), so the exact value
# matters little -- but the grid must span low values, because over-shrinking
# ratings leaks team strength into the HFA coefficient and inflates it.
LAMBDA_GRID = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32]

# Selected by sweeping the leak-free walk-forward in backtest.py, which is the
# authoritative evaluation. An earlier value of 0.25 came from select_lambda()
# while that function still ordered on (season, week) -- so January bowl games
# sat in the training slice for the season that followed and lambda was chosen
# under a live look-ahead leak.
#
# Measured on the clean walk-forward at production settings (conf_shrink on,
# decay 0.20): 0.25 -> 16.057, 0.5 -> 16.002, 1.0 -> 15.958, 1.5 -> 15.966,
# 2.0 -> 16.006, 4.0 -> 16.291. A flat optimum at 1.0. The fitted HFA under
# the SAME production settings is 3.28 at lambda=1.0, 3.43 at 2.0, 3.92 at
# 5.0 -- rising HFA with over-shrinking is the signature of team strength
# leaking into the home-field term, and 3.28 sits comfortably inside the
# published 2.4-4.1 range. (An earlier version of this comment quoted HFA
# from a conf_shrink=False fit -- 3.92/4.69/6.07 -- which is not the
# configuration that ships.)
DEFAULT_LAMBDA = 1.0


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

FBS_CONFERENCES = {
    "SEC", "Big Ten", "Big 12", "ACC", "American", "Sun Belt",
    "MAC", "CUSA", "Mountain West", "Pac-12", "FBS Indep.",
}

# A team promoted from FCS to FBS is NOT a typical FCS team. Measured on the
# three promotions inside this data window (Kennesaw State 2024, Delaware 2025,
# Missouri State 2025), year-one average margin against FBS competition was
# -8.25 (sd 5.97, SE 3.45, n=3). The pooled FCS node sits near -24, so treating
# a promoted team as generic FCS is a ~15-point error on every one of its games.
# n=3 is a thin sample, so this is a deliberately weak prior and any team
# carrying it is flagged low_confidence.
PROMOTED_PRIOR_POWER = -8.25
PROMOTED_PRIOR_SE = 3.45


def canonicalize_team_names(games, extra=None):
    """Collapse ESPN's renamed teams onto one canonical name per team_id.

    ESPN changed several display names inside the 2023-2025 window, e.g.
    'Appalachian State' -> 'App State', 'Texas A&M-Commerce' -> 'East Texas
    A&M', 'St. Francis (PA)' -> 'Saint Francis'. Keyed by name, one real team
    then becomes TWO rating entities, each fitted on a fraction of its games.
    This was live: Appalachian State was rated #86 and App State #124 in the
    same table -- the same program, split in half, both halves wrong.

    team_id is stable across renames, so the fix is to key on it and adopt the
    most recent name. Returns (games_copy, {old_name: canonical_name}).
    """
    if "home_id" not in games.columns:
        return games.copy(), {}

    rows = pd.concat([
        games[["season", "home_team", "home_id"]].rename(
            columns={"home_team": "team", "home_id": "tid"}),
        games[["season", "away_team", "away_id"]].rename(
            columns={"away_team": "team", "away_id": "tid"}),
    ], ignore_index=True)
    rows["tid"] = rows["tid"].astype(str)

    canon = (rows.sort_values("season")
                 .drop_duplicates("tid", keep="last")
                 .set_index("tid")["team"].to_dict())

    mapping = {}
    for tid, grp in rows.groupby("tid"):
        target = canon.get(tid)
        for nm in grp["team"].unique():
            if nm != target:
                mapping[nm] = target
    if extra:
        mapping.update(extra)
    if not mapping:
        return games.copy(), {}

    g = games.copy()
    g["home_team"] = g["home_team"].replace(mapping)
    g["away_team"] = g["away_team"].replace(mapping)
    return g, mapping


def detect_promotions(history, schedule):
    """Teams appearing in an FBS conference on `schedule` whose history holds
    almost no FBS-LEVEL games.

    Returns {team: reason}. The count is of FBS-context appearances -- games
    where at least one side is in an FBS conference -- not raw appearances.
    That distinction matters once FCS-vs-FCS schedules are augmented into the
    training data: a promoted team like North Dakota State then has 30+ real
    games and a genuinely fitted rating, but every one of them is at the FCS
    level, so the level-jump uncertainty (and the AVOID flag) must survive.
    fit_ratings only overwrites the rating with the promoted-team prior when
    the team has no fitted rating at all; when FCS games gave it one, the
    fitted rating is kept and only the low-confidence flag applies.
    """
    if schedule is None or "home_conf" not in schedule.columns:
        return {}
    sched_conf = {}
    for _, r in schedule.iterrows():
        if r.get("home_conf") in FBS_CONFERENCES:
            sched_conf[r["home_team"]] = r["home_conf"]
        if r.get("away_conf") in FBS_CONFERENCES:
            sched_conf[r["away_team"]] = r["away_conf"]

    if "home_conf" in history.columns and "away_conf" in history.columns:
        fbs_ctx = history[history["home_conf"].isin(FBS_CONFERENCES)
                          | history["away_conf"].isin(FBS_CONFERENCES)]
    else:
        fbs_ctx = history
    counts = pd.concat([fbs_ctx["home_team"], fbs_ctx["away_team"]]).value_counts()
    out = {}
    for team, conf in sched_conf.items():
        n = int(counts.get(team, 0))
        if n < DEFAULT_MIN_GAMES:
            out[team] = f"new to FBS ({conf}); only {n} prior FBS-level games in data"
    return out


# ---------------------------------------------------------------------------
# FCS schedule augmentation (optional training-only data)
# ---------------------------------------------------------------------------

def load_fcs_games(path=None):
    """FCS-vs-FCS results fetched by `fetch_espn_data.py --fcs`, or None.

    These games enter the TRAINING data only (never predicted or graded), so
    every FCS opponent can carry its own fitted rating instead of sharing one
    pooled node. Measured motivation: with the pooled node the model loses to
    the closing line by 2.19 RMSE on FCS-involved games versus 0.57 on
    FBS-vs-FBS -- the market rates each FCS team individually and the model
    cannot. Whether augmentation actually helps is decided empirically by
    `python backtest.py --fcs-ab`, which writes its verdict into
    calibration.json; nothing turns on by default.
    """
    from paths import DATA_DIR as _D
    p = path or os.path.join(_D, "fcs_games.csv")
    if not os.path.exists(p):
        return None
    try:
        fcs = pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None
    need = {"home_team", "away_team", "home_score", "away_score",
            "season", "neutral_site", "date"}
    if not need.issubset(fcs.columns) or len(fcs) == 0:
        return None
    return fcs


def _fcs_tier_map(games, rare):
    """Split pooled non-FBS teams into two competitive tiers.

    One pooled FCS node is too coarse. Measured on 2023-2025: non-FBS teams
    whose BEST single result against FBS competition was within 14 points
    average -22.94 against FBS, while the rest average -39.27 -- a 16.3-point
    separation across 106 teams and 365 games. Collapsing them into a single
    node (mean -31.44) therefore makes good FCS teams look ~8.5 points worse
    than they are and weak ones ~7.8 points better, and those errors land
    directly on the FBS opponent's rating.

    Tiering uses only the games passed in, so a walk-forward training slice
    stays leak-free.
    """
    best = {}
    for _, r in games.iterrows():
        h, a = r["home_team"], r["away_team"]
        h_rare, a_rare = h in rare, a in rare
        if h_rare == a_rare:
            continue                      # both pooled or neither: uninformative
        if h_rare:
            m = r["home_score"] - r["away_score"]
            best[h] = max(best.get(h, -999), m)
        else:
            m = r["away_score"] - r["home_score"]
            best[a] = max(best.get(a, -999), m)

    tier = {}
    for t in rare:
        b = best.get(t)
        # teams with no measured game against a rated opponent default to the
        # lower tier: most such teams are genuine cupcakes, and being wrong in
        # the optimistic direction inflates the FBS opponent's rating
        tier[t] = FCS_HI_LABEL if (b is not None and b >= FCS_TIER_THRESHOLD) else FCS_LO_LABEL
    return tier


def resolve_team(ratings, team, history=None):
    """Map a team name onto a rated entity.

    An unrated team is one the model never saw enough of. It resolves to an
    FCS tier rather than a single generic node -- which tier is decided by the
    team's best result against rated opposition if we have one, else the lower
    tier (see _fcs_tier_map for why the pessimistic default is correct).
    """
    if ratings.has(team):
        return team
    alias = getattr(ratings, "name_map", {}).get(team)
    if alias and ratings.has(alias):
        return alias
    tier = getattr(ratings, "fcs_tier_of", {}).get(team)
    if tier and ratings.has(tier):
        return tier
    for lbl in (FCS_LO_LABEL, FCS_HI_LABEL, FCS_LABEL):
        if ratings.has(lbl):
            return lbl
    return team


def pool_rare_teams(games, min_games=DEFAULT_MIN_GAMES, tiered=False):
    """Collapse teams with very few appearances into replacement-level
    entities. Returns (games_copy, set_of_pooled_names).

    NEGATIVE RESULT -- tiered defaults to False. Splitting the pooled FCS node
    into two competitive tiers looks obviously right descriptively: teams whose
    best result against FBS was within 14 points average -22.9, the rest -39.3,
    a 16.3-point gap. But measured on walk-forward prediction it does NOT help:

        single node : ALL 15.967 | FBS-FBS 15.790 | FCS games 17.071
        two tiers   : ALL 15.987 | FBS-FBS 15.807 | FCS games 17.108

    The gap is largely SELECTION, not skill. Tiering keys on max(margin vs
    FBS), an order statistic with high variance -- a team lands in the top tier
    partly because it got one lucky result, and that luck does not repeat. The
    descriptive split is real; the predictive content is not. Kept available
    for experimentation, off by default.
    """
    counts = pd.concat([games["home_team"], games["away_team"]]).value_counts()
    rare = set(counts[counts < min_games].index)
    if not rare:
        return games.copy(), rare
    g = games.copy()
    if tiered:
        tmap = _fcs_tier_map(g, rare)
        g["home_team"] = g["home_team"].map(lambda t: tmap.get(t, t))
        g["away_team"] = g["away_team"].map(lambda t: tmap.get(t, t))
    else:
        g["home_team"] = g["home_team"].where(~g["home_team"].isin(rare), FCS_LABEL)
        g["away_team"] = g["away_team"].where(~g["away_team"].isin(rare), FCS_LABEL)
    return g, rare


def season_weights(games, current_season=None, decay=DEFAULT_DECAY,
                   adaptive=False, decay_early=DECAY_EARLY, k_games=ADAPT_K):
    """Recency weight per game: weight = decay ** years_back.

    Walk-forward CV on 2023-2025 puts the optimum at decay=0.20 (last season
    worth 20% of the current one) -- much lower than intuition suggests, but
    college rosters turn over fast.

    NEGATIVE RESULT (adaptive=False is the default for this reason): an
    adaptive scheme that raises the effective decay early in the season, on
    the theory that week 1 has no current-season games and therefore needs
    more history, was tested and LOSES on every measure -- including on
    early-season games specifically (wk<=4 RMSE 17.35-17.52 adaptive vs
    17.32 fixed; overall 16.09-16.31 vs 16.04).

    The reason is that in week 1 ALL training data is prior seasons, so decay
    controls the ratio BETWEEN those prior seasons, not history-vs-current.
    decay=0.20 correctly says "last season matters 5x more than two seasons
    ago"; raising it toward 1.0 flattens that and lets stale seasons count
    nearly as much as recent ones. The adaptive path is retained for
    experimentation but is off by default.
    """
    if "season" not in games.columns:
        return np.ones(len(games))
    cur = current_season if current_season is not None else int(games["season"].max())
    years_back = (cur - games["season"]).clip(lower=0).values.astype(float)

    if adaptive:
        n_cur = int((games["season"] == cur).sum())
        decay_eff = decay + (decay_early - decay) * (k_games / (k_games + n_cur))
    else:
        decay_eff = decay
    return decay_eff ** years_back


# ---------------------------------------------------------------------------
# Ridge solve
# ---------------------------------------------------------------------------

@dataclass
class Ratings:
    off: dict = field(default_factory=dict)
    defn: dict = field(default_factory=dict)
    power: dict = field(default_factory=dict)
    sos: dict = field(default_factory=dict)
    conf_strength: dict = field(default_factory=dict)
    conf_of: dict = field(default_factory=dict)
    teams: list = field(default_factory=list)
    mu: float = 0.0            # league average points per team per game
    hfa: float = 0.0           # total home-field advantage in points (margin)
    lam: float = 0.0
    resid_std: float = 0.0     # std of per-team-score residuals
    resid_corr: float = 0.0    # corr of the two teams' residuals within a game
    margin_resid_std: float = 0.0
    total_resid_std: float = 0.0
    n_games: int = 0
    pooled: set = field(default_factory=set)
    promoted: dict = field(default_factory=dict)   # team -> reason (prior-rated, low confidence)
    fcs_tier_of: dict = field(default_factory=dict)  # unrated team -> FCS tier label
    name_map: dict = field(default_factory=dict)     # alias -> canonical team name

    def has(self, team):
        return team in self.off


def _build_conf_map(games):
    """Team -> conference, using each team's MOST RECENT season label.

    Conference realignment makes a naive full-frame map wrong: Texas and
    Oklahoma were Big 12 in 2023 and SEC from 2024, and the Pac-12 collapsed.
    Iterating the frame and letting the last row win is order-dependent and
    silently mislabels history. Sorting by season and taking the latest label
    per team makes "which conference is this team in now" explicit.
    """
    if "season" not in games.columns:
        rows = pd.concat([
            games[["home_team", "home_conf"]].rename(
                columns={"home_team": "team", "home_conf": "conf"}),
            games[["away_team", "away_conf"]].rename(
                columns={"away_team": "team", "away_conf": "conf"}),
        ], ignore_index=True)
        m = rows.drop_duplicates("team", keep="last").set_index("team")["conf"].to_dict()
    else:
        rows = pd.concat([
            games[["season", "home_team", "home_conf"]].rename(
                columns={"home_team": "team", "home_conf": "conf"}),
            games[["season", "away_team", "away_conf"]].rename(
                columns={"away_team": "team", "away_conf": "conf"}),
        ], ignore_index=True)
        rows = rows.sort_values("season")
        m = rows.drop_duplicates("team", keep="last").set_index("team")["conf"].to_dict()
    m[FCS_LABEL] = "FCS/Other"
    m[FCS_HI_LABEL] = "FCS/Other"
    m[FCS_LO_LABEL] = "FCS/Other"
    return m


def _build_design(games, teams):
    """Two rows per game (home perf, away perf).

    Columns: [OFF_0..OFF_{n-1} | DEF_0..DEF_{n-1} | HFA]
    Row for team t vs opponent o:
        points(t) - mu = OFF[t] - DEF[o] + hfa_sign * (HFA/2)
    HFA is expressed as the full margin advantage, so each side carries half.
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    m = len(games)
    rows = 2 * m

    X = np.zeros((rows, 2 * n + 1))
    y = np.zeros(rows)

    h = games["home_team"].map(idx).values
    a = games["away_team"].map(idx).values
    hs = games["home_score"].values.astype(float)
    as_ = games["away_score"].values.astype(float)
    neutral = games["neutral_site"].values.astype(bool)
    sign = np.where(neutral, 0.0, 1.0)

    r = np.arange(m)
    # home performance rows
    X[r, h] = 1.0                      # OFF[home]
    X[r, n + a] = -1.0                 # -DEF[away]
    X[r, 2 * n] = sign * 0.5           # +HFA/2
    y[r] = hs
    # away performance rows
    X[m + r, a] = 1.0                  # OFF[away]
    X[m + r, n + h] = -1.0             # -DEF[home]
    X[m + r, 2 * n] = -sign * 0.5      # -HFA/2
    y[m + r] = as_

    return X, y


def _solve_ridge(X, y, w, lam, n_teams, mu, target=None):
    """Weighted ridge with an optional shrinkage TARGET.

    Minimizes  ||y - mu - X.beta||^2_w  +  lam * ||beta - m||^2
    giving     (X'WX + lam.I) beta = X'W(y - mu) + lam.m

    With m = 0 this shrinks every team toward league average. With m set to
    each team's conference mean (see fit_ratings) it shrinks toward the
    conference instead, which is a much better prior: a Sun Belt team with a
    3-1 record is far more likely to be a good Sun Belt team than a national
    contender.

    The HFA column is never penalized (it is a real physical effect, not a
    noisy team parameter). An additive shift (OFF+c, DEF+c) leaves every
    prediction unchanged, so the system is rank-deficient without
    regularization; the ridge penalty resolves this by selecting the
    minimum-norm solution.
    """
    P = np.eye(X.shape[1]) * lam
    P[-1, -1] = 0.0
    Xw = X * w[:, None]
    A = X.T @ Xw + P
    b = Xw.T @ (y - mu)
    if target is not None:
        t = np.asarray(target, dtype=float).copy()
        t[-1] = 0.0                     # never pull HFA toward a target
        b = b + lam * t
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # A is singular. The realistic cause is an unidentifiable HFA column:
        # it is deliberately unpenalized, so if every game in the slice is at
        # a neutral site that column is all zeros and carries no information.
        # lstsq returns the minimum-norm solution, which sets the
        # unidentifiable coefficient to 0 -- exactly the right answer ("no
        # evidence of home-field advantage in this data").
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol


def _conference_targets(off, defn, teams, conf_of, k_conf=2.0):
    """Empirical-Bayes conference means, shrunk toward the league mean.

    m_c = mean(ratings in c) * n_c / (n_c + k_conf)

    k_conf is deliberately small (2.0). A large value would drag a genuinely
    elite team in a weak conference toward mediocrity; a small one lets that
    team pull its own conference mean up instead.
    """
    by_conf_off, by_conf_def = {}, {}
    for t in teams:
        if t in (FCS_LABEL, FCS_HI_LABEL, FCS_LO_LABEL):
            continue
        c = conf_of.get(t, "UNK")
        by_conf_off.setdefault(c, []).append(off[t])
        by_conf_def.setdefault(c, []).append(defn[t])

    m_off, m_def = {}, {}
    for c, vals in by_conf_off.items():
        n_c = len(vals)
        m_off[c] = float(np.mean(vals)) * n_c / (n_c + k_conf)
    for c, vals in by_conf_def.items():
        n_c = len(vals)
        m_def[c] = float(np.mean(vals)) * n_c / (n_c + k_conf)
    return m_off, m_def


def fit_ratings(games, lam=None, decay=DEFAULT_DECAY, current_season=None,
                min_games=DEFAULT_MIN_GAMES, cv=True, verbose=False,
                adaptive=False, decay_early=DECAY_EARLY, k_games=ADAPT_K,
                conf_shrink=True, k_conf=2.0, conf_iters=6, schedule=None):
    """Fit ridge OFF/DEF ratings. If lam is None and cv=True, lambda is chosen
    by walk-forward cross-validation on held-out weeks."""
    games = games.dropna(subset=["home_score", "away_score"]).copy()
    games["neutral_site"] = games["neutral_site"].astype(bool)
    games, name_map = canonicalize_team_names(games)
    _pre_pool = games
    games, pooled = pool_rare_teams(games, min_games=min_games)
    fcs_tier_of = _fcs_tier_map(_pre_pool, pooled) if pooled else {}

    if lam is None:
        lam = select_lambda(games, decay=decay, current_season=current_season,
                            verbose=verbose) if cv else 8.0

    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    n = len(teams)
    mu = float(pd.concat([games["home_score"], games["away_score"]]).mean())

    X, y = _build_design(games, teams)
    gw = season_weights(games, current_season=current_season, decay=decay,
                        adaptive=adaptive, decay_early=decay_early, k_games=k_games)
    w = np.concatenate([gw, gw])

    beta = _solve_ridge(X, y, w, lam, n, mu)
    off = {t: float(beta[i]) for i, t in enumerate(teams)}
    defn = {t: float(beta[n + i]) for i, t in enumerate(teams)}
    hfa = float(beta[2 * n])

    # --- two-level shrinkage: team -> conference -> league --------------
    # Shrinking toward zero says "we know nothing about this team". Shrinking
    # toward its conference mean says "absent evidence, it looks like its
    # peers", which is a far better prior and the single largest accuracy
    # gain available here. Iterate: conference means are themselves computed
    # from the shrunk ratings, so an elite team drags its conference up
    # rather than being flattened by it.
    if conf_shrink:
        conf_map0 = _build_conf_map(games)
        for _ in range(conf_iters):
            m_off, m_def = _conference_targets(off, defn, teams, conf_map0, k_conf=k_conf)
            target = np.zeros(2 * n + 1)
            for i, t in enumerate(teams):
                c = conf_map0.get(t, "UNK")
                target[i] = m_off.get(c, 0.0)
                target[n + i] = m_def.get(c, 0.0)
            beta_new = _solve_ridge(X, y, w, lam, n, mu, target=target)
            off_new = {t: float(beta_new[i]) for i, t in enumerate(teams)}
            def_new = {t: float(beta_new[n + i]) for i, t in enumerate(teams)}
            shift = max(max(abs(off_new[t] - off[t]) for t in teams),
                        max(abs(def_new[t] - defn[t]) for t in teams))
            off, defn, beta = off_new, def_new, beta_new
            if shift < 0.01:
                break
        hfa = float(beta[2 * n])

    # residual diagnostics -- these drive the simulator's noise model
    pred = mu + X @ beta
    resid = y - pred
    m = len(games)
    rh, ra = resid[:m], resid[m:]
    resid_std = float(np.sqrt(np.average(resid ** 2, weights=w)))
    resid_corr = float(np.corrcoef(rh, ra)[0, 1])
    margin_resid_std = float(np.std(rh - ra))
    total_resid_std = float(np.std(rh + ra))

    power = {t: off[t] + defn[t] for t in teams}
    conf_of = _build_conf_map(games)

    # SOS = recency-weighted average POWER of opponents actually played
    opp_rows = pd.DataFrame({
        "team": pd.concat([games["home_team"], games["away_team"]], ignore_index=True),
        "opp": pd.concat([games["away_team"], games["home_team"]], ignore_index=True),
        "w": w,
    })
    opp_rows["opp_power"] = opp_rows["opp"].map(power)
    num = (opp_rows["opp_power"] * opp_rows["w"]).groupby(opp_rows["team"]).sum()
    den = opp_rows["w"].groupby(opp_rows["team"]).sum()
    sos = (num / den).to_dict()

    conf_strength = {}
    for t in teams:
        if t in (FCS_LABEL, FCS_HI_LABEL, FCS_LO_LABEL):
            continue
        conf_strength.setdefault(conf_of.get(t, "UNK"), []).append(power[t])
    conf_strength = {c: float(np.mean(v)) for c, v in conf_strength.items()}

    # --- newly-promoted FBS teams get a prior, not the FCS pooled node -----
    promoted = {}
    if schedule is not None:
        promoted = detect_promotions(games, schedule)
        for team, reason in promoted.items():
            if team in off:
                # FCS augmentation gave this team a genuinely fitted rating
                # (e.g. North Dakota State with 30+ FCS games). Keep it --
                # real results beat an n=3 prior -- but keep the promoted
                # flag too: everything it earned was at the FCS level, and
                # the level jump is exactly the uncertainty AVOID exists for.
                promoted[team] = reason + " (rated from FCS results)"
                continue
            # split the prior power evenly across offense and defense; we have
            # no basis to say a promoted team is lopsided either way
            off[team] = PROMOTED_PRIOR_POWER / 2.0
            defn[team] = PROMOTED_PRIOR_POWER / 2.0
            power[team] = PROMOTED_PRIOR_POWER
            sos.setdefault(team, 0.0)
            if team not in teams:
                teams = sorted(teams + [team])
            conf_of.setdefault(team, "FBS (new)")

    return Ratings(
        off=off, defn=defn, power=power, sos=sos, conf_strength=conf_strength,
        conf_of=conf_of, teams=teams, mu=mu, hfa=hfa, lam=lam,
        resid_std=resid_std, resid_corr=resid_corr,
        margin_resid_std=margin_resid_std, total_resid_std=total_resid_std,
        n_games=len(games), pooled=pooled, promoted=promoted, fcs_tier_of=fcs_tier_of,
        name_map=name_map,
    )


# ---------------------------------------------------------------------------
# Lambda selection by walk-forward CV (never fit lambda on the test set)
# ---------------------------------------------------------------------------

def _season_week_key(games):
    return games["season"].astype(int) * 100 + games["week"].astype(int)


def select_lambda(games, grid=LAMBDA_GRID, decay=DEFAULT_DECAY,
                  current_season=None, min_train=300, verbose=False):
    """Walk-forward: for each holdout block of weeks, fit on everything
    strictly earlier and score margin RMSE on the block. Picks the lambda with
    the lowest pooled out-of-sample margin RMSE."""
    if "season" not in games.columns or "week" not in games.columns:
        return 8.0

    # Order on KICKOFF DATE, never on (season, week).
    #
    # ESPN restarts week numbering at 1 for the postseason, so every bowl and
    # playoff game -- including the January national championship -- is stored
    # as week 1 of its season. A (season, week) key therefore places those
    # results in the TRAINING slice for every later week of the same season.
    #
    # backtest.py was hardened against exactly this and orders on kickoff_utc.
    # select_lambda was not, so lambda itself was chosen under a live
    # look-ahead leak even while the backtest that reported its performance was
    # clean. Same defect, same file family, missed because fixing one call site
    # felt like fixing the bug.
    if "kickoff_utc" in games.columns:
        g = games.copy()
        g["_ord"] = pd.to_datetime(g["kickoff_utc"], errors="coerce", utc=True)
        g = g.dropna(subset=["_ord"]).sort_values("_ord").reset_index(drop=True)
        key = g["_ord"]
    else:
        g = games.copy()
        g["_ord"] = pd.to_datetime(g["date"], errors="coerce")
        g = g.dropna(subset=["_ord"]).sort_values("_ord").reset_index(drop=True)
        key = g["_ord"]
    # blocks of consecutive PLAYING DAYS, evaluated on the back half.
    # Work in integer day numbers so a tz-aware kickoff column and a tz-naive
    # date column behave identically here.
    ordv = pd.Series(key).reset_index(drop=True)
    day_num = (ordv.astype("int64") // 86_400_000_000_000).to_numpy()
    uniq = np.unique(day_num)
    start_i = max(1, len(uniq) // 2)
    blocks = [uniq[i:i + 7] for i in range(start_i, len(uniq), 7)]

    errs = {lam: [] for lam in grid}
    for blk in blocks:
        cutoff = blk[0]
        train = g[day_num < cutoff]
        test = g[np.isin(day_num, blk)]
        if len(train) < min_train or len(test) == 0:
            continue
        # same guarantee backtest.py makes: nothing in training kicked off at
        # or after the first game being scored
        assert day_num[day_num < cutoff].max() < cutoff
        tr, _ = pool_rare_teams(train)
        teams = sorted(set(tr["home_team"]) | set(tr["away_team"]))
        tset = set(teams)
        n = len(teams)
        mu = float(pd.concat([tr["home_score"], tr["away_score"]]).mean())
        X, y = _build_design(tr, teams)
        gw = season_weights(tr, current_season=current_season, decay=decay)
        w = np.concatenate([gw, gw])

        # only score test games whose both teams were seen in training
        te = test.copy()
        te["home_team"] = te["home_team"].where(te["home_team"].isin(tset), FCS_LABEL)
        te["away_team"] = te["away_team"].where(te["away_team"].isin(tset), FCS_LABEL)
        te = te[te["home_team"].isin(tset) & te["away_team"].isin(tset)]
        if len(te) == 0:
            continue

        actual = (te["home_score"] - te["away_score"]).values
        sign = np.where(te["neutral_site"].astype(bool).values, 0.0, 1.0)

        for lam in grid:
            beta = _solve_ridge(X, y, w, lam, n, mu)
            off = {t: beta[i] for i, t in enumerate(teams)}
            dfn = {t: beta[n + i] for i, t in enumerate(teams)}
            hfa = beta[2 * n]
            ph = np.array([off[t] for t in te["home_team"]]) - np.array([dfn[t] for t in te["away_team"]])
            pa = np.array([off[t] for t in te["away_team"]]) - np.array([dfn[t] for t in te["home_team"]])
            pred_margin = (ph - pa) + sign * hfa
            errs[lam].append((pred_margin - actual) ** 2)

    scores = {}
    for lam, chunks in errs.items():
        if chunks:
            scores[lam] = float(np.sqrt(np.mean(np.concatenate(chunks))))
    if not scores:
        return 8.0
    best = min(scores, key=scores.get)
    if verbose:
        print("  walk-forward lambda selection (out-of-sample margin RMSE):")
        for lam in grid:
            if lam in scores:
                mark = " <-- best" if lam == best else ""
                print(f"    lambda={lam:<7g} RMSE={scores[lam]:6.3f}{mark}")
    return best


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_games.csv"
    games = pd.read_csv(path)
    r = fit_ratings(games, verbose=True)
    print(f"\nFit on {r.n_games} games, {len(r.teams)} rated entities "
          f"({len(r.pooled)} pooled into {FCS_LABEL})")
    print(f"  lambda            = {r.lam}")
    print(f"  league avg pts    = {r.mu:.2f}")
    print(f"  home field adv    = {r.hfa:+.2f} pts")
    print(f"  score resid std   = {r.resid_std:.2f}")
    print(f"  resid correlation = {r.resid_corr:+.3f}")
    print(f"  margin resid std  = {r.margin_resid_std:.2f}")
    print(f"  total  resid std  = {r.total_resid_std:.2f}")

    tbl = pd.DataFrame({
        "team": r.teams,
        "conf": [r.conf_of.get(t, "") for t in r.teams],
        "off": [round(r.off[t], 2) for t in r.teams],
        "def": [round(r.defn[t], 2) for t in r.teams],
        "power": [round(r.power[t], 2) for t in r.teams],
        "sos": [round(r.sos.get(t, 0), 2) for t in r.teams],
    }).sort_values("power", ascending=False).reset_index(drop=True)
    tbl.index += 1
    print("\nTop 25:")
    print(tbl.head(25).to_string())
