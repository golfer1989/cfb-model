#!/usr/bin/env python3
"""PLAY-BY-PLAY LAB -- rate teams on how they play, not just on final scores.

Owner's decision (2026-11-09): rebuild the predictions on play-by-play
efficiency. Final scores carry a lot of noise -- garbage-time touchdowns,
return scores, a fumble at the 1 -- while success rate and EPA per play
measure every snap. CFBD's /stats/game/advanced (garbage time excluded) is
collected per game by collect_game_stats.py into
data/raw_cache/cfbd/advgame_<season>_wk<week>[_postseason].json.

Method: one walk-forward over 2024, 2025 and 2026 in 7-day blocks. Every
block refits on games kicked off strictly before it, with the SAME ridge
engine (cfb_ratings.fit_ratings) for every stream:

    R0   points -- the current model (final scores)
    PPA  offensive EPA per play, garbage time excluded
    SR   offensive success rate
    XPL  offensive explosiveness (EPA on successful plays)

Each stream gives a predicted home-minus-away differential for every game.
Margins are then predicted by an OLS blend fitted on OTHER seasons only:

    blend:    margin ~ 1 + R0 + PPA + SR + XPL
    pbp-only: margin ~ 1 + PPA + SR + XPL

PREREGISTERED -- written 2026-11-10, before any play-by-play data exists in
the repo; nothing below may be changed after the data arrives:
  fits: 2024 tested with weights fitted on 2025, 2025 with weights fitted on
        2024, 2026 with weights fitted on 2024+2025. Test games: FBS vs FBS.
  PASS requires BOTH:
    (1) blend margin RMSE < R0 margin RMSE in each of 2024, 2025 and 2026;
    (2) pooled paired t of (R0 squared error - blend squared error) >= 2.0.
  Reported, not gating: pbp-only RMSE; straight-up accuracy; ATS against the
  2026 DraftKings closing lines in the ledger (the only sharp lines on file
  -- the 2024-25 ESPN BET closes are soft, see cfb-market-lab-2026-10-30);
  early-season blocks 0-2 separately (weeks 1-2 were 2026's worst stretch).
  Ship value on PASS: blend weights refitted on 2024+2025+2026.

Exploratory only (needs its own preregistered confirmation before any use):
  totals with PPA/SR sums added to the shipped efficiency total.

Usage: python experiments/pbp_lab.py [data_dir]
"""
import glob
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import cfb_predict as P  # noqa: E402
import cfb_ratings as R  # noqa: E402

SEASONS = (2024, 2025, 2026)
STREAMS = {"ppa": "ppa", "sr": "successRate", "xpl": "explosiveness"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and").replace("state", "st")
    return re.sub(r"[^a-z0-9]", "", s)


def _num(d, key):
    try:
        v = (d or {}).get(key)
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_adv(data_dir):
    """One row per team-game: offensive and defensive play-by-play metrics."""
    rows = []
    for path in glob.glob(os.path.join(data_dir, "raw_cache", "cfbd", "advgame_*.json")):
        try:
            payload = json.load(open(path, encoding="utf-8")) or []
        except (OSError, ValueError):
            continue
        for r in payload:
            off, dfn = r.get("offense") or {}, r.get("defense") or {}
            rec = {"event_id": str(r.get("gameId", r.get("game_id", r.get("id")))),
                   "team": r.get("team"), "opponent": r.get("opponent")}
            for k, col in STREAMS.items():
                rec[f"off_{k}"] = _num(off, col)
                rec[f"def_{k}"] = _num(dfn, col)
            rec["off_plays"] = _num(off, "plays")
            rows.append(rec)
    return pd.DataFrame(rows)


def attach(games, adv):
    """Put each game's two team rows on the home/away sides of games.
    Returns games with home_<k>/away_<k> offensive metrics, plus a count of
    games whose team names could not be matched (reported, never guessed)."""
    g = games.copy()
    if adv.empty:
        return g, 0
    adv = adv.copy()
    adv["_t"] = adv["team"].map(_norm)
    by = {e: x for e, x in adv.groupby("event_id")}
    unmatched = 0
    cols = {f"{side}_{k}": [] for side in ("home", "away") for k in list(STREAMS) + ["plays"]}
    for e, h, a in zip(g["event_id"], g["home_team"], g["away_team"]):
        x = by.get(e)
        ok = x is not None and len(x) >= 2
        if ok:
            hr = x[x["_t"] == _norm(h)]
            ar = x[x["_t"] == _norm(a)]
            ok = len(hr) == 1 and len(ar) == 1
            if not ok:
                unmatched += 1
        for side, rr in (("home", hr if ok else None), ("away", ar if ok else None)):
            for k in list(STREAMS) + ["plays"]:
                cols[f"{side}_{k}"].append(float(rr.iloc[0][f"off_{k}"]) if ok else np.nan)
    for c, v in cols.items():
        g[c] = v
    return g, unmatched


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def _diff(rt, h, a, neutral):
    hh, aa = R.resolve_team(rt, h), R.resolve_team(rt, a)
    if not (rt.has(hh) and rt.has(aa)):
        return np.nan, np.nan
    hfa = 0.0 if neutral else rt.hfa / 2.0
    eh = rt.mu + rt.off[hh] - rt.defn[aa] + hfa
    ea = rt.mu + rt.off[aa] - rt.defn[hh] - hfa
    return eh - ea, eh + ea


def _fit_stream(train, k, scale, season):
    x = train[["home_team", "away_team", "home_conf", "away_conf", "neutral_site",
               "season", "week", "date", f"home_{k}", f"away_{k}"]].dropna().copy()
    x["home_score"] = x[f"home_{k}"] * scale
    x["away_score"] = x[f"away_{k}"] * scale
    return R.fit_ratings(x, lam=R.DEFAULT_LAMBDA, cv=False, current_season=season)


def walk_forward(data_dir):
    games = pd.read_csv(os.path.join(data_dir, "real_games.csv"))
    games["event_id"] = games["event_id"].astype(str)
    games["_kick"] = pd.to_datetime(games["kickoff_utc"], utc=True, errors="coerce")
    games = games[games["home_score"].notna()]
    adv = load_adv(data_dir)
    g, unmatched = attach(games, adv)
    have = g["home_ppa"].notna()
    print(f"play-by-play rows: {len(adv)} team-games; games matched {int(have.sum())}"
          f" of {len(g)}; name mismatches {unmatched}")
    print("coverage by season:", g[have].groupby("season").size().to_dict())
    pts_sd = pd.concat([g.home_score, g.away_score]).std()
    scale = {k: pts_sd / pd.concat([g[f"home_{k}"], g[f"away_{k}"]]).std() for k in STREAMS}

    lines = _closing_lines(data_dir)
    rows = []
    for season in SEASONS:
        s = g[g.season == season]
        if not len(s):
            continue
        k0 = s["_kick"].min().normalize()
        blk = ((s["_kick"] - k0).dt.days // 7).astype(int)
        for b in sorted(blk.unique()):
            test = s[blk == b]
            cutoff = test["_kick"].min()
            train = g[g["_kick"] < cutoff]
            tr_pbp = train[train["home_ppa"].notna()]
            if len(tr_pbp) < 150:
                continue
            rt0 = R.fit_ratings(train, lam=R.DEFAULT_LAMBDA, cv=False, current_season=season)
            rts = {k: _fit_stream(tr_pbp, k, scale[k], season) for k in STREAMS}
            for _, t in test.iterrows():
                if t.home_conf not in R.FBS_CONFERENCES or t.away_conf not in R.FBS_CONFERENCES:
                    continue
                neu = bool(t.neutral_site)
                h0, a0 = R.resolve_team(rt0, t.home_team), R.resolve_team(rt0, t.away_team)
                if not (rt0.has(h0) and rt0.has(a0)):
                    continue
                eh, ea = P.expected_scores(rt0, h0, a0, neutral=neu, floor=True)
                rec = {"event_id": t.event_id, "season": season, "block": int(b),
                       "R0": eh - ea, "act": t.home_score - t.away_score,
                       "close": lines.get(t.event_id) if season == 2026 else np.nan}
                for k, rt in rts.items():
                    d, _ = _diff(rt, t.home_team, t.away_team, neu)
                    rec[k] = d / scale[k] if pd.notna(d) else np.nan
                rows.append(rec)
            print(f"  {season} block {b}: {len(test)} games", flush=True)
    return pd.DataFrame(rows).dropna(subset=["R0"] + list(STREAMS))


def _closing_lines(data_dir):
    """2026 DraftKings closing home margins from the graded ledger."""
    led = pd.read_csv(os.path.join(data_dir, "ledger.csv"))
    led = led[led["close_spread_home"].notna()]
    return dict(zip(led["event_id"].astype(str), -led["close_spread_home"].astype(float)))


# ---------------------------------------------------------------------------
# the preregistered evaluation
# ---------------------------------------------------------------------------

BLEND = ["R0", "ppa", "sr", "xpl"]
PBP_ONLY = ["ppa", "sr", "xpl"]


def _fit(d, cols):
    X = np.c_[np.ones(len(d)), d[cols].values]
    b, *_ = np.linalg.lstsq(X, d["act"].values, rcond=None)
    return b


def _pred(d, cols, b):
    return np.c_[np.ones(len(d)), d[cols].values] @ b


def evaluate(df):
    plan = {2024: [2025], 2025: [2024], 2026: [2024, 2025]}
    ok_each, pooled = True, []
    print("\ntest  n     RMSE R0  blend  pbp-only | SU R0  blend | ATS vs DK close R0  blend")
    for test, fit in plan.items():
        d = df[df.season == test].copy()
        f = df[df.season.isin(fit)]
        if not len(d) or not len(f):
            print(f"{test}: no data"); ok_each = False
            continue
        d["blend"] = _pred(d, BLEND, _fit(f, BLEND))
        d["pbp"] = _pred(d, PBP_ONLY, _fit(f, PBP_ONLY))
        rm = {c: float(np.sqrt(((d[c] - d.act) ** 2).mean())) for c in ("R0", "blend", "pbp")}
        su = {c: float(((d[c] > 0) == (d.act > 0)).mean()) for c in ("R0", "blend")}
        ats = ""
        m = d.dropna(subset=["close"])
        m = m[m.act != m.close]
        if len(m):
            a0 = ((m.R0 > m.close) == (m.act > m.close)).mean()
            ab = ((m.blend > m.close) == (m.act > m.close)).mean()
            ats = f"{a0:.1%}  {ab:.1%} (n={len(m)})"
        print(f"{test}  {len(d):>4}  {rm['R0']:6.2f}  {rm['blend']:6.2f}  {rm['pbp']:6.2f}  | "
              f"{su['R0']:.1%}  {su['blend']:.1%} | {ats}")
        early = d[d.block <= 2]
        if len(early):
            print(f"      early blocks 0-2 (n={len(early)}): R0 {np.sqrt(((early.R0-early.act)**2).mean()):.2f}"
                  f"  blend {np.sqrt(((early.blend-early.act)**2).mean()):.2f}")
        ok_each &= rm["blend"] < rm["R0"]
        pooled.append((d.R0 - d.act) ** 2 - (d.blend - d.act) ** 2)
    p = pd.concat(pooled) if pooled else pd.Series(dtype=float)
    t = float(p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))) if len(p) > 2 else float("nan")
    passed = bool(ok_each and t >= 2.0)
    print(f"\npooled paired t (R0 SE - blend SE): {t:+.2f} over {len(p)} games")
    print(f"PREREGISTERED VERDICT: {'PASS' if passed else 'FAIL'}"
          f"  (blend beats R0 every season: {ok_each}; t >= 2: {t >= 2.0})")
    if passed:
        b = _fit(df, BLEND)
        print("ship weights [1, R0, ppa, sr, xpl] fitted on 2024-2026:",
              [round(float(x), 6) for x in b])
    return passed


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")
    df = walk_forward(data_dir)
    df.to_csv(os.path.join(HERE, "pbp_walkforward.csv"), index=False)
    print(f"{len(df)} out-of-sample FBS-vs-FBS games:", df.groupby("season").size().to_dict())
    evaluate(df)
