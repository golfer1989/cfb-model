#!/usr/bin/env python3
"""
Double-click entry point for the CFB model report.

Packaged into CFB_Report.exe by build_exe.py. Running it:
  1. refreshes only the schedule weeks that overlap the next few days
     (not the whole season -- that would be ~22 needless requests)
  2. pulls the live market line, roster and injury report for each game
  3. runs the Monte Carlo model
  4. writes report.html and opens it in the browser

It prompts for the day window so the same executable covers "today's slate"
and "the coming week" without command-line arguments.
"""

import os
import sys
import traceback

# When frozen by PyInstaller the working directory is wherever the user
# double-clicked, so anchor every relative path to the executable's folder.
if getattr(sys, "frozen", False):
    BASE = os.path.dirname(sys.executable)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)


BANNER = r"""
+------------------------------------------------------------+
|          COLLEGE FOOTBALL MODEL  --  BETTING REPORT         |
+------------------------------------------------------------+
"""


def _pause():
    """Keep the console window open after a double-click, but do not crash
    when stdin is not a terminal (piped input, scheduled task)."""
    try:
        input("\n  Press Enter to close...")
    except (EOFError, OSError):
        pass


def refresh_all(days):
    """Bring every input up to date BEFORE the report is calculated.

    Order matters. Each step feeds the next, and the ratings must be fitted on
    the freshest completed results before any game is priced:

      1. completed results   -> new final scores enter the ratings
      2. team season stats   -> style/efficiency splits for the matchup readout
      3. re-backtest          -> re-derives simulator sigma and the totals
                                 calibration from the updated results
      4. refit market edge    -> re-measures how much of the model's
                                 disagreement with the closing line is real

    Steps 3 and 4 are what stop the report quoting stale calibration. Any step
    may fail (network, rate limit); a failure is reported and the run
    continues on the last good data rather than aborting, because a report
    built on slightly stale inputs beats no report ten minutes before kickoff.
    """
    import io
    import contextlib

    steps = [
        ("Refreshing completed game results", _refresh_results),
        ("Refreshing team season statistics", _refresh_team_stats),
        ("Recalibrating simulator from latest results", _refresh_backtest),
        ("Re-measuring edge vs the betting market", _refresh_market_edge),
    ]
    print("\n  Updating model inputs before calculating the report:")
    for label, fn in steps:
        print(f"    - {label} ... ", end="", flush=True)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                note = fn()
            print(note or "done")
        except Exception as e:  # noqa: BLE001
            print(f"SKIPPED ({type(e).__name__}: {str(e)[:60]})")
            print("      using last good data for this input")


def _refresh_results():
    """Pull any newly-completed games into real_games.csv.

    Two sources for the same rows:
      * website builds (DATA_SOURCE=cfbd): one CFBD call covers the whole
        season -- GitHub's servers are rate-limited to death by ESPN, and
        CFBD is not (see fetch_cfbd_live's header for the measurements).
        The merge below is IDENTICAL either way; only the fetch differs,
        and the CFBD path refuses to merge until its selfcheck has proven
        the event-id space matches.
      * desktop (default): the ESPN weekly scoreboards, unchanged.
    """
    import pandas as pd
    from paths import DATA_DIR

    from paths import current_cfb_season
    season = current_cfb_season()
    path = os.path.join(DATA_DIR, "real_games.csv")

    import fetch_cfbd_live as CF
    if CF.enabled():
        if not CF.selfcheck():
            return "CFBD selfcheck FAILED -- not merging; see data/cfbd_selfcheck.txt"
        rows = CF.season_games(season)
    else:
        import fetch_espn_data as F
        conf = F.fetch_conference_map(season, os.path.join(DATA_DIR, "conferences.json"))
        rows = []
        for wk in range(1, 18):
            try:
                raw, _ = F.fetch_week_cached(season, 2, wk, refresh=True)
            except Exception:  # noqa: BLE001
                continue
            rows += F.extract_games(raw, conf)
    if not rows:
        return "no new games"
    fresh = pd.DataFrame(rows).drop_duplicates("event_id")
    done = fresh[fresh["status"] == "STATUS_FINAL"].dropna(
        subset=["home_score", "away_score"])
    if not len(done):
        return "season not started; nothing to add"

    base = pd.read_csv(path)
    # event_id must be compared as STRINGS on both sides: the CSV reads back
    # as int64 while both fetchers produce strings, and an unnormalized isin()
    # matches nothing -- every already-stored final would be appended again on
    # every run, silently double-counting games in the ratings. Same defect
    # class as the schedule-doubling bug documented in run_report's
    # refresh_schedule_window; the dedupe is belt-and-braces on top.
    base["event_id"] = base["event_id"].astype(str)
    done = done.copy()
    done["event_id"] = done["event_id"].astype(str)
    keep = base[~base["event_id"].isin(done["event_id"])]
    cols = [c for c in base.columns if c in done.columns]
    merged = pd.concat([keep, done[cols]], ignore_index=True)
    merged = merged.drop_duplicates("event_id", keep="last").sort_values("date")
    merged.to_csv(path, index=False)
    added = len(merged) - len(base)
    return f"{len(done)} final {season} games ({added} new)"


def _refresh_stat_seasons():
    """Current season plus the one before it -- the two that can still change."""
    from paths import current_cfb_season
    cur = current_cfb_season()
    return (cur - 1, cur)


def _refresh_team_stats():
    """Refresh the CURRENT season's team stats and MERGE them into the file.

    This used to rebuild data/team_stats.csv from scratch out of the two
    seasons it refetches, which silently deleted every earlier season on every
    run. It destroyed 2023 and 2024 in production on 2026-08-18, taking the
    file from 403 rows to 136; only the raw_cache JSONs made recovery
    possible, which was luck rather than design.

    History matters here even though the ratings never read this file: prior
    seasons are what any year-over-year stability or prior-season-strength
    check is measured on, and those checks cannot be re-run against data that
    no longer exists. A refresh step must never be able to lose data it did
    not fetch.

    So: replace only the (season, team_id) rows actually refetched, keep every
    other row untouched, and if the fetch yields nothing leave the file
    completely alone rather than truncating it.
    """
    import fetch_team_stats as T
    from paths import DATA_DIR
    import pandas as pd

    # Website builds skip this step entirely: team_stats.csv feeds only the
    # desktop matchup/style readout (run_report never loads it), CFBD has no
    # equivalent of ESPN's byteam endpoint, and ESPN refuses GitHub's IPs --
    # so on the website this fetch could only ever waste minutes to fail.
    import fetch_cfbd_live as CF
    if CF.enabled():
        return "skipped on website builds (desktop-only input; report unaffected)"

    path = os.path.join(DATA_DIR, "team_stats.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    rows = []
    got = []
    for s in _refresh_stat_seasons():
        try:
            parsed = T.parse_season(
                T.fetch_season(s, refresh=(s == max(_refresh_stat_seasons()))), s)
        except Exception:  # noqa: BLE001
            continue
        if parsed:
            rows += parsed
            got.append(s)

    if not rows:
        # Nothing came back. Writing here would be pure data loss.
        return (f"unavailable; kept {len(old)} existing rows"
                if len(old) else "unavailable")

    fresh = T.add_derived(pd.DataFrame(rows))

    if len(old) and {"season", "team_id"} <= set(old.columns):
        # drop only the seasons we actually just refetched
        keep = old[~old["season"].isin(got)]
        merged = pd.concat([keep, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["season", "team_id"], keep="last")
    else:
        merged = fresh

    merged = merged.sort_values(["season", "team_id"]).reset_index(drop=True)

    # Final guard: a refresh may only ever ADD seasons, never remove one.
    if len(old) and "season" in old.columns:
        lost = set(old["season"].unique()) - set(merged["season"].unique())
        if lost:
            return f"refresh would drop seasons {sorted(lost)}; file left unchanged"

    merged.to_csv(path, index=False)
    seasons = sorted(int(s) for s in merged["season"].unique())
    return f"{len(merged)} team-seasons across {seasons}"


def _refresh_backtest():
    import backtest as B
    _run_backtest(B)
    return "sigma + totals calibration updated"


def _run_backtest(B):
    old = sys.argv
    try:
        sys.argv = ["backtest.py", "--quiet"]
        B.main()
    finally:
        sys.argv = old


def _refresh_market_edge():
    from paths import DATA_DIR
    if not os.path.exists(os.path.join(DATA_DIR, "odds.csv")):
        return "no historical odds on disk; edge stays uncalibrated"
    import evaluate_vs_market as EV
    old = sys.argv
    try:
        sys.argv = ["evaluate_vs_market.py"]
        EV.main()
    finally:
        sys.argv = old
    return "edge shrinkage refitted"


def ask_days(default=5):
    print("How many days ahead should the report cover?")
    print("  [Enter] = 5 days   |   1 = today/tomorrow only   |   7 = full week")
    try:
        raw = input("  days > ").strip()
    except (EOFError, OSError):
        return default
    if not raw:
        return default
    try:
        n = int(raw)
        return max(1, min(n, 21))
    except ValueError:
        print(f"  (not a number -- using {default})")
        return default


def main():
    print(BANNER)

    data_dir = os.path.join(BASE, "data")
    needed = ["real_games.csv", "calibration.json"]
    missing = [f for f in needed if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print("  ERROR: missing required data files in ./data:")
        for f in missing:
            print(f"    - {f}")
        print("\n  Run these once from the project folder to build them:")
        print("    python fetch_espn_data.py")
        print("    python backtest.py")
        _pause()
        return 1

    days = ask_days()
    print(f"\n  Building report for the next {days} day(s).")

    refresh_all(days)

    print("\n  Pulling live lines, rosters and injury reports per game...\n")
    import run_report
    argv = ["run_report.py", "--days", str(days), "--open"]
    old = sys.argv
    try:
        sys.argv = argv
        rc = run_report.main()
    finally:
        sys.argv = old

    print("\n  Done. The report should have opened in your browser.")
    print("  It is also saved at: data\\report.html")
    _pause()
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(1)
    except Exception:
        print("\n  Something went wrong:\n")
        traceback.print_exc()
        _pause()
        sys.exit(1)
