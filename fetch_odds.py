#!/usr/bin/env python3
"""
Fetch historical betting lines (opening AND closing) from ESPN's odds endpoint.

The closing line is the single most important benchmark in sports modeling:
it is the market's final consensus after all information and money have moved
it, and it is very hard to beat. Without it, any "against the spread" claim is
unverifiable. With it we can measure honestly:
    * how close the model lands to the closing line
    * whether the model beats the closing line ATS
    * whether the model adds information the market lacks

IMPORTANT -- two things this fetcher is careful about:

  1. LIVE ODDS ARE EXCLUDED. ESPN returns both a pregame provider and an
     in-game "Live Odds" provider. The live line reflects the score at the
     time it was sampled, so scoring it against the final result would be
     massive leakage and would produce absurdly good fake accuracy. Any
     provider whose name contains "live" is dropped.

  2. OPEN vs CLOSE ARE KEPT SEPARATE. Line movement between them is real
     signal, and conflating them would misstate what the market knew.

Output: data/odds.csv, one row per game with event_id joinable to real_games.csv.
Fully resumable -- every event's raw payload is cached under data/raw_cache/odds/.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}  # see fetch_espn_data.py -- do not embellish
ODDS_URL = ("http://sports.core.api.espn.com/v2/sports/football/leagues/"
            "college-football/events/{eid}/competitions/{eid}/odds")

from paths import DATA_DIR  # frozen-aware; see paths.py
ODDS_CACHE = os.path.join(DATA_DIR, "raw_cache", "odds")

REQUEST_DELAY = 0.8
COOLDOWN_403 = 60.0
MAX_ATTEMPTS = 3


def _get_json(url, label=""):
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
            time.sleep(REQUEST_DELAY)
            return data
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                return None          # plenty of games simply have no odds
            if e.code in (403, 429):
                wait = COOLDOWN_403 * (attempt + 1)
                print(f"    [rate-limited {label}] cooling {wait:.0f}s", flush=True)
                time.sleep(wait)
            else:
                time.sleep(3.0)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(3.0)
    print(f"    ! give up {label}: {last}", flush=True)
    return None


def _num(x):
    """ESPN mixes '+16.5', '-3', 'EVEN', and floats. Normalize to float."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("+", "")
    if s.upper() in ("EVEN", "PK", "PICK", ""):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def fetch_event_odds(eid, refresh=False):
    os.makedirs(ODDS_CACHE, exist_ok=True)
    path = os.path.join(ODDS_CACHE, f"{eid}.json")
    if os.path.exists(path) and not refresh:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f), True
        except (json.JSONDecodeError, OSError):
            pass
    data = _get_json(ODDS_URL.format(eid=eid), label=str(eid))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data if data is not None else {"items": []}, f)
    return data, False


# Provider preference, best first.
#
# ESPN's own `priority` field is USELESS for this: every provider in every
# payload carries priority 0, so sorting by it is a no-op and simply returns
# whichever book ESPN happened to list first. That was frequently a stale
# regional Caesars record. On Ohio State-Michigan 2024 it gave OSU -10.5 when
# the real closing line was -20.5 -- a ten-point error on a marquee game, and
# the same defect corrupted Michigan-Texas, Alabama-Georgia and others.
#
# ESPN BET is ESPN's own book and is the most consistently populated and
# current; the regional Caesars feeds are the ones observed going stale.
PROVIDER_RANK = [
    "espn bet",
    "draftkings",
    "caesars sportsbook",          # the plain one, not a state-specific feed
    "william hill",
    "unibet",
    "sugarhouse",
]

MAX_PLAUSIBLE_SPREAD = 70.0        # no CFB line approaches this
MIN_PLAUSIBLE_TOTAL = 20.0
MAX_PLAUSIBLE_TOTAL = 110.0


def _provider_rank(name):
    n = str(name or "").lower()
    for i, p in enumerate(PROVIDER_RANK):
        if p in n:
            # a state-specific Caesars feed ranks below the plain one
            return i + (0.5 if "(" in n else 0.0)
    return len(PROVIDER_RANK) + 1


def _spread_from_details(details, home_abbr=None):
    """ESPN's `details` string ("OSU -20.5", "PK") is the line it actually
    displays, and it stays correct in payloads where the structured open/close
    block is malformed. Used as a cross-check, not blindly."""
    if not details:
        return None
    s = str(details).strip()
    if s.upper() in ("PK", "PICK", "EVEN"):
        return 0.0
    parts = s.split()
    if len(parts) < 2:
        return None
    try:
        val = float(parts[-1].replace("+", ""))
    except ValueError:
        return None
    return val, parts[0]


def parse_odds(data, eid):
    """Pull the pregame line out of an odds payload.

    Returns None if the only entries are live in-game lines. Applies three
    guards that were each added after a real corruption was found:
      1. choose the book by a real preference order, not ESPN's dead
         `priority` field
      2. reject implausible magnitudes -- one payload wrote American juice
         (-115) into the point-spread field
      3. cross-check the structured spread against the `details` string and
         the favourite flag, since a few payloads have the home/away legs
         swapped
    """
    if not data:
        return None
    items = [it for it in data.get("items", [])
             if "live" not in str(it.get("provider", {}).get("name", "")).lower()]
    if not items:
        return None
    items.sort(key=lambda it: _provider_rank(it.get("provider", {}).get("name")))
    it = items[0]

    home = it.get("homeTeamOdds", {}) or {}
    away = it.get("awayTeamOdds", {}) or {}

    def leg(side, phase, field):
        return ((side.get(phase) or {}).get(field) or {}).get("american")

    close_total = ((it.get("close") or {}).get("total") or {}).get("american")
    open_total = ((it.get("open") or {}).get("total") or {}).get("american")

    def _sane_spread(v):
        """Reject values that cannot be a point spread.

        One payload wrote American odds into the spread field
        (spread_close_home = -115.0, total_close = -110.0). Without a guard the
        parser copied it faithfully and a -115 'spread' entered the dataset."""
        v = _num(v)
        if v is None or abs(v) > MAX_PLAUSIBLE_SPREAD:
            return None
        return v

    def _sane_total(v):
        v = _num(v)
        if v is None or not (MIN_PLAUSIBLE_TOTAL <= v <= MAX_PLAUSIBLE_TOTAL):
            return None
        return v

    sp_close = _sane_spread(leg(home, "close", "pointSpread"))
    sp_open = _sane_spread(leg(home, "open", "pointSpread"))
    sp_cur = _sane_spread(it.get("spread"))

    # SIGN CHECK. A few payloads have the home and away legs swapped, which
    # inverts the closing spread and silently flips the favourite. The
    # `details` string and the favourite flag are independent of the leg
    # structure, so they can catch it.
    flipped = False
    det = _spread_from_details(it.get("details"))
    fav_home = home.get("favorite")
    if sp_close is not None and det:
        det_val = det[0] if isinstance(det, tuple) else det
        # details quotes the FAVOURITE laying points, so its sign is negative
        # for whoever is favoured; compare magnitudes and the implied side
        if isinstance(fav_home, bool):
            implied_home_close = -abs(det_val) if fav_home else abs(det_val)
            if abs(implied_home_close - sp_close) > 0.6 and \
                    abs(implied_home_close + sp_close) < 0.6:
                # our stored value is the exact negation of what details and
                # the favourite flag agree on -> the legs are swapped
                sp_close = implied_home_close
                sp_open = -sp_open if sp_open is not None else None
                flipped = True

    # last resort: if the structured block gave nothing usable, fall back to
    # the top-level current spread, which is populated far more reliably
    if sp_close is None and sp_cur is not None:
        sp_close = sp_cur

    return {
        "event_id": eid,
        "provider": it.get("provider", {}).get("name"),
        "details": it.get("details"),
        # top-level `spread` is quoted from the HOME side (negative = home favored)
        "spread_current": sp_cur,
        "spread_open_home": sp_open,
        "spread_close_home": sp_close,
        "spread_open_away": (-sp_open if sp_open is not None else None),
        "spread_close_away": (-sp_close if sp_close is not None else None),
        "total_current": _sane_total(it.get("overUnder")),
        "total_open": _sane_total(open_total),
        "total_close": _sane_total(close_total) or _sane_total(it.get("overUnder")),
        "ml_home": _num(home.get("moneyLine")),
        "ml_away": _num(away.get("moneyLine")),
        "home_favorite": fav_home,
        # PROVENANCE, read carefully: this flag marks ONLY rows repaired by
        # the leg-swap detector above. It is False on every row of the
        # current odds.csv -- NOT because the raw feeds were clean, but
        # because the provider re-ranking (ESPN BET first) already routed
        # around the corrupted regional-Caesars payloads before this check
        # ran. The 2024-11 refetch changed 292 closing spreads vs the old
        # file (max error removed: a -115.0 moneyline stored as a spread);
        # those repairs came from provider selection, not from this flag.
        # An all-False column is a statement about THIS mechanism only.
        "leg_swap_corrected": flipped,
    }


def main():
    ap = argparse.ArgumentParser(description="Fetch historical ESPN betting lines.")
    ap.add_argument("--games", default=os.path.join(DATA_DIR, "real_games.csv"))
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "odds.csv"))
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    games = pd.read_csv(args.games)
    if args.seasons:
        games = games[games["season"].isin(args.seasons)]
    eids = games["event_id"].astype(int).tolist()
    if args.limit:
        eids = eids[:args.limit]

    print(f"Fetching odds for {len(eids)} games...", flush=True)
    rows, cached_n, live_n, none_n = [], 0, 0, 0
    for i, eid in enumerate(eids, 1):
        data, from_cache = fetch_event_odds(eid, refresh=args.refresh)
        cached_n += from_cache
        rec = parse_odds(data, eid)
        if rec is None:
            if data and data.get("items"):
                live_n += 1      # had only live-odds entries
            else:
                none_n += 1
        else:
            rows.append(rec)
        if i % 100 == 0 or i == len(eids):
            print(f"  {i}/{len(eids)}  parsed={len(rows)} "
                  f"cache={cached_n} live-only={live_n} none={none_n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}: {len(df)} games with pregame lines "
          f"({len(df)/max(len(eids),1)*100:.1f}% coverage)")
    if len(df):
        print(f"  closing spread present: {df['spread_close_home'].notna().sum()}")
        print(f"  closing total present : {df['total_close'].notna().sum()}")
        print(f"  providers: {df['provider'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
