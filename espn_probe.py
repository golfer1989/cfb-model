#!/usr/bin/env python3
"""ESPN speed probe -- measures how ESPN treats THIS machine's requests.

Why this exists: the daily website build runs on GitHub's cloud servers, and
ESPN rate-limits by IP reputation. The exact same ~20-request refresh that
takes a couple of minutes on a home connection can take hours on a GitHub
runner if ESPN answers 403 and the fetchers sit in their polite retry
cooldowns (75s/150s/225s in fetch_espn_data, 60s/120s in fetch_odds).

This probe fires ONE attempt at each of 12 representative production URLs --
the same three ESPN hosts, same User-Agent, same request spacing -- with no
retries and no cooldowns, so it finishes in under a minute and reports the
raw 403 rate. From that it computes what a real daily build should cost on
this machine, using the production retry arithmetic.

It writes the verdict to data/espn_probe_result.txt (so the workflow can
commit it and it can be read without opening the Actions logs) and prints
the same thing to stdout.

Run cost: 13 requests, ~30-60 seconds. Safe to run anywhere.
"""

import datetime as dt
import json
import os
import socket
import time
import urllib.error
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0"}   # identical to every production fetcher

SB = ("https://site.api.espn.com/apis/site/v2/sports/football/college-football/"
      "scoreboard?dates={year}&seasontype=2&week={week}&groups=80&limit=300")
STATS = ("https://site.web.api.espn.com/apis/common/v3/sports/football/"
         "college-football/statistics/byteam?region=us&lang=en"
         "&contentorigin=espn&season=2025&seasontype=2&limit=200")
ODDS = ("http://sports.core.api.espn.com/v2/sports/football/leagues/"
        "college-football/events/{eid}/competitions/{eid}/odds")

# real 2025 games that have odds payloads
EIDS = [401769071, 401831583, 401778332, 401778333,
        401778334, 401769075, 401769074, 401769076]

REQUESTS = (
    [("scoreboard wk1", SB.format(year=2026, week=1)),
     ("scoreboard wk2", SB.format(year=2026, week=2)),
     ("scoreboard wk3", SB.format(year=2026, week=3)),
     ("team stats 2025", STATS)]
    + [(f"odds {e}", ODDS.format(eid=e)) for e in EIDS]
)

# production retry arithmetic (what a blocked request really costs a build)
BLOCKED_COST_SCOREBOARD = 75 + 150 + 225          # fetch_espn_data, 4 attempts
BLOCKED_COST_ODDS = 60 + 120                      # fetch_odds, 3 attempts
SPACING = 1.0                                     # polite delay between requests


def probe_one(label, url):
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read()
            json.loads(body)                      # confirm it is real JSON
            return label, 200, time.monotonic() - t0, len(body)
    except urllib.error.HTTPError as e:
        return label, e.code, time.monotonic() - t0, 0
    except (urllib.error.URLError, TimeoutError, socket.timeout,
            json.JSONDecodeError) as e:
        return label, f"ERR:{type(e).__name__}", time.monotonic() - t0, 0


def main():
    lines = []
    say = lambda s: (print(s, flush=True), lines.append(s))
    where = "GitHub Actions runner" if os.environ.get("GITHUB_ACTIONS") else "local machine"
    say(f"ESPN speed probe -- {where} -- "
        f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    say(f"{len(REQUESTS)} single-attempt requests, production headers/spacing\n")
    say(f"{'request':<18}{'status':<10}{'seconds':<9}bytes")

    results = []
    for label, url in REQUESTS:
        r = probe_one(label, url)
        results.append(r)
        say(f"{r[0]:<18}{str(r[1]):<10}{r[2]:<9.2f}{r[3]}")
        time.sleep(SPACING)

    ok = [r for r in results if r[1] == 200]
    blocked = [r for r in results if r[1] in (403, 429)]
    other = [r for r in results if r not in ok and r not in blocked]
    n = len(results)
    ok_rate = len(ok) / n
    block_rate = len(blocked) / n
    med = sorted(r[2] for r in ok)[len(ok) // 2] if ok else None

    say(f"\nOK {len(ok)}/{n}   rate-limited {len(blocked)}/{n}   "
        f"other-failed {len(other)}/{n}")
    if med is not None:
        say(f"median successful request: {med:.2f}s")

    # what one production request costs at THIS block rate, retries included
    clean = (med or 1.0) + SPACING
    sb_cost = ok_rate * clean + block_rate * BLOCKED_COST_SCOREBOARD
    od_cost = ok_rate * clean + block_rate * BLOCKED_COST_ODDS

    say("\nExpected fetch time inside a real build on this machine")
    say("(compute -- ratings refit, backtest, sims, checkout -- is extra):")
    say(f"  daily refresh, ~20 scoreboard/stats requests : "
        f"{20 * sb_cost / 60:6.1f} min")
    say(f"  quiet day odds,  ~5 games                    : "
        f"{5 * od_cost / 60:6.1f} min")
    say(f"  full Saturday odds, ~70 games                : "
        f"{70 * od_cost / 60:6.1f} min")
    say(f"  -> full Saturday build fetch total           : "
        f"{(20 * sb_cost + 70 * od_cost) / 60:6.1f} min")

    if block_rate == 0:
        say("\nVERDICT: ESPN is serving this machine normally.")
    elif block_rate < 0.4:
        say("\nVERDICT: partial rate-limiting; builds run but slower.")
    else:
        say("\nVERDICT: heavy rate-limiting from this machine's IP. Builds "
            "spend nearly all their time in retry cooldowns; fix the source "
            "of requests rather than the schedule.")

    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "espn_probe_result.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote data/espn_probe_result.txt")


if __name__ == "__main__":
    main()
