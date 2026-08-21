#!/usr/bin/env python3
"""
Results report: how the model's LOCKED predictions actually did.

Deliberately separate from run_report.py. That one looks forward and must not
be cluttered with history; this one looks backward and never makes a
prediction. The only thing they share is data/ledger.csv -- this tool reads
and grades it, and never writes a prediction into it.

What it does:
  1. grades any locked prediction whose game has finished, against the CLOSING
     line and total
  2. writes results.html: final scores, who covered, who won the O/U, whether
     the straight-up pick was right, plus the season record and a week-by-week
     breakdown

Usage:
  python results_report.py              # everything graded so far
  python results_report.py --week 3     # one week
  python results_report.py --no-grade   # display only, skip the network pull
"""

import argparse
import datetime as dt
import html
import os
import sys
import webbrowser

import numpy as np
import pandas as pd

import ledger as LG
from paths import DATA_DIR

BREAKEVEN_110 = 0.5238

CSS = """
:root{--bg:#ffffff;--fg:#16191d;--muted:#5b636d;--line:#e3e6ea;--card:#f7f8fa;
--pos:#0a7d3f;--neg:#b3261e;--warn:#8a6100;--accent:#1a4f8a;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0f1216;--fg:#e8eaed;--muted:#9aa3ad;--line:#272c33;--card:#161a20;
--pos:#4ade80;--neg:#f87171;--warn:#fbbf24;--accent:#7fb2ee;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:24px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 8px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 16px;min-width:170px;flex:1}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.card .v{font-size:22px;font-weight:600;margin-top:3px;font-variant-numeric:tabular-nums}
.card .n{font-size:11px;color:var(--muted);margin-top:2px}
.note{background:var(--card);border-left:3px solid var(--warn);padding:12px 14px;
border-radius:4px;margin:16px 0;font-size:13px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:980px;font-variant-numeric:tabular-nums}
th{background:var(--card);text-align:right;padding:9px 10px;font-size:11px;
text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
border-bottom:1px solid var(--line);white-space:nowrap;position:sticky;top:0}
th.l,td.l{text-align:left}
td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--card)}
.game{font-weight:600}
.dim{color:var(--muted);font-size:12px}
.sep{border-left:2px solid var(--line)}
.win{color:var(--pos);font-weight:600}
.loss{color:var(--neg);font-weight:600}
.push{color:var(--muted);font-weight:600}
.good{color:var(--pos)} .bad{color:var(--neg)}
footer{margin-top:24px;color:var(--muted);font-size:12px;line-height:1.7}
"""


def _res(v):
    v = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    cls = {"WIN": "win", "LOSS": "loss", "PUSH": "push"}.get(v)
    return f'<span class="{cls}">{v}</span>' if cls else '<span class="dim">-</span>'


def _rec_card(label, r):
    if not r:
        return ""
    pct = r["pct"] * 100
    roi = r["roi"] * 100
    cls = "good" if r["roi"] > 0 else "bad"
    push = f"-{r['push']}" if r.get("push") else ""
    return f"""<div class="card"><div class="k">{html.escape(label)}</div>
<div class="v">{r['w']}-{r['l']}{push}</div>
<div class="n">{pct:.1f}%  &middot;  ROI <span class="{cls}">{roi:+.1f}%</span>
<br>95% CI [{r['lo']*100:.1f}, {r['hi']*100:.1f}]</div></div>"""


def build_html(graded, summ, week=None):
    su = summ.get("straight_up")
    cards = []
    if su:
        cards.append(f"""<div class="card"><div class="k">Straight up</div>
<div class="v">{su['w']}-{su['n']-su['w']}</div>
<div class="n">{su['pct']*100:.1f}%<br>95% CI [{su['lo']*100:.1f}, {su['hi']*100:.1f}]</div></div>""")
    # the all-picks records now live in the visual strip up top; the cards
    # keep what the strip does not show
    cards.append(_rec_card("ATS &mdash; PLAY/LEAN only", summ.get("spread_recommended")))
    cards.append(_rec_card("O/U &mdash; PLAY/LEAN only", summ.get("total_recommended")))

    acc = ""
    if "margin_rmse" in summ:
        mk = summ.get("market_rmse")
        acc = f"""<div class="card"><div class="k">Margin RMSE</div>
<div class="v">{summ['margin_rmse']:.2f}</div>
<div class="n">{'market ' + format(mk, '.2f') if mk else 'model'}
{'&mdash; ' + ('model closer' if mk and summ['margin_rmse'] < mk else 'market closer') if mk else ''}</div></div>"""
    cards.append(acc)

    head = """<tr>
<th class="l">Game</th><th class="l">Date</th>
<th>Final</th><th>We projected</th><th>Winner</th>
<th class="sep">Close spread</th><th>Our side</th><th>ATS</th>
<th class="sep">Close O/U</th><th>Our side</th><th>O/U</th>
<th class="sep">Call</th>
</tr>"""

    body = []
    for _, r in graded.iterrows():
        sp, tot = r.get("close_spread_home"), r.get("close_total")
        pick = r.get("spread_pick")
        if isinstance(pick, str) and pick:
            side = r["home_team"] if pick.upper().startswith("HOME") else r["away_team"]
            side_s = html.escape(str(side))
        else:
            side_s = '<span class="dim">-</span>'
        su_v = r.get("su_correct")
        su_cell = _res("WIN" if su_v == 1 else ("LOSS" if su_v == 0 else None))
        body.append(f"""<tr>
<td class="l game">{html.escape(str(r['away_team']))} at {html.escape(str(r['home_team']))}</td>
<td class="l dim">{html.escape(str(r.get('kickoff', ''))[:10])}</td>
<td><b>{r['final_away']:.0f} &ndash; {r['final_home']:.0f}</b></td>
<td class="dim">{r['exp_away']:.1f} &ndash; {r['exp_home']:.1f}</td>
<td>{su_cell}</td>
<td class="sep">{'-' if pd.isna(sp) else html.escape(str(r['home_team'])) + f' {sp:+g}'}</td>
<td>{side_s}</td><td>{_res(r.get('spread_result'))}</td>
<td class="sep">{'-' if pd.isna(tot) else f'{tot:g}'}</td>
<td>{html.escape(str(r.get('total_pick') or '-'))}</td><td>{_res(r.get('total_result'))}</td>
<td class="dim">{html.escape(str(r.get('spread_call') or ''))}</td>
</tr>""")

    # week-by-week
    wk_rows = []
    if "week" in graded.columns:
        for wk, sub in graded.groupby("week"):
            sr = sub["spread_result"].dropna()
            w = int((sr == "WIN").sum()); l = int((sr == "LOSS").sum())
            tr = sub["total_result"].dropna()
            tw = int((tr == "WIN").sum()); tl = int((tr == "LOSS").sum())
            sux = sub["su_correct"].dropna()
            wk_rows.append(f"""<tr><td class="l">Week {int(wk)}</td>
<td>{len(sub)}</td><td>{int(sux.sum())}-{len(sux)-int(sux.sum())}</td>
<td>{w}-{l}</td><td>{tw}-{tl}</td></tr>""")

    weekly = ""
    if wk_rows:
        weekly = f"""<h2>By week</h2><div class="wrap"><table>
<tr><th class="l">Week</th><th>Games</th><th>Straight up</th><th>ATS</th><th>O/U</th></tr>
{''.join(wk_rows)}</table></div>"""

    n_bets = summ.get("spread_all", {}).get("n", 0)
    caveat = ""
    if 0 < n_bets < 700:
        caveat = f"""<div class="note">
<b>{n_bets} graded bets is not enough to conclude anything.</b> Detecting a
2-point against-the-spread edge reliably needs roughly 700 bets; confirming you
beat &minus;110 juice needs about 3,500 &mdash; several seasons. A hot or cold
stretch here is overwhelmingly likely to be variance. Read this as bookkeeping,
not as proof either way.</div>"""

    scope = f"Week {week}" if week else "Season to date"
    return f"""<title>CFB Results</title>
<style>{CSS}</style>
{LG.site_nav_html('results')}
<h1>College Football Model &mdash; Results</h1>
{LG.record_strip_html(summ)}
<div class="sub">{scope} &middot; {len(graded)} completed games &middot;
generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="cards">{''.join(c for c in cards if c)}</div>
{caveat}

<h2>Game by game</h2>
<div class="wrap"><table>{head}{''.join(body)}</table></div>
{weekly}

<footer>
Every row here was <b>locked before kickoff</b> and cannot be edited afterwards.
Spread and O/U results are graded against the <b>closing</b> line; the line you
would actually have bet at lock time is stored separately in
<code>data/ledger.csv</code> if you want to compare.
Pushes are excluded from win percentages rather than counted as wins.
ROI assumes flat stakes at &minus;110, where break-even is
{BREAKEVEN_110:.2%} &mdash; not 50%.
</footer>"""


def main():
    ap = argparse.ArgumentParser(description="Show how locked predictions actually did.")
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--no-grade", action="store_true",
                    help="skip fetching results; just display what is already graded")
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    if not args.no_grade:
        print("Checking for newly completed games...")
        try:
            LG.grade()
        except Exception as e:  # noqa: BLE001
            print(f"  (could not fetch results: {type(e).__name__}: {str(e)[:70]})")
            print("  showing what is already on record")

    led = LG._load()
    graded = led[led["graded_at"].notna()].copy()
    if args.season is not None:
        graded = graded[graded["season"] == args.season]
    if args.week is not None:
        graded = graded[graded["week"] == args.week]

    if not len(graded):
        pending = int(led["graded_at"].isna().sum())
        print("\nNo completed games on record yet.")
        if pending:
            print(f"  {pending} prediction(s) are locked and waiting for their games "
                  f"to finish.")
        else:
            print("  Run CFB_Report.exe first -- it locks predictions within 30 hours")
            print("  of kickoff, and those are what get graded here.")
        # still write the page (the website needs a Results tab on day zero):
        # empty strip + an honest explanation instead of a dead link
        out = args.out or os.path.join(DATA_DIR, "results.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"""<title>CFB Model &mdash; Results</title>
<style>{CSS}</style>
{LG.site_nav_html('results')}
<h1>College Football Model &mdash; Results</h1>
{LG.record_strip_html(LG.summarize(led))}
<div class="sub">No completed games on record yet.
{f"{pending} locked prediction(s) are waiting for their games to finish." if pending else
"Picks lock automatically within 30 hours of kickoff; graded results appear here the morning after they play."}</div>""")
        print(f"Wrote {out} (empty-state page)")
        return 0

    graded = graded.sort_values("kickoff", ascending=False)
    summ = LG.summarize(led if args.week is None else graded)

    out = args.out or os.path.join(DATA_DIR, "results.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(graded, summ, week=args.week))

    print("\n" + LG.format_summary(summ))
    print(f"\nWrote {out}")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
