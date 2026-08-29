#!/usr/bin/env python3
"""Headless daily runner for the website.

Refreshes every model input (completed results, team stats, simulator
recalibration, market-edge refit -- same steps the desktop exe runs), builds
the report for TODAY'S games only, and stages it into docs/index.html, which
GitHub Pages serves as the site.

Run by .github/workflows/daily.yml at 8:00 AM America/Chicago.
REPORT_DAYS env var widens the window (default 1 = today's slate).
"""
import datetime as dt
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
DOCS = os.path.join(HERE, "docs")
DAYS = int(os.environ.get("REPORT_DAYS", "1"))

sys.argv = ["daily_web.py"]
import launcher as L
import run_report as RR
from paths import DATA_DIR

print(f"=== daily_web: refreshing inputs (window {DAYS} day(s)) ===")
L.refresh_all(DAYS)

report = os.path.join(DATA_DIR, "report.html")
if os.path.exists(report):
    os.remove(report)  # so a fresh write is detectable

print("=== daily_web: building the report ===")
# Lock window: on the website, builds fire ~45-75 min before each kickoff
# wave, so a short window means picks enter the permanent ledger WITH the
# final injury/QB information -- exactly what the ledger's own docstring
# wants. Earlier runs of the day show those games as previews instead.
lock_hours = os.environ.get("REPORT_LOCK_HOURS", "3")
sys.argv = ["run_report.py", "--days", str(DAYS), "--lock-hours", lock_hours]
RR.main()

os.makedirs(DOCS, exist_ok=True)
out = os.path.join(DOCS, "index.html")
today = dt.datetime.now().strftime("%A, %B %d, %Y")

if os.path.exists(report):
    shutil.copyfile(report, out)
    print(f"Published report -> docs/index.html")
else:
    # off day: publish an honest placeholder instead of yesterday's slate
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><meta charset="utf-8">
<title>CFB Model Report</title>
<style>body{{font:16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;
max-width:640px;margin:15vh auto;padding:0 24px;color:#16191d}}
h1{{font-size:22px}} .dim{{color:#5b636d}}</style>
<h1>No college football games today</h1>
<p class="dim">Checked {today}. The model found no games kicking off in the
next {DAYS} day(s). This page rebuilds automatically every morning at
8:00 AM Central; the next slate will appear here the morning it plays.</p>""")
    print("No games today -> published placeholder page")

# ---- Results tab: grade + publish the season-record page -------------------
print("=== daily_web: building the results page ===")
sys.argv = ["results_report.py"]           # grades any newly finished games
import results_report as RES
try:
    RES.main()
except Exception as e:  # noqa: BLE001
    print(f"  results page failed ({type(e).__name__}: {str(e)[:80]}) -- "
          f"keeping the previous one")
res = os.path.join(DATA_DIR, "results.html")
if os.path.exists(res):
    shutil.copyfile(res, os.path.join(DOCS, "results.html"))
    print("Published season results -> docs/results.html")

# ---- Schedule tab: the full upcoming slate, grouped by day ------------------
# (every game, including unpriced ones the report deliberately hides)
import ledger as LG

_PAGE_CSS = """:root{--bg:#ffffff;--fg:#16191d;--muted:#5b636d;--line:#e3e6ea;
--card:#f7f8fa;--pos:#0a7d3f;--neg:#b3261e;--warn:#8a6100;--accent:#1a4f8a;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#0f1216;--fg:#e8eaed;--muted:#9aa3ad;--line:#272c33;--card:#161a20;
--pos:#4ade80;--neg:#f87171;--warn:#fbbf24;--accent:#7fb2ee;}}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif;padding:22px 26px 60px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);font-size:12.5px;margin:0 0 18px}
h2{font-size:15px;margin:24px 0 6px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
td{padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--card)}
.dim{color:var(--muted);font-size:12px}
.game{font-weight:600}
.wrap{overflow-x:auto}
.live{color:var(--pos);font-weight:700}
.fin{color:var(--muted);font-weight:600}"""


def build_schedule_page():
    import html as H
    import pandas as pd
    from zoneinfo import ZoneInfo
    CT = ZoneInfo("America/Chicago")
    from paths import current_cfb_season
    season = current_cfb_season()
    path = os.path.join(DATA_DIR, f"schedule_{season}.csv")
    if not os.path.exists(path):
        return
    s = pd.read_csv(path)
    s = s[s["home_team"].notna() & (s["home_team"] != "TBD")
          & (s["away_team"] != "TBD")]
    kick = pd.to_datetime(s["kickoff_utc"], errors="coerce", utc=True)
    now = dt.datetime.now(dt.timezone.utc)
    s = s[kick >= now - dt.timedelta(hours=6)].copy()
    s["_k"] = kick[kick >= now - dt.timedelta(hours=6)]
    s = s.sort_values("_k")
    parts = []
    for day, sub in s.groupby(s["_k"].dt.tz_convert(CT).dt.date, sort=True):
        rows = []
        for _, r in sub.iterrows():
            t = r["_k"].tz_convert(CT).strftime("%-I:%M %p") if pd.notna(r["_k"]) else ""
            rows.append(
                f'<tr><td class="dim">{t} CT</td>'
                f'<td class="game">{H.escape(str(r["away_team"]))} at '
                f'{H.escape(str(r["home_team"]))}</td>'
                f'<td class="dim">Week {int(r["week"])}</td></tr>')
        label = day.strftime("%A, %B %d")
        parts.append(f"<h2>{label}</h2>"
                     f'<div class="wrap"><table>{"".join(rows)}</table></div>')
    page = (f"<title>CFB Schedule</title>\n<style>{_PAGE_CSS}</style>\n"
            + LG.site_nav_html("schedule")
            + "<h1>Schedule</h1>"
            + f'<div class="sub">{len(s)} upcoming games &middot; times are US Central '
            + f'&middot; updated {dt.datetime.now(CT).strftime("%Y-%m-%d %-I:%M %p")}</div>'
            + "".join(parts))
    with open(os.path.join(DOCS, "schedule.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print("Published schedule -> docs/schedule.html")


def build_live_page():
    """Static page whose JavaScript pulls live scores in the VIEWER'S browser.

    The builds themselves can't do live -- the site only rebuilds around
    kickoffs -- and GitHub's servers are rate-limited by ESPN anyway. But the
    viewer's own browser is neither: ESPN's public scoreboard endpoint allows
    cross-origin reads, so the page fetches FBS+FCS scores client-side and
    refreshes itself every 60 seconds."""
    nav = LG.site_nav_html("live")
    page = f"""<title>CFB Live Scores</title>
<style>{_PAGE_CSS}</style>
{nav}
<h1>Live Scores</h1>
<div class="sub" id="stamp">loading&hellip;</div>
<div id="out" class="wrap"></div>
<script>
const HOSTS=[80,81].map(g=>`https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=${{g}}&limit=300`);
function fmt(ev){{
  try{{
    const c=ev.competitions[0];
    const home=c.competitors.find(x=>x.homeAway==='home');
    const away=c.competitors.find(x=>x.homeAway==='away');
    const st=ev.status.type;
    let cls='dim', stat=st.shortDetail||st.description;
    if(st.state==='in'){{cls='live';}}
    else if(st.state==='post'){{cls='fin'; stat='Final'+(st.shortDetail&&st.shortDetail.includes('OT')?' (OT)':'');}}
    else{{stat=new Date(ev.date).toLocaleTimeString('en-US',{{hour:'numeric',minute:'2-digit',timeZone:'America/Chicago'}})+' CT';}}
    const score=(st.state==='pre')?'':`${{away.score}} – ${{home.score}}`;
    return {{state:st.state, html:`<tr><td class="game">${{away.team.location}} at ${{home.team.location}}</td>`+
      `<td><b>${{score}}</b></td><td class="${{cls}}">${{stat}}</td></tr>`}};
  }}catch(e){{return null;}}
}}
async function load(){{
  try{{
    const seen=new Set(); const rows={{in:[],pre:[],post:[]}};
    for(const u of HOSTS){{
      const r=await fetch(u); const j=await r.json();
      for(const ev of (j.events||[])){{
        if(seen.has(ev.id)) continue; seen.add(ev.id);
        const f=fmt(ev); if(f) rows[f.state].push(f.html);
      }}
    }}
    const blocks=[];
    if(rows.in.length) blocks.push('<h2>In progress</h2><table>'+rows.in.join('')+'</table>');
    if(rows.pre.length) blocks.push('<h2>Upcoming today</h2><table>'+rows.pre.join('')+'</table>');
    if(rows.post.length) blocks.push('<h2>Final</h2><table>'+rows.post.join('')+'</table>');
    document.getElementById('out').innerHTML=blocks.join('')||'<p class="dim">No college football games today.</p>';
    document.getElementById('stamp').textContent='FBS + FCS · refreshes every 60 seconds · updated '+new Date().toLocaleTimeString('en-US',{{timeZone:'America/Chicago'}})+' CT';
  }}catch(e){{
    document.getElementById('stamp').textContent='Could not reach the scores feed — retrying…';
  }}
}}
load(); setInterval(load, 60000);
</script>"""
    with open(os.path.join(DOCS, "live.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print("Published live scores page -> docs/live.html")


try:
    build_schedule_page()
except Exception as e:  # noqa: BLE001
    print(f"  schedule page failed ({type(e).__name__}: {str(e)[:80]})")
try:
    build_live_page()
except Exception as e:  # noqa: BLE001
    print(f"  live page failed ({type(e).__name__}: {str(e)[:80]})")
