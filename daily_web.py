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
sys.argv = ["run_report.py", "--days", str(DAYS)]
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
