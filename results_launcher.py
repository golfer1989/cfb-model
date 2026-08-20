#!/usr/bin/env python3
"""
Double-click entry point for the RESULTS report (CFB_Results.exe).

This one only looks backward. It grades locked predictions against final
scores and closing lines and opens a scorecard. It never makes a prediction
and never writes one into the ledger, so running it can't affect what the
prediction tool recorded.

CFB_Report.exe  -> what the model thinks will happen (locks picks within 30h)
CFB_Results.exe -> what actually happened to those picks
"""

import os
import sys
import traceback

if getattr(sys, "frozen", False):
    BASE = os.path.dirname(sys.executable)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

BANNER = r"""
+------------------------------------------------------------+
|        COLLEGE FOOTBALL MODEL  --  RESULTS SCORECARD        |
+------------------------------------------------------------+
"""


def _pause():
    try:
        input("\n  Press Enter to close...")
    except (EOFError, OSError):
        pass


def ask_scope():
    """Whole season, or a single week."""
    print("What would you like to see?")
    print("  [Enter] = whole season to date   |   a number = that week only")
    try:
        raw = input("  week > ").strip()
    except (EOFError, OSError):
        return None
    if not raw:
        return None
    try:
        return max(1, min(int(raw), 20))
    except ValueError:
        print("  (not a number -- showing the whole season)")
        return None


def main():
    print(BANNER)

    ledger = os.path.join(BASE, "data", "ledger.csv")
    if not os.path.exists(ledger):
        print("  No prediction ledger yet.\n")
        print("  Run CFB_Report.exe first. It locks a prediction for every game")
        print("  kicking off within 30 hours, and those locked picks are what")
        print("  this scorecard grades once the games finish.")
        _pause()
        return 0

    week = ask_scope()
    print(f"\n  Building {'week ' + str(week) if week else 'season'} scorecard.")
    print("  Fetching final scores and closing lines...\n")

    import results_report
    argv = ["results_report.py", "--open"]
    if week is not None:
        argv += ["--week", str(week)]
    old = sys.argv
    try:
        sys.argv = argv
        rc = results_report.main()
    finally:
        sys.argv = old

    print("\n  Done. The scorecard should have opened in your browser.")
    print("  It is also saved at: data\\results.html")
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
