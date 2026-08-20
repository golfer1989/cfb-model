#!/usr/bin/env python3
"""
Build both executables into one folder with a SHARED data directory.

    dist/CFB/
        CFB_Report.exe    predictions; locks picks kicking off within 30h
        CFB_Results.exe   scorecard: how those locked picks actually did
        _internal/        shared runtime
        data/             results, schedule, calibration, and the ledger

The shared data folder is the point. Both tools read and write
data/ledger.csv -- CFB_Report.exe locks predictions into it, CFB_Results.exe
grades them. Ship them to separate folders and each gets its own ledger, so
the scorecard would grade nothing and quietly report an empty season.

Run:  python build_exe.py
Out:  dist/CFB/
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = "CFB"
APPS = [
    ("CFB_Report", "launcher.py"),           # forward-looking
    ("CFB_Results", "results_launcher.py"),  # backward-looking
]

# Build on LOCAL disk, then copy the finished product to the project folder.
#
# Building directly into a Google Drive / OneDrive path produces a corrupt
# executable: the cloud client syncs the file while PyInstaller is still
# writing it, and it fails at launch with
#     "Could not load PyInstaller's embedded PKG archive"
# even after being copied elsewhere. The bytes are already wrong on disk.
LOCAL_BUILD = os.path.join(tempfile.gettempdir(), "cfb_report_build")

# imported dynamically or reached only through another module, so
# PyInstaller's static analysis can miss them
HIDDEN = [
    "paths", "ledger",
    "cfb_ratings", "cfb_predict", "pregame", "run_report", "results_report",
    "fetch_espn_data", "fetch_odds", "fetch_team_stats",
    "backtest", "evaluate_vs_market",
    "pandas", "numpy",
]

EXCLUDE = ("matplotlib", "scipy", "tkinter", "PIL", "PyQt5", "IPython",
           "notebook", "pytest", "sqlalchemy")

DATA_FILES = ("real_games.csv", "calibration.json", "team_stats.csv",
              "conferences.json", "odds.csv", "ledger.csv")


def build_one(name, entry, onefile, local_dist):
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile" if onefile else "--onedir",
        "--console", "--name", name,
        "--distpath", local_dist,
        "--workpath", os.path.join(LOCAL_BUILD, "work"),
        "--specpath", os.path.join(LOCAL_BUILD, "spec"),
        "--noconfirm", "--clean",
    ]
    for h in HIDDEN:
        cmd += ["--hidden-import", h]
    for x in EXCLUDE:
        cmd += ["--exclude-module", x]
    cmd.append(os.path.join(BASE, entry))

    print(f"\n--- building {name} from {entry} ---")
    r = subprocess.run(cmd, cwd=BASE)
    if r.returncode != 0:
        return None
    exe = (os.path.join(local_dist, name + ".exe") if onefile
           else os.path.join(local_dist, name, name + ".exe"))
    return exe if os.path.exists(exe) else None


def verify(exe):
    """Actually launch it. A build that produces a broken binary must fail
    here rather than at the user's double-click."""
    try:
        p = subprocess.run([exe], input="\n", capture_output=True, text=True,
                           timeout=60, cwd=os.path.dirname(exe))
        blob = (p.stdout or "") + (p.stderr or "")
        if "Could not load PyInstaller" in blob or "Failed to load Python DLL" in blob:
            print("  FAILED to start: " + blob.strip().splitlines()[0][:150])
            return False
        print("  OK -- it starts.")
        return True
    except subprocess.TimeoutExpired:
        print("  OK -- reached an interactive prompt.")
        return True


def main():
    ap = argparse.ArgumentParser(description="Build both CFB executables.")
    ap.add_argument("--onefile", action="store_true",
                    help="single-file exes (self-extracting; more fragile)")
    args = ap.parse_args()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Run:\n    pip install pyinstaller")
        return 1

    local_dist = os.path.join(LOCAL_BUILD, "dist")
    built = []
    for name, entry in APPS:
        exe = build_one(name, entry, args.onefile, local_dist)
        if not exe:
            print(f"\nBuild failed for {name}.")
            return 1
        print(f"  verifying {name} ...")
        if not verify(exe):
            return 1
        built.append((name, exe))

    # stage both into ONE folder with ONE data directory
    out_dir = os.path.join(BASE, "dist", BUNDLE)
    if os.path.exists(os.path.join(BASE, "dist")):
        shutil.rmtree(os.path.join(BASE, "dist"), ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    for name, exe in built:
        if args.onefile:
            shutil.copy2(exe, os.path.join(out_dir, name + ".exe"))
        else:
            src_dir = os.path.dirname(exe)
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(out_dir, item)
                if os.path.isdir(s):
                    if not os.path.exists(d):
                        shutil.copytree(s, d)
                else:
                    if not os.path.exists(d):
                        shutil.copy2(s, d)

    data_out = os.path.join(out_dir, "data")
    os.makedirs(data_out, exist_ok=True)
    for f in DATA_FILES:
        src = os.path.join(BASE, "data", f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(data_out, f))
    for f in os.listdir(os.path.join(BASE, "data")):
        if f.startswith("schedule_") and f.endswith(".csv"):
            shutil.copy2(os.path.join(BASE, "data", f), os.path.join(data_out, f))
        # the first-year-head-coach adjustment reads these next to the exe;
        # without them it silently no-ops (with a printed note)
        if f.startswith("cfbd_coaches_") and f.endswith(".csv"):
            shutil.copy2(os.path.join(BASE, "data", f), os.path.join(data_out, f))
        # optional FCS training augmentation, if fetched and validated
        if f == "fcs_games.csv":
            shutil.copy2(os.path.join(BASE, "data", f), os.path.join(data_out, f))

    print("\n" + "=" * 62)
    print(f"  Built into {out_dir}")
    print("=" * 62)
    for name, _ in built:
        p = os.path.join(out_dir, name + ".exe")
        if os.path.exists(p):
            print(f"    {name}.exe   {os.path.getsize(p)/1048576:.1f} MB")
    print(f"\n  Shared data folder: {data_out}")
    print("\n  CFB_Report.exe  -> predictions (locks picks within 30h of kickoff)")
    print("  CFB_Results.exe -> scorecard for those locked picks")
    print("\n  Keep both exes and the data folder together -- they share the ledger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
