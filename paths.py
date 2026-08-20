#!/usr/bin/env python3
"""
Single source of truth for where the data directory lives.

This exists because `os.path.dirname(__file__)` is WRONG inside a PyInstaller
bundle. When frozen, every bundled module reports a path inside the temporary
extraction folder (`_internal/`), so a DATA_DIR derived from `__file__`
resolves to `_internal/data` -- which does not exist. The data ships next to
the executable, not inside it, precisely so it can be refreshed without
rebuilding.

Import DATA_DIR from here rather than recomputing it per module.
"""

import os
import sys


def app_dir():
    """The folder the user actually launched from.

    Frozen  -> the directory containing the .exe
    Source  -> the project directory containing this file
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def data_dir():
    d = os.path.join(app_dir(), "data")
    os.makedirs(d, exist_ok=True)
    return d


DATA_DIR = data_dir()


def current_cfb_season(today=None):
    """The season a given date belongs to.

    A college football season is labelled by the calendar year it STARTS in,
    but runs into January: the January 2027 playoff belongs to the 2026
    season. So anything before August rolls back to the previous year.

    This exists so the tool does not need editing every August. Hardcoding
    2026 would have quietly produced wrong recency weights on 2027-08-01 --
    every 2026 game would have been treated as "last season" and shrunk to
    20%, while a season with no games yet got full weight.
    """
    import datetime as _dt
    d = today or _dt.date.today()
    return d.year if d.month >= 8 else d.year - 1
