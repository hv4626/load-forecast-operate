#!/usr/bin/env python3
"""Prepare data, train TiDE, write Operate dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, SOURCE_XLSX
from export_web import write_dashboard
from prepare import prepare
from train import run


def main() -> None:
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE_XLSX
    panel = DATA_DIR / "hourly_panel.csv"
    if not panel.exists():
        prepare(xlsx)
    report = run(panel)
    write_dashboard(report)


if __name__ == "__main__":
    main()
