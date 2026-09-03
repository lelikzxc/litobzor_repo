"""Dispatch to a paper-specific prediction entry point.

Usage:
    python predict.py <paper> [args...]

If the paper has no predict.py, prints available entry points.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "papers"


def _usage() -> None:
    available = sorted(
        p.name for p in PAPERS.iterdir() if p.is_dir() and (p / "predict.py").exists()
    )
    print("Usage: python predict.py <paper> [args...]")
    if available:
        print("Available papers with predict.py:")
        for name in available:
            print(f"  - {name}")
    else:
        print("No papers currently expose predict.py; use paper-specific demos or evaluate.py.")
    print("\nExample:")
    print("  python predict.py <paper> --help")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    paper = sys.argv[1]
    paper_dir = PAPERS / paper if not paper.startswith("papers/") else ROOT / paper
    script = paper_dir / "predict.py"
    if not script.exists():
        print(f"No predict.py found for '{paper}' at {script}")
        _usage()
        sys.exit(1)

    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
