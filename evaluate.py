"""Dispatch to a paper-specific evaluation entry point.

Usage:
    python evaluate.py papers/semiwafernet
    python evaluate.py semiwafernet --config papers/semiwafernet/configs/config.yaml
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "papers"


def _usage() -> None:
    available = sorted(
        p.name for p in PAPERS.iterdir() if p.is_dir() and (p / "evaluate.py").exists()
    )
    print("Usage: python evaluate.py <paper> [args...]")
    print("Available papers with evaluate.py:")
    for name in available:
        print(f"  - {name}")
    print("\nExample:")
    print("  python evaluate.py semiwafernet --config papers/semiwafernet/configs/config.yaml")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    paper = sys.argv[1]
    paper_dir = PAPERS / paper if not paper.startswith("papers/") else ROOT / paper
    script = paper_dir / "evaluate.py"
    if not script.exists():
        print(f"No evaluate.py found for '{paper}' at {script}")
        _usage()
        sys.exit(1)

    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
