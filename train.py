"""Dispatch to a paper-specific training entry point.

Usage:
    python train.py papers/semiwafernet
    python train.py papers/semiwafernet --config papers/semiwafernet/configs/config.yaml
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "papers"


def _usage() -> None:
    available = sorted(
        p.name for p in PAPERS.iterdir() if p.is_dir() and (p / "train.py").exists()
    )
    print("Usage: python train.py <paper> [args...]")
    print("Available papers with train.py:")
    for name in available:
        print(f"  - {name}")
    print("\nExample:")
    print("  python train.py semiwafernet --config papers/semiwafernet/configs/config.yaml")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _usage()
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    paper = sys.argv[1]
    paper_dir = PAPERS / paper if not paper.startswith("papers/") else ROOT / paper
    train_script = paper_dir / "train.py"
    if not train_script.exists():
        print(f"No train.py found for '{paper}' at {train_script}")
        _usage()
        sys.exit(1)

    # Forward remaining args to the paper entry point
    sys.argv = [str(train_script), *sys.argv[2:]]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
