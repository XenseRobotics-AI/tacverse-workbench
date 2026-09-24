#!/usr/bin/env python3
"""Install Python dependencies and the vendored viewer for full Workbench use."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), f"(cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_optional(cmd: list[str], cwd: Path = ROOT) -> int:
    print("+", " ".join(cmd), f"(cwd={cwd})", flush=True)
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def install_requirements(req: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-r", req]
    if run_optional(cmd) == 0:
        return
    print("pip install failed; retrying via official PyPI.", flush=True)
    run(cmd + ["--index-url", "https://pypi.org/simple"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install TacVerse Workbench dependencies.")
    parser.add_argument(
        "--skip-lerobot", action="store_true",
        help="Install the GUI/checker dependencies but skip LeRobot dataset ops.")
    parser.add_argument(
        "--skip-viewer", action="store_true",
        help="Skip viewer runtime preparation and Bun dependencies.")
    args = parser.parse_args()

    req = "requirements.txt" if args.skip_lerobot else "requirements-full.txt"
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    install_requirements(req)

    if not args.skip_viewer:
        run([sys.executable, "scripts/install_viewer.py"])

    print("Install complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
