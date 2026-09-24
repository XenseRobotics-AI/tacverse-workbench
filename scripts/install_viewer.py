#!/usr/bin/env python3
"""Materialize the viewer runtime copy and install its Bun dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import viewer_service as vsvc

SOURCE_DIR = vsvc.VIEWER_SOURCE_DIR
RUNTIME_DIR = vsvc.VIEWER_RUNTIME_DIR


def run(cmd: list[str], cwd: Path = ROOT, check: bool = True) -> int:
    print("+", " ".join(cmd), f"(cwd={cwd})", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if check and proc.returncode:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> int:
    run(["git", "submodule", "update", "--init", "--recursive",
         "third_party/lerobot_viewer"])
    try:
        viewer_dir = vsvc.prepare_viewer_runtime(SOURCE_DIR, RUNTIME_DIR)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    bun = vsvc.find_bun()
    if not bun:
        print("Bun is required for the viewer. Install it from https://bun.sh/",
              file=sys.stderr)
        return 1

    rc = run([bun, "install"], cwd=viewer_dir, check=False)
    if rc:
        print(
            "bun install failed; clearing Bun cache and retrying via npmjs registry.",
            flush=True)
        run([bun, "pm", "cache", "rm"], cwd=viewer_dir, check=False)
        rc = run([bun, "install", "--registry", "https://registry.npmjs.org"],
                 cwd=viewer_dir, check=False)
    if rc:
        return rc

    print(f"Viewer ready: {viewer_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
