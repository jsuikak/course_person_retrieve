#!/usr/bin/env python3
"""Create runtime folder structure for retrieval workflow."""

from __future__ import annotations

import argparse
from pathlib import Path


RUNTIME_DIRS = [
    "data_runtime/query",
    "data_runtime/gallery/images",
    "data_runtime/gallery/videos",
    "indexes/image_index",
    "indexes/video_index",
    "outputs/retrieval",
    "outputs/annotate",
]


def create_runtime_folders(project_root: str = ".") -> None:
    root = Path(project_root).resolve()
    for rel in RUNTIME_DIRS:
        target = root / rel
        existed = target.exists()
        target.mkdir(parents=True, exist_ok=True)
        status = "exists" if existed else "created"
        print(f"[{status}] {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create runtime folder structure.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory. Defaults to current directory.",
    )
    args = parser.parse_args()
    create_runtime_folders(project_root=args.project_root)


if __name__ == "__main__":
    main()

