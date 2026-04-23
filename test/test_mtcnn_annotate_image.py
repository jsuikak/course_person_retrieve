"""MTCNN 标记图片接口测试脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.mtcnn_detector import MTCNNDetector


def _find_first_image(root: Path) -> Path:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in exts:
            return file
    raise FileNotFoundError(f"No image found under: {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MTCNN annotate_image interface.")
    parser.add_argument("--image", type=str, default="", help="Input image path. If empty, auto-pick from data/")
    parser.add_argument("--output-dir", type=str, default="outputs/mtcnn_annotated_test", help="Annotated output dir")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else _find_first_image(Path("data"))
    detector = MTCNNDetector()
    output_path = detector.annotate_image(str(image_path), output_dir=args.output_dir)

    print(f"input:  {image_path.name}")
    print(f"output: {Path(output_path).name}")
    print(f"path:   {output_path}")


if __name__ == "__main__":
    main()
