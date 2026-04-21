"""检索系统输入输出工具。

作用：
- 读取图像/视频 manifest（CSV）并转换为结构化记录。
- 提供通用 CSV 读写函数给索引与评估模块复用。

典型用法：
```python
images = load_image_manifest("gallery.csv")
videos = load_video_manifest("videos.csv")
```
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .types import BBox, ImageRecord, VideoRecord


REQUIRED_IMAGE_COLUMNS = {"image_path"}
REQUIRED_VIDEO_COLUMNS = {"video_path"}


def _resolve_path(path_value: str, root_dir: Optional[str]) -> str:
    """将 manifest 中相对路径解析为绝对或可执行路径。"""
    path = Path(path_value)
    if path.is_absolute() or root_dir is None:
        return str(path)
    return str(Path(root_dir) / path)


def load_image_manifest(csv_path: str, root_dir: Optional[str] = None) -> List[ImageRecord]:
    """读取图像 manifest 并转换为 `ImageRecord` 列表。"""
    records: List[ImageRecord] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {csv_path}")
        missing = REQUIRED_IMAGE_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Image manifest missing columns: {sorted(missing)}")

        for row in reader:
            image_path = _resolve_path(row["image_path"], root_dir)
            image_id = row.get("image_id") or Path(image_path).stem
            identity = row.get("identity") or None
            bbox = _parse_bbox(row)
            records.append(ImageRecord(image_path=image_path, image_id=image_id, identity=identity, bbox=bbox))
    return records


def load_video_manifest(csv_path: str, root_dir: Optional[str] = None) -> List[VideoRecord]:
    """读取视频 manifest 并转换为 `VideoRecord` 列表。"""
    records: List[VideoRecord] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {csv_path}")
        missing = REQUIRED_VIDEO_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Video manifest missing columns: {sorted(missing)}")

        for row in reader:
            video_path = _resolve_path(row["video_path"], root_dir)
            video_id = row.get("video_id") or Path(video_path).stem
            identity = row.get("identity") or None
            records.append(VideoRecord(video_path=video_path, video_id=video_id, identity=identity))
    return records


def _parse_bbox(row: Dict[str, str]) -> Optional[BBox]:
    """从 CSV 行中解析可选的人脸框字段。"""
    keys = ["x", "y", "w", "h"]
    if not all(row.get(k) not in (None, "") for k in keys):
        return None
    return BBox(
        x=int(float(row["x"])),
        y=int(float(row["y"])),
        w=int(float(row["w"])),
        h=int(float(row["h"])),
    )


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    """写入字典列表到 CSV 文件。"""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    """读取 CSV 为字典列表。"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
