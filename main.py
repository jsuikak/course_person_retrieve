"""本地脚本入口：直接在文件里写参数，不走命令行解析。"""

from __future__ import annotations

import json
from pathlib import Path

from src.app_retrieval import run_app_retrieval_flow


def _default_index_name(gallery_path: str) -> str:
    p = Path(gallery_path)
    return p.name if p.is_dir() else p.stem


# 在这里直接改参数即可
CONFIG = {
    "query_path": "data_runtime/query/ikura_person.png",
    "gallery_path": "data_runtime/gallery/images/yoasobi_TFT",
    "feature_mode": "person",  # "face" or "person"
    "topk": 5,
    "index_name": None,  # None 时自动使用 gallery 名称
    "device": "cpu",
    "arcface_weight_path": "./models/weights/arcface.pt",
    "indexes_root": "indexes",
    "retrieval_output_root": "outputs/retrieval",
    # person 模式配置
    "yolo_weights": "./models/weights/yolo11n.pt",
    "yolo_conf": 0.25,
    "yolo_iou": 0.7,
    "yolo_max_det": 100,
}


def main() -> None:
    query_path = str(CONFIG["query_path"])
    gallery_path = str(CONFIG["gallery_path"])
    index_name = CONFIG["index_name"] or _default_index_name(gallery_path)

    result = run_app_retrieval_flow(
        query_path=query_path,
        gallery_path=gallery_path,
        index_name=str(index_name),
        topk=int(CONFIG["topk"]),
        arcface_weight_path=str(CONFIG["arcface_weight_path"]),
        device=str(CONFIG["device"]),
        indexes_root=str(CONFIG["indexes_root"]),
        retrieval_output_root=str(CONFIG["retrieval_output_root"]),
        feature_mode=str(CONFIG["feature_mode"]),
        yolo_weights=str(CONFIG["yolo_weights"]),
        yolo_conf=float(CONFIG["yolo_conf"]),
        yolo_iou=float(CONFIG["yolo_iou"]),
        yolo_max_det=int(CONFIG["yolo_max_det"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
