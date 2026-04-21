from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Optional

import torch

from retrieval.evaluation import run_ablation_experiments, save_experiment_table
from retrieval.feature_extractor import ArcFaceFeatureExtractor, ExtractorConfig
from retrieval.image_index import ImageIndex, build_image_index
from retrieval.io_utils import load_image_manifest, load_video_manifest
from retrieval.types import BBox
from retrieval.video_annotator import annotate_video_with_retrieval
from retrieval.video_index import VideoIndex, build_video_index


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_bbox(args: argparse.Namespace) -> Optional[BBox]:
    if args.qx is None and args.qy is None and args.qw is None and args.qh is None:
        return None
    if None in (args.qx, args.qy, args.qw, args.qh):
        raise ValueError("If bbox is provided, all of --qx --qy --qw --qh must be set.")
    return BBox(x=int(args.qx), y=int(args.qy), w=int(args.qw), h=int(args.qh))


def _make_extractor(args: argparse.Namespace) -> ArcFaceFeatureExtractor:
    # 将 CLI 参数映射为统一提特征配置。
    cfg = ExtractorConfig(
        weight_path=args.weights,
        device=args.device,
        flip_test=args.flip_test,
        detect_face=args.detect_face,
        blur_threshold=args.blur_threshold,
        min_face_size=args.min_face_size,
    )
    return ArcFaceFeatureExtractor(cfg)


def cmd_build_image_index(args: argparse.Namespace) -> None:
    records = load_image_manifest(args.manifest, root_dir=args.root_dir)
    extractor = _make_extractor(args)
    stats = build_image_index(records, extractor, index_dir=args.index_dir, batch_size=args.batch_size)
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


def cmd_search_image(args: argparse.Namespace) -> None:
    extractor = _make_extractor(args)
    index = ImageIndex(args.index_dir)
    results = index.search_image(
        query_image_path=args.query,
        extractor=extractor,
        topk=args.topk,
        threshold=args.threshold,
        bbox=_make_bbox(args),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_build_video_index(args: argparse.Namespace) -> None:
    records = load_video_manifest(args.manifest, root_dir=args.root_dir)
    extractor = _make_extractor(args)
    stats = build_video_index(
        records,
        extractor,
        index_dir=args.index_dir,
        sample_fps=args.sample_fps,
        iou_threshold=args.iou_threshold,
        max_track_gap=args.max_track_gap,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


def cmd_search_video(args: argparse.Namespace) -> None:
    extractor = _make_extractor(args)
    index = VideoIndex(args.index_dir)
    results = index.search_video(
        query_image_path=args.query,
        extractor=extractor,
        topk=args.topk,
        threshold=args.threshold,
        bbox=_make_bbox(args),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    gallery_records = load_image_manifest(args.gallery_manifest, root_dir=args.root_dir)
    query_records = load_image_manifest(args.query_manifest, root_dir=args.root_dir)
    video_records = None
    if args.video_manifest:
        video_records = load_video_manifest(args.video_manifest, root_dir=args.root_dir)

    rows = run_ablation_experiments(
        weight_path=args.weights,
        device=args.device,
        gallery_records=gallery_records,
        query_records=query_records,
        video_records=video_records,
        output_dir=args.output_dir,
        detect_face=args.detect_face,
        blur_threshold=args.blur_threshold,
        min_face_size=args.min_face_size,
        sample_fps=args.sample_fps,
        iou_threshold=args.iou_threshold,
    )

    output_csv = os.path.join(args.output_dir, "ablation_results.csv")
    save_experiment_table(rows, output_csv)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Saved: {output_csv}")


def cmd_annotate_video(args: argparse.Namespace) -> None:
    extractor = _make_extractor(args)
    index = ImageIndex(args.index_dir)
    stats = annotate_video_with_retrieval(
        video_path=args.video,
        output_video_path=args.output_video,
        image_index=index,
        extractor=extractor,
        threshold=args.threshold,
        topk=1,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ArcFace retrieval pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared_extractor_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--weights", default="./models/weights/arcface.pt", help="ArcFace weight path")
        p.add_argument("--device", default=_default_device(), help="cpu or cuda")
        p.add_argument("--flip-test", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--detect-face", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--blur-threshold", type=float, default=0.0)
        p.add_argument("--min-face-size", type=int, default=0)

    def add_query_bbox_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--qx", type=int, default=None)
        p.add_argument("--qy", type=int, default=None)
        p.add_argument("--qw", type=int, default=None)
        p.add_argument("--qh", type=int, default=None)

    p_build_img = sub.add_parser("build-image-index", help="Build image feature index")
    p_build_img.add_argument("--manifest", required=True, help="CSV with image_path, optional image_id/identity/x/y/w/h")
    p_build_img.add_argument("--root-dir", default=None, help="Prefix for relative paths in manifest")
    p_build_img.add_argument("--index-dir", required=True)
    p_build_img.add_argument("--batch-size", type=int, default=32)
    add_shared_extractor_args(p_build_img)
    p_build_img.set_defaults(func=cmd_build_image_index)

    p_search_img = sub.add_parser("search-image", help="Search image index by query image")
    p_search_img.add_argument("--query", required=True)
    p_search_img.add_argument("--index-dir", required=True)
    p_search_img.add_argument("--topk", type=int, default=5)
    p_search_img.add_argument("--threshold", type=float, default=0.3)
    add_shared_extractor_args(p_search_img)
    add_query_bbox_args(p_search_img)
    p_search_img.set_defaults(func=cmd_search_image)

    p_build_vid = sub.add_parser("build-video-index", help="Build video frame/track feature index")
    p_build_vid.add_argument("--manifest", required=True, help="CSV with video_path, optional video_id/identity")
    p_build_vid.add_argument("--root-dir", default=None, help="Prefix for relative paths in manifest")
    p_build_vid.add_argument("--index-dir", required=True)
    p_build_vid.add_argument("--sample-fps", type=float, default=1.0)
    p_build_vid.add_argument("--iou-threshold", type=float, default=0.3)
    p_build_vid.add_argument("--max-track-gap", type=int, default=1)
    add_shared_extractor_args(p_build_vid)
    p_build_vid.set_defaults(func=cmd_build_video_index)

    p_search_vid = sub.add_parser("search-video", help="Search video index by query image")
    p_search_vid.add_argument("--query", required=True)
    p_search_vid.add_argument("--index-dir", required=True)
    p_search_vid.add_argument("--topk", type=int, default=5)
    p_search_vid.add_argument("--threshold", type=float, default=0.3)
    add_shared_extractor_args(p_search_vid)
    add_query_bbox_args(p_search_vid)
    p_search_vid.set_defaults(func=cmd_search_video)

    p_eval = sub.add_parser("evaluate", help="Run 4 fixed ablation experiments")
    p_eval.add_argument("--gallery-manifest", required=True)
    p_eval.add_argument("--query-manifest", required=True)
    p_eval.add_argument("--video-manifest", default=None)
    p_eval.add_argument("--root-dir", default=None)
    p_eval.add_argument("--output-dir", required=True)
    p_eval.add_argument("--sample-fps", type=float, default=1.0)
    p_eval.add_argument("--iou-threshold", type=float, default=0.3)
    add_shared_extractor_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_annotate = sub.add_parser("annotate-video", help="Run MTCNN+retrieval and save annotated video")
    p_annotate.add_argument("--video", required=True, help="Input video path")
    p_annotate.add_argument("--output-video", required=True, help="Output annotated video path")
    p_annotate.add_argument("--index-dir", required=True, help="Image gallery index directory")
    p_annotate.add_argument("--threshold", type=float, default=0.3, help="Match threshold")
    add_shared_extractor_args(p_annotate)
    p_annotate.set_defaults(func=cmd_annotate_video, detect_face=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
