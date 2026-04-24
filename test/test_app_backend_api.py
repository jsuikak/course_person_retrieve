from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.backend import services
from src.app.backend.app import app


class AppBackendApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _write_result_json(self, tmp_dir: Path) -> Path:
        out_dir = tmp_dir / "retrieval"
        out_dir.mkdir(parents=True, exist_ok=True)
        crop_path = services.WEB_OUTPUT_DIR / "retrieval" / "demo_crop.jpg"
        anno_path = services.WEB_OUTPUT_DIR / "retrieval" / "demo_anno.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop_path.write_bytes(b"x")
        anno_path.write_bytes(b"y")

        payload = {
            "results": [
                {
                    "rank": 1,
                    "row_index": 0,
                    "score": 0.99,
                    "source_name": "dummy.jpg",
                    "frame_index": -1,
                    "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                    "crop_path": str(crop_path),
                    "annotated_path": str(anno_path),
                }
            ]
        }
        path = out_dir / "results.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_import_backend_app(self) -> None:
        __import__("src.app.backend.app")

    def test_status_route(self) -> None:
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("weights", data)
        self.assertIn("defaults", data)
        self.assertIn("indexes", data)
        self.assertIn("feature_modes", data)
        self.assertIn("person_models", data)
        self.assertIn("torchreid_importable", data["runtime"])

    def test_rebuild_gallery_index_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            gallery_dir = tmp_dir / "gallery"
            gallery_dir.mkdir(parents=True, exist_ok=True)
            (gallery_dir / "a.jpg").write_bytes(b"a")

            fake_result = mock.Mock()
            fake_result.library_type = "image"
            fake_result.feature_mode = "face"
            fake_result.output_paths = {
                "features_path": str(tmp_dir / "face_features.npy"),
                "meta_path": str(tmp_dir / "face_meta.csv"),
                "info_path": str(tmp_dir / "face_info.json"),
            }
            fake_result.total_items = 1
            fake_result.feature_dim = 512

            with mock.patch("src.app.backend.services.build_feature_index", return_value=fake_result) as m_build:
                resp = self.client.post(
                    "/api/admin/rebuild-gallery-index",
                    json={
                        "gallery_path": str(gallery_dir),
                        "feature_mode": "face",
                        "index_name": "demo_idx",
                        "sample_fps": 2.0,
                    },
                )

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["index_name"], "demo_idx")
            self.assertEqual(data["feature_mode"], "face")
            self.assertIn("index_paths", data)

            kwargs = m_build.call_args.kwargs
            self.assertEqual(Path(kwargs["library_path"]).resolve(), gallery_dir.resolve())
            self.assertEqual(kwargs["prefix"], "demo_idx")
            self.assertEqual(kwargs["sample_fps"], 2.0)
            self.assertEqual(kwargs["person_model"], "resnet")
            self.assertEqual(kwargs["resnet_backbone"], "resnet18")

    def test_rebuild_gallery_index_person_model_suffix_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            gallery_dir = tmp_dir / "gallery"
            gallery_dir.mkdir(parents=True, exist_ok=True)
            (gallery_dir / "a.jpg").write_bytes(b"a")

            fake_result = mock.Mock()
            fake_result.library_type = "image"
            fake_result.feature_mode = "person"
            fake_result.output_paths = {
                "features_path": str(tmp_dir / "person_features.npy"),
                "meta_path": str(tmp_dir / "person_meta.csv"),
                "info_path": str(tmp_dir / "person_info.json"),
            }
            fake_result.total_items = 1
            fake_result.feature_dim = 512

            with mock.patch("src.app.backend.services.build_feature_index", return_value=fake_result) as m_build:
                resp = self.client.post(
                    "/api/admin/rebuild-gallery-index",
                    json={
                        "gallery_path": str(gallery_dir),
                        "feature_mode": "person",
                        "person_model": "osnet",
                        "index_name": "demo_idx",
                    },
                )

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["index_name"], "demo_idx_osnet_x1_0")
            self.assertEqual(data["person_model"], "osnet")

            kwargs = m_build.call_args.kwargs
            self.assertEqual(kwargs["prefix"], "demo_idx_osnet_x1_0")
            self.assertEqual(kwargs["person_model"], "osnet")

    def test_index_status_uses_effective_index_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            gallery_dir = tmp_dir / "gallery"
            gallery_dir.mkdir(parents=True, exist_ok=True)
            (gallery_dir / "a.jpg").write_bytes(b"a")

            index_dir = tmp_dir / "image_index"
            index_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("features.npy", "meta.csv", "info.json"):
                (index_dir / f"demo_resnet18_person_{suffix}").write_bytes(b"x")

            with mock.patch("src.app.backend.services.IMAGE_INDEX_DIR", index_dir):
                resp = self.client.post(
                    "/api/index/status",
                    json={
                        "gallery_path": str(gallery_dir),
                        "feature_mode": "person",
                        "person_model": "resnet",
                        "index_name": "demo",
                    },
                )

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["exists"])
            self.assertEqual(data["index_name"], "demo_resnet18")
            self.assertEqual(data["library_type"], "image")

    def test_index_status_for_uploaded_video_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            index_dir = tmp_dir / "video_index"
            index_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch("src.app.backend.services.VIDEO_INDEX_DIR", index_dir):
                resp = self.client.post(
                    "/api/index/status",
                    json={
                        "library_type": "video",
                        "source_name": "clip.mp4",
                        "feature_mode": "person",
                        "person_model": "osnet",
                    },
                )

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data["exists"])
            self.assertEqual(data["index_name"], "clip_osnet_x1_0")
            self.assertEqual(data["library_type"], "video")

    def test_rebuild_uploaded_video_index_mapping(self) -> None:
        fake_summary = {
            "library_type": "video",
            "feature_mode": "person",
            "person_model": "osnet",
            "index_name": "clip_osnet_x1_0",
            "total_items": 2,
            "feature_dim": 512,
            "index_paths": {},
        }

        with mock.patch("src.app.backend.app.rebuild_uploaded_video_index", return_value=fake_summary) as m_rebuild:
            files = {"video": ("clip.mp4", b"video-data", "video/mp4")}
            data = {
                "feature_mode": "person",
                "person_model": "osnet",
                "sample_fps": "1.5",
            }
            resp = self.client.post("/api/admin/rebuild-uploaded-video-index", files=files, data=data)

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["index_name"], "clip_osnet_x1_0")
        self.assertIn("status", payload)

        kwargs = m_rebuild.call_args.kwargs
        self.assertEqual(kwargs["video_name"], "clip.mp4")
        self.assertEqual(kwargs["options"].feature_mode, "person")
        self.assertEqual(kwargs["options"].person_model, "osnet")
        self.assertEqual(kwargs["options"].sample_fps, 1.5)

    def test_rebuild_gallery_index_missing_dependency_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            gallery_dir = tmp_dir / "gallery"
            gallery_dir.mkdir(parents=True, exist_ok=True)
            (gallery_dir / "a.jpg").write_bytes(b"a")

            with mock.patch(
                "src.app.backend.services.build_feature_index",
                side_effect=ImportError("ultralytics is required for YOLOPersonDetector"),
            ):
                resp = self.client.post(
                    "/api/admin/rebuild-gallery-index",
                    json={
                        "gallery_path": str(gallery_dir),
                        "feature_mode": "person",
                        "index_name": "demo_idx",
                    },
                )

            self.assertEqual(resp.status_code, 400)
            self.assertIn("ultralytics", resp.json()["detail"])

    def test_search_gallery_auto_build_branch_and_url_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            gallery_dir = tmp_dir / "gallery"
            gallery_dir.mkdir(parents=True, exist_ok=True)
            (gallery_dir / "a.jpg").write_bytes(b"a")

            result_json_path = self._write_result_json(tmp_dir)
            fake_summary = {
                "library_type": "image",
                "feature_mode": "face",
                "person_model": "resnet",
                "index_name": "demo",
                "build": {"status": "built"},
                "retrieval": {"result_json": str(result_json_path), "output_dir": "x"},
            }

            with mock.patch("src.app.backend.services.run_app_retrieval_flow", return_value=fake_summary) as m_flow:
                files = {"query": ("query.jpg", b"abc", "image/jpeg")}
                data = {
                    "gallery_path": str(gallery_dir),
                    "feature_mode": "face",
                    "index_name": "demo",
                    "topk": "5",
                }
                resp = self.client.post("/api/search/gallery", files=files, data=data)

            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            self.assertEqual(payload["result_count"], 1)
            self.assertTrue(payload["results"][0]["crop_url"].startswith("/outputs-static/"))
            self.assertEqual(payload["results"][0]["source_name"], "dummy.jpg")

            kwargs = m_flow.call_args.kwargs
            self.assertEqual(kwargs["feature_mode"], "face")
            self.assertEqual(kwargs["index_name"], "demo")
            self.assertEqual(kwargs["person_model"], "resnet")
            self.assertEqual(kwargs["resnet_backbone"], "resnet18")

    def test_search_uploaded_video_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            result_json_path = self._write_result_json(tmp_dir)
            fake_summary = {
                "library_type": "video",
                "feature_mode": "person",
                "person_model": "osnet",
                "index_name": "v_idx_osnet_x1_0",
                "build": {"status": "skipped"},
                "retrieval": {"result_json": str(result_json_path), "output_dir": "x"},
            }

            with mock.patch("src.app.backend.services.run_app_retrieval_flow", return_value=fake_summary) as m_flow:
                files = {
                    "query": ("query.jpg", b"abc", "image/jpeg"),
                    "video": ("clip.mp4", b"video-data", "video/mp4"),
                }
                data = {
                    "feature_mode": "person",
                    "person_model": "osnet",
                    "index_name": "v_idx",
                    "topk": "3",
                    "sample_fps": "1.5",
                }
                resp = self.client.post("/api/search/uploaded-video", files=files, data=data)

            self.assertEqual(resp.status_code, 200)
            payload = resp.json()
            self.assertEqual(payload["feature_mode"], "person")
            self.assertEqual(payload["person_model"], "osnet")
            self.assertEqual(payload["index_name"], "v_idx_osnet_x1_0")
            self.assertEqual(payload["video_name"], "clip.mp4")
            self.assertEqual(payload["result_count"], 1)

            kwargs = m_flow.call_args.kwargs
            self.assertEqual(kwargs["feature_mode"], "person")
            self.assertEqual(kwargs["person_model"], "osnet")
            self.assertEqual(kwargs["index_name"], "v_idx_osnet_x1_0")
            self.assertEqual(kwargs["sample_fps"], 1.5)

    def test_clear_web_outputs_only_clears_temp(self) -> None:
        services.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        services.RETRIEVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        services.IMAGE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        upload_file = services.UPLOAD_DIR / "tmp.bin"
        retrieval_file = services.RETRIEVAL_OUTPUT_DIR / "tmp.json"
        index_file = services.IMAGE_INDEX_DIR / "keep.txt"

        upload_file.write_bytes(b"a")
        retrieval_file.write_text("{}", encoding="utf-8")
        index_file.write_text("keep", encoding="utf-8")

        resp = self.client.delete("/api/admin/clear-web-outputs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["cleared"])

        self.assertFalse(upload_file.exists())
        self.assertFalse(retrieval_file.exists())
        self.assertTrue(index_file.exists())


if __name__ == "__main__":
    unittest.main()
