from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image


class MTCNNFaceDetector:
    """MTCNN detector exposed from models package."""

    def __init__(
        self,
        min_face_size: float = 20.0,
        thresholds: tuple[float, float, float] = (0.6, 0.7, 0.8),
        nms_thresholds: tuple[float, float, float] = (0.7, 0.7, 0.7),
    ):
        project_root = Path(__file__).resolve().parent / "mtcnn_project"
        if not project_root.exists():
            raise FileNotFoundError(f"MTCNN project root not found: {project_root}")

        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

        from mtcnn_pytorch.src.get_nets import ONet, PNet, RNet
        from mtcnn_pytorch.src.box_utils import calibrate_box, convert_to_square, get_image_boxes, nms
        from mtcnn_pytorch.src.first_stage import run_first_stage

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.onet.eval()

        self.calibrate_box = calibrate_box
        self.convert_to_square = convert_to_square
        self.get_image_boxes = get_image_boxes
        self.nms = nms
        self.run_first_stage = run_first_stage

        self.min_face_size = float(min_face_size)
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds

    def detect(self, image_bgr: np.ndarray) -> List["MTCNNBox"]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        width, height = image_pil.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707
        m = min_detection_size / self.min_face_size
        min_length *= m

        scales: List[float] = []
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * (factor ** factor_count))
            min_length *= factor
            factor_count += 1

        if not scales:
            return []

        bounding_boxes = []
        with torch.no_grad():
            for s in scales:
                boxes = self.run_first_stage(image_pil, self.pnet, scale=s, threshold=self.thresholds[0])
                if boxes is not None and len(boxes) > 0:
                    bounding_boxes.append(boxes)

            if not bounding_boxes:
                return []

            bounding_boxes = np.vstack(bounding_boxes)
            if bounding_boxes.size == 0:
                return []

            keep = self.nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])
            if len(keep) == 0:
                return []
            bounding_boxes = bounding_boxes[keep]

            bounding_boxes = self.calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            bounding_boxes = self.convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            img_boxes = self.get_image_boxes(bounding_boxes, image_pil, size=24)
            if len(img_boxes) == 0:
                return []
            img_boxes = torch.FloatTensor(img_boxes)

            output = self.rnet(img_boxes)
            offsets = output[0].data.numpy()
            probs = output[1].data.numpy()

            keep = np.where(probs[:, 1] > self.thresholds[1])[0]
            if len(keep) == 0:
                return []
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = self.nms(bounding_boxes, self.nms_thresholds[1])
            if len(keep) == 0:
                return []
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = self.calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = self.convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            img_boxes = self.get_image_boxes(bounding_boxes, image_pil, size=48)
            if len(img_boxes) == 0:
                return []
            img_boxes = torch.FloatTensor(img_boxes)

            output = self.onet(img_boxes)
            offsets = output[1].data.numpy()
            probs = output[2].data.numpy()

            keep = np.where(probs[:, 1] > self.thresholds[2])[0]
            if len(keep) == 0:
                return []
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            bounding_boxes = self.calibrate_box(bounding_boxes, offsets)
            keep = self.nms(bounding_boxes, self.nms_thresholds[2], mode="min")
            if len(keep) == 0:
                return []
            bounding_boxes = bounding_boxes[keep]

        out: List[MTCNNBox] = []
        h_img, w_img = image_bgr.shape[:2]
        for bb in bounding_boxes:
            x1, y1, x2, y2 = [int(v) for v in bb[:4]]
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(0, min(w_img - 1, x2))
            y2 = max(0, min(h_img - 1, y2))
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue
            out.append(MTCNNBox(x=x1, y=y1, w=w, h=h))

        return out


@dataclass(slots=True)
class MTCNNBox:
    x: int
    y: int
    w: int
    h: int
