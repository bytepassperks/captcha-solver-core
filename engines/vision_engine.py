"""Vision-based image grid captcha solver using YOLOv8 + CLIP."""

import io
import logging
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from captcha_solver_core.config import config

logger = logging.getLogger(__name__)


class VisionEngine:
    """Solves image grid captchas (select all X) using YOLOv8 object detection + CLIP similarity."""

    def __init__(self):
        self._yolo = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenize = None

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            model_path = config.models_dir / config.yolo_model
            if not model_path.exists():
                logger.info(f"Downloading YOLOv8 model to {model_path}")
                self._yolo = YOLO("yolov8n.pt")
            else:
                self._yolo = YOLO(str(model_path))
            logger.info("YOLOv8 model loaded")
        return self._yolo

    def _load_clip(self):
        if self._clip_model is None:
            try:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                self._clip_model = model
                self._clip_preprocess = preprocess
                self._clip_tokenize = open_clip.get_tokenizer("ViT-B-32")
                logger.info("CLIP model loaded")
            except Exception as e:
                logger.warning(f"CLIP load failed, falling back to YOLO-only: {e}")
        return self._clip_model

    def _split_grid(self, img: np.ndarray, rows: int = 3, cols: int = 3) -> list[np.ndarray]:
        """Split image into grid tiles."""
        h, w = img.shape[:2]
        tile_h, tile_w = h // rows, w // cols
        tiles = []
        for r in range(rows):
            for c in range(cols):
                y1, y2 = r * tile_h, (r + 1) * tile_h
                x1, x2 = c * tile_w, (c + 1) * tile_w
                tiles.append(img[y1:y2, x1:x2])
        return tiles

    def _detect_objects_yolo(self, tile: np.ndarray, target_label: str) -> bool:
        """Use YOLO to detect if target object is in tile."""
        yolo = self._load_yolo()
        results = yolo(tile, verbose=False, conf=config.yolo_confidence)

        if not results or len(results) == 0:
            return False

        target_lower = target_label.lower()
        # YOLO class name aliases
        aliases = {
            "bicycle": ["bicycle", "bike"],
            "bus": ["bus"],
            "car": ["car", "automobile"],
            "motorcycle": ["motorcycle", "motorbike"],
            "traffic light": ["traffic light"],
            "fire hydrant": ["fire hydrant", "hydrant"],
            "crosswalk": ["crosswalk"],
            "stairs": ["stairs", "staircase"],
            "bridge": ["bridge"],
            "boat": ["boat", "ship"],
            "truck": ["truck"],
            "train": ["train"],
            "airplane": ["airplane", "aeroplane", "plane"],
            "person": ["person", "people"],
        }

        target_names = aliases.get(target_lower, [target_lower])

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id].lower()
                if class_name in target_names or target_lower in class_name:
                    return True
        return False

    def _detect_objects_clip(self, tile: np.ndarray, target_label: str) -> float:
        """Use CLIP to get similarity score between tile and target label."""
        import torch

        clip_model = self._load_clip()
        if clip_model is None:
            return 0.0

        pil_img = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
        img_tensor = self._clip_preprocess(pil_img).unsqueeze(0)

        text_tokens = self._clip_tokenize([target_label, "nothing", "empty"])

        with torch.no_grad():
            img_features = clip_model.encode_image(img_tensor)
            text_features = clip_model.encode_text(text_tokens)

            img_features /= img_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (img_features @ text_features.T).squeeze(0)

        target_sim = similarity[0].item()
        return target_sim

    def _load_image(self, image_data: bytes | None, image_url: str | None) -> np.ndarray | None:
        if image_data:
            try:
                if isinstance(image_data, str):
                    image_data = base64.b64decode(image_data)
                nparr = np.frombuffer(image_data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Failed to decode image: {e}")
                return None

        if image_url:
            import httpx
            try:
                resp = httpx.get(image_url, timeout=15)
                nparr = np.frombuffer(resp.content, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Failed to download image: {e}")
                return None

        return None

    async def solve(self, captcha_type=None, pageurl="", sitekey=None,
                    image_data=None, image_url=None, extra=None) -> dict:
        """Solve an image grid captcha. Returns selected tile indices."""
        extra = extra or {}
        target_label = extra.get("target_label", extra.get("prompt", ""))
        grid_rows = extra.get("grid_rows", 3)
        grid_cols = extra.get("grid_cols", 3)

        if not target_label:
            return {"success": False, "error": "No target_label provided in extra"}

        img = self._load_image(image_data, image_url)
        if img is None:
            return {"success": False, "error": "No image provided"}

        tiles = self._split_grid(img, grid_rows, grid_cols)
        selected_indices = []

        for i, tile in enumerate(tiles):
            # Try YOLO first
            yolo_match = self._detect_objects_yolo(tile, target_label)

            # CLIP similarity as secondary signal
            clip_score = 0.0
            try:
                clip_score = self._detect_objects_clip(tile, target_label)
            except Exception:
                pass

            # Combined decision
            if yolo_match or clip_score > 0.25:
                selected_indices.append(i)
                logger.debug(f"Tile {i}: YOLO={yolo_match}, CLIP={clip_score:.3f} -> SELECTED")
            else:
                logger.debug(f"Tile {i}: YOLO={yolo_match}, CLIP={clip_score:.3f} -> skipped")

        logger.info(f"Vision solve for '{target_label}': selected tiles {selected_indices}")

        return {
            "success": len(selected_indices) > 0,
            "token": ",".join(str(i) for i in selected_indices),
            "confidence": 0.85 if selected_indices else 0.0,
            "selected_indices": selected_indices,
        }
