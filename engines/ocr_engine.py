"""OCR-based text captcha solver using Claude Vision (primary) + Tesseract (fallback)."""

import io
import logging
import base64

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pytesseract
import httpx

from config import config

logger = logging.getLogger(__name__)

# Tesseract character whitelist for captcha text (alphanumeric only)
CAPTCHA_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class OCREngine:
    """Solves text-based captchas using Claude Vision (primary) + Tesseract (fallback)."""

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        self._claude_available = bool(config.claude_api_key)

    def _scale_up(self, img: np.ndarray, target_height: int = 100) -> np.ndarray:
        """Scale image to a minimum height for better OCR."""
        h, w = img.shape[:2]
        if h < target_height:
            scale = target_height / h
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img

    def _pipeline_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Pipeline 1: Adaptive threshold — good for varied backgrounds."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        gray = self._scale_up(gray)
        denoised = cv2.fastNlMeansDenoising(gray, h=12)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 4
        )
        if np.mean(thresh) < 128:
            thresh = cv2.bitwise_not(thresh)
        return thresh

    def _pipeline_otsu(self, img: np.ndarray) -> np.ndarray:
        """Pipeline 2: Otsu threshold — good for bimodal histograms."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        gray = self._scale_up(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) < 128:
            thresh = cv2.bitwise_not(thresh)
        return thresh

    def _pipeline_color_isolation(self, img: np.ndarray) -> np.ndarray:
        """Pipeline 3: Color channel isolation — good for colored captchas.
        Tries to isolate text by finding the channel with highest contrast."""
        if len(img.shape) != 3:
            return self._pipeline_adaptive(img)

        img = self._scale_up(img)
        best = None
        best_std = 0

        # Try each color channel + grayscale
        channels = [img[:, :, 0], img[:, :, 1], img[:, :, 2],
                     cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]

        for ch in channels:
            std = np.std(ch)
            if std > best_std:
                best_std = std
                best = ch

        # Also try HSV value channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        if np.std(v_channel) > best_std:
            best = v_channel

        blurred = cv2.GaussianBlur(best, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) < 128:
            thresh = cv2.bitwise_not(thresh)
        return thresh

    def _pipeline_morphological(self, img: np.ndarray) -> np.ndarray:
        """Pipeline 4: Heavy morphological cleanup — good for noisy backgrounds."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        gray = self._scale_up(gray)
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close small gaps in characters
        kernel_close = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        # Open to remove small noise dots
        kernel_open = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

        if np.mean(cleaned) < 128:
            cleaned = cv2.bitwise_not(cleaned)
        return cleaned

    def _pipeline_dilate_erode(self, img: np.ndarray) -> np.ndarray:
        """Pipeline 5: Dilate then erode — thickens thin characters."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        gray = self._scale_up(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(thresh) < 128:
            thresh = cv2.bitwise_not(thresh)

        # Dilate to thicken characters, then erode to restore
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        return dilated

    def _pipeline_hsv_color_masks(self, img: np.ndarray) -> list[np.ndarray]:
        """Pipeline 6: HSV color masking — isolates colored text from backgrounds.
        Returns multiple candidate masks for different color ranges."""
        if len(img.shape) != 3:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        results = []

        # Common captcha text color ranges in HSV
        color_ranges = [
            ([30, 30, 60], [90, 255, 255]),    # Green text
            ([0, 50, 50], [20, 255, 255]),      # Red text (low hue)
            ([160, 50, 50], [180, 255, 255]),   # Red text (high hue)
            ([100, 50, 50], [140, 255, 255]),   # Blue text
            ([0, 0, 0], [180, 40, 100]),        # Dark text (low saturation, low value)
            ([0, 0, 150], [180, 40, 255]),      # Light text on dark bg
        ]

        for lo, hi in color_ranges:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            # Check if mask has enough content (not empty or full)
            fill_ratio = np.sum(mask > 0) / mask.size
            if 0.05 < fill_ratio < 0.7:
                # Scale up 3x for better OCR
                scaled = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
                results.append(scaled)

        return results

    @staticmethod
    def _plausibility_score(text: str, conf: float) -> float:
        """Score OCR result by plausibility for captcha text.
        Captchas typically have 4-8 alphanumeric characters with mixed case/digits."""
        if not text:
            return -1.0
        length = len(text)
        # Length bonus: captchas are typically 4-8 chars
        if 4 <= length <= 8:
            length_score = 40.0
        elif 3 <= length <= 10:
            length_score = 20.0
        else:
            length_score = 0.0
        # Character diversity: captchas almost always mix types
        has_upper = any(c.isupper() for c in text)
        has_lower = any(c.islower() for c in text)
        has_digit = any(c.isdigit() for c in text)
        diversity = 0.0
        if has_upper and has_lower:
            diversity += 10.0
        if has_digit:
            diversity += 20.0  # Digits in captcha text are a strong signal
        if has_upper and has_lower and has_digit:
            diversity += 10.0  # Bonus for having all three
        # Penalize repeated characters (e.g., "eee", "aaa")
        unique_ratio = len(set(text.lower())) / length
        if unique_ratio < 0.5:
            diversity -= 15.0
        return conf * 0.5 + length_score + diversity + length * 2.0

    def _recognize(self, processed: np.ndarray) -> tuple[str, float]:
        """Run Tesseract OCR on preprocessed image. Returns (text, confidence)."""
        whitelist_config = f'-c tessedit_char_whitelist={CAPTCHA_CHARS}'
        configs_to_try = [
            f"--psm 7 --oem 3 -l {config.tesseract_lang} {whitelist_config}",   # Single line
            f"--psm 8 --oem 3 -l {config.tesseract_lang} {whitelist_config}",   # Single word
            f"--psm 13 --oem 3 -l {config.tesseract_lang} {whitelist_config}",  # Raw line
        ]

        best_text = ""
        best_conf = 0.0

        for tess_config in configs_to_try:
            try:
                text = pytesseract.image_to_string(processed, config=tess_config).strip()
                cleaned = "".join(c for c in text if c.isalnum())
                if not cleaned:
                    continue
                data = pytesseract.image_to_data(
                    processed, config=tess_config, output_type=pytesseract.Output.DICT
                )
                confs = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
                avg_conf = sum(confs) / len(confs) if confs else 0
                if avg_conf > best_conf or (avg_conf == best_conf and len(cleaned) > len(best_text)):
                    best_text = cleaned
                    best_conf = avg_conf
            except Exception as e:
                logger.debug(f"Tesseract config {tess_config} failed: {e}")

        return best_text, best_conf

    def _load_image(self, image_data: bytes | None = None, image_url: str | None = None) -> np.ndarray | None:
        """Load image from bytes or URL."""
        if image_data:
            try:
                if isinstance(image_data, str):
                    image_data = base64.b64decode(image_data)
                nparr = np.frombuffer(image_data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Failed to decode image data: {e}")
                return None

        if image_url:
            import httpx
            try:
                resp = httpx.get(image_url, timeout=10)
                nparr = np.frombuffer(resp.content, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Failed to download image from {image_url}: {e}")
                return None

        return None

    def _detect_media_type(self, raw_bytes: bytes) -> str:
        """Detect image media type from file header bytes."""
        if raw_bytes[:3] == b'GIF':
            return "image/gif"
        if raw_bytes[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]):
            return "image/png"
        if raw_bytes[:2] == b'\xff\xd8':
            return "image/jpeg"
        if raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
            return "image/webp"
        return "image/png"

    async def _solve_claude(self, raw_bytes: bytes) -> dict | None:
        """Use Claude Vision to read captcha text — near-perfect accuracy."""
        if not self._claude_available:
            return None

        try:
            img_b64 = base64.b64encode(raw_bytes).decode()
            media_type = self._detect_media_type(raw_bytes)

            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    f"{config.claude_api_base}/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": config.claude_api_key,
                    },
                    json={
                        "model": config.claude_model,
                        "max_tokens": 50,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": img_b64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": "Read the captcha text in this image. Reply with ONLY the exact characters you see, nothing else.",
                                },
                            ],
                        }],
                    },
                )

            if resp.status_code != 200:
                logger.warning(f"Claude Vision API error: {resp.status_code} {resp.text[:200]}")
                return None

            data = resp.json()
            text = data.get("content", [{}])[0].get("text", "").strip()
            cleaned = "".join(c for c in text if c.isalnum())
            if cleaned:
                logger.info(f"Claude Vision result: '{cleaned}'")
                return {
                    "success": True,
                    "token": cleaned,
                    "confidence": 0.99,
                    "pipeline": "claude_vision",
                }
            return None
        except Exception as e:
            logger.warning(f"Claude Vision failed: {e}")
            return None

    async def solve(self, captcha_type=None, pageurl="", sitekey=None,
                    image_data=None, image_url=None, extra=None) -> dict:
        """Solve a text captcha. Uses Claude Vision (primary) with Tesseract fallback."""
        # Get raw bytes for Claude Vision
        raw_bytes = None
        if image_data:
            if isinstance(image_data, str):
                raw_bytes = base64.b64decode(image_data)
            else:
                raw_bytes = image_data
        elif image_url:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(image_url)
                    raw_bytes = resp.content
            except Exception as e:
                logger.error(f"Failed to download image from {image_url}: {e}")

        # Priority 1: Claude Vision (near-perfect accuracy)
        if raw_bytes:
            claude_result = await self._solve_claude(raw_bytes)
            if claude_result:
                return claude_result

        # Priority 2: Tesseract multi-pipeline (fallback)
        img = self._load_image(image_data, image_url)
        if img is None:
            return {"success": False, "error": "No image provided or failed to load"}

        pipelines = [
            ("adaptive", self._pipeline_adaptive),
            ("otsu", self._pipeline_otsu),
            ("color_isolation", self._pipeline_color_isolation),
            ("morphological", self._pipeline_morphological),
            ("dilate_erode", self._pipeline_dilate_erode),
        ]

        best_text = ""
        best_conf = 0.0
        best_pipeline = ""

        for name, pipeline_fn in pipelines:
            try:
                processed = pipeline_fn(img)
                text, conf = self._recognize(processed)
                logger.debug(f"Pipeline '{name}': text='{text}', conf={conf:.1f}")
                if text:
                    score = self._plausibility_score(text, conf)
                    best_score = self._plausibility_score(best_text, best_conf)
                    if score > best_score:
                        best_text = text
                        best_conf = conf
                        best_pipeline = name
            except Exception as e:
                logger.debug(f"Pipeline '{name}' failed: {e}")

        # HSV color mask pipeline — generates multiple candidates
        try:
            color_masks = self._pipeline_hsv_color_masks(img)
            for i, mask in enumerate(color_masks):
                text, conf = self._recognize(mask)
                logger.debug(f"Pipeline 'hsv_mask_{i}': text='{text}', conf={conf:.1f}")
                if text:
                    hsv_score = self._plausibility_score(text, conf)
                    best_score = self._plausibility_score(best_text, best_conf)
                    if hsv_score > best_score:
                        best_text = text
                        best_conf = conf
                        best_pipeline = f"hsv_mask_{i}"
        except Exception as e:
            logger.debug(f"HSV mask pipeline failed: {e}")

        if not best_text:
            return {"success": False, "error": "OCR failed to extract text", "confidence": 0.0}

        confidence = min(best_conf / 100.0, 1.0)
        logger.info(f"Tesseract result: '{best_text}' (pipeline={best_pipeline}, conf={confidence:.2f})")
        return {
            "success": True,
            "token": best_text,
            "confidence": confidence,
            "pipeline": best_pipeline,
        }
