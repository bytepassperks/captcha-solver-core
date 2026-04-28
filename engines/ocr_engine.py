"""OCR-based text captcha solver using Tesseract + image preprocessing."""

import io
import logging
import base64

import cv2
import numpy as np
from PIL import Image
import pytesseract

from config import config

logger = logging.getLogger(__name__)


class OCREngine:
    """Solves text-based captchas using image preprocessing + Tesseract OCR."""

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline: grayscale, threshold, denoise, morphology."""
        # Grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Resize for better OCR (scale up small images)
        h, w = gray.shape
        if w < 200:
            scale = 200 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Invert if mostly black background
        if np.mean(cleaned) < 128:
            cleaned = cv2.bitwise_not(cleaned)

        return cleaned

    def _recognize(self, processed: np.ndarray) -> str:
        """Run Tesseract OCR on preprocessed image."""
        # Try multiple PSM modes and pick best
        configs_to_try = [
            f"--psm 7 --oem 3 -l {config.tesseract_lang}",  # Single line
            f"--psm 8 --oem 3 -l {config.tesseract_lang}",  # Single word
            f"--psm 13 --oem 3 -l {config.tesseract_lang}",  # Raw line
        ]

        results = []
        for tess_config in configs_to_try:
            try:
                text = pytesseract.image_to_string(processed, config=tess_config).strip()
                # Clean: remove non-alphanumeric except common captcha chars
                cleaned = "".join(c for c in text if c.isalnum())
                if cleaned:
                    # Score by confidence
                    data = pytesseract.image_to_data(
                        processed, config=tess_config, output_type=pytesseract.Output.DICT
                    )
                    confs = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
                    avg_conf = sum(confs) / len(confs) if confs else 0
                    results.append((cleaned, avg_conf))
            except Exception as e:
                logger.debug(f"Tesseract config {tess_config} failed: {e}")

        if not results:
            return ""

        # Return highest confidence result
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0][0]

    def _load_image(self, image_data: bytes | None = None, image_url: str | None = None) -> np.ndarray | None:
        """Load image from bytes or URL."""
        if image_data:
            # Could be base64 encoded
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

    async def solve(self, captcha_type=None, pageurl="", sitekey=None,
                    image_data=None, image_url=None, extra=None) -> dict:
        """Solve a text captcha from image data or URL."""
        img = self._load_image(image_data, image_url)
        if img is None:
            return {"success": False, "error": "No image provided or failed to load"}

        processed = self._preprocess(img)
        text = self._recognize(processed)

        if not text:
            return {"success": False, "error": "OCR failed to extract text", "confidence": 0.0}

        logger.info(f"OCR result: '{text}'")
        return {
            "success": True,
            "token": text,
            "confidence": 0.85,
        }
