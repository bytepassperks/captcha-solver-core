"""Dataset builder using Firecrawl API to harvest captcha tiles and prompts."""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import httpx

from captcha_solver_core.config import config

logger = logging.getLogger(__name__)


class FirecrawlCollector:
    """
    Collects captcha datasets (tiles, prompts, sitekeys) using Firecrawl
    web scraping API. Stores locally for training YOLOv8 and prompt classifiers.
    """

    def __init__(self):
        self.api_key = config.firecrawl_api_key
        self.base_url = "https://api.firecrawl.dev/v1"
        self.output_dir = config.dataset_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.jsonl"

    async def scrape_page(self, url: str) -> dict | None:
        """Scrape a page using Firecrawl and extract captcha-related elements."""
        if not self.api_key:
            logger.warning("No Firecrawl API key configured")
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/scrape",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "url": url,
                        "formats": ["html", "screenshot"],
                        "waitFor": 5000,
                    },
                )
                resp.raise_for_status()
                return resp.json().get("data", {})
        except Exception as e:
            logger.error(f"Firecrawl scrape failed for {url}: {e}")
            return None

    async def crawl_for_captchas(self, urls: list[str]) -> list[dict]:
        """Crawl multiple URLs and collect captcha data."""
        results = []
        for url in urls:
            data = await self.scrape_page(url)
            if data:
                html = data.get("html", "")
                # Check for captcha signals
                from captcha_solver_core.detector.captcha_detector import detect_from_html
                detection = detect_from_html(html)
                if detection.captcha_type.value != "none":
                    entry = {
                        "url": url,
                        "captcha_type": detection.captcha_type.value,
                        "sitekey": detection.sitekey,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    results.append(entry)
                    self._save_metadata(entry)
                    logger.info(f"Found {detection.captcha_type.value} captcha at {url}")

        return results

    def save_tile(self, image_data: bytes, label: str, source: str = "") -> str:
        """Save a captcha tile image to the dataset."""
        img_hash = hashlib.md5(image_data).hexdigest()[:12]
        label_dir = self.output_dir / label.replace(" ", "_")
        label_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{img_hash}.png"
        filepath = label_dir / filename

        with open(filepath, "wb") as f:
            f.write(image_data)

        entry = {
            "file": str(filepath),
            "label": label,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._save_metadata(entry)
        logger.debug(f"Saved tile: {filepath}")
        return str(filepath)

    def _save_metadata(self, entry: dict):
        """Append metadata entry to JSONL file."""
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_dataset_stats(self) -> dict:
        """Get statistics about the collected dataset."""
        stats = {"total_tiles": 0, "labels": {}, "sources": set()}

        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if "label" in entry:
                            stats["total_tiles"] += 1
                            label = entry["label"]
                            stats["labels"][label] = stats["labels"].get(label, 0) + 1
                        if "source" in entry:
                            stats["sources"].add(entry["source"])
                    except json.JSONDecodeError:
                        continue

        stats["sources"] = list(stats["sources"])
        stats["unique_labels"] = len(stats["labels"])
        return stats

    def export_yolo_format(self, output_dir: str | None = None) -> str:
        """Export dataset in YOLO training format (images/ + labels/ directories)."""
        out = Path(output_dir) if output_dir else self.output_dir / "yolo_export"
        images_dir = out / "images"
        labels_dir = out / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Create classes.txt
        label_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name != "yolo_export"]
        class_names = sorted(d.name for d in label_dirs)

        with open(out / "classes.txt", "w") as f:
            for name in class_names:
                f.write(name + "\n")

        # Create data.yaml
        with open(out / "data.yaml", "w") as f:
            f.write(f"path: {out}\n")
            f.write(f"train: images\n")
            f.write(f"val: images\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write(f"names: {class_names}\n")

        count = 0
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.output_dir / class_name
            if not class_dir.exists():
                continue
            for img_file in class_dir.glob("*.png"):
                import shutil
                shutil.copy2(img_file, images_dir / img_file.name)
                # YOLO label: class_id center_x center_y width height (normalized)
                label_file = labels_dir / img_file.with_suffix(".txt").name
                with open(label_file, "w") as f:
                    f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
                count += 1

        logger.info(f"Exported {count} tiles in YOLO format to {out}")
        return str(out)
