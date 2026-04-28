"""Central configuration for the captcha solver system."""

import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).parent

@dataclass
class Config:
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    base_dir: Path = BASE_DIR
    profiles_dir: Path = BASE_DIR / "profiles" / "cookie_farm"
    dataset_dir: Path = BASE_DIR / "dataset" / "captcha_tiles"
    models_dir: Path = BASE_DIR / "models"
    logs_dir: Path = BASE_DIR / "logs"

    # Browser (xvfb-run provides virtual display, so headed mode works in Docker)
    browser_headless: bool = bool(os.getenv("BROWSER_HEADLESS", "false").lower() in ("true", "1"))
    browser_user_data_base: str = str(BASE_DIR / "profiles" / "chrome_data")
    viewport_width: int = 1280
    viewport_height: int = 800
    default_timeout_ms: int = 30000

    # OCR
    tesseract_cmd: str = "tesseract"
    tesseract_lang: str = "eng"

    # Vision (YOLOv8 + CLIP) — model preloaded in Docker build
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.25
    clip_model: str = "ViT-B/32"

    # Audio
    whisper_model: str = "base"
    vapi_api_key: str = os.getenv("VAPI_API_KEY", "")
    vapi_base_url: str = "https://api.vapi.ai"

    # External APIs
    firecrawl_api_key: str = os.getenv("FIRECRAWL_API_KEY", "")
    reducto_api_key: str = os.getenv("REDUCTO_API_KEY", "")

    # Token cache
    cache_db_path: str = str(BASE_DIR / "cache" / "tokens.db")
    token_ttl_seconds: int = 110  # reCAPTCHA tokens expire in ~120s

    # Behavior simulation
    mouse_jitter_range: tuple = (1, 5)
    typing_delay_range: tuple = (0.05, 0.15)
    scroll_delay_range: tuple = (0.3, 1.2)

    # Cookie farm
    farm_warmup_sites: list = field(default_factory=lambda: [
        "https://www.google.com",
        "https://www.youtube.com",
        "https://news.ycombinator.com",
        "https://en.wikipedia.org",
        "https://www.reddit.com",
        "https://stackoverflow.com",
    ])
    farm_profile_count: int = 5
    farm_warmup_duration_min: int = 3

    # Scheduler
    profile_rotation_interval_hours: int = 12

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True

    def ensure_dirs(self):
        for d in [self.profiles_dir, self.dataset_dir, self.models_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
        Path(self.browser_user_data_base).mkdir(parents=True, exist_ok=True)


config = Config()
config.ensure_dirs()
