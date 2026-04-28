# Captcha Solver Core

A modular, locally-hosted captcha solving system with multiple solver engines and a FastAPI REST API.

## Architecture

```
POST /solve  →  Detector  →  Router  →  Engine  →  Token Cache  →  Response
```

### Engines

| Engine | Captcha Types | Technology |
|--------|---------------|------------|
| **OCR** | Text captchas | Tesseract + OpenCV preprocessing |
| **Vision** | Image grid (select all X) | YOLOv8 + CLIP |
| **Audio** | reCAPTCHA audio challenges | Whisper (local) + Vapi (cloud) |
| **Token Harvest** | reCAPTCHA/hCaptcha checkbox | Auto-pass with trusted profiles |
| **Behavior** | Turnstile / behavioral | Mouse simulation, scrolling, timing |

### Support Modules

- **Captcha Detector** — Identifies captcha type from page HTML/JS signals
- **Router/Dispatcher** — Routes solve requests to the best engine with fallback
- **Token Cache** — SQLite store for reusing valid tokens (avoids re-solving)
- **Cookie Reputation Farm** — Warms browser profiles on safe sites for trust
- **Behavior Simulator** — Human-like mouse, scroll, typing patterns
- **Pre-harvest Daemon** — Background worker that collects tokens proactively
- **Profile Scheduler** — Rotates and warms browser profiles on schedule
- **Firecrawl Collector** — Harvests captcha tiles for dataset building
- **Reducto Classifier** — Extracts object labels from challenge prompts
- **Telemetry** — Logs success rates, engine usage, solve times

## Quick Start

### Local

```bash
# Install system deps
sudo apt install tesseract-ocr tesseract-ocr-eng

# Install Python deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn playwright pytesseract opencv-python-headless \
    Pillow httpx aiosqlite ultralytics open-clip-torch openai-whisper python-multipart

# Install Playwright browser
playwright install chromium

# Run server
cd captcha_solver_core
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t captcha-solver .
docker run -p 8000:8000 captcha-solver
```

## API Usage

### Solve a text captcha

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "captcha_type": "text",
    "captcha_image_url": "https://example.com/captcha.png"
  }'
```

### Solve from uploaded image

```bash
curl -X POST http://localhost:8000/solve/image \
  -F "file=@captcha.png" \
  -F "captcha_type=text"
```

### Solve reCAPTCHA v2

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "captcha_type": "recaptcha_v2",
    "pageurl": "https://example.com/login",
    "sitekey": "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-"
  }'
```

### Detect captcha type

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"html": "<div class=\"g-recaptcha\" data-sitekey=\"6Le...\"></div>"}'
```

### Health check

```bash
curl http://localhost:8000/health
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VAPI_API_KEY` | Vapi API key for audio transcription | (empty) |
| `FIRECRAWL_API_KEY` | Firecrawl API key for dataset collection | (empty) |
| `REDUCTO_API_KEY` | Reducto API key for prompt classification | (empty) |

## Expected Success Rates

| Captcha Type | Success Rate |
|-------------|-------------|
| Text captcha | 95-99% |
| Image grid | 85-92% |
| hCaptcha | 65-80% |
| reCAPTCHA v2 | 70-88% |
| reCAPTCHA checkbox | ~60% |
| reCAPTCHA v3 | Low |
| Turnstile | Low |

## Project Structure

```
captcha_solver_core/
├── api/server.py              # FastAPI REST server
├── config.py                  # Central configuration
├── detector/captcha_detector.py   # Captcha type detection
├── router/dispatcher.py       # Request routing + fallback
├── engines/
│   ├── ocr_engine.py          # Tesseract OCR solver
│   ├── vision_engine.py       # YOLOv8 + CLIP grid solver
│   ├── audio_engine.py        # Whisper + Vapi audio solver
│   ├── token_engine.py        # Token harvest (checkbox)
│   ├── behavior_engine.py     # Behavior simulation
│   └── preharvest_daemon.py   # Background token farmer
├── browser/persistent_runner.py   # Playwright browser manager
├── cache/token_cache.py       # SQLite token cache
├── classifier/reducto_prompt_parser.py  # Prompt classification
├── dataset/firecrawl_collector.py  # Dataset harvesting
├── scheduler/profile_scheduler.py  # Profile rotation
├── logs/                      # Telemetry + logging
├── models/                    # ML model files
├── profiles/                  # Browser profiles
├── Dockerfile                 # Container build
├── pyproject.toml             # Python project config
└── README.md                  # This file
```
