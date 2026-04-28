FROM python:3.12-slim

# System deps for Tesseract, OpenCV, Playwright, xvfb
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    wget \
    xvfb \
    dbus \
    xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core Python deps first (lightweight)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pytesseract \
    opencv-python-headless \
    Pillow \
    httpx \
    aiohttp \
    aiosqlite \
    pydantic \
    python-multipart

# Install PyTorch CPU-only (smaller footprint)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install ML packages (these depend on torch)
RUN pip install --no-cache-dir \
    ultralytics \
    open-clip-torch

# Install Whisper separately (pulls in additional deps)
RUN pip install --no-cache-dir openai-whisper

# Install Playwright and Chromium
RUN pip install --no-cache-dir playwright && \
    playwright install chromium && \
    playwright install-deps chromium

# Preload YOLOv8 weights during build (avoids 30s download on first request)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    mv yolov8n.pt /tmp/yolov8n.pt || true

# Copy application
COPY . .

# Create required directories and move preloaded model
RUN mkdir -p profiles/cookie_farm profiles/chrome_data dataset/captcha_tiles models logs cache && \
    (mv /tmp/yolov8n.pt models/yolov8n.pt 2>/dev/null || true)

# Persistent volume mount point for browser profiles (cookies survive restarts)
VOLUME /app/profiles

EXPOSE 8000

# Start with xvfb-run so Playwright can run headed Chromium (needed for token harvest + behavior engines)
CMD ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x800x24", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
