FROM python:3.12-slim

# System deps for Tesseract, OpenCV, Playwright
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi uvicorn[standard] playwright pytesseract \
    opencv-python-headless Pillow httpx aiohttp aiosqlite pydantic python-multipart \
    ultralytics open-clip-torch openai-whisper && \
    playwright install chromium && \
    playwright install-deps chromium

# Copy application
COPY . .

# Create required directories
RUN mkdir -p profiles/cookie_farm profiles/chrome_data dataset/captcha_tiles models logs cache

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
