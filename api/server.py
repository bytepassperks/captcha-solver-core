"""FastAPI server exposing the captcha solver as an API."""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from captcha_solver_core.config import config
from captcha_solver_core.detector.captcha_detector import detect_from_html, CaptchaType
from captcha_solver_core.router.dispatcher import Dispatcher, SolveRequest, SolveResult
from captcha_solver_core.engines.ocr_engine import OCREngine
from captcha_solver_core.engines.vision_engine import VisionEngine
from captcha_solver_core.engines.audio_engine import AudioEngine
from captcha_solver_core.engines.token_engine import TokenEngine
from captcha_solver_core.engines.behavior_engine import BehaviorEngine
from captcha_solver_core.cache.token_cache import TokenCache
from captcha_solver_core.browser.persistent_runner import PersistentBrowserRunner
from captcha_solver_core.engines.preharvest_daemon import PreharvestDaemon
from captcha_solver_core.scheduler.profile_scheduler import ProfileScheduler
from captcha_solver_core.logs import setup_logging, log_solve, get_stats

logger = logging.getLogger(__name__)

# Globals
cache = TokenCache()
browser_runner = PersistentBrowserRunner()
scheduler = ProfileScheduler()
daemon = PreharvestDaemon(cache=cache)


def _build_engines() -> dict:
    return {
        "ocr": OCREngine(),
        "vision": VisionEngine(),
        "audio": AudioEngine(),
        "token_harvest": TokenEngine(browser_runner=browser_runner),
        "behavior": BehaviorEngine(browser_runner=browser_runner),
    }


dispatcher = Dispatcher(engines=_build_engines())


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Captcha Solver Core starting up...")

    # Start browser (lazy — only when first solve request comes in)
    # Start pre-harvest daemon if targets are configured
    # Start scheduler in background
    scheduler_task = asyncio.create_task(scheduler.run_scheduler())

    yield

    # Shutdown
    scheduler.stop()
    scheduler_task.cancel()
    await daemon.stop()
    await browser_runner.stop()
    logger.info("Captcha Solver Core shut down.")


app = FastAPI(
    title="Captcha Solver Core",
    description="Modular local captcha solving API — OCR, Vision, Audio, Token Harvest, Behavior Simulation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────

class SolveRequestModel(BaseModel):
    captcha_type: str = Field("auto", description="Captcha type: auto, text, image_grid, recaptcha_v2, hcaptcha, turnstile")
    pageurl: str = Field("", description="URL of the page with captcha")
    sitekey: str | None = Field(None, description="Captcha sitekey (for reCAPTCHA/hCaptcha/Turnstile)")
    captcha_image_base64: str | None = Field(None, description="Base64-encoded captcha image")
    captcha_image_url: str | None = Field(None, description="URL to captcha image")
    extra: dict | None = Field(None, description="Additional params (target_label, audio_url, etc.)")


class SolveResponseModel(BaseModel):
    success: bool
    token: str | None = None
    engine_used: str = ""
    confidence: float = 0.0
    solve_time_ms: int = 0
    error: str | None = None
    cached: bool = False


class DetectRequestModel(BaseModel):
    html: str = Field(..., description="HTML source to analyze")
    url: str = Field("", description="Page URL for context")


class DetectResponseModel(BaseModel):
    captcha_type: str
    sitekey: str | None = None
    iframe_src: str | None = None
    confidence: float = 0.0


class HarvestTargetModel(BaseModel):
    sitekey: str
    domain: str
    captcha_type: str
    pageurl: str
    min_tokens: int = 3
    max_tokens: int = 10


# ─── Endpoints ───────────────────────────────────────────────────────

@app.post("/solve", response_model=SolveResponseModel)
async def solve_captcha(req: SolveRequestModel):
    """Solve a captcha. Main endpoint."""

    # Check cache first
    if req.sitekey and req.pageurl:
        from urllib.parse import urlparse
        domain = urlparse(req.pageurl).netloc
        cached_token = cache.get(req.sitekey, domain, req.captcha_type)
        if cached_token:
            log_solve(req.captcha_type, "cache", True, 1.0, 0, req.pageurl, req.sitekey or "")
            return SolveResponseModel(
                success=True,
                token=cached_token,
                engine_used="cache",
                confidence=1.0,
                solve_time_ms=0,
                cached=True,
            )

    # Build solve request
    image_data = None
    if req.captcha_image_base64:
        image_data = base64.b64decode(req.captcha_image_base64)

    solve_req = SolveRequest(
        captcha_type=req.captcha_type,
        pageurl=req.pageurl,
        sitekey=req.sitekey,
        captcha_image=image_data,
        captcha_image_url=req.captcha_image_url,
        extra=req.extra,
    )

    # Auto-detect if needed
    detection = None
    if req.captcha_type == "auto" and req.pageurl:
        # Could fetch the page and detect, but for API use we expect type to be provided
        pass

    result = await dispatcher.solve(solve_req, detection)

    # Log telemetry
    log_solve(
        req.captcha_type, result.engine_used, result.success,
        result.confidence, result.solve_time_ms,
        req.pageurl, req.sitekey or "", result.error or ""
    )

    # Cache successful tokens
    if result.success and result.token and req.sitekey and req.pageurl:
        from urllib.parse import urlparse
        domain = urlparse(req.pageurl).netloc
        cache.store(req.sitekey, domain, req.captcha_type, result.token)

    return SolveResponseModel(
        success=result.success,
        token=result.token,
        engine_used=result.engine_used,
        confidence=result.confidence,
        solve_time_ms=result.solve_time_ms,
        error=result.error,
    )


@app.post("/solve/image", response_model=SolveResponseModel)
async def solve_image_captcha(file: UploadFile = File(...), captcha_type: str = "text"):
    """Solve a captcha from an uploaded image file."""
    image_data = await file.read()

    solve_req = SolveRequest(
        captcha_type=captcha_type,
        captcha_image=image_data,
    )

    result = await dispatcher.solve(solve_req)

    log_solve(captcha_type, result.engine_used, result.success,
              result.confidence, result.solve_time_ms)

    return SolveResponseModel(
        success=result.success,
        token=result.token,
        engine_used=result.engine_used,
        confidence=result.confidence,
        solve_time_ms=result.solve_time_ms,
        error=result.error,
    )


@app.post("/detect", response_model=DetectResponseModel)
async def detect_captcha(req: DetectRequestModel):
    """Detect captcha type from HTML source."""
    result = detect_from_html(req.html)
    return DetectResponseModel(
        captcha_type=result.captcha_type.value,
        sitekey=result.sitekey,
        iframe_src=result.iframe_src,
        confidence=result.confidence,
    )


@app.post("/harvest/add")
async def add_harvest_target(target: HarvestTargetModel):
    """Add a target for the token pre-harvest daemon."""
    daemon.add_target(
        sitekey=target.sitekey,
        domain=target.domain,
        captcha_type=target.captcha_type,
        pageurl=target.pageurl,
        min_tokens=target.min_tokens,
        max_tokens=target.max_tokens,
    )
    return {"status": "added", "targets": len(daemon.targets)}


@app.post("/harvest/start")
async def start_harvest():
    """Start the token pre-harvest daemon."""
    await daemon.start()
    return {"status": "started", "targets": len(daemon.targets)}


@app.post("/harvest/stop")
async def stop_harvest():
    """Stop the token pre-harvest daemon."""
    await daemon.stop()
    return {"status": "stopped"}


@app.get("/harvest/status")
async def harvest_status():
    """Get pre-harvest daemon status."""
    return daemon.get_status()


@app.get("/cache/stats")
async def cache_stats():
    """Get token cache statistics."""
    return cache.get_stats()


@app.post("/cache/cleanup")
async def cache_cleanup():
    """Clean up expired tokens from cache."""
    cache.cleanup_expired()
    return cache.get_stats()


@app.get("/stats")
async def solver_stats(hours: int = 24):
    """Get solver telemetry statistics."""
    return get_stats(hours)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "engines": list(dispatcher.engines.keys()),
        "cache": cache.get_stats(),
    }


@app.get("/")
async def root():
    return {
        "service": "Captcha Solver Core",
        "version": "1.0.0",
        "endpoints": {
            "POST /solve": "Solve a captcha (main endpoint)",
            "POST /solve/image": "Solve from uploaded image",
            "POST /detect": "Detect captcha type from HTML",
            "POST /harvest/add": "Add pre-harvest target",
            "POST /harvest/start": "Start token pre-harvest daemon",
            "GET /harvest/status": "Pre-harvest daemon status",
            "GET /cache/stats": "Token cache statistics",
            "GET /stats": "Solver telemetry",
            "GET /health": "Health check",
        },
    }
