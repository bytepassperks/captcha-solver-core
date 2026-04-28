"""FastAPI server exposing the captcha solver as an API — v2.3.0."""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import config
from detector.captcha_detector import detect_from_html, detect_from_page, CaptchaType
from router.dispatcher import Dispatcher, SolveRequest, SolveResult, get_cached_detection
from engines.ocr_engine import OCREngine
from engines.vision_engine import VisionEngine
from engines.audio_engine import AudioEngine
from engines.token_engine import TokenEngine
from engines.behavior_engine import BehaviorEngine
from cache.token_cache import TokenCache
from browser.persistent_runner import PersistentBrowserRunner
from browser.browser_pool import BrowserPool
from engines.preharvest_daemon import PreharvestDaemon
from scheduler.profile_scheduler import ProfileScheduler
from test_runner.random_site_runner import RandomSiteRunner
from logs import setup_logging, log_solve, get_stats
from vpn_router.manager import VPNManager
from vpn_router.profile_map import ProfileRegionMap
from vpn_router.scheduler import VPNRotationScheduler

logger = logging.getLogger(__name__)

# Globals
cache = TokenCache()
browser_pool = BrowserPool()
browser_runner = PersistentBrowserRunner()
scheduler = ProfileScheduler()
daemon = PreharvestDaemon(cache=cache)
vpn_manager = VPNManager()
vpn_profile_map = ProfileRegionMap()
vpn_scheduler = VPNRotationScheduler(manager=vpn_manager, profile_map=vpn_profile_map)


def _build_engines() -> dict:
    return {
        "ocr": OCREngine(),
        "vision": VisionEngine(),
        "audio": AudioEngine(),
        "token_harvest": TokenEngine(browser_runner=browser_runner),
        "behavior": BehaviorEngine(browser_runner=browser_runner),
    }


dispatcher = Dispatcher(engines=_build_engines())
dispatcher.set_stats_provider(get_stats)
test_runner = RandomSiteRunner(dispatcher=dispatcher)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Captcha Solver Core v2.3.0 starting up...")

    # Speed Boost: YOLO warm-start at boot (CLIP lazy-loads on first vision request — too heavy for boot)
    logger.info("Warm-starting YOLO model...")
    try:
        vision_engine = dispatcher.engines.get("vision")
        if vision_engine:
            vision_engine._load_yolo()
            logger.info("YOLO model warm-started successfully")
    except Exception as e:
        logger.warning(f"YOLO warm-start failed (will lazy-load on first request): {e}")

    # Start browser pool (adaptive scaling enabled)
    try:
        await browser_pool.start()
        logger.info(f"Browser pool started: {browser_pool.active_count} contexts (max={browser_pool.max_size})")
    except Exception as e:
        logger.warning(f"Browser pool start failed: {e}")

    # Start profile scheduler in background (delayed to avoid boot OOM)
    async def _delayed_scheduler():
        await asyncio.sleep(60)  # wait 60s for boot to stabilize before warming profiles
        await scheduler.run_scheduler()
    scheduler_task = asyncio.create_task(_delayed_scheduler())

    # Start random site benchmark scheduler
    test_runner.start_scheduler(interval_hours=config.test_runner_interval_hours)

    # Initialize VPN identity routing
    if config.vpn_enabled:
        available = vpn_manager.available_providers
        logger.info(f"VPN routing enabled, available providers: {available or 'none (will use direct connection)'}")
        # Assign regions to browser pool profiles
        for i in range(browser_pool.active_count):
            profile_name = f"pool_{i:03d}"
            provider = available[0] if available else "none"
            vpn_profile_map.assign(profile_name, provider)
        # Start rotation scheduler
        vpn_scheduler.start()
    else:
        logger.info("VPN routing disabled")

    yield

    # Shutdown
    vpn_scheduler.stop()
    scheduler.stop()
    scheduler_task.cancel()
    test_runner.stop_scheduler()
    await daemon.stop()
    await vpn_manager.disconnect()
    await browser_pool.stop()
    await browser_runner.stop()
    logger.info("Captcha Solver Core shut down.")


app = FastAPI(
    title="Captcha Solver Core",
    description="Modular captcha solving API with engine racing, adaptive routing, VPN identity routing, and continuous benchmarking",
    version="2.3.0",
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
    cached: bool = False


class HarvestTargetModel(BaseModel):
    sitekey: str
    domain: str
    captcha_type: str
    pageurl: str
    min_tokens: int = 3
    max_tokens: int = 10


class VerifySiteRequestModel(BaseModel):
    url: str = Field(..., description="URL to verify captcha on")


class VerifySiteResponseModel(BaseModel):
    captcha_detected: bool = False
    captcha_type: str | None = None
    sitekey_present: bool = False
    engine_selected: str | None = None
    fallback_chain: list[str] = []
    solve_attempted: bool = False
    solve_success: bool = False
    token: str | None = None
    latency_ms: int = 0
    confidence_score: float = 0.0
    error: str | None = None


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

    detection = None
    result = await dispatcher.solve(solve_req, detection)

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
    """Detect captcha type from HTML source. Uses domain cache for repeat lookups."""
    # Check detector cache first
    from_cache = False
    if req.url and config.detector_cache_enabled:
        from urllib.parse import urlparse
        domain = urlparse(req.url).netloc
        cached_type = get_cached_detection(domain)
        if cached_type:
            from_cache = True
            return DetectResponseModel(
                captcha_type=cached_type.value,
                confidence=0.95,
                cached=True,
            )

    result = detect_from_html(req.html)

    # Cache the detection result for this domain
    if req.url and config.detector_cache_enabled:
        from urllib.parse import urlparse
        domain = urlparse(req.url).netloc
        if domain:
            from router.dispatcher import cache_detection
            cache_detection(domain, result.captcha_type)

    return DetectResponseModel(
        captcha_type=result.captcha_type.value,
        sitekey=result.sitekey,
        iframe_src=result.iframe_src,
        confidence=result.confidence,
        cached=False,
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


@app.post("/run_random_tests")
async def run_random_tests():
    """Trigger a single random-site test evaluation pass. Returns results."""
    summary = await test_runner.run_once()
    return summary


@app.post("/benchmark/run")
async def run_benchmark():
    """Run a full benchmark evaluation pass. Returns summary with fastest/slowest engines."""
    summary = await test_runner.run_once()
    return {
        "runs_completed": summary["tests_run"],
        "detections_successful": summary["detections_successful"],
        "solves_successful": summary["solves_successful"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "fastest_engine": summary.get("fastest_engine"),
        "slowest_engine": summary.get("slowest_engine"),
        "rolling_stats": summary.get("rolling_stats", {}),
    }


@app.get("/benchmark/history")
async def benchmark_history(days: int = 7):
    """Get rolling 7-day benchmark performance history."""
    return test_runner.get_history(days)


@app.get("/benchmark/rolling")
async def benchmark_rolling_stats():
    """Get current rolling performance statistics per engine."""
    return test_runner.get_rolling_stats()


@app.get("/test_runner/results")
async def test_runner_results(count: int = 20):
    """Get recent random-site test results."""
    return test_runner.get_latest_results(count)


@app.get("/test_runner/status")
async def test_runner_status():
    """Get test runner scheduler status."""
    return {
        "scheduler_running": test_runner._running,
        "targets": test_runner.targets,
        "interval_hours": config.test_runner_interval_hours,
    }


@app.get("/pool/status")
async def pool_status():
    """Get browser pool status (including adaptive scaling info)."""
    return {
        "active": browser_pool.active_count,
        "busy": browser_pool.busy_count,
        "pool_size": browser_pool.size,
        "max_size": browser_pool.max_size,
        "queue_depth": browser_pool.queue_depth,
    }


@app.get("/vpn/status")
async def vpn_status():
    """Get VPN identity routing status."""
    manager_status = await vpn_manager.get_status()
    scheduler_status = vpn_scheduler.get_status()
    return {
        **manager_status,
        "rotation": scheduler_status,
        "profile_mapping": vpn_profile_map.to_dict(),
    }


@app.post("/vpn/rotate")
async def vpn_force_rotate():
    """Force an immediate VPN region rotation."""
    result = await vpn_scheduler.force_rotate()
    return result


@app.post("/vpn/connect")
async def vpn_connect(region: str = "us"):
    """Connect to a specific VPN region."""
    success = await vpn_manager.connect(region)
    return {
        "success": success,
        "provider": vpn_manager.active_provider_name,
        "region": vpn_manager.active_region,
    }


@app.post("/verify-site", response_model=VerifySiteResponseModel)
async def verify_site(req: VerifySiteRequestModel):
    """
    Full pipeline site verification: render page with browser pool (JS enabled),
    detect captcha (including JS-loaded widgets like MTCaptcha), attempt solve.
    """
    start = time.time()
    result = VerifySiteResponseModel()

    slot_idx = -1
    page = None

    try:
        # Step 1: Acquire browser from pool and navigate with full JS rendering
        runner, slot_idx = await browser_pool.acquire()
        page = await runner.get_page()
        logger.info(f"verify-site: navigating to {req.url}")
        await page.goto(req.url, wait_until="networkidle", timeout=30000)
        # Extra wait for JS-loaded captcha widgets (MTCaptcha, etc.)
        await asyncio.sleep(3)

        # Step 2: Use detect_from_page which checks rendered DOM + JS globals
        detection = await detect_from_page(page)
        logger.info(f"verify-site: detection result = {detection.captcha_type.value}, sitekey={detection.sitekey}")

        result.captcha_type = detection.captcha_type.value
        result.captcha_detected = detection.captcha_type != CaptchaType.NONE
        result.sitekey_present = bool(detection.sitekey)
        result.confidence_score = detection.confidence

        if not result.captcha_detected:
            # Check iframes (MTCaptcha loads via iframe from service.mtcaptcha.com)
            for frame in page.frames:
                if frame == page.main_frame:
                    continue
                try:
                    frame_html = await frame.content()
                    frame_det = detect_from_html(frame_html)
                    if frame_det.captcha_type != CaptchaType.NONE:
                        detection = frame_det
                        result.captcha_type = detection.captcha_type.value
                        result.captcha_detected = True
                        result.sitekey_present = bool(detection.sitekey)
                        result.confidence_score = detection.confidence
                        break
                except Exception:
                    continue

        if not result.captcha_detected:
            # Visual element fallback
            captcha_el = await page.query_selector(
                'div[id*="mtcaptcha"], div[class*="mtcaptcha"], '
                'img[src*="captcha"], canvas[class*="captcha"], '
                'div[id*="captcha"] img'
            )
            if captcha_el:
                result.captcha_detected = True
                result.captcha_type = "mtcaptcha"
                result.confidence_score = 0.8

        if not result.captcha_detected:
            result.latency_ms = int((time.time() - start) * 1000)
            return result

        # Step 3: Attempt to solve
        result.solve_attempted = True

        is_text_captcha = result.captcha_type in ("text", "text_image", "image_grid", "mtcaptcha")

        if is_text_captcha:
            img_bytes = None
            captcha_el = None

            # Strategy 1: For MTCaptcha, screenshot the widget container from main page
            # (cross-origin iframes block direct element access, but we can screenshot the visible widget)
            if result.captcha_type == "mtcaptcha":
                # Find the MTCaptcha container div on the main page
                for selector in [
                    'div[id*="mtcap"]',
                    'div[class*="mtcaptcha"]',
                    'div[id*="mtcaptcha"]',
                    'iframe[src*="mtcaptcha"]',
                ]:
                    captcha_el = await page.query_selector(selector)
                    if captcha_el:
                        break

                if captcha_el:
                    img_bytes = await captcha_el.screenshot()
                    logger.info("verify-site: screenshotted MTCaptcha widget container")

            # Strategy 2: Search iframe contents for captcha image elements
            if not img_bytes:
                for frame in page.frames:
                    if frame == page.main_frame:
                        continue
                    try:
                        el = await frame.query_selector(
                            'img.mtcaptcha-image-text, '
                            'img[class*="mtcaptcha"], '
                            'canvas.mtcaptcha-canvas, '
                            'div.mtcaptcha-image img, '
                            'img[src*="captcha"]'
                        )
                        if el:
                            img_bytes = await el.screenshot()
                            logger.info("verify-site: screenshotted captcha image from iframe")
                            break
                    except Exception:
                        continue

            # Strategy 3: Check main page for captcha images
            if not img_bytes:
                for selector in [
                    'div[id*="mtcaptcha"] img',
                    'img[src*="captcha"]',
                    'canvas[class*="captcha"]',
                    'div[id*="captcha"] img',
                ]:
                    captcha_el = await page.query_selector(selector)
                    if captcha_el:
                        img_bytes = await captcha_el.screenshot()
                        logger.info(f"verify-site: screenshotted captcha element via {selector}")
                        break

            if img_bytes:
                # Solve via OCR engine
                ocr_engine = dispatcher.engines.get("ocr")
                if ocr_engine:
                    solve_result = await ocr_engine.solve(
                        captcha_type="text",
                        pageurl=req.url,
                        sitekey=detection.sitekey,
                        image_data=img_bytes,
                    )
                    result.engine_selected = "ocr"
                    result.fallback_chain = ["ocr", "vision"]
                    result.solve_success = solve_result.get("success", False)
                    result.token = solve_result.get("token")
                    result.confidence_score = solve_result.get("confidence", 0.0)

                    # If OCR solved, try to type into the captcha input
                    if result.solve_success and result.token:
                        captcha_input = None
                        # Search all frames for input field
                        for frame in page.frames:
                            try:
                                inp = await frame.query_selector(
                                    'input[name*="captcha"], '
                                    'input[id*="mtcaptcha"], '
                                    'input[placeholder*="captcha"], '
                                    'input[aria-label*="captcha"]'
                                )
                                if inp:
                                    captcha_input = inp
                                    break
                            except Exception:
                                continue
                        if captcha_input:
                            await captcha_input.click()
                            await captcha_input.fill("")
                            await captcha_input.type(result.token, delay=80)
                        else:
                            result.error = "Solved but could not find captcha input field"
                else:
                    result.error = "OCR engine not available"

                # Fallback to vision if OCR failed
                if not result.solve_success:
                    vision_engine = dispatcher.engines.get("vision")
                    if vision_engine:
                        try:
                            v_result = await vision_engine.solve(
                                captcha_type="image_grid",
                                pageurl=req.url,
                                image_data=img_bytes,
                            )
                            if v_result.get("success"):
                                result.engine_selected = "vision"
                                result.solve_success = True
                                result.token = v_result.get("token")
                                result.confidence_score = v_result.get("confidence", 0.0)
                        except Exception as ve:
                            logger.debug(f"Vision fallback failed: {ve}")
            else:
                result.error = "Captcha detected but could not locate image element for OCR"
        else:
            # Widget-based captchas (reCAPTCHA, hCaptcha, Turnstile) — use dispatcher racing
            solve_req = SolveRequest(
                captcha_type=detection.captcha_type.value,
                pageurl=req.url,
                sitekey=detection.sitekey,
            )
            solve_result = await dispatcher.solve(solve_req, detection)
            result.engine_selected = solve_result.engine_used
            result.fallback_chain = [solve_result.engine_used] if solve_result.engine_used else []
            result.solve_success = solve_result.success
            result.token = solve_result.token
            result.confidence_score = solve_result.confidence
            if solve_result.error:
                result.error = solve_result.error

    except Exception as e:
        result.error = str(e)
        logger.error(f"verify-site error for {req.url}: {e}")
    finally:
        if page:
            try:
                await page.close()
            except Exception:
                pass
        if slot_idx >= 0:
            browser_pool.release(slot_idx)
        result.latency_ms = int((time.time() - start) * 1000)

    return result


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "2.3.0",
        "engines": list(dispatcher.engines.keys()),
        "cache": cache.get_stats(),
        "browser_pool": {
            "active": browser_pool.active_count,
            "busy": browser_pool.busy_count,
            "max_size": browser_pool.max_size,
            "queue_depth": browser_pool.queue_depth,
        },
        "vpn": {
            "enabled": config.vpn_enabled,
            "connected": vpn_manager.is_connected,
            "provider": vpn_manager.active_provider_name,
            "region": vpn_manager.active_region,
            "available_providers": vpn_manager.available_providers,
            "rotation_interval": config.vpn_rotation_interval,
        },
        "features": {
            "engine_racing": config.engine_racing_enabled,
            "arbitration_window_ms": config.arbitration_window_ms,
            "adaptive_routing": config.adaptive_routing_enabled,
            "detector_cache": config.detector_cache_enabled,
            "memory_cache": config.cache_use_memory,
            "whisper_model": config.whisper_model,
            "browser_pool_size": config.browser_pool_size,
            "browser_pool_max_size": config.browser_pool_max_size,
            "clip_mode": "lazy_load",
            "vpn_identity_routing": config.vpn_enabled,
        },
    }


@app.get("/")
async def root():
    return {
        "service": "Captcha Solver Core",
        "version": "2.3.0",
        "endpoints": {
            "POST /solve": "Solve a captcha (main endpoint)",
            "POST /solve/image": "Solve from uploaded image",
            "POST /detect": "Detect captcha type from HTML (cached by domain)",
            "POST /benchmark/run": "Run full benchmark evaluation",
            "GET /benchmark/history": "Rolling 7-day benchmark history",
            "GET /benchmark/rolling": "Rolling engine performance stats",
            "POST /run_random_tests": "Run random site captcha tests",
            "GET /test_runner/results": "Get recent test results",
            "GET /test_runner/status": "Test runner scheduler status",
            "POST /harvest/add": "Add pre-harvest target",
            "POST /harvest/start": "Start token pre-harvest daemon",
            "GET /harvest/status": "Pre-harvest daemon status",
            "GET /cache/stats": "Token cache statistics",
            "GET /stats": "Solver telemetry",
            "GET /pool/status": "Browser pool status (adaptive scaling)",
            "GET /vpn/status": "VPN identity routing status",
            "POST /vpn/rotate": "Force VPN region rotation",
            "POST /vpn/connect": "Connect to specific VPN region",
            "GET /health": "Health check",
        },
    }
