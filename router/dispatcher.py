"""Route captcha solve requests to the appropriate engine."""

import logging
from dataclasses import dataclass

from captcha_solver_core.detector.captcha_detector import CaptchaType, DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class SolveRequest:
    captcha_type: str = "auto"
    pageurl: str = ""
    sitekey: str | None = None
    captcha_image: bytes | None = None
    captcha_image_url: str | None = None
    proxy: str | None = None
    user_agent: str | None = None
    extra: dict | None = None


@dataclass
class SolveResult:
    success: bool
    token: str | None = None
    engine_used: str = ""
    confidence: float = 0.0
    solve_time_ms: int = 0
    error: str | None = None


class Dispatcher:
    """Routes captcha solving requests to the correct engine."""

    def __init__(self, engines: dict):
        self.engines = engines

    def _select_engine(self, captcha_type: CaptchaType):
        """Select the best engine for a given captcha type."""
        routing_table = {
            CaptchaType.TEXT: ["ocr"],
            CaptchaType.IMAGE_GRID: ["vision"],
            CaptchaType.RECAPTCHA_V2: ["token_harvest", "audio"],
            CaptchaType.RECAPTCHA_V3: ["token_harvest"],
            CaptchaType.HCAPTCHA: ["vision", "audio"],
            CaptchaType.TURNSTILE: ["behavior"],
        }

        engine_priority = routing_table.get(captcha_type, [])

        for engine_name in engine_priority:
            if engine_name in self.engines:
                return self.engines[engine_name], engine_name

        return None, None

    async def solve(self, request: SolveRequest, detection: DetectionResult | None = None) -> SolveResult:
        """Solve a captcha using the appropriate engine."""
        import time
        start = time.time()

        if detection is None:
            ctype = CaptchaType(request.captcha_type) if request.captcha_type != "auto" else CaptchaType.NONE
        else:
            ctype = detection.captcha_type

        if ctype == CaptchaType.NONE:
            return SolveResult(
                success=False,
                error="Could not determine captcha type. Provide captcha_type explicitly.",
                engine_used="none",
            )

        engine, engine_name = self._select_engine(ctype)
        if engine is None:
            return SolveResult(
                success=False,
                error=f"No engine available for captcha type: {ctype.value}",
                engine_used="none",
            )

        logger.info(f"Routing {ctype.value} to engine: {engine_name}")

        try:
            result = await engine.solve(
                captcha_type=ctype,
                pageurl=request.pageurl,
                sitekey=request.sitekey or (detection.sitekey if detection else None),
                image_data=request.captcha_image,
                image_url=request.captcha_image_url,
                extra=request.extra,
            )

            elapsed_ms = int((time.time() - start) * 1000)

            return SolveResult(
                success=result.get("success", False),
                token=result.get("token"),
                engine_used=engine_name,
                confidence=result.get("confidence", 0.0),
                solve_time_ms=elapsed_ms,
                error=result.get("error"),
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            logger.exception(f"Engine {engine_name} failed: {e}")

            # Try fallback engines
            routing_table = {
                CaptchaType.RECAPTCHA_V2: ["audio", "token_harvest"],
                CaptchaType.HCAPTCHA: ["audio", "vision"],
            }
            fallbacks = routing_table.get(ctype, [])
            for fb_name in fallbacks:
                if fb_name != engine_name and fb_name in self.engines:
                    logger.info(f"Trying fallback engine: {fb_name}")
                    try:
                        fb_engine = self.engines[fb_name]
                        result = await fb_engine.solve(
                            captcha_type=ctype,
                            pageurl=request.pageurl,
                            sitekey=request.sitekey or (detection.sitekey if detection else None),
                            image_data=request.captcha_image,
                            image_url=request.captcha_image_url,
                            extra=request.extra,
                        )
                        elapsed_ms = int((time.time() - start) * 1000)
                        return SolveResult(
                            success=result.get("success", False),
                            token=result.get("token"),
                            engine_used=fb_name,
                            confidence=result.get("confidence", 0.0),
                            solve_time_ms=elapsed_ms,
                            error=result.get("error"),
                        )
                    except Exception as fb_e:
                        logger.warning(f"Fallback engine {fb_name} also failed: {fb_e}")

            return SolveResult(
                success=False,
                error=f"All engines failed: {str(e)}",
                engine_used=engine_name,
                solve_time_ms=elapsed_ms,
            )
