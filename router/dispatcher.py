"""Route captcha solve requests to the appropriate engine with adaptive routing and racing."""

import asyncio
import logging
import time
from dataclasses import dataclass

from config import config
from detector.captcha_detector import CaptchaType, DetectionResult

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


# Default routing table
DEFAULT_ROUTING = {
    CaptchaType.TEXT: ["ocr"],
    CaptchaType.IMAGE_GRID: ["vision"],
    CaptchaType.RECAPTCHA_V2: ["token_harvest", "audio"],
    CaptchaType.RECAPTCHA_V3: ["token_harvest"],
    CaptchaType.HCAPTCHA: ["vision", "audio"],
    CaptchaType.TURNSTILE: ["behavior"],
}


class Dispatcher:
    """Routes captcha solving requests with adaptive engine priority and optional racing."""

    def __init__(self, engines: dict):
        self.engines = engines
        self._stats_fn = None  # set externally for adaptive routing

    def set_stats_provider(self, stats_fn):
        """Set a callable that returns solve stats for adaptive routing."""
        self._stats_fn = stats_fn

    def _get_adaptive_priority(self, ctype: CaptchaType) -> list[str]:
        """Reorder engine priority based on historical success rate and latency."""
        base_priority = DEFAULT_ROUTING.get(ctype, [])
        if not config.adaptive_routing_enabled or not self._stats_fn or len(base_priority) <= 1:
            return base_priority

        try:
            stats = self._stats_fn(hours=24)
            by_engine = stats.get("by_engine", {})

            scored = []
            for eng_name in base_priority:
                eng_stats = by_engine.get(eng_name, {})
                total = eng_stats.get("total", 0)
                if total < config.adaptive_min_samples:
                    scored.append((eng_name, -1))  # not enough data, keep original position
                    continue
                success_rate = eng_stats.get("success_rate", 0)
                avg_ms = eng_stats.get("avg_solve_ms", 99999)
                # Score: higher success rate is better, lower latency is better
                score = success_rate * 1000 - avg_ms * 0.1
                scored.append((eng_name, score))

            # Sort: engines with data by score (desc), engines without data keep relative order
            has_data = [(n, s) for n, s in scored if s >= 0]
            no_data = [n for n, s in scored if s < 0]

            has_data.sort(key=lambda x: x[1], reverse=True)
            reordered = [n for n, _ in has_data] + no_data

            if reordered != base_priority:
                logger.info(f"Adaptive routing for {ctype.value}: {base_priority} -> {reordered}")
            return reordered

        except Exception as e:
            logger.debug(f"Adaptive routing failed, using default: {e}")
            return base_priority

    async def _solve_single_engine(self, engine, engine_name: str, request: SolveRequest,
                                     detection: DetectionResult | None) -> dict:
        """Run a single engine solve attempt."""
        return await engine.solve(
            captcha_type=CaptchaType(request.captcha_type) if request.captcha_type != "auto" else CaptchaType.NONE,
            pageurl=request.pageurl,
            sitekey=request.sitekey or (detection.sitekey if detection else None),
            image_data=request.captcha_image,
            image_url=request.captcha_image_url,
            extra=request.extra,
        )

    async def _solve_racing(self, engines_to_race: list[tuple], request: SolveRequest,
                             detection: DetectionResult | None, start: float) -> SolveResult | None:
        """Race multiple engines in parallel, return first success."""
        tasks = {}
        for engine, engine_name in engines_to_race:
            task = asyncio.create_task(
                self._solve_single_engine(engine, engine_name, request, detection),
                name=f"race_{engine_name}",
            )
            tasks[task] = engine_name

        last_error = None
        try:
            pending = set(tasks.keys())
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    eng_name = tasks[task]
                    try:
                        result = task.result()
                        if result.get("success"):
                            # Cancel remaining tasks
                            for p in pending:
                                p.cancel()
                            elapsed_ms = int((time.time() - start) * 1000)
                            return SolveResult(
                                success=True,
                                token=result.get("token"),
                                engine_used=eng_name,
                                confidence=result.get("confidence", 0.0),
                                solve_time_ms=elapsed_ms,
                            )
                        last_error = result.get("error", f"{eng_name} failed")
                        logger.info(f"Racing: {eng_name} returned success=false: {last_error}")
                    except Exception as e:
                        last_error = str(e)
                        logger.exception(f"Racing: {eng_name} raised exception: {e}")
        except Exception as e:
            last_error = str(e)

        return None  # no engine succeeded

    async def solve(self, request: SolveRequest, detection: DetectionResult | None = None) -> SolveResult:
        """Solve a captcha using adaptive routing with optional engine racing."""
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

        engine_priority = self._get_adaptive_priority(ctype)
        available_engines = [(self.engines[n], n) for n in engine_priority if n in self.engines]

        if not available_engines:
            return SolveResult(
                success=False,
                error=f"No engine available for captcha type: {ctype.value}",
                engine_used="none",
            )

        first_engine_name = available_engines[0][1]
        logger.info(f"Routing {ctype.value} — priority: {[n for _, n in available_engines]}")

        # Racing mode: if enabled AND multiple engines available, race them
        if config.engine_racing_enabled and len(available_engines) > 1:
            result = await self._solve_racing(available_engines, request, detection, start)
            if result:
                return result
            # All engines failed in racing mode
            elapsed_ms = int((time.time() - start) * 1000)
            return SolveResult(
                success=False,
                error=f"All engines failed (racing mode) for {ctype.value}",
                engine_used=first_engine_name,
                solve_time_ms=elapsed_ms,
            )

        # Sequential fallback mode
        last_error = None
        for engine, try_name in available_engines:
            try:
                result = await self._solve_single_engine(engine, try_name, request, detection)

                if result.get("success"):
                    elapsed_ms = int((time.time() - start) * 1000)
                    return SolveResult(
                        success=True,
                        token=result.get("token"),
                        engine_used=try_name,
                        confidence=result.get("confidence", 0.0),
                        solve_time_ms=elapsed_ms,
                    )

                last_error = result.get("error", f"{try_name} failed")
                logger.info(f"Engine {try_name} returned success=false: {last_error}, trying next...")

            except Exception as e:
                last_error = str(e)
                logger.exception(f"Engine {try_name} raised exception: {e}")

        elapsed_ms = int((time.time() - start) * 1000)
        return SolveResult(
            success=False,
            error=last_error or f"No engine could solve {ctype.value}",
            engine_used=first_engine_name,
            solve_time_ms=elapsed_ms,
        )
