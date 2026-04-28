"""Route captcha solve requests to the appropriate engine with adaptive routing, racing, and arbitration."""

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
    CaptchaType.MTCAPTCHA: ["ocr", "vision"],
}

# Domain -> captcha_type detection cache (Speed Boost: detector caching)
_detection_cache: dict[str, CaptchaType] = {}


def cache_detection(domain: str, ctype: CaptchaType):
    """Cache a detection result for a domain (saves 150-400ms on repeat visits)."""
    if ctype != CaptchaType.NONE:
        _detection_cache[domain] = ctype


def get_cached_detection(domain: str) -> CaptchaType | None:
    """Get cached detection for a domain."""
    return _detection_cache.get(domain)


class Dispatcher:
    """Routes captcha solving with adaptive priority, parallel racing, and arbitration window."""

    def __init__(self, engines: dict):
        self.engines = engines
        self._stats_fn = None
        # Rolling engine performance for telemetry-driven ordering
        self._engine_perf: dict[str, dict] = {}

    def set_stats_provider(self, stats_fn):
        """Set a callable that returns solve stats for adaptive routing."""
        self._stats_fn = stats_fn

    def _record_engine_result(self, engine_name: str, success: bool, latency_ms: int):
        """Record engine performance for rolling telemetry-driven ordering."""
        if engine_name not in self._engine_perf:
            self._engine_perf[engine_name] = {"successes": 0, "total": 0, "latency_sum": 0}
        perf = self._engine_perf[engine_name]
        perf["total"] += 1
        if success:
            perf["successes"] += 1
        perf["latency_sum"] += latency_ms

    def _get_adaptive_priority(self, ctype: CaptchaType) -> list[str]:
        """Reorder engine priority based on historical success rate and latency."""
        base_priority = DEFAULT_ROUTING.get(ctype, [])
        if not config.adaptive_routing_enabled or len(base_priority) <= 1:
            return base_priority

        # Use rolling stats first, fall back to telemetry DB
        scored = []
        for eng_name in base_priority:
            perf = self._engine_perf.get(eng_name, {})
            total = perf.get("total", 0)

            if total < config.adaptive_min_samples:
                scored.append((eng_name, -1))
                continue

            success_rate = (perf["successes"] / total * 100) if total > 0 else 0
            avg_ms = (perf["latency_sum"] / total) if total > 0 else 99999
            score = success_rate * 1000 - avg_ms * 0.1
            scored.append((eng_name, score))

        has_data = [(n, s) for n, s in scored if s >= 0]
        no_data = [n for n, s in scored if s < 0]

        if not has_data:
            # Fall back to stats provider
            if self._stats_fn:
                return self._get_priority_from_stats(ctype, base_priority)
            return base_priority

        has_data.sort(key=lambda x: x[1], reverse=True)
        reordered = [n for n, _ in has_data] + no_data

        if reordered != base_priority:
            logger.info(f"Adaptive routing for {ctype.value}: {base_priority} -> {reordered}")
        return reordered

    def _get_priority_from_stats(self, ctype: CaptchaType, base_priority: list[str]) -> list[str]:
        """Fall back to external stats for routing decisions."""
        try:
            stats = self._stats_fn(hours=24)
            by_engine = stats.get("by_engine", {})

            scored = []
            for eng_name in base_priority:
                eng_stats = by_engine.get(eng_name, {})
                total = eng_stats.get("total", 0)
                if total < config.adaptive_min_samples:
                    scored.append((eng_name, -1))
                    continue
                success_rate = eng_stats.get("success_rate", 0)
                avg_ms = eng_stats.get("avg_solve_ms", 99999)
                score = success_rate * 1000 - avg_ms * 0.1
                scored.append((eng_name, score))

            has_data = [(n, s) for n, s in scored if s >= 0]
            no_data = [n for n, s in scored if s < 0]
            has_data.sort(key=lambda x: x[1], reverse=True)
            return [n for n, _ in has_data] + no_data
        except Exception:
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
        """Race engines in parallel with 250ms arbitration window for best-result selection."""
        tasks = {}
        for engine, engine_name in engines_to_race:
            task = asyncio.create_task(
                self._solve_single_engine(engine, engine_name, request, detection),
                name=f"race_{engine_name}",
            )
            tasks[task] = engine_name

        first_success: SolveResult | None = None
        last_error = None

        try:
            pending = set(tasks.keys())
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    eng_name = tasks[task]
                    try:
                        result = task.result()
                        elapsed_ms = int((time.time() - start) * 1000)

                        if result.get("success"):
                            candidate = SolveResult(
                                success=True,
                                token=result.get("token"),
                                engine_used=eng_name,
                                confidence=result.get("confidence", 0.0),
                                solve_time_ms=elapsed_ms,
                            )
                            self._record_engine_result(eng_name, True, elapsed_ms)

                            if first_success is None:
                                first_success = candidate

                                # Arbitration window: wait up to 250ms for potentially better results
                                if pending and config.arbitration_window_ms > 0:
                                    try:
                                        arb_done, pending = await asyncio.wait(
                                            pending,
                                            timeout=config.arbitration_window_ms / 1000,
                                        )
                                        for arb_task in arb_done:
                                            arb_name = tasks[arb_task]
                                            try:
                                                arb_result = arb_task.result()
                                                arb_elapsed = int((time.time() - start) * 1000)
                                                if arb_result.get("success"):
                                                    arb_candidate = SolveResult(
                                                        success=True,
                                                        token=arb_result.get("token"),
                                                        engine_used=arb_name,
                                                        confidence=arb_result.get("confidence", 0.0),
                                                        solve_time_ms=arb_elapsed,
                                                    )
                                                    self._record_engine_result(arb_name, True, arb_elapsed)
                                                    # Pick higher confidence or faster result
                                                    if arb_candidate.confidence > first_success.confidence:
                                                        first_success = arb_candidate
                                                else:
                                                    self._record_engine_result(arb_name, False, arb_elapsed)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                # Cancel remaining and return best
                                for p in pending:
                                    p.cancel()
                                return first_success
                            else:
                                # Already have a winner, compare
                                if candidate.confidence > first_success.confidence:
                                    first_success = candidate
                        else:
                            self._record_engine_result(eng_name, False, elapsed_ms)
                            last_error = result.get("error", f"{eng_name} failed")
                            logger.info(f"Racing: {eng_name} returned success=false: {last_error}")
                    except Exception as e:
                        last_error = str(e)
                        logger.exception(f"Racing: {eng_name} raised exception: {e}")
        except Exception as e:
            last_error = str(e)

        if first_success:
            return first_success
        return None

    async def solve(self, request: SolveRequest, detection: DetectionResult | None = None) -> SolveResult:
        """Solve a captcha using adaptive routing with engine racing and arbitration."""
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

        # Cache detection for this domain
        if request.pageurl:
            from urllib.parse import urlparse
            domain = urlparse(request.pageurl).netloc
            if domain:
                cache_detection(domain, ctype)

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
                    self._record_engine_result(try_name, True, elapsed_ms)
                    return SolveResult(
                        success=True,
                        token=result.get("token"),
                        engine_used=try_name,
                        confidence=result.get("confidence", 0.0),
                        solve_time_ms=elapsed_ms,
                    )

                elapsed_ms = int((time.time() - start) * 1000)
                self._record_engine_result(try_name, False, elapsed_ms)
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
