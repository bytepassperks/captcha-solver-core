"""Token pre-harvest daemon — background worker that collects tokens before they're needed."""

import asyncio
import logging
import time
from dataclasses import dataclass

from captcha_solver_core.config import config
from captcha_solver_core.cache.token_cache import TokenCache
from captcha_solver_core.browser.persistent_runner import PersistentBrowserRunner
from captcha_solver_core.engines.token_engine import TokenEngine
from captcha_solver_core.detector.captcha_detector import CaptchaType

logger = logging.getLogger(__name__)


@dataclass
class HarvestTarget:
    """A target to pre-harvest tokens for."""
    sitekey: str
    domain: str
    captcha_type: CaptchaType
    pageurl: str
    min_tokens: int = 3
    max_tokens: int = 10


class PreharvestDaemon:
    """
    Background worker that:
    1. Launches browser profiles
    2. Opens captcha widgets
    3. Collects tokens early
    4. Stores tokens in cache
    5. Refreshes before expiry

    Transforms solver from reactive to a token reservoir.
    """

    def __init__(self, cache: TokenCache | None = None):
        self.cache = cache or TokenCache()
        self.targets: list[HarvestTarget] = []
        self._running = False
        self._tasks: list[asyncio.Task] = []

    def add_target(self, sitekey: str, domain: str, captcha_type: str,
                   pageurl: str, min_tokens: int = 3, max_tokens: int = 10):
        """Add a sitekey/domain pair to pre-harvest tokens for."""
        target = HarvestTarget(
            sitekey=sitekey,
            domain=domain,
            captcha_type=CaptchaType(captcha_type),
            pageurl=pageurl,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        self.targets.append(target)
        logger.info(f"Added harvest target: {domain}/{sitekey[:8]}... (min={min_tokens})")

    async def _harvest_loop(self, target: HarvestTarget, profile_name: str):
        """Continuously harvest tokens for a single target."""
        runner = PersistentBrowserRunner(profile_name=profile_name)
        engine = TokenEngine(browser_runner=runner)

        try:
            await runner.start()

            while self._running:
                available = self.cache.get_available_count(
                    target.sitekey, target.domain, target.captcha_type.value
                )

                if available >= target.max_tokens:
                    logger.debug(f"Token pool full for {target.domain} ({available}/{target.max_tokens})")
                    await asyncio.sleep(30)
                    continue

                if available < target.min_tokens:
                    needed = target.max_tokens - available
                    logger.info(f"Harvesting {needed} tokens for {target.domain}")

                    for _ in range(needed):
                        if not self._running:
                            break

                        result = await engine.solve(
                            captcha_type=target.captcha_type,
                            pageurl=target.pageurl,
                            sitekey=target.sitekey,
                        )

                        if result.get("success") and result.get("token"):
                            self.cache.store(
                                sitekey=target.sitekey,
                                domain=target.domain,
                                captcha_type=target.captcha_type.value,
                                token=result["token"],
                            )
                            logger.info(f"Pre-harvested token for {target.domain}")
                        else:
                            logger.debug(f"Harvest attempt failed: {result.get('error')}")

                        await asyncio.sleep(5)  # Brief pause between attempts

                # Cleanup expired tokens
                self.cache.cleanup_expired()

                # Wait before next check
                await asyncio.sleep(config.token_ttl_seconds // 3)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Harvest loop failed for {target.domain}: {e}")
        finally:
            await runner.stop()

    async def start(self):
        """Start the pre-harvest daemon."""
        self._running = True
        logger.info(f"Pre-harvest daemon starting with {len(self.targets)} targets")

        for i, target in enumerate(self.targets):
            profile_name = f"harvester_{i:03d}"
            task = asyncio.create_task(
                self._harvest_loop(target, profile_name),
                name=f"harvest_{target.domain}",
            )
            self._tasks.append(task)

    async def stop(self):
        """Stop the daemon and all harvest loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Pre-harvest daemon stopped")

    def get_status(self) -> dict:
        """Get current daemon status."""
        return {
            "running": self._running,
            "targets": len(self.targets),
            "active_tasks": len([t for t in self._tasks if not t.done()]),
            "cache_stats": self.cache.get_stats(),
        }
