"""Persistent browser pool — pre-launched Chromium contexts for instant page creation."""

import asyncio
import logging
import random
from pathlib import Path

from config import config
from browser.persistent_runner import PersistentBrowserRunner

logger = logging.getLogger(__name__)


class BrowserPool:
    """
    Maintains a pool of pre-launched browser contexts.
    Eliminates 2.5s browser launch overhead per request.

    Each pool slot is a PersistentBrowserRunner with its own profile directory,
    allowing cookie/session isolation between concurrent tasks.
    """

    def __init__(self, size: int | None = None):
        self.size = size or config.browser_pool_size
        self._runners: list[PersistentBrowserRunner] = []
        self._locks: list[asyncio.Lock] = []
        self._initialized = False

    async def start(self):
        """Pre-launch all browser contexts in the pool."""
        if self._initialized:
            return

        logger.info(f"Starting browser pool with {self.size} contexts...")
        for i in range(self.size):
            profile_name = f"pool_{i:03d}"
            runner = PersistentBrowserRunner(profile_name=profile_name)
            try:
                await runner.start()
                self._runners.append(runner)
                self._locks.append(asyncio.Lock())
                logger.info(f"Pool slot {i} ready (profile={profile_name})")
            except Exception as e:
                logger.error(f"Failed to start pool slot {i}: {e}")

        self._initialized = True
        logger.info(f"Browser pool started: {len(self._runners)}/{self.size} slots active")

    async def acquire(self) -> tuple[PersistentBrowserRunner, int]:
        """
        Acquire the first available browser runner from the pool.
        Returns (runner, slot_index). Caller must call release(slot_index) when done.
        """
        if not self._initialized:
            await self.start()

        # Try to acquire any unlocked slot
        for idx, lock in enumerate(self._locks):
            if not lock.locked():
                await lock.acquire()
                return self._runners[idx], idx

        # All slots busy — wait for the first available
        slot_idx = random.randint(0, len(self._locks) - 1)
        await self._locks[slot_idx].acquire()
        return self._runners[slot_idx], slot_idx

    def release(self, slot_index: int):
        """Release a browser slot back to the pool."""
        if 0 <= slot_index < len(self._locks):
            if self._locks[slot_index].locked():
                self._locks[slot_index].release()

    async def get_page(self):
        """Get a page from any available pool runner (convenience method)."""
        if not self._initialized:
            await self.start()

        if not self._runners:
            return None

        # Round-robin through runners without locking (for non-exclusive use)
        runner = random.choice(self._runners)
        return await runner.get_page()

    async def stop(self):
        """Shut down all browser contexts in the pool."""
        for runner in self._runners:
            try:
                await runner.stop()
            except Exception as e:
                logger.error(f"Error stopping pool runner: {e}")
        self._runners.clear()
        self._locks.clear()
        self._initialized = False
        logger.info("Browser pool stopped")

    @property
    def active_count(self) -> int:
        return len(self._runners)

    @property
    def busy_count(self) -> int:
        return sum(1 for lock in self._locks if lock.locked())
