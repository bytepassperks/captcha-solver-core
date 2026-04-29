"""Persistent browser pool with adaptive scaling — pre-launched Chromium contexts for instant page creation."""

import asyncio
import logging
import random
from pathlib import Path

from config import config
from browser.persistent_runner import PersistentBrowserRunner

logger = logging.getLogger(__name__)


class BrowserPool:
    """
    Maintains a pool of pre-launched browser contexts with adaptive scaling.
    Eliminates 2.5s browser launch overhead per request.

    When all slots are busy and queue_length > 2, auto-scales up to max_size.
    """

    def __init__(self, size: int | None = None, max_size: int | None = None):
        self.size = size or config.browser_pool_size
        self.max_size = max_size or config.browser_pool_max_size
        self._runners: list[PersistentBrowserRunner] = []
        self._locks: list[asyncio.Lock] = []
        self._initialized = False
        self._scaling_lock = asyncio.Lock()
        self._queue_depth = 0

    async def start(self):
        """Pre-launch all browser contexts in the pool."""
        if self._initialized:
            return

        logger.info(f"Starting browser pool with {self.size} contexts (max={self.max_size})...")
        for i in range(self.size):
            await self._add_slot(i)

        self._initialized = True
        logger.info(f"Browser pool started: {len(self._runners)}/{self.size} slots active")

    async def _add_slot(self, index: int | None = None) -> bool:
        """Add a new browser slot to the pool."""
        if index is None:
            index = len(self._runners)
        if index >= self.max_size:
            return False

        profile_name = f"pool_{index:03d}"
        runner = PersistentBrowserRunner(profile_name=profile_name)
        try:
            await runner.start()
            self._runners.append(runner)
            self._locks.append(asyncio.Lock())
            logger.info(f"Pool slot {index} ready (profile={profile_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to start pool slot {index}: {e}")
            return False

    async def _maybe_scale_up(self):
        """Auto-scale the pool if all slots are busy and demand is high."""
        if len(self._runners) >= self.max_size:
            return
        if self._queue_depth <= 2:
            return

        async with self._scaling_lock:
            if len(self._runners) >= self.max_size:
                return
            new_idx = len(self._runners)
            logger.info(f"Adaptive scaling: adding pool slot {new_idx} (queue_depth={self._queue_depth})")
            await self._add_slot(new_idx)

    async def acquire(self) -> tuple[PersistentBrowserRunner, int]:
        """
        Acquire the first available browser runner from the pool.
        Returns (runner, slot_index). Caller must call release(slot_index) when done.
        """
        if not self._initialized:
            await self.start()

        self._queue_depth += 1

        # Try to acquire any unlocked slot
        for idx, lock in enumerate(self._locks):
            if not lock.locked():
                await lock.acquire()
                self._queue_depth -= 1
                return self._runners[idx], idx

        # All slots busy — try adaptive scaling
        await self._maybe_scale_up()

        # Wait for the first available
        slot_idx = random.randint(0, len(self._locks) - 1)
        await self._locks[slot_idx].acquire()
        self._queue_depth -= 1
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

    @property
    def queue_depth(self) -> int:
        return self._queue_depth
