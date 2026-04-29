"""VPN rotation scheduler — rotates VPN regions for browser profiles on a configurable interval."""

import asyncio
import logging
from datetime import datetime

from config import config
from vpn_router.manager import VPNManager
from vpn_router.profile_map import ProfileRegionMap

logger = logging.getLogger(__name__)


class VPNRotationScheduler:
    """Rotates VPN regions every N seconds (default 4 hours = 14400s) for all active profiles."""

    def __init__(self, manager: VPNManager, profile_map: ProfileRegionMap):
        self.manager = manager
        self.profile_map = profile_map
        self._running = False
        self._task: asyncio.Task | None = None
        self._rotation_count = 0
        self._last_rotation: datetime | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._rotation_loop())
        logger.info(
            f"VPN rotation scheduler started (interval={config.vpn_rotation_interval}s)"
        )

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("VPN rotation scheduler stopped")

    async def _rotation_loop(self):
        # Initial delay to let browser pool warm up
        await asyncio.sleep(120)

        while self._running:
            try:
                await self._rotate()
            except Exception as e:
                logger.error(f"VPN rotation error: {e}")

            await asyncio.sleep(config.vpn_rotation_interval)

    async def _rotate(self):
        if not config.vpn_enabled:
            return

        rotated = self.profile_map.rotate_all()
        if not rotated:
            logger.debug("No profiles to rotate")
            return

        # Connect to the region of the first profile (shared connection model)
        # In a multi-node deployment, each node would manage its own region
        first = rotated[0]
        success = await self.manager.connect(first.region)

        self._rotation_count += 1
        self._last_rotation = datetime.utcnow()

        logger.info(
            f"VPN rotation #{self._rotation_count}: "
            f"{len(rotated)} profiles rotated, "
            f"connected={'yes' if success else 'no'} → {first.region}"
        )

    async def force_rotate(self) -> dict:
        await self._rotate()
        return {
            "rotation_count": self._rotation_count,
            "last_rotation": self._last_rotation.isoformat() if self._last_rotation else None,
            "profiles": self.profile_map.to_dict(),
        }

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "rotation_count": self._rotation_count,
            "last_rotation": self._last_rotation.isoformat() if self._last_rotation else None,
            "interval_seconds": config.vpn_rotation_interval,
            "profiles": self.profile_map.to_dict(),
        }
