"""Profile scheduler — rotates browser profiles and warms cookies on schedule."""

import asyncio
import random
import logging
import json
import time
from pathlib import Path
from datetime import datetime

from config import config
from browser.persistent_runner import PersistentBrowserRunner
from engines.behavior_engine import BehaviorEngine

logger = logging.getLogger(__name__)


class ProfileScheduler:
    """
    Manages a pool of browser profiles. Periodically warms them up
    by visiting safe domains to build cookie reputation and session trust.
    """

    def __init__(self):
        self.profiles: list[str] = []
        self.state_file = config.profiles_dir / "scheduler_state.json"
        self._running = False
        self._load_state()

    def _load_state(self):
        """Load scheduler state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.profiles = state.get("profiles", [])
            except Exception:
                self.profiles = []

        # Ensure minimum profile count
        while len(self.profiles) < config.farm_profile_count:
            name = f"profile_{len(self.profiles):03d}"
            self.profiles.append(name)

        self._save_state()

    def _save_state(self):
        """Persist scheduler state to disk."""
        state = {
            "profiles": self.profiles,
            "last_updated": datetime.utcnow().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def get_best_profile(self) -> str:
        """Select the profile with the best reputation (most recently warmed)."""
        profile_scores = []
        for name in self.profiles:
            profile_dir = Path(config.browser_user_data_base) / name
            if profile_dir.exists():
                # Score based on recency of cookie data
                cookie_files = list(profile_dir.rglob("Cookies*"))
                if cookie_files:
                    latest = max(f.stat().st_mtime for f in cookie_files)
                    profile_scores.append((name, latest))
                else:
                    profile_scores.append((name, 0))
            else:
                profile_scores.append((name, 0))

        if profile_scores:
            profile_scores.sort(key=lambda x: x[1], reverse=True)
            return profile_scores[0][0]

        return self.profiles[0]

    async def warm_profile(self, profile_name: str):
        """Warm a single profile by browsing safe sites."""
        logger.info(f"Warming profile: {profile_name}")

        runner = PersistentBrowserRunner(profile_name=profile_name)
        behavior = BehaviorEngine()

        try:
            await runner.start()
            page = await runner.get_page()

            sites = config.farm_warmup_sites.copy()
            random.shuffle(sites)

            for site in sites[:4]:
                try:
                    await page.goto(site, timeout=15000, wait_until="domcontentloaded")
                    # Simulate natural browsing
                    await behavior.warm_page(page, site, duration_seconds=config.farm_warmup_duration_min * 10)
                    await asyncio.sleep(random.uniform(1, 3))
                except Exception as e:
                    logger.debug(f"Failed to warm {site}: {e}")

            logger.info(f"Profile {profile_name} warmed successfully")
        except Exception as e:
            logger.error(f"Failed to warm profile {profile_name}: {e}")
        finally:
            await runner.stop()

    async def warm_all_profiles(self):
        """Warm all profiles sequentially."""
        for name in self.profiles:
            await self.warm_profile(name)
            await asyncio.sleep(random.uniform(2, 5))

    async def run_scheduler(self):
        """Run the profile warming scheduler as a background task."""
        self._running = True
        logger.info("Profile scheduler started")

        while self._running:
            try:
                await self.warm_all_profiles()
            except Exception as e:
                logger.error(f"Scheduler cycle failed: {e}")

            # Sleep until next cycle
            interval = config.profile_rotation_interval_hours * 3600
            jitter = random.uniform(-300, 300)
            sleep_time = max(60, interval + jitter)
            logger.info(f"Next warmup cycle in {sleep_time / 3600:.1f} hours")
            await asyncio.sleep(sleep_time)

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        self._save_state()
        logger.info("Profile scheduler stopped")
