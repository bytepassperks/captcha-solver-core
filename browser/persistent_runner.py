"""Persistent browser runner using Playwright with stateful profiles."""

import asyncio
import logging
import random
from pathlib import Path

from captcha_solver_core.config import config

logger = logging.getLogger(__name__)


class PersistentBrowserRunner:
    """
    Manages persistent Playwright browser instances with stateful profiles.
    Uses user-data-dir to persist cookies, localStorage, and session state.
    """

    def __init__(self, profile_name: str = "default"):
        self.profile_name = profile_name
        self.profile_dir = Path(config.browser_user_data_base) / profile_name
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = None
        self._browser = None
        self._context = None

    async def start(self):
        """Launch persistent browser context."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=config.browser_headless,
            viewport={"width": config.viewport_width + random.randint(-10, 10),
                       "height": config.viewport_height + random.randint(-10, 10)},
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-web-security",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )

        # Apply stealth patches to every page
        self._context.on("page", self._apply_stealth)

        logger.info(f"Browser started with profile: {self.profile_name}")

    async def _apply_stealth(self, page):
        """Apply anti-detection patches to a new page."""
        await page.add_init_script("""
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5].map(() => ({
                    name: 'Chrome PDF Plugin',
                    description: 'Portable Document Format',
                    filename: 'internal-pdf-viewer',
                    length: 1,
                }))
            });

            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });

            // Mock permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) =>
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters);

            // Chrome runtime mock
            window.chrome = { runtime: {} };

            // WebGL vendor/renderer
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) return 'Intel Inc.';
                if (parameter === 37446) return 'Intel Iris OpenGL Engine';
                return getParameter.call(this, parameter);
            };
        """)

    async def get_page(self):
        """Get a new page from the persistent context."""
        if self._context is None:
            await self.start()
        page = await self._context.new_page()
        return page

    async def get_pages(self) -> list:
        """Get all open pages."""
        if self._context is None:
            return []
        return self._context.pages

    async def connect_cdp(self, cdp_url: str):
        """Connect to an existing Chrome instance via CDP."""
        from playwright.async_api import async_playwright

        if self._playwright is None:
            self._playwright = await async_playwright().start()

        browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        if contexts:
            self._context = contexts[0]
        else:
            self._context = await browser.new_context()
        self._browser = browser
        logger.info(f"Connected to Chrome via CDP: {cdp_url}")

    async def stop(self):
        """Close browser and cleanup."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info(f"Browser stopped for profile: {self.profile_name}")
