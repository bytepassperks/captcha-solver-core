"""Behavior simulation engine for Turnstile avoidance and trust score boosting."""

import asyncio
import random
import math
import logging

from captcha_solver_core.config import config

logger = logging.getLogger(__name__)


class BehaviorEngine:
    """
    Simulates human-like browser behavior to boost trust scores and
    bypass behavioral captchas like Cloudflare Turnstile.

    Techniques:
    - Mouse jitter and natural movement curves
    - Realistic typing with variable delays
    - Random scrolling patterns
    - Viewport resize entropy
    - Focus/blur tab switching
    """

    def __init__(self, browser_runner=None):
        self._browser_runner = browser_runner

    @staticmethod
    def _bezier_points(start: tuple, end: tuple, control: tuple, steps: int = 20) -> list[tuple]:
        """Generate points along a quadratic Bezier curve for natural mouse movement."""
        points = []
        for i in range(steps + 1):
            t = i / steps
            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
            points.append((int(x), int(y)))
        return points

    async def simulate_mouse_movement(self, page, start: tuple, end: tuple):
        """Move mouse along a natural Bezier curve with jitter."""
        # Random control point for curve
        mid_x = (start[0] + end[0]) / 2 + random.randint(-100, 100)
        mid_y = (start[1] + end[1]) / 2 + random.randint(-50, 50)
        control = (mid_x, mid_y)

        points = self._bezier_points(start, end, control, steps=random.randint(15, 30))

        for x, y in points:
            # Add micro-jitter
            jx = x + random.randint(*config.mouse_jitter_range)
            jy = y + random.randint(*config.mouse_jitter_range)
            await page.mouse.move(jx, jy)
            await asyncio.sleep(random.uniform(0.005, 0.025))

    async def simulate_typing(self, page, selector: str, text: str):
        """Type text with human-like variable delays."""
        element = page.locator(selector)
        await element.click()
        await asyncio.sleep(random.uniform(0.1, 0.3))

        for char in text:
            await page.keyboard.type(char)
            base_delay = random.uniform(*config.typing_delay_range)
            # Occasional longer pauses (thinking)
            if random.random() < 0.05:
                base_delay += random.uniform(0.3, 0.8)
            await asyncio.sleep(base_delay)

    async def simulate_scroll(self, page, direction: str = "down", amount: int | None = None):
        """Simulate natural scrolling."""
        if amount is None:
            amount = random.randint(100, 500)

        scroll_y = amount if direction == "down" else -amount

        # Smooth scroll in increments
        steps = random.randint(3, 8)
        per_step = scroll_y / steps

        for _ in range(steps):
            delta = per_step + random.uniform(-20, 20)
            await page.mouse.wheel(0, delta)
            await asyncio.sleep(random.uniform(*config.scroll_delay_range))

    async def simulate_viewport_entropy(self, page):
        """Slightly randomize viewport size to add entropy."""
        width = config.viewport_width + random.randint(-20, 20)
        height = config.viewport_height + random.randint(-20, 20)
        await page.set_viewport_size({"width": width, "height": height})

    async def simulate_focus_switching(self, page):
        """Simulate tab focus/blur events."""
        await page.evaluate("() => { document.dispatchEvent(new Event('visibilitychange')); }")
        await asyncio.sleep(random.uniform(0.5, 2.0))
        await page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', { value: true, writable: true });
            document.dispatchEvent(new Event('visibilitychange'));
        }""")
        await asyncio.sleep(random.uniform(1.0, 3.0))
        await page.evaluate("""() => {
            Object.defineProperty(document, 'hidden', { value: false, writable: true });
            document.dispatchEvent(new Event('visibilitychange'));
        }""")

    async def warm_page(self, page, url: str, duration_seconds: int = 10):
        """
        Warm a page with natural behavior to build trust before captcha interaction.
        """
        await page.goto(url, wait_until="domcontentloaded")
        await asyncio.sleep(random.uniform(1, 3))

        # Simulate viewport entropy
        await self.simulate_viewport_entropy(page)

        elapsed = 0
        while elapsed < duration_seconds:
            action = random.choice(["scroll", "mouse_move", "wait", "focus"])

            if action == "scroll":
                direction = random.choice(["down", "up"])
                await self.simulate_scroll(page, direction)
                elapsed += 2

            elif action == "mouse_move":
                vp = page.viewport_size or {"width": 1280, "height": 800}
                start = (random.randint(100, vp["width"] - 100),
                         random.randint(100, vp["height"] - 100))
                end = (random.randint(100, vp["width"] - 100),
                       random.randint(100, vp["height"] - 100))
                await self.simulate_mouse_movement(page, start, end)
                elapsed += 1

            elif action == "wait":
                wait_time = random.uniform(0.5, 3.0)
                await asyncio.sleep(wait_time)
                elapsed += wait_time

            elif action == "focus":
                await self.simulate_focus_switching(page)
                elapsed += 3

    async def solve(self, captcha_type=None, pageurl="", sitekey=None,
                    image_data=None, image_url=None, extra=None) -> dict:
        """
        Attempt to bypass behavioral captchas (Turnstile) through simulation.
        """
        if self._browser_runner is None:
            return {"success": False, "error": "Browser runner not available"}

        page = await self._browser_runner.get_page()
        if page is None:
            return {"success": False, "error": "Could not get browser page"}

        try:
            # Navigate and warm up
            await self.warm_page(page, pageurl, duration_seconds=15)

            # Check if Turnstile widget exists and try to interact
            turnstile_token = await page.evaluate("""() => {
                const input = document.querySelector('[name="cf-turnstile-response"]');
                return input ? input.value : null;
            }""")

            if turnstile_token:
                return {
                    "success": True,
                    "token": turnstile_token,
                    "confidence": 0.70,
                }

            # Try clicking Turnstile checkbox if present
            try:
                ts_frame = page.frame_locator("iframe[src*='challenges.cloudflare.com']")
                checkbox = ts_frame.locator("input[type='checkbox'], .cb-i")
                if await checkbox.is_visible(timeout=3000):
                    # Natural movement to checkbox
                    box = await checkbox.bounding_box()
                    if box:
                        await self.simulate_mouse_movement(
                            page,
                            (random.randint(100, 300), random.randint(100, 300)),
                            (int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
                        )
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                        await checkbox.click()
                        await asyncio.sleep(3)

                        turnstile_token = await page.evaluate("""() => {
                            const input = document.querySelector('[name="cf-turnstile-response"]');
                            return input ? input.value : null;
                        }""")

                        if turnstile_token:
                            return {
                                "success": True,
                                "token": turnstile_token,
                                "confidence": 0.65,
                            }
            except Exception as e:
                logger.debug(f"Turnstile checkbox interaction failed: {e}")

            return {"success": False, "error": "Behavioral bypass did not produce token"}

        finally:
            await page.close()
