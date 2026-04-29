"""Token harvest engine — attempts checkbox auto-pass to extract captcha tokens."""

import asyncio
import logging
import time

from config import config
from detector.captcha_detector import CaptchaType

logger = logging.getLogger(__name__)


class TokenEngine:
    """
    Harvests captcha tokens by loading the widget in a trusted browser profile
    and waiting for checkbox auto-pass (no puzzle required).

    Works best with warmed-up cookie profiles that have high reputation scores.
    """

    def __init__(self, browser_runner=None):
        self._browser_runner = browser_runner

    async def _try_checkbox_pass(self, page, sitekey: str, pageurl: str,
                                  captcha_type: CaptchaType) -> str | None:
        """Load captcha widget and attempt checkbox auto-pass."""
        if captcha_type == CaptchaType.RECAPTCHA_V2:
            return await self._harvest_recaptcha_v2(page, sitekey, pageurl)
        elif captcha_type == CaptchaType.HCAPTCHA:
            return await self._harvest_hcaptcha(page, sitekey, pageurl)
        return None

    async def _harvest_recaptcha_v2(self, page, sitekey: str, pageurl: str) -> str | None:
        """Attempt reCAPTCHA v2 checkbox auto-pass."""
        # Build a minimal HTML page with the widget
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <script src="https://www.google.com/recaptcha/api.js" async defer></script>
        </head>
        <body>
            <div class="g-recaptcha" data-sitekey="{sitekey}" data-callback="onSolved"></div>
            <script>
                window.__captchaToken = null;
                function onSolved(token) {{
                    window.__captchaToken = token;
                }}
            </script>
        </body>
        </html>"""

        await page.set_content(html)
        await asyncio.sleep(2)

        # Click the checkbox
        try:
            checkbox_frame = page.frame_locator("iframe[title='reCAPTCHA']")
            checkbox = checkbox_frame.locator("#recaptcha-anchor")
            await checkbox.click(timeout=5000)
        except Exception as e:
            logger.debug(f"Could not click reCAPTCHA checkbox: {e}")
            return None

        # Wait for auto-pass (token appears without puzzle)
        for _ in range(20):
            await asyncio.sleep(0.5)
            token = await page.evaluate("() => window.__captchaToken")
            if token:
                logger.info(f"reCAPTCHA token harvested (auto-pass): {token[:30]}...")
                return token

        # Check if challenge appeared (not auto-pass)
        try:
            challenge_frame = page.frame_locator("iframe[title*='challenge']")
            visible = await challenge_frame.locator("body").is_visible(timeout=1000)
            if visible:
                logger.info("Challenge appeared — auto-pass failed, needs puzzle solver")
                return None
        except Exception:
            pass

        return None

    async def _harvest_hcaptcha(self, page, sitekey: str, pageurl: str) -> str | None:
        """Attempt hCaptcha checkbox auto-pass."""
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <script src="https://js.hcaptcha.com/1/api.js" async defer></script>
        </head>
        <body>
            <div class="h-captcha" data-sitekey="{sitekey}" data-callback="onSolved"></div>
            <script>
                window.__captchaToken = null;
                function onSolved(token) {{
                    window.__captchaToken = token;
                }}
            </script>
        </body>
        </html>"""

        await page.set_content(html)
        await asyncio.sleep(2)

        try:
            checkbox_frame = page.frame_locator("iframe[title*='hCaptcha']")
            checkbox = checkbox_frame.locator("#checkbox")
            await checkbox.click(timeout=5000)
        except Exception as e:
            logger.debug(f"Could not click hCaptcha checkbox: {e}")
            return None

        for _ in range(20):
            await asyncio.sleep(0.5)
            token = await page.evaluate("() => window.__captchaToken")
            if token:
                logger.info(f"hCaptcha token harvested (auto-pass): {token[:30]}...")
                return token

        return None

    async def solve(self, captcha_type=None, pageurl="", sitekey=None,
                    image_data=None, image_url=None, extra=None) -> dict:
        """Attempt to harvest a token via checkbox auto-pass."""
        if not sitekey:
            return {"success": False, "error": "sitekey required for token harvesting"}

        if self._browser_runner is None:
            return {"success": False, "error": "Browser runner not available"}

        page = await self._browser_runner.get_page()
        if page is None:
            return {"success": False, "error": "Could not get browser page"}

        try:
            token = await self._try_checkbox_pass(page, sitekey, pageurl, captcha_type)
            if token:
                return {
                    "success": True,
                    "token": token,
                    "confidence": 0.95,
                }
            return {"success": False, "error": "Auto-pass failed, challenge appeared"}
        finally:
            await page.close()
