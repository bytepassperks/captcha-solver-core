"""Detect captcha type from page HTML signals."""

import re
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CaptchaType(str, Enum):
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    TURNSTILE = "turnstile"
    MTCAPTCHA = "mtcaptcha"
    TEXT = "text"
    IMAGE_GRID = "image_grid"
    NONE = "none"


@dataclass
class DetectionResult:
    captcha_type: CaptchaType
    sitekey: str | None = None
    iframe_src: str | None = None
    confidence: float = 0.0
    extra: dict | None = None


# Patterns to detect captcha type from page source
DETECTION_PATTERNS = {
    CaptchaType.RECAPTCHA_V2: [
        r'google\.com/recaptcha/api\.js',
        r'google\.com/recaptcha/enterprise\.js',
        r'g-recaptcha',
        r'data-sitekey="([^"]+)"',
        r'grecaptcha\.render',
        r'recaptcha/api2/anchor',
    ],
    CaptchaType.RECAPTCHA_V3: [
        r'grecaptcha\.execute\(',
        r'recaptcha.*action\s*[:=]',
        r'recaptcha-v3',
    ],
    CaptchaType.HCAPTCHA: [
        r'hcaptcha\.com/1/api\.js',
        r'h-captcha',
        r'data-hcaptcha-widget-id',
        r'hcaptcha\.render',
        r'hcaptcha\.com/checksiteconfig',
    ],
    CaptchaType.TURNSTILE: [
        r'challenges\.cloudflare\.com/turnstile',
        r'cf-turnstile',
        r'turnstile\.render',
    ],
    CaptchaType.TEXT: [
        r'captcha.*img',
        r'img.*captcha',
        r'captcha[-_]image',
        r'captchaImage',
        r'verification.*code.*image',
    ],
    CaptchaType.IMAGE_GRID: [
        r'captcha.*grid',
        r'select.*all.*images',
        r'click.*all.*images',
        r'image.*challenge',
    ],
    CaptchaType.MTCAPTCHA: [
        r'mtcaptcha',
        r'service\.mtcaptcha\.com',
        r'MTPublic-',
        r'mtcap-',
        r'mtcaptcha-verifiedtoken',
        r'data-sitekey.*MTPublic',
    ],
}

SITEKEY_PATTERNS = {
    CaptchaType.RECAPTCHA_V2: [
        r'data-sitekey="([^"]+)"',
        r"data-sitekey='([^']+)'",
        r'sitekey["\s:=]+([0-9A-Za-z_-]{40})',
        r'render=([0-9A-Za-z_-]{40})',
    ],
    CaptchaType.HCAPTCHA: [
        r'data-sitekey="([^"]+)"',
        r"data-sitekey='([^']+)'",
        r'sitekey["\s:=]+([0-9a-f-]{36})',
    ],
    CaptchaType.TURNSTILE: [
        r'data-sitekey="([^"]+)"',
        r"data-sitekey='([^']+)'",
        r'sitekey["\s:=]+([0-9A-Za-z_-]+)',
    ],
    CaptchaType.MTCAPTCHA: [
        r'data-sitekey="(MTPublic-[^"]+)"',
        r"data-sitekey='(MTPublic-[^']+)'",
        r'sitekey["\s:=]+(MTPublic-[A-Za-z0-9]+)',
        r"'sitekey'\s*:\s*'(MTPublic-[^']+)'",
        r'"sitekey"\s*:\s*"(MTPublic-[^"]+)"',
    ],
}


def detect_from_html(html: str) -> DetectionResult:
    """Detect captcha type and extract sitekey from raw HTML."""
    html_lower = html.lower()
    scores: dict[CaptchaType, float] = {}

    for ctype, patterns in DETECTION_PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                score += 1.0 / len(patterns)
        if score > 0:
            scores[ctype] = min(score, 1.0)

    if not scores:
        return DetectionResult(captcha_type=CaptchaType.NONE, confidence=1.0)

    best_type = max(scores, key=scores.get)
    confidence = scores[best_type]

    # reCAPTCHA v3 usually co-exists with v2 indicators; differentiate
    if CaptchaType.RECAPTCHA_V3 in scores and CaptchaType.RECAPTCHA_V2 in scores:
        if scores[CaptchaType.RECAPTCHA_V3] > scores[CaptchaType.RECAPTCHA_V2]:
            best_type = CaptchaType.RECAPTCHA_V3

    # Extract sitekey
    sitekey = None
    if best_type in SITEKEY_PATTERNS:
        for pattern in SITEKEY_PATTERNS[best_type]:
            match = re.search(pattern, html)
            if match:
                sitekey = match.group(1)
                break

    # Extract iframe src
    iframe_src = None
    iframe_match = re.search(
        r'<iframe[^>]+src="([^"]*(?:recaptcha|hcaptcha|turnstile)[^"]*)"',
        html, re.IGNORECASE
    )
    if iframe_match:
        iframe_src = iframe_match.group(1)

    logger.info(f"Detected captcha: {best_type.value} (confidence={confidence:.2f}, sitekey={sitekey})")

    return DetectionResult(
        captcha_type=best_type,
        sitekey=sitekey,
        iframe_src=iframe_src,
        confidence=confidence,
    )


async def detect_from_page(page) -> DetectionResult:
    """Detect captcha type from a Playwright page object."""
    html = await page.content()
    result = detect_from_html(html)

    # Enhanced: check for dynamically loaded captcha widgets via JS
    if result.captcha_type == CaptchaType.NONE:
        try:
            has_recaptcha = await page.evaluate(
                "() => typeof grecaptcha !== 'undefined'"
            )
            if has_recaptcha:
                sitekey = await page.evaluate("""() => {
                    const el = document.querySelector('[data-sitekey]');
                    return el ? el.getAttribute('data-sitekey') : null;
                }""")
                return DetectionResult(
                    captcha_type=CaptchaType.RECAPTCHA_V2,
                    sitekey=sitekey,
                    confidence=0.9,
                )

            has_hcaptcha = await page.evaluate(
                "() => typeof hcaptcha !== 'undefined'"
            )
            if has_hcaptcha:
                sitekey = await page.evaluate("""() => {
                    const el = document.querySelector('[data-sitekey]');
                    return el ? el.getAttribute('data-sitekey') : null;
                }""")
                return DetectionResult(
                    captcha_type=CaptchaType.HCAPTCHA,
                    sitekey=sitekey,
                    confidence=0.9,
                )

            has_turnstile = await page.evaluate(
                "() => typeof turnstile !== 'undefined'"
            )
            if has_turnstile:
                return DetectionResult(
                    captcha_type=CaptchaType.TURNSTILE,
                    confidence=0.8,
                )

            # Check for MTCaptcha (loaded via iframe from service.mtcaptcha.com)
            has_mtcaptcha = await page.evaluate("""
                () => {
                    const el = document.querySelector('[id*="mtcaptcha"], [class*="mtcaptcha"], div[id*="mtcap"]');
                    if (el) return true;
                    const iframes = document.querySelectorAll('iframe');
                    for (const f of iframes) {
                        if (f.src && f.src.includes('mtcaptcha')) return true;
                    }
                    return false;
                }
            """)
            if has_mtcaptcha:
                sitekey = await page.evaluate("""
                    () => {
                        const el = document.querySelector('[data-sitekey]');
                        if (el) return el.getAttribute('data-sitekey');
                        const scripts = document.querySelectorAll('script');
                        for (const s of scripts) {
                            const m = s.textContent.match(/sitekey['"]?\s*[:=]\s*['"]?(MTPublic-[^'"\s,}]+)/);
                            if (m) return m[1];
                        }
                        return null;
                    }
                """)
                return DetectionResult(
                    captcha_type=CaptchaType.MTCAPTCHA,
                    sitekey=sitekey,
                    confidence=0.95,
                )
        except Exception as e:
            logger.debug(f"JS detection failed: {e}")

    return result
