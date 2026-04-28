"""Prompt classifier using Reducto API or local NLP fallback."""

import re
import logging

import httpx

from captcha_solver_core.config import config

logger = logging.getLogger(__name__)

# Common captcha prompt patterns and their object labels
PROMPT_PATTERNS = [
    (r"select\s+(?:all\s+)?(?:images?\s+(?:of|with|containing)\s+)?(?:a\s+)?(.+)", None),
    (r"click\s+(?:on\s+)?(?:all\s+)?(?:the\s+)?(.+)", None),
    (r"which\s+(?:images?\s+)?(?:contain|show|have)\s+(?:a\s+)?(.+)", None),
    (r"identify\s+(?:all\s+)?(?:the\s+)?(.+)", None),
    (r"choose\s+(?:all\s+)?(?:the\s+)?(.+)", None),
    (r"pick\s+(?:all\s+)?(?:the\s+)?(.+)", None),
    (r"find\s+(?:all\s+)?(?:the\s+)?(.+)", None),
]

# Known object label normalizations
LABEL_NORMALIZATIONS = {
    "bicycles": "bicycle",
    "bikes": "bicycle",
    "buses": "bus",
    "busses": "bus",
    "cars": "car",
    "vehicles": "car",
    "motorcycles": "motorcycle",
    "motorbikes": "motorcycle",
    "traffic lights": "traffic light",
    "stoplights": "traffic light",
    "fire hydrants": "fire hydrant",
    "hydrants": "fire hydrant",
    "crosswalks": "crosswalk",
    "zebra crossings": "crosswalk",
    "stairs": "stairs",
    "staircases": "stairs",
    "bridges": "bridge",
    "boats": "boat",
    "ships": "boat",
    "trucks": "truck",
    "lorries": "truck",
    "trains": "train",
    "airplanes": "airplane",
    "planes": "airplane",
    "palm trees": "palm tree",
    "mountains": "mountain",
    "chimneys": "chimney",
    "parking meters": "parking meter",
    "taxis": "taxi",
    "cabs": "taxi",
}


def parse_prompt_local(prompt: str) -> dict:
    """Parse captcha prompt locally using regex patterns."""
    prompt_lower = prompt.lower().strip()

    # Try each pattern
    for pattern, _ in PROMPT_PATTERNS:
        match = re.search(pattern, prompt_lower, re.IGNORECASE)
        if match:
            raw_label = match.group(1).strip().rstrip(".")
            # Normalize
            label = LABEL_NORMALIZATIONS.get(raw_label, raw_label)
            # Remove trailing articles/prepositions
            label = re.sub(r"\s+(in|on|at|the|a|an)\s*$", "", label)

            # Determine challenge type
            challenge_type = "grid"
            if "click" in prompt_lower or "tap" in prompt_lower:
                challenge_type = "click"

            return {
                "object": label,
                "challenge_type": challenge_type,
                "raw_prompt": prompt,
                "confidence": 0.85,
            }

    # Fallback: just use the whole prompt as label
    return {
        "object": prompt_lower,
        "challenge_type": "grid",
        "raw_prompt": prompt,
        "confidence": 0.3,
    }


async def parse_prompt_reducto(prompt: str) -> dict:
    """Parse captcha prompt using Reducto API for better classification."""
    if not config.reducto_api_key:
        logger.debug("No Reducto API key, falling back to local parser")
        return parse_prompt_local(prompt)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://api.reducto.ai/classify",
                headers={"Authorization": f"Bearer {config.reducto_api_key}"},
                json={
                    "text": prompt,
                    "categories": [
                        "select_object_grid",
                        "click_object",
                        "text_identification",
                        "audio_challenge",
                        "other",
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract classification
            category = data.get("category", "other")
            challenge_type = "grid" if "grid" in category else "click"

            # Still use local parser for the object label
            local = parse_prompt_local(prompt)
            local["challenge_type"] = challenge_type
            local["confidence"] = max(local["confidence"], data.get("confidence", 0))
            return local

    except Exception as e:
        logger.warning(f"Reducto API failed, using local parser: {e}")
        return parse_prompt_local(prompt)


async def classify(prompt: str) -> dict:
    """Main entry: classify a captcha prompt and extract the target object."""
    # Try Reducto first, fall back to local
    result = await parse_prompt_reducto(prompt)
    logger.info(f"Prompt '{prompt}' -> object='{result['object']}', type={result['challenge_type']}")
    return result
