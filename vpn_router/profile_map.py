"""Sticky profile-to-region mapping — ensures each browser profile keeps the same region across its session lifecycle."""

import logging
import random
from datetime import datetime
from dataclasses import dataclass, field

from vpn_router.regions import VPN_REGIONS

logger = logging.getLogger(__name__)


@dataclass
class ProfileAssignment:
    profile_name: str
    region: str
    provider: str
    assigned_at: datetime
    last_rotated: datetime


class ProfileRegionMap:
    """Maintains sticky region assignments for browser profiles."""

    def __init__(self, available_regions: list[str] | None = None):
        self._assignments: dict[str, ProfileAssignment] = {}
        self._available_regions = available_regions or list(VPN_REGIONS.keys())
        self._region_index = 0

    def assign(self, profile_name: str, provider: str, region: str | None = None) -> ProfileAssignment:
        if region is None:
            region = self._next_region()

        now = datetime.utcnow()
        assignment = ProfileAssignment(
            profile_name=profile_name,
            region=region,
            provider=provider,
            assigned_at=now,
            last_rotated=now,
        )
        self._assignments[profile_name] = assignment
        logger.info(f"Assigned profile {profile_name} → {region} via {provider}")
        return assignment

    def get(self, profile_name: str) -> ProfileAssignment | None:
        return self._assignments.get(profile_name)

    def rotate(self, profile_name: str) -> ProfileAssignment | None:
        assignment = self._assignments.get(profile_name)
        if not assignment:
            return None

        old_region = assignment.region
        new_region = self._next_region(exclude=old_region)
        assignment.region = new_region
        assignment.last_rotated = datetime.utcnow()
        logger.info(f"Rotated profile {profile_name}: {old_region} → {new_region}")
        return assignment

    def rotate_all(self) -> list[ProfileAssignment]:
        rotated = []
        for profile_name in list(self._assignments.keys()):
            result = self.rotate(profile_name)
            if result:
                rotated.append(result)
        return rotated

    def get_all(self) -> dict[str, ProfileAssignment]:
        return dict(self._assignments)

    def get_region_for_captcha_type(self, captcha_type: str) -> str:
        from vpn_router.regions import CAPTCHA_REGION_PREFERENCES
        return CAPTCHA_REGION_PREFERENCES.get(captcha_type, "us")

    def _next_region(self, exclude: str | None = None) -> str:
        candidates = [r for r in self._available_regions if r != exclude]
        if not candidates:
            candidates = self._available_regions

        region = candidates[self._region_index % len(candidates)]
        self._region_index += 1
        return region

    def to_dict(self) -> dict:
        return {
            name: {
                "region": a.region,
                "provider": a.provider,
                "assigned_at": a.assigned_at.isoformat(),
                "last_rotated": a.last_rotated.isoformat(),
            }
            for name, a in self._assignments.items()
        }
