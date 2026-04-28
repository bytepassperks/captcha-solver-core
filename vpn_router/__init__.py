"""VPN identity routing layer — multi-region, multi-provider identity management for browser profiles."""

from vpn_router.regions import VPN_REGIONS, RegionConfig
from vpn_router.manager import VPNManager
from vpn_router.profile_map import ProfileRegionMap
from vpn_router.scheduler import VPNRotationScheduler

__all__ = [
    "VPN_REGIONS",
    "RegionConfig",
    "VPNManager",
    "ProfileRegionMap",
    "VPNRotationScheduler",
]
