"""VPN connection manager — handles connect/disconnect/status across multiple VPN providers with fallback."""

import asyncio
import logging
import shutil
from dataclasses import dataclass

from config import config
from vpn_router.regions import VPN_REGIONS, RegionConfig

logger = logging.getLogger(__name__)


@dataclass
class VPNStatus:
    connected: bool
    provider: str | None
    region: str | None
    ip_address: str | None
    error: str | None


class VPNProvider:
    """Base interface for VPN CLI operations."""

    name: str = "base"

    def is_available(self) -> bool:
        return False

    async def connect(self, region_config: RegionConfig) -> bool:
        raise NotImplementedError

    async def disconnect(self) -> bool:
        raise NotImplementedError

    async def status(self) -> VPNStatus:
        raise NotImplementedError


class ExpressVPNProvider(VPNProvider):
    name = "expressvpn"

    def is_available(self) -> bool:
        return shutil.which("expressvpnctl") is not None or shutil.which("expressvpn") is not None

    def _cmd(self) -> str:
        if shutil.which("expressvpnctl"):
            return "expressvpnctl"
        return "expressvpn"

    async def connect(self, region_config: RegionConfig) -> bool:
        cmd = self._cmd()
        region_id = region_config.expressvpn_id
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd, "connect", region_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            success = proc.returncode == 0
            if success:
                logger.info(f"ExpressVPN connected to {region_id}")
            else:
                logger.warning(f"ExpressVPN connect failed: {stderr.decode().strip()}")
            return success
        except Exception as e:
            logger.error(f"ExpressVPN connect error: {e}")
            return False

    async def disconnect(self) -> bool:
        cmd = self._cmd()
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd, "disconnect",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"ExpressVPN disconnect error: {e}")
            return False

    async def status(self) -> VPNStatus:
        cmd = self._cmd()
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd, "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode().strip()
            connected = "connected" in output.lower()
            return VPNStatus(
                connected=connected,
                provider="expressvpn",
                region=output if connected else None,
                ip_address=None,
                error=None,
            )
        except Exception as e:
            return VPNStatus(connected=False, provider="expressvpn", region=None, ip_address=None, error=str(e))


class ProtonVPNProvider(VPNProvider):
    name = "protonvpn"

    def is_available(self) -> bool:
        return shutil.which("protonvpn-cli") is not None

    async def connect(self, region_config: RegionConfig) -> bool:
        region_id = region_config.protonvpn_id
        try:
            proc = await asyncio.create_subprocess_exec(
                "protonvpn-cli", "connect", "--cc", region_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            success = proc.returncode == 0
            if success:
                logger.info(f"ProtonVPN connected to {region_id}")
            else:
                logger.warning(f"ProtonVPN connect failed: {stderr.decode().strip()}")
            return success
        except Exception as e:
            logger.error(f"ProtonVPN connect error: {e}")
            return False

    async def disconnect(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "protonvpn-cli", "disconnect",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"ProtonVPN disconnect error: {e}")
            return False

    async def status(self) -> VPNStatus:
        try:
            proc = await asyncio.create_subprocess_exec(
                "protonvpn-cli", "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode().strip()
            connected = "connected" in output.lower()
            return VPNStatus(
                connected=connected,
                provider="protonvpn",
                region=output if connected else None,
                ip_address=None,
                error=None,
            )
        except Exception as e:
            return VPNStatus(connected=False, provider="protonvpn", region=None, ip_address=None, error=str(e))


class WindscribeProvider(VPNProvider):
    name = "windscribe"

    def is_available(self) -> bool:
        return shutil.which("windscribe") is not None

    async def connect(self, region_config: RegionConfig) -> bool:
        region_id = region_config.windscribe_id
        try:
            proc = await asyncio.create_subprocess_exec(
                "windscribe", "connect", region_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            success = proc.returncode == 0
            if success:
                logger.info(f"Windscribe connected to {region_id}")
            else:
                logger.warning(f"Windscribe connect failed: {stderr.decode().strip()}")
            return success
        except Exception as e:
            logger.error(f"Windscribe connect error: {e}")
            return False

    async def disconnect(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "windscribe", "disconnect",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"Windscribe disconnect error: {e}")
            return False

    async def status(self) -> VPNStatus:
        try:
            proc = await asyncio.create_subprocess_exec(
                "windscribe", "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode().strip()
            connected = "connected" in output.lower()
            return VPNStatus(
                connected=connected,
                provider="windscribe",
                region=output if connected else None,
                ip_address=None,
                error=None,
            )
        except Exception as e:
            return VPNStatus(connected=False, provider="windscribe", region=None, ip_address=None, error=str(e))


PROVIDER_CLASSES: dict[str, type[VPNProvider]] = {
    "expressvpn": ExpressVPNProvider,
    "protonvpn": ProtonVPNProvider,
    "windscribe": WindscribeProvider,
}


class VPNManager:
    """Manages VPN connections with provider fallback and region routing."""

    def __init__(self):
        self._providers: dict[str, VPNProvider] = {}
        self._active_provider: VPNProvider | None = None
        self._active_region: str | None = None
        self._connected = False
        self._init_providers()

    def _init_providers(self):
        for name in config.vpn_provider_priority:
            cls = PROVIDER_CLASSES.get(name)
            if cls:
                provider = cls()
                self._providers[name] = provider
                available = provider.is_available()
                logger.info(f"VPN provider {name}: {'available' if available else 'not found'}")

    @property
    def available_providers(self) -> list[str]:
        return [name for name, p in self._providers.items() if p.is_available()]

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def active_provider_name(self) -> str | None:
        return self._active_provider.name if self._active_provider else None

    @property
    def active_region(self) -> str | None:
        return self._active_region

    async def connect(self, region: str) -> bool:
        if not config.vpn_enabled:
            logger.debug("VPN disabled in config")
            return True

        region_config = VPN_REGIONS.get(region)
        if not region_config:
            logger.error(f"Unknown VPN region: {region}")
            return False

        # Disconnect existing connection first
        if self._connected:
            await self.disconnect()

        # Try providers in priority order with fallback
        for provider_name in config.vpn_provider_priority:
            provider = self._providers.get(provider_name)
            if not provider or not provider.is_available():
                logger.debug(f"Skipping unavailable provider: {provider_name}")
                continue

            success = await provider.connect(region_config)
            if success:
                self._active_provider = provider
                self._active_region = region
                self._connected = True
                logger.info(f"VPN connected: {provider_name} → {region} ({region_config.name})")
                return True

            logger.warning(f"Provider {provider_name} failed, trying next...")

        logger.error(f"All VPN providers failed for region {region}")
        return False

    async def disconnect(self) -> bool:
        if self._active_provider:
            success = await self._active_provider.disconnect()
            if success:
                logger.info(f"VPN disconnected from {self._active_region}")
            self._active_provider = None
            self._active_region = None
            self._connected = False
            return success
        return True

    async def get_status(self) -> dict:
        status = {
            "enabled": config.vpn_enabled,
            "connected": self._connected,
            "provider": self.active_provider_name,
            "region": self._active_region,
            "available_providers": self.available_providers,
        }

        if self._active_provider:
            vpn_status = await self._active_provider.status()
            status["provider_status"] = {
                "connected": vpn_status.connected,
                "region": vpn_status.region,
                "ip_address": vpn_status.ip_address,
                "error": vpn_status.error,
            }

        return status
