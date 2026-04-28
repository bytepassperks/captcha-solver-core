"""VPN region definitions and provider-specific connection commands."""

from dataclasses import dataclass


@dataclass
class RegionConfig:
    name: str
    expressvpn_id: str
    protonvpn_id: str
    windscribe_id: str

    def get_connect_args(self, provider: str) -> str:
        mapping = {
            "expressvpn": self.expressvpn_id,
            "protonvpn": self.protonvpn_id,
            "windscribe": self.windscribe_id,
        }
        return mapping.get(provider, self.expressvpn_id)


VPN_REGIONS: dict[str, RegionConfig] = {
    "us": RegionConfig(
        name="United States",
        expressvpn_id="smart",
        protonvpn_id="US",
        windscribe_id="US",
    ),
    "uk": RegionConfig(
        name="United Kingdom",
        expressvpn_id="uk",
        protonvpn_id="GB",
        windscribe_id="GB",
    ),
    "germany": RegionConfig(
        name="Germany",
        expressvpn_id="germany",
        protonvpn_id="DE",
        windscribe_id="DE",
    ),
    "canada": RegionConfig(
        name="Canada",
        expressvpn_id="canada",
        protonvpn_id="CA",
        windscribe_id="CA",
    ),
    "singapore": RegionConfig(
        name="Singapore",
        expressvpn_id="singapore",
        protonvpn_id="SG",
        windscribe_id="SG",
    ),
    "japan": RegionConfig(
        name="Japan",
        expressvpn_id="japan",
        protonvpn_id="JP",
        windscribe_id="JP",
    ),
    "netherlands": RegionConfig(
        name="Netherlands",
        expressvpn_id="netherlands",
        protonvpn_id="NL",
        windscribe_id="NL",
    ),
    "australia": RegionConfig(
        name="Australia",
        expressvpn_id="australia",
        protonvpn_id="AU",
        windscribe_id="AU",
    ),
}

CAPTCHA_REGION_PREFERENCES: dict[str, str] = {
    "recaptcha_v2": "us",
    "recaptcha_v3": "us",
    "turnstile": "uk",
    "hcaptcha": "canada",
    "text_image": "germany",
}
