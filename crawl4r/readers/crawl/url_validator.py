"""URL validation with SSRF protection for web crawling."""

import ipaddress
import re
from urllib.parse import urlparse


class ValidationError(ValueError):
    """Raised when URL validation fails."""

    pass


# Blocked hostnames for SSRF protection (case-insensitive)
# Note: localhost is handled separately by allow_localhost flag
BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata",
}


class UrlValidator:
    """Validates URLs and prevents SSRF attacks.

    Args:
        allow_private_ips: Allow private IP addresses (e.g., 192.168.x.x)
        allow_localhost: Allow localhost/127.0.0.1
    """

    def __init__(
        self,
        allow_private_ips: bool = False,
        allow_localhost: bool = False,
    ) -> None:
        self.allow_private_ips = allow_private_ips
        self.allow_localhost = allow_localhost

    def validate(self, url: str) -> None:
        """Validate URL and check for SSRF risks.

        Args:
            url: URL to validate

        Raises:
            ValidationError: If URL is invalid or poses SSRF risk
        """
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Malformed URL: {url}") from e

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            raise ValidationError(f"URL must use http or https scheme: {url}")

        # Extract hostname
        hostname = parsed.hostname
        if not hostname:
            raise ValidationError(f"URL missing hostname: {url}")

        hostname_lower = hostname.lower()

        # Check blocked hostnames (cloud metadata, etc.)
        if hostname_lower in BLOCKED_HOSTNAMES:
            raise ValidationError(f"Blocked hostname: {url}")

        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            ip = None
            # Check for decimal/hex IP notation (SSRF bypass attempts)
            # Decimal: 2130706433 = 127.0.0.1
            # Hex: 0x7f000001 = 127.0.0.1
            if re.match(r"^(0x[0-9a-fA-F]+|\d{8,})$", hostname):
                raise ValidationError(
                    f"IP address in alternate notation not allowed: {url}"
                )

        # Check for localhost
        if not self.allow_localhost:
            if hostname_lower in ("localhost", "127.0.0.1", "::1"):
                raise ValidationError(f"Localhost access not allowed: {url}")
            if ip is not None and ip.is_loopback:
                raise ValidationError(f"Localhost access not allowed: {url}")

        # Check for private IP addresses
        if not self.allow_private_ips and ip is not None and (
            ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValidationError(
                f"Non-public IP addresses not allowed: {url}"
            )
