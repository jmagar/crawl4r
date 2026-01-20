"""URL validation with SSRF protection for web crawling."""

import ipaddress
from urllib.parse import urlparse


class ValidationError(ValueError):
    """Raised when URL validation fails."""

    pass


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
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            raise ValidationError(f"URL must use http or https scheme: {url}")

        # Extract hostname
        hostname = parsed.hostname
        if not hostname:
            raise ValidationError(f"URL missing hostname: {url}")

        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError:
            ip = None

        # Check for localhost
        if not self.allow_localhost:
            if hostname in ("localhost", "127.0.0.1", "::1"):
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
