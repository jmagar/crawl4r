"""URL validation utilities for SSRF prevention.

This module provides URL validation functions to prevent Server-Side Request
Forgery (SSRF) attacks by rejecting URLs that target internal/private resources.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """Validate URL is safe for crawling (SSRF prevention).

    Prevents Server-Side Request Forgery (SSRF) attacks by validating URLs
    before forwarding them to crawling services. Rejects:
    - Private IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x)
    - Cloud metadata endpoints (169.254.169.254)
    - Non-HTTP(S) schemes (file://, ftp://, gopher://)
    - Localhost hostnames
    - IP addresses in alternate notations (decimal, hex, octal)
    - URLs that resolve to private IPs via DNS (DNS rebinding prevention)

    Args:
        url: URL to validate. Must be a well-formed URL string.

    Returns:
        True if URL is safe for crawling, False otherwise.

    Examples:
        Safe URLs:
            >>> validate_url("https://example.com/")
            True

        Blocked URLs:
            >>> validate_url("http://192.168.1.1/admin")
            False
            >>> validate_url("http://169.254.169.254/")
            False
            >>> validate_url("not-a-url")
            False
    """
    # Blocked hostnames (case-insensitive)
    blocked_hostnames = {
        "localhost",
        "metadata.google.internal",
        "metadata",
    }

    # Allowed schemes
    allowed_schemes = {"http", "https"}

    try:
        parsed = urlparse(url)

        # Check scheme
        if not parsed.scheme or parsed.scheme.lower() not in allowed_schemes:
            return False

        # Check hostname exists
        hostname = parsed.hostname
        if not hostname:
            return False

        hostname_lower = hostname.lower()

        # Check blocked hostnames
        if hostname_lower in blocked_hostnames:
            return False

        # Check if hostname is an IP address
        try:
            # Handle IPv6 addresses (already unwrapped by urlparse)
            ip = ipaddress.ip_address(hostname)

            # Block private IP ranges
            if ip.is_private:
                return False

            # Block loopback (127.x.x.x, ::1)
            if ip.is_loopback:
                return False

            # Block link-local (169.254.x.x - AWS metadata)
            if ip.is_link_local:
                return False

            # Block reserved ranges
            if ip.is_reserved:
                return False

        except ValueError:
            # Not an IP address, check for decimal/hex/octal IP notation
            # Decimal: 2130706433 = 127.0.0.1
            # Hex: 0x7f000001 = 127.0.0.1
            # Octal: 0177.0.0.1 = 127.0.0.1
            if re.match(r"^(0x[0-9a-fA-F]+|\d{8,}|0[0-7]+(\.|$))$", hostname):
                return False

            # DNS rebinding prevention: resolve hostname and validate IPs
            try:
                addr_info = socket.getaddrinfo(hostname, None)
                for addr in addr_info:
                    resolved_ip = ipaddress.ip_address(addr[4][0])

                    # Block if ANY resolved IP is private/reserved
                    if (
                        resolved_ip.is_private
                        or resolved_ip.is_loopback
                        or resolved_ip.is_link_local
                        or resolved_ip.is_reserved
                    ):
                        return False

            except (socket.gaierror, ValueError, OSError):
                # DNS resolution failed - reject URL
                return False

        return True

    except Exception:
        # Malformed URL
        return False
