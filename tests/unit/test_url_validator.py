"""Tests for crawl4r.readers.crawl.url_validator module."""

import pytest

from crawl4r.readers.crawl.url_validator import UrlValidator, ValidationError


def test_validate_accepts_valid_https_url() -> None:
    """Verify validator accepts standard HTTPS URLs."""
    validator = UrlValidator()
    validator.validate("https://example.com")  # Should not raise


def test_validate_accepts_valid_http_url() -> None:
    """Verify validator accepts HTTP URLs."""
    validator = UrlValidator()
    validator.validate("http://example.com")  # Should not raise


def test_validate_rejects_non_http_scheme() -> None:
    """Verify validator rejects non-HTTP(S) schemes."""
    validator = UrlValidator()

    with pytest.raises(ValidationError, match="URL must use http or https"):
        validator.validate("ftp://example.com")


def test_validate_rejects_private_ip() -> None:
    """Verify validator rejects private IP addresses (SSRF protection)."""
    validator = UrlValidator(allow_private_ips=False)

    with pytest.raises(ValidationError, match="Private IP addresses not allowed"):
        validator.validate("http://192.168.1.1")


def test_validate_allows_private_ip_when_configured() -> None:
    """Verify validator allows private IPs when explicitly enabled."""
    validator = UrlValidator(allow_private_ips=True)
    validator.validate("http://192.168.1.1")  # Should not raise


def test_validate_rejects_localhost() -> None:
    """Verify validator rejects localhost (SSRF protection)."""
    validator = UrlValidator(allow_localhost=False)

    with pytest.raises(ValidationError, match="Localhost access not allowed"):
        validator.validate("http://localhost:8000")


def test_validate_allows_localhost_when_configured() -> None:
    """Verify validator allows localhost when explicitly enabled."""
    validator = UrlValidator(allow_localhost=True)
    validator.validate("http://localhost:8000")  # Should not raise


def test_validate_rejects_loopback_ip_when_localhost_disallowed() -> None:
    """Verify validator rejects loopback IPs when localhost is disallowed."""
    validator = UrlValidator(allow_localhost=False)

    with pytest.raises(ValidationError, match="Localhost access not allowed"):
        validator.validate("http://127.0.0.2")


def test_validate_rejects_link_local_ip_when_private_disallowed() -> None:
    """Verify validator rejects link-local IPs when private IPs are disallowed."""
    validator = UrlValidator(allow_private_ips=False)

    with pytest.raises(ValidationError, match="Private IP addresses not allowed"):
        validator.validate("http://169.254.1.1")


def test_validate_rejects_unspecified_ip_when_private_disallowed() -> None:
    """Verify validator rejects unspecified IPs when private IPs are disallowed."""
    validator = UrlValidator(allow_private_ips=False)

    with pytest.raises(ValidationError, match="Private IP addresses not allowed"):
        validator.validate("http://0.0.0.0")


def test_validate_rejects_ipv6_loopback_when_localhost_disallowed() -> None:
    """Verify validator rejects IPv6 loopback when localhost is disallowed."""
    validator = UrlValidator(allow_localhost=False)

    with pytest.raises(ValidationError, match="Localhost access not allowed"):
        validator.validate("http://[::1]")
