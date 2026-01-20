"""Test URL validation utility."""

from __future__ import annotations

import pytest

from crawl4r.core.url_validation import validate_url


class TestValidateUrl:
    """Test URL validation for SSRF prevention."""

    def test_accepts_valid_http_url(self):
        """Should accept valid HTTP URLs."""
        assert validate_url("http://example.com") is True
        assert validate_url("http://example.com/path") is True
        assert validate_url("http://example.com:8080/path") is True

    def test_accepts_valid_https_url(self):
        """Should accept valid HTTPS URLs."""
        assert validate_url("https://example.com") is True
        assert validate_url("https://example.com/path") is True
        assert validate_url("https://example.com:443/path") is True

    def test_rejects_non_http_schemes(self):
        """Should reject non-HTTP(S) schemes."""
        assert validate_url("ftp://example.com") is False
        assert validate_url("file:///etc/passwd") is False
        assert validate_url("gopher://example.com") is False
        assert validate_url("javascript:alert(1)") is False
        assert validate_url("data:text/html,<script>alert(1)</script>") is False

    def test_rejects_localhost_hostname(self):
        """Should reject localhost hostname."""
        assert validate_url("http://localhost") is False
        assert validate_url("http://localhost:8080") is False
        assert validate_url("https://localhost/admin") is False
        assert validate_url("http://LOCALHOST") is False  # Case insensitive

    def test_rejects_loopback_ips(self):
        """Should reject loopback IP addresses."""
        assert validate_url("http://127.0.0.1") is False
        assert validate_url("http://127.0.0.2") is False
        assert validate_url("http://127.255.255.255") is False
        assert validate_url("http://[::1]") is False  # IPv6 loopback

    def test_rejects_private_ip_ranges(self):
        """Should reject private IP addresses."""
        # 10.0.0.0/8
        assert validate_url("http://10.0.0.1") is False
        assert validate_url("http://10.255.255.255") is False

        # 172.16.0.0/12
        assert validate_url("http://172.16.0.1") is False
        assert validate_url("http://172.31.255.255") is False

        # 192.168.0.0/16
        assert validate_url("http://192.168.0.1") is False
        assert validate_url("http://192.168.255.255") is False

    def test_rejects_link_local_ips(self):
        """Should reject link-local IP addresses (AWS metadata)."""
        assert validate_url("http://169.254.169.254") is False
        assert validate_url("http://169.254.0.1") is False

    def test_rejects_cloud_metadata_hostnames(self):
        """Should reject cloud metadata hostnames."""
        assert validate_url("http://metadata.google.internal") is False
        assert validate_url("http://metadata") is False

    def test_rejects_decimal_ip_notation(self):
        """Should reject decimal IP notation."""
        # 2130706433 = 127.0.0.1
        assert validate_url("http://2130706433") is False

    def test_rejects_hex_ip_notation(self):
        """Should reject hexadecimal IP notation."""
        # 0x7f000001 = 127.0.0.1
        assert validate_url("http://0x7f000001") is False

    def test_rejects_malformed_urls(self):
        """Should reject malformed URLs."""
        assert validate_url("not-a-url") is False
        assert validate_url("") is False
        assert validate_url("http://") is False
        assert validate_url("://example.com") is False

    def test_rejects_missing_hostname(self):
        """Should reject URLs without hostname."""
        assert validate_url("http:///path") is False

    def test_accepts_public_ips(self):
        """Should accept public IP addresses."""
        assert validate_url("http://8.8.8.8") is True  # Google DNS
        assert validate_url("http://1.1.1.1") is True  # Cloudflare DNS

    def test_accepts_valid_domains(self):
        """Should accept valid public domain names."""
        assert validate_url("https://google.com") is True
        assert validate_url("https://www.example.org") is True
        assert validate_url("https://api.github.com") is True
        assert validate_url("https://sub.domain.example.com") is True

    def test_accepts_ipv6_public_addresses(self):
        """Should accept public IPv6 addresses."""
        # Google Public DNS IPv6
        assert validate_url("http://[2001:4860:4860::8888]") is True

    def test_rejects_ipv6_private_addresses(self):
        """Should reject private IPv6 addresses."""
        assert validate_url("http://[fc00::1]") is False  # Unique local
        assert validate_url("http://[fd00::1]") is False  # Unique local
        assert validate_url("http://[fe80::1]") is False  # Link local

    def test_case_insensitive_hostname_blocking(self):
        """Should block hostnames case-insensitively."""
        assert validate_url("http://LOCALHOST") is False
        assert validate_url("http://Localhost") is False
        assert validate_url("http://LocalHost") is False
        assert validate_url("http://METADATA") is False
