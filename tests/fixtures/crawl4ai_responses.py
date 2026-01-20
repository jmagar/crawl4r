"""
Test fixtures for Crawl4AI API response mocking.

This module provides realistic mock data for Crawl4AI API responses used in
unit and integration tests. Fixtures cover success cases, HTTP error codes,
and timeout scenarios.

Usage:
    from tests.fixtures.crawl4ai_responses import MOCK_CRAWL_RESULT_SUCCESS

    # In test with respx:
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=MOCK_CRAWL_RESULT_SUCCESS)
    )
"""

from typing import Any

# Success case: 200 OK with complete markdown content and metadata
MOCK_CRAWL_RESULT_SUCCESS = {
    "url": "https://example.com",
    "success": True,
    "status_code": 200,
    "markdown": {
        "fit_markdown": (
            "# Example Domain\n\n"
            "This domain is for use in illustrative examples in documents.\n\n"
            "## Overview\n\n"
            "You may use this domain in literature without prior coordination "
            "or asking for permission.\n\n"
            "## More information\n\n"
            "[More information...](https://www.iana.org/domains/example)"
        ),
        "raw_markdown": (
            "# Example Domain\n\n"
            "This domain is for use in illustrative examples in documents.\n\n"
            "## Overview\n\n"
            "You may use this domain in literature without prior coordination "
            "or asking for permission.\n\n"
            "## More information\n\n"
            "[More information...](https://www.iana.org/domains/example)\n\n"
            "---\nFooter: Copyright 2024"
        ),
    },
    "metadata": {
        "title": "Example Domain",
        "description": "Example Domain for documentation and testing purposes",
    },
    "links": {
        "internal": [
            {"href": "/about"},
            {"href": "/contact"},
            {"href": "/privacy"},
        ],
        "external": [
            {"href": "https://www.iana.org/domains/example"},
            {"href": "https://www.rfc-editor.org/info/rfc2606"},
        ],
    },
}

# Fallback case: Success with raw_markdown only (fit_markdown missing)
MOCK_CRAWL_RESULT_RAW_MARKDOWN_ONLY = {
    "url": "https://example.com/page",
    "success": True,
    "status_code": 200,
    "markdown": {
        "raw_markdown": "# Page Title\n\nSome content here."
    },
    "metadata": {
        "title": "Page Title",
        "description": "",
    },
    "links": {
        "internal": [],
        "external": [],
    },
}

# Error case: 404 Not Found
MOCK_CRAWL_RESULT_404 = {
    "url": "https://example.com/missing",
    "success": False,
    "status_code": 404,
    "error_message": "Page not found",
}

# Error case: 500 Internal Server Error
MOCK_CRAWL_RESULT_500 = {
    "url": "https://example.com/broken",
    "success": False,
    "status_code": 500,
    "error_message": "Internal server error occurred during crawl",
}

# Error case: Service unavailable
MOCK_CRAWL_RESULT_503 = {
    "url": "https://example.com/overload",
    "success": False,
    "status_code": 503,
    "error_message": "Service temporarily unavailable",
}

# Edge case: Success but empty markdown
MOCK_CRAWL_RESULT_EMPTY_MARKDOWN = {
    "url": "https://example.com/empty",
    "success": True,
    "status_code": 200,
    "markdown": {
        "fit_markdown": "",
        "raw_markdown": "",
    },
    "metadata": {
        "title": "Empty Page",
        "description": "",
    },
    "links": {
        "internal": [],
        "external": [],
    },
}

# Edge case: Success with minimal metadata
MOCK_CRAWL_RESULT_MINIMAL = {
    "url": "https://example.com/minimal",
    "success": True,
    "status_code": 200,
    "markdown": {
        "fit_markdown": "# Minimal Page\n\nBasic content.",
    },
    "metadata": {},
    "links": {},
}


def get_success_response() -> dict[str, Any]:
    """
    Return a mock successful Crawl4AI response.

    Returns:
        dict: Complete CrawlResult with 200 status and markdown content
    """
    return MOCK_CRAWL_RESULT_SUCCESS


def get_404_response() -> dict[str, Any]:
    """
    Return a mock 404 Not Found response.

    Returns:
        dict: CrawlResult with success=False and 404 status
    """
    return MOCK_CRAWL_RESULT_404


def get_500_response() -> dict[str, Any]:
    """
    Return a mock 500 Internal Server Error response.

    Returns:
        dict: CrawlResult with success=False and 500 status
    """
    return MOCK_CRAWL_RESULT_500


def get_timeout_response() -> None:
    """
    Return a mock timeout scenario (no actual response object).

    This function exists for consistency but should be used with respx to
    simulate timeout exceptions rather than returning actual data.

    Returns:
        None: Indicates timeout scenario for test setup

    Usage:
        respx.post("http://localhost:52004/crawl").mock(
            side_effect=httpx.TimeoutException("Connection timeout")
        )
    """
    return None


# MapperService fixtures: Simplified link discovery responses for /crawl endpoint
MOCK_MAP_RESULT_SUCCESS = {
    "success": True,
    "results": [
        {
            "url": "https://example.com",
            "links": {
                "internal": [{"href": "/a"}, {"href": "/b"}, {"href": "/c"}],
                "external": [
                    {"href": "https://www.iana.org/domains/example"},
                    {"href": "https://example.net"},
                ],
            },
        }
    ],
}

# MapperService: Response with duplicate internal links
MOCK_MAP_RESULT_WITH_DUPLICATES = {
    "success": True,
    "results": [
        {
            "url": "https://example.com",
            "links": {
                "internal": [{"href": "/about"}, {"href": "/about"}],
                "external": [],
            },
        }
    ],
}

# MapperService: Response for nested page discovery
MOCK_MAP_RESULT_NESTED = {
    "success": True,
    "results": [
        {
            "url": "https://example.com",
            "links": {
                "internal": [{"href": "/team"}],
                "external": [],
            },
        }
    ],
}

# MapperService: Empty links response
MOCK_MAP_RESULT_EMPTY = {
    "success": True,
    "results": [
        {
            "url": "https://example.com",
            "links": {
                "internal": [],
                "external": [],
            },
        }
    ],
}
