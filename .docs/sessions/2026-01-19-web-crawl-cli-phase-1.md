21:05:20 | 01/19/2026
- Issue: `python -m crawl4r.cli.app scrape https://example.com` failed with "Missing command" because subcommand groups required a command.
- Fix: Enabled `invoke_without_command=True` on scrape/crawl/status Typer groups so callbacks run directly.
- Note: POC commands run with env overrides for host endpoints: CRAWL4AI_BASE_URL, REDIS_URL, QDRANT_URL, TEI_ENDPOINT.
