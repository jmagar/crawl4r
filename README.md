# Crawl4AI Deployment (Docker Compose)

This directory is a minimal deployment wrapper for the prebuilt Crawl4AI image.

## Quick Start

1. Create the environment file:
   ```bash
   cp .env.example .env
   ```

2. Start the service:
   ```bash
   docker compose -f docker-compose.yaml up -d
   ```

3. Verify health:
   ```bash
   # From another container (like code-server):
   # Use the Docker gateway IP from `ip route` (default via ...).
   curl -f http://10.2.0.1:52001/health

   # From the host:
   curl -f http://localhost:52001/health
   ```

## Notes

- Host port is remapped to `52001` to avoid default ports.
- Set only the LLM provider keys you intend to use in `.env`.
- Stop the service with:
  ```bash
  docker compose -f docker-compose.yaml down
  ```
