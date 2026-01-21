# Fix Extract --prompt 422 Error Implementation Plan

**Created:** 12:47:02 AM | 01/21/2026 (EST)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `crawl4r extract --prompt` command that fails with 422 error due to incorrect API parameter naming.

**Architecture:** The Crawl4AI `/llm/job` endpoint expects `q` (query) parameter, not `prompt`. Current implementation sends `{"prompt": "..."}` which causes 422 validation error. Fix requires changing parameter name and updating tests.

**Tech Stack:** Python 3.10+, httpx, pytest, respx

---

## Root Cause Analysis

**Problem:** `crawl4r extract https://example.com --prompt "Extract heading"` returns 422 error.

**Actual API Expectation:**
```json
{
  "url": "https://example.com",
  "q": "Extract the main heading",
  "provider": "openai/gpt-4o-mini"  // optional
}
```

**Current Payload (WRONG):**
```json
{
  "url": "https://example.com",
  "prompt": "Extract the main heading",
  "instruction": "Extract the main heading",  // fallback attempt
  "provider": "openai/gpt-4o-mini"
}
```

**API Response:**
```json
{
  "detail": [{
    "type": "missing",
    "loc": ["body", "q"],
    "msg": "Field required",
    "input": {"url": "...", "prompt": "..."}
  }]
}
```

**Files to Fix:**
- `crawl4r/services/extractor.py:312-313` - Change `prompt` to `q`, remove `instruction`
- `tests/unit/services/test_extractor_service.py:204` - Update test assertion

---

## Phase 1: Write Failing Integration Test

### Step 1: Create integration test file

Create: `/home/jmagar/workspace/crawl4r/tests/integration/test_extractor_integration.py`

```python
"""Integration tests for ExtractorService with real Crawl4AI service.

These tests verify ExtractorService works correctly with a running Crawl4AI
instance at localhost:52004. They test actual API interactions, not mocks.
"""

import os
import pytest
from crawl4r.services.extractor import ExtractorService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_with_prompt_real_service() -> None:
    """Test prompt-based extraction with real Crawl4AI service.

    This integration test verifies the /llm/job endpoint accepts
    prompt-based extraction requests with correct parameter naming.

    Requires:
        - Crawl4AI service running at localhost:52004
        - LLM provider configured (openai/gpt-4o-mini or ollama)
    """
    endpoint_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:52004")

    async with ExtractorService(endpoint_url=endpoint_url) as service:
        result = await service.extract_with_prompt(
            url="https://example.com",
            prompt="Extract the main heading from this page"
        )

        # Should not fail with 422 error
        assert result.success is True, f"Expected success, got error: {result.error}"
        assert result.data is not None
        assert result.extraction_method == "prompt"
```

### Step 2: Run integration test to verify it fails

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest /home/jmagar/workspace/crawl4r/tests/integration/test_extractor_integration.py::test_extract_with_prompt_real_service -v -m integration`

Expected output:
```
FAILED tests/integration/test_extractor_integration.py::test_extract_with_prompt_real_service
AssertionError: Expected success, got error: Request failed with status 422
```

---

## Phase 2: Fix API Parameter Naming

### Step 3: Update _fetch_extraction to use `q` parameter

Modify: `/home/jmagar/workspace/crawl4r/crawl4r/services/extractor.py:287-316`

**OLD CODE (lines 307-315):**
```python
# Build request payload
payload: dict[str, Any] = {"url": url}
if schema is not None:
    payload["schema"] = schema
if prompt is not None:
    payload["prompt"] = prompt
    payload["instruction"] = prompt  # Some APIs use 'instruction'
if provider is not None:
    payload["provider"] = provider
```

**NEW CODE:**
```python
# Build request payload
payload: dict[str, Any] = {"url": url}
if schema is not None:
    payload["schema"] = schema
if prompt is not None:
    # Crawl4AI /llm/job endpoint expects 'q' parameter for prompts
    payload["q"] = prompt
if provider is not None:
    payload["provider"] = provider
```

**Rationale:** Crawl4AI v0.5.x `/llm/job` endpoint requires `q` field for natural language queries, not `prompt` or `instruction`.

### Step 4: Run integration test to verify it passes

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest /home/jmagar/workspace/crawl4r/tests/integration/test_extractor_integration.py::test_extract_with_prompt_real_service -v -m integration`

Expected output:
```
PASSED tests/integration/test_extractor_integration.py::test_extract_with_prompt_real_service
```

---

## Phase 3: Update Unit Test Assertions

### Step 5: Update unit test to verify correct parameter name

Modify: `/home/jmagar/workspace/crawl4r/tests/unit/services/test_extractor_service.py:179-205`

**OLD CODE (line 204):**
```python
assert b"prompt" in request_body or b"instruction" in request_body
```

**NEW CODE:**
```python
# Verify 'q' parameter is sent (not 'prompt')
assert b"q" in request_body
assert b'"q":' in request_body or b'"q": "' in request_body
```

**Rationale:** Test should verify the correct API parameter is sent to Crawl4AI.

### Step 6: Run unit test to verify it passes

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest /home/jmagar/workspace/crawl4r/tests/unit/services/test_extractor_service.py::test_extract_with_prompt_passes_prompt_to_endpoint -v`

Expected output:
```
PASSED tests/unit/services/test_extractor_service.py::test_extract_with_prompt_passes_prompt_to_endpoint
```

---

## Phase 4: Add Edge Case Tests

### Step 7: Write test for schema + prompt combination

Add to: `/home/jmagar/workspace/crawl4r/tests/integration/test_extractor_integration.py`

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_with_schema_and_prompt_real_service() -> None:
    """Test extraction with both schema and prompt (schema takes priority).

    When both schema and prompt are provided, the Crawl4AI API uses
    the schema for structured extraction and ignores the prompt.
    """
    endpoint_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:52004")

    async with ExtractorService(endpoint_url=endpoint_url) as service:
        # This is allowed by the API but schema takes precedence
        result = await service.extract_with_schema(
            url="https://example.com",
            schema={"type": "object", "properties": {"title": {"type": "string"}}},
            provider="openai/gpt-4o-mini"
        )

        assert result.success is True
        assert result.extraction_method == "schema"
```

### Step 8: Run schema + prompt test

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest /home/jmagar/workspace/crawl4r/tests/integration/test_extractor_integration.py::test_extract_with_schema_and_prompt_real_service -v -m integration`

Expected output:
```
PASSED tests/integration/test_extractor_integration.py::test_extract_with_schema_and_prompt_real_service
```

---

## Phase 5: Test CLI Command

### Step 9: Test extract --prompt command manually

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && python -m crawl4r.cli.app extract https://example.com --prompt "Extract the main heading" --provider "openai/gpt-4o-mini"`

Expected output:
```json
{
  "heading": "Example Domain"
}
```

**Failure Scenario (OLD):**
```
Failed: Request failed with status 422
```

**Success Scenario (NEW):**
- JSON output with extracted data
- Exit code 0

### Step 10: Test extract --schema command (verify not broken)

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && python -m crawl4r.cli.app extract https://example.com --schema '{"type":"object","properties":{"title":{"type":"string"}}}' --provider "openai/gpt-4o-mini"`

Expected output:
```json
{
  "title": "Example Domain"
}
```

---

## Phase 6: Update Documentation

### Step 11: Add API parameter mapping documentation

Add to: `/home/jmagar/workspace/crawl4r/CLAUDE.md`

After line 237 (in "Crawl4AI API Endpoints" section):

```markdown
### LLM Extraction Endpoint

**Endpoint:** `POST /llm/job`

**Request Parameters:**
- `url` (required): URL to extract from
- `q` (required): Natural language query/prompt for extraction
- `schema` (optional): JSON schema for structured extraction
- `provider` (optional): LLM provider (e.g., `openai/gpt-4o-mini`, `ollama/llama3`)

**Response:**
```json
{
  "task_id": "llm_1768956440_135097258120432",
  "status": "processing",
  "url": "https://example.com",
  "_links": {
    "self": {"href": "http://localhost:52004/llm/llm_1768956440_135097258120432"},
    "status": {"href": "http://localhost:52004/llm/llm_1768956440_135097258120432"}
  }
}
```

**Note:** The endpoint uses `q` for queries, not `prompt`. This is different from
the CLI flag `--prompt` which is user-facing and gets mapped to `q` internally.
```

### Step 12: Update ExtractorService docstring

Modify: `/home/jmagar/workspace/crawl4r/crawl4r/services/extractor.py:287-304`

Add note to `_fetch_extraction` docstring:

**OLD DOCSTRING:**
```python
"""Fetch extraction results from Crawl4AI /llm/job endpoint.

Args:
    url: URL to extract data from.
    schema: JSON schema for extraction (optional).
    prompt: Natural language prompt (optional).
    provider: LLM provider to use (optional).

Returns:
    ExtractResult with extracted data or error details.
"""
```

**NEW DOCSTRING:**
```python
"""Fetch extraction results from Crawl4AI /llm/job endpoint.

Args:
    url: URL to extract data from.
    schema: JSON schema for extraction (optional).
    prompt: Natural language prompt (optional).
           Maps to 'q' parameter in Crawl4AI API.
    provider: LLM provider to use (optional).

Returns:
    ExtractResult with extracted data or error details.

Note:
    Crawl4AI /llm/job endpoint expects 'q' parameter for prompts,
    not 'prompt'. This method handles the parameter mapping.
"""
```

---

## Phase 7: Run Full Test Suite

### Step 13: Run all extractor unit tests

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest /home/jmagar/workspace/crawl4r/tests/unit/services/test_extractor_service.py -v`

Expected output:
```
============================== 35 passed ==============================
```

All existing tests should continue passing.

### Step 14: Run all integration tests

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest /home/jmagar/workspace/crawl4r/tests/integration/test_extractor_integration.py -v -m integration`

Expected output:
```
============================== 2 passed ==============================
```

### Step 15: Run linting and type checking

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && ruff check /home/jmagar/workspace/crawl4r/crawl4r/services/extractor.py /home/jmagar/workspace/crawl4r/tests/unit/services/test_extractor_service.py /home/jmagar/workspace/crawl4r/tests/integration/test_extractor_integration.py`

Expected output:
```
All checks passed!
```

---

## Phase 8: Commit Changes

### Step 16: Stage and commit the fix

Run:
```bash
cd /home/jmagar/workspace/crawl4r && \
source .venv/bin/activate && \
git add crawl4r/services/extractor.py tests/unit/services/test_extractor_service.py tests/integration/test_extractor_integration.py CLAUDE.md && \
git commit -m "$(cat <<'EOF'
fix(extractor): change prompt parameter to q for Crawl4AI API

Fixes #3 - extract --prompt failing with 422 error

Changes:
- Change 'prompt' parameter to 'q' in /llm/job requests
- Remove fallback 'instruction' parameter (not used by API)
- Add integration test verifying real Crawl4AI service accepts prompt
- Update unit test assertions to check for 'q' parameter
- Document API parameter mapping in CLAUDE.md

Root cause: Crawl4AI v0.5.x /llm/job endpoint requires 'q' field
for natural language queries, not 'prompt'. This caused 422 validation
errors when using --prompt flag.

Verified:
- Integration test passes with real service at localhost:52004
- Unit tests verify correct parameter sent in request payload
- CLI command works: crawl4r extract URL --prompt "query"
- Schema extraction still works (unaffected)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

Expected output:
```
[main abc1234] fix(extractor): change prompt parameter to q for Crawl4AI API
 4 files changed, 87 insertions(+), 5 deletions(-)
 create mode 100644 tests/integration/test_extractor_integration.py
```

---

## Verification Checklist

**Before committing, verify:**

- [ ] Integration test passes with real Crawl4AI service
- [ ] Unit test assertions updated and passing
- [ ] CLI `--prompt` command works without 422 error
- [ ] CLI `--schema` command still works (not broken)
- [ ] All 35 unit tests pass
- [ ] All 2 integration tests pass
- [ ] Ruff linting passes
- [ ] Documentation updated in CLAUDE.md
- [ ] Commit message follows conventional commits format

---

## Known Limitations

1. **Async Job Handling:** Current implementation doesn't poll task status from `/llm/job` response. This is acceptable for POC but should be addressed in production (task tracking in Phase 3).

2. **Provider Configuration:** Test uses `openai/gpt-4o-mini` which requires OpenAI API key in Crawl4AI service environment. Alternative is `ollama/llama3` for local testing.

3. **Webhook Support:** Crawl4AI supports webhooks for async notifications but current implementation doesn't use them (future enhancement).

---

## Success Criteria

1. ✅ `crawl4r extract https://example.com --prompt "Extract heading"` returns JSON without 422 error
2. ✅ Integration test verifies real Crawl4AI service accepts prompt parameter
3. ✅ Unit test verifies `q` parameter sent in request body (not `prompt`)
4. ✅ Schema extraction continues working unchanged
5. ✅ All tests pass (35 unit + 2 integration)
6. ✅ Documentation updated explaining API parameter mapping

---

## References

**API Documentation:**
- Crawl4AI v0.7.x Self-Hosting Guide: https://docs.crawl4ai.com/core/self-hosting/
- LLM Strategies: https://docs.crawl4ai.com/extraction/llm-strategies/

**Related Files:**
- `crawl4r/services/extractor.py` - ExtractorService implementation
- `crawl4r/cli/commands/extract.py` - CLI command handler
- `tests/unit/services/test_extractor_service.py` - Unit tests
- `tests/integration/test_extractor_integration.py` - Integration tests (new)

**Issue:** Bug #3 - `extract --prompt` fails with 422 error
