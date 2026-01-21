# Fix Scrape Command Flag Parsing Implementation Plan

**Created:** 12:47:01 AM | 01/21/2026 (EST)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix Typer argument configuration so flags like `--output` and `--concurrent` are parsed correctly instead of being captured as URLs.

**Architecture:** Update Typer argument definition to use ellipsis `...` instead of `None` default, ensuring variadic arguments don't capture option flags. Add comprehensive tests validating all flag combinations.

**Tech Stack:** Python 3.11+, Typer CLI framework, pytest, typer.testing.CliRunner

---

## Root Cause Analysis

**Current Problem:**
```python
# Line 23: crawl4r/cli/commands/scrape.py
urls: list[str] = typer.Argument(None)
```

This configuration allows the variadic argument to capture **everything** after `scrape`, including flags like `--output` and `--concurrent`.

**Example Failure:**
```bash
scrape https://example.com --output file.md
# Parses as: urls=['https://example.com', '--output', 'file.md']
# Instead of: urls=['https://example.com'], output='file.md'
```

**Fix:**
```python
urls: list[str] = typer.Argument(...)
```

Using ellipsis `...` instead of `None` tells Typer this is a **required** variadic argument that should **stop** when it encounters flags.

---

## Phase 1: Test-Driven Fix (RED-GREEN-REFACTOR)

### Step 1: Write failing test for single URL with --output flag

**Create:** `tests/unit/cli/test_scrape_flag_parsing.py`

```python
"""Tests for scrape command flag parsing (Bug #2)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from crawl4r.cli.app import app


@pytest.fixture
def mock_scraper_service():
    """Mock ScraperService to avoid real HTTP calls."""
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock:
        service_instance = MagicMock()
        service_instance.scrape_url = AsyncMock(
            return_value=MagicMock(
                success=True,
                url="https://example.com",
                markdown="# Example\n\nContent",
                error=None,
            )
        )
        mock.return_value = service_instance
        yield service_instance


def test_scrape_single_url_with_output_flag(mock_scraper_service, tmp_path):
    """Test that --output flag is parsed correctly after URL."""
    runner = CliRunner()
    output_file = tmp_path / "result.md"

    result = runner.invoke(
        app, ["scrape", "https://example.com", "--output", str(output_file)]
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert output_file.exists(), "Output file not created"
    assert "Example" in output_file.read_text()
    mock_scraper_service.scrape_url.assert_called_once_with("https://example.com")
```

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_output_flag -v`

**Expected:** FAIL with URLs capturing `--output` and `str(output_file)` as positional arguments

---

### Step 2: Run test to verify it fails

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_output_flag -v`

**Expected Output:**
```
FAILED tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_output_flag
AssertionError: URLs list incorrectly contains ['https://example.com', '--output', '/tmp/.../result.md']
```

---

### Step 3: Fix Typer argument configuration

**Modify:** `crawl4r/cli/commands/scrape.py:23`

**OLD:**
```python
urls: list[str] = typer.Argument(None),
```

**NEW:**
```python
urls: list[str] = typer.Argument(...),
```

**Explanation:**
- `None` default makes argument optional and greedy (captures flags)
- `...` (ellipsis) makes argument required and stops at first flag
- This is the standard Typer pattern for variadic arguments with options

---

### Step 4: Run test to verify it passes

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_output_flag -v`

**Expected Output:**
```
PASSED tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_output_flag
```

---

### Step 5: Write failing test for --concurrent flag

**Modify:** `tests/unit/cli/test_scrape_flag_parsing.py`

**ADD:**
```python
def test_scrape_single_url_with_concurrent_flag(mock_scraper_service, tmp_path):
    """Test that --concurrent flag is parsed correctly after URL."""
    runner = CliRunner()

    result = runner.invoke(
        app, ["scrape", "https://example.com", "--concurrent", "3"]
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    # Verify concurrent limit was respected (check it's not treated as URL)
    mock_scraper_service.scrape_url.assert_called_once_with("https://example.com")
```

---

### Step 6: Run test to verify it passes (should pass with previous fix)

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_concurrent_flag -v`

**Expected Output:**
```
PASSED tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_concurrent_flag
```

---

### Step 7: Write failing test for combined flags

**Modify:** `tests/unit/cli/test_scrape_flag_parsing.py`

**ADD:**
```python
def test_scrape_single_url_with_combined_flags(mock_scraper_service, tmp_path):
    """Test that multiple flags work together after URL."""
    runner = CliRunner()
    output_file = tmp_path / "result.md"

    result = runner.invoke(
        app,
        [
            "scrape",
            "https://example.com",
            "--output",
            str(output_file),
            "--concurrent",
            "10",
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert output_file.exists(), "Output file not created"
    mock_scraper_service.scrape_url.assert_called_once_with("https://example.com")
```

---

### Step 8: Run test to verify it passes

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_combined_flags -v`

**Expected Output:**
```
PASSED tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_single_url_with_combined_flags
```

---

### Step 9: Write failing test for multiple URLs with flags

**Modify:** `tests/unit/cli/test_scrape_flag_parsing.py`

**ADD:**
```python
def test_scrape_multiple_urls_with_output_flag(mock_scraper_service, tmp_path):
    """Test that flags work with multiple URLs."""
    runner = CliRunner()
    output_dir = tmp_path / "results"

    # Mock returns different content for each URL
    async def mock_scrape(url):
        return MagicMock(
            success=True,
            url=url,
            markdown=f"# {url}\n\nContent",
            error=None,
        )

    mock_scraper_service.scrape_url = AsyncMock(side_effect=mock_scrape)

    result = runner.invoke(
        app,
        [
            "scrape",
            "https://example.com",
            "https://example.org",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert output_dir.exists(), "Output directory not created"
    assert len(list(output_dir.glob("*.md"))) == 2, "Should create 2 markdown files"
    assert mock_scraper_service.scrape_url.call_count == 2
```

---

### Step 10: Run test to verify it passes

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_multiple_urls_with_output_flag -v`

**Expected Output:**
```
PASSED tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_multiple_urls_with_output_flag
```

---

## Phase 2: Backward Compatibility Tests

### Step 11: Write test for URL-only invocation (no flags)

**Modify:** `tests/unit/cli/test_scrape_flag_parsing.py`

**ADD:**
```python
def test_scrape_url_only_no_flags(mock_scraper_service):
    """Test backward compatibility: URL without any flags."""
    runner = CliRunner()

    result = runner.invoke(app, ["scrape", "https://example.com"])

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert "# Example" in result.output  # Should print to stdout
    mock_scraper_service.scrape_url.assert_called_once_with("https://example.com")
```

---

### Step 12: Run test to verify backward compatibility

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_url_only_no_flags -v`

**Expected Output:**
```
PASSED tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_url_only_no_flags
```

---

### Step 13: Write test for --file flag (existing feature)

**Modify:** `tests/unit/cli/test_scrape_flag_parsing.py`

**ADD:**
```python
def test_scrape_with_file_flag(mock_scraper_service, tmp_path):
    """Test backward compatibility: --file flag for URL list."""
    runner = CliRunner()
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com\nhttps://example.org\n")

    # Mock returns different content for each URL
    async def mock_scrape(url):
        return MagicMock(
            success=True,
            url=url,
            markdown=f"# {url}\n\nContent",
            error=None,
        )

    mock_scraper_service.scrape_url = AsyncMock(side_effect=mock_scrape)

    result = runner.invoke(app, ["scrape", "--file", str(url_file)])

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert mock_scraper_service.scrape_url.call_count == 2
```

---

### Step 14: Run test to verify --file flag still works

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_with_file_flag -v`

**Expected Output:**
```
PASSED tests/unit/cli/test_scrape_flag_parsing.py::test_scrape_with_file_flag
```

---

### Step 15: Handle edge case - no URLs provided

**IMPORTANT:** With `typer.Argument(...)`, Typer will now **require** at least one URL. This changes behavior when using `--file` alone.

**Check current behavior:**

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && python -m crawl4r.cli.app scrape --help`

**Expected:** See if `[URLS]...` is marked as required

**Decision Point:**
- If `--file` flag should work **without** positional URLs, we need to adjust the fix
- Current code at line 29 merges URLs from both sources: `_merge_urls(urls or [], file)`
- With `...`, `urls` can never be empty when invoked, so `--file`-only won't work

---

### Step 16: Fix argument to make URLs optional when --file is provided

**Modify:** `crawl4r/cli/commands/scrape.py:23`

**Change approach - use callback validation instead of Typer defaults:**

**REPLACE:**
```python
@app.callback()
def scrape(
    urls: list[str] = typer.Argument(...),
    file: Path | None = typer.Option(None, "-f", "--file"),
    output: Path | None = typer.Option(None, "-o", "--output"),
    concurrent: int = typer.Option(5, "-c", "--concurrent"),
) -> None:
```

**WITH:**
```python
@app.callback()
def scrape(
    urls: list[str] | None = typer.Argument(None),
    file: Path | None = typer.Option(None, "-f", "--file"),
    output: Path | None = typer.Option(None, "-o", "--output"),
    concurrent: int = typer.Option(5, "-c", "--concurrent"),
) -> None:
```

**WAIT - This brings back the original bug!**

**Better Solution:** Use explicit `metavar` and `help` to guide Typer parser:

**FINAL FIX:**
```python
@app.callback()
def scrape(
    urls: list[str] | None = typer.Argument(
        None,
        metavar="[URLS]...",
        help="URLs to scrape (or use --file)",
    ),
    file: Path | None = typer.Option(None, "-f", "--file", help="File containing URLs"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file/directory"),
    concurrent: int = typer.Option(5, "-c", "--concurrent", help="Max concurrent requests"),
) -> None:
```

**NOTE:** Adding `metavar` and `help` doesn't fix the parsing bug. The real issue is Typer's behavior with `Argument(None)`.

---

### Step 17: Research Typer's argument handling

**Investigation needed:**

The bug occurs because Typer doesn't know where arguments end and options begin when using `Argument(None)` for a list.

**Typer Documentation Pattern:**
```python
# Correct pattern for variadic arguments with options
def command(
    names: list[str] = typer.Argument(None),  # This CAN capture flags!
):
    pass
```

**The issue:** Typer processes arguments left-to-right. When it sees `scrape URL --output file.md`, it:
1. Sees `scrape` (command)
2. Sees `URL` (matches `urls` argument)
3. Sees `--output` - should be option, but `urls` is still accepting arguments!
4. Captures `--output` and `file.md` as additional URLs

**Solution:** Use `--` separator or make URLs required with `...`

**Typer Best Practice:**
```python
# Force users to use -- separator
scrape -- https://example.com --output file.md  # Won't work, flags after --

# OR make URLs required (our approach)
urls: list[str] = typer.Argument(...)  # At least one required, stops at first flag
```

**Trade-off:**
- Using `...` breaks `scrape --file urls.txt` (no positional URLs)
- Using `None` captures flags as URLs (current bug)

---

### Step 18: Implement hybrid solution with validation

**Best Solution:** Keep `Argument(None)` but add runtime validation and update help text

**Modify:** `crawl4r/cli/commands/scrape.py:22-32`

**OLD:**
```python
@app.callback()
def scrape(
    urls: list[str] = typer.Argument(None),
    file: Path | None = typer.Option(None, "-f", "--file"),
    output: Path | None = typer.Option(None, "-o", "--output"),
    concurrent: int = typer.Option(5, "-c", "--concurrent"),
) -> None:
    """Scrape URLs and output markdown."""
    resolved_urls = _merge_urls(urls or [], file)
    if not resolved_urls:
        typer.echo("No URLs provided")
        raise typer.Exit(code=1)
```

**NEW:**
```python
@app.callback()
def scrape(
    urls: list[str] = typer.Argument(
        None,
        help="URLs to scrape (place before flags or use --file)",
    ),
    file: Path | None = typer.Option(
        None,
        "-f",
        "--file",
        help="File containing URLs (one per line)",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file or directory",
    ),
    concurrent: int = typer.Option(
        5,
        "-c",
        "--concurrent",
        help="Maximum concurrent requests",
    ),
) -> None:
    """Scrape URLs and output markdown.

    Examples:
        scrape https://example.com
        scrape https://example.com -o output.md
        scrape https://a.com https://b.com -o results/
        scrape -f urls.txt -o results/
    """
    # Validate URLs don't start with '-' (likely captured flags)
    if urls:
        invalid_urls = [url for url in urls if url.startswith("-")]
        if invalid_urls:
            typer.echo(
                f"Error: URLs cannot start with '-'. "
                f"Did you mean to use a flag? Invalid: {invalid_urls}\n"
                f"Place URLs before flags: scrape URL [URL...] [OPTIONS]",
                err=True,
            )
            raise typer.Exit(code=1)

    resolved_urls = _merge_urls(urls or [], file)
    if not resolved_urls:
        typer.echo("No URLs provided. Use positional URLs or --file option.")
        raise typer.Exit(code=1)
```

**This adds runtime detection but doesn't fix the root cause!**

---

### Step 19: CORRECT FIX - Update help text and document usage pattern

**After research, the REAL fix is:**

1. Keep `Argument(None)` for optional variadic args
2. **Document** that URLs must come BEFORE flags (standard CLI convention)
3. Add validation to detect when flags are captured
4. Update tests to verify correct usage patterns

**The bug is a USAGE issue, not a code issue:**
- WRONG: `scrape https://example.com --output file.md`
- RIGHT: `scrape --output file.md https://example.com`
- ALSO RIGHT: `scrape https://example.com` then pipe to file

**Wait - let me verify Typer's actual behavior:**

---

### Step 20: Test Typer's actual parsing behavior

**Create temporary test script:**

**Create:** `/tmp/test_typer_parsing.py`

```python
import typer

app = typer.Typer()

@app.command()
def test(
    urls: list[str] = typer.Argument(None),
    output: str = typer.Option(None, "-o", "--output"),
):
    print(f"URLs: {urls}")
    print(f"Output: {output}")

if __name__ == "__main__":
    app()
```

**Test 1 - Flags after URLs:**
**Run:** `python /tmp/test_typer_parsing.py https://example.com --output file.md`

**Test 2 - Flags before URLs:**
**Run:** `python /tmp/test_typer_parsing.py --output file.md https://example.com`

**Test 3 - Multiple URLs with flags:**
**Run:** `python /tmp/test_typer_parsing.py https://a.com https://b.com --output file.md`

**Expected:** Determine if Typer correctly parses flags in different positions

---

### Step 21: Run Typer parsing tests manually

**Run:**
```bash
cat > /tmp/test_typer_parsing.py << 'EOF'
import typer

app = typer.Typer()

@app.command()
def test(
    urls: list[str] = typer.Argument(None),
    output: str = typer.Option(None, "-o", "--output"),
):
    print(f"URLs: {urls}")
    print(f"Output: {output}")

if __name__ == "__main__":
    app()
EOF

python /tmp/test_typer_parsing.py https://example.com --output file.md
```

**Expected Output (if bug exists):**
```
URLs: ['https://example.com', '--output', 'file.md']
Output: None
```

**Expected Output (if Typer works correctly):**
```
URLs: ['https://example.com']
Output: file.md
```

---

### Step 22: Based on test results, implement correct fix

**IF Typer captures flags (bug confirmed):**

The fix is to use Click's `nargs=-1` with explicit termination:

**Modify:** `crawl4r/cli/commands/scrape.py:22`

**REPLACE:**
```python
urls: list[str] = typer.Argument(None),
```

**WITH:**
```python
urls: list[str] = typer.Argument(None, click_type=click.Tuple()),  # Wrong approach
```

**ACTUALLY, use native Typer solution - check Typer version:**

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && python -c "import typer; print(typer.__version__)"`

**Then research Typer issue tracker for this bug**

---

### Step 23: FINAL SOLUTION - Use documented Typer pattern

After research, the **correct Typer pattern** for variadic arguments with options is:

**Modify:** `crawl4r/cli/commands/scrape.py:22-26`

**FINAL FIX:**
```python
@app.callback()
def scrape(
    urls: list[str] = typer.Argument(None),  # Keep as-is
    file: Path | None = typer.Option(None, "-f", "--file"),
    output: Path | None = typer.Option(None, "-o", "--output"),
    concurrent: int = typer.Option(5, "-c", "--concurrent"),
) -> None:
```

**The fix is actually in how Typer is invoked - need to ensure options are registered BEFORE arguments are parsed.**

**ROOT CAUSE FOUND:** The `@app.callback()` decorator might be the issue!

**Callbacks in Typer run BEFORE subcommands, not AS commands**

**Change:**

**OLD:**
```python
app = typer.Typer(no_args_is_help=True, invoke_without_command=True)

@app.callback()
def scrape(...):
```

**NEW:**
```python
app = typer.Typer(no_args_is_help=True)

@app.command()  # Use command(), not callback()
def main(...):  # Rename to avoid confusion
```

---

### Step 24: Implement actual fix - change callback to command

**Modify:** `crawl4r/cli/commands/scrape.py:18-27`

**OLD:**
```python
app = typer.Typer(no_args_is_help=True, invoke_without_command=True)


@app.callback()
def scrape(
    urls: list[str] = typer.Argument(None),
```

**NEW:**
```python
app = typer.Typer(help="Scrape URLs and output markdown")


@app.command(name="scrape")
def scrape_command(
    urls: list[str] = typer.Argument(None, help="URLs to scrape"),
```

**WAIT - This changes the CLI interface!**

**Current:** `crawl4r scrape https://example.com`
**After change:** `crawl4r scrape scrape https://example.com` (double scrape!)

**The issue is how scrape is registered in main CLI**

---

### Step 25: Check how scrape is registered in main app

**Read:** `crawl4r/cli/app.py` to see how `scrape` command is added

**Run:**
```bash
source /home/jmagar/workspace/crawl4r/.venv/bin/activate && grep -n "scrape" /home/jmagar/workspace/crawl4r/crawl4r/cli/app.py
```

**Expected:** See `app.add_typer(scrape.app, name="scrape")` or similar

---

### Step 26: Verify registration pattern

**Read registration:** `/home/jmagar/workspace/crawl4r/crawl4r/cli/app.py:*-*` (find scrape registration)

Based on registration pattern, determine correct fix:

**Option A:** If using `add_typer()`, keep `@app.callback()` but fix Typer version
**Option B:** If using `@app.command()`, the bug is elsewhere
**Option C:** Need to restructure command registration

---

## Summary of Investigation Steps

**Steps 1-14:** Initial TDD cycle with assumed fix (Argument(...))
**Steps 15-26:** Deep investigation of root cause - discovered it's likely:
1. Typer version issue
2. Registration pattern issue (`callback` vs `command`)
3. Need to check how scrape command is mounted in main CLI

**NEXT:** Read main CLI file to understand registration, then apply correct fix

---

## ACTUAL Implementation (After Investigation)

### Step 27: Read main CLI app structure

**Read:** `crawl4r/cli/app.py`

---

### Step 28: Apply researched fix based on CLI structure

**[Will be determined after Step 27]**

---

### Step 29: Run all flag parsing tests

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_flag_parsing.py -v`

**Expected Output:**
```
test_scrape_single_url_with_output_flag PASSED
test_scrape_single_url_with_concurrent_flag PASSED
test_scrape_single_url_with_combined_flags PASSED
test_scrape_multiple_urls_with_output_flag PASSED
test_scrape_url_only_no_flags PASSED
test_scrape_with_file_flag PASSED
======================== 6 passed in 0.5s ========================
```

---

### Step 30: Run existing scrape tests to ensure no regression

**Run:** `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_scrape_command.py -v`

**Expected:** All existing tests pass

---

### Step 31: Manual CLI testing

**Test real commands:**

```bash
source /home/jmagar/workspace/crawl4r/.venv/bin/activate

# Test 1: Single URL with output
python -m crawl4r.cli.app scrape https://example.com --output /tmp/test1.md

# Test 2: Multiple URLs with output directory
python -m crawl4r.cli.app scrape https://example.com https://example.org --output /tmp/results

# Test 3: File input with concurrent flag
echo "https://example.com" > /tmp/urls.txt
python -m crawl4r.cli.app scrape --file /tmp/urls.txt --concurrent 3

# Test 4: Combined flags
python -m crawl4r.cli.app scrape https://example.com --output /tmp/test4.md --concurrent 10
```

**Expected:** All commands execute successfully, flags are not treated as URLs

---

### Step 32: Update documentation

**Modify:** `README.md` (if scrape command is documented)

**ADD usage examples:**
```markdown
### Scrape Command

Scrape URLs and output markdown:

```bash
# Single URL (print to stdout)
crawl4r scrape https://example.com

# Save to file
crawl4r scrape https://example.com --output result.md

# Multiple URLs (saves to directory)
crawl4r scrape https://example.com https://example.org --output results/

# From file
crawl4r scrape --file urls.txt --output results/

# Concurrent requests
crawl4r scrape https://example.com --concurrent 10
```

**Note:** URLs can be placed before or after flags.
```

---

### Step 33: Commit changes

```bash
cd /home/jmagar/workspace/crawl4r
source .venv/bin/activate

git add crawl4r/cli/commands/scrape.py
git add tests/unit/cli/test_scrape_flag_parsing.py
git add README.md

git commit -m "$(cat <<'EOF'
fix(cli): correct flag parsing in scrape command

Fixed Bug #2 where flags like --output and --concurrent were captured
as positional URL arguments instead of being parsed as options.

Changes:
- Updated Typer argument configuration to prevent flag capture
- Added comprehensive flag parsing tests (6 new test cases)
- Validated backward compatibility with existing usage patterns
- Updated documentation with correct usage examples

Test coverage:
- Single URL with --output flag
- Single URL with --concurrent flag
- Combined flags (--output + --concurrent)
- Multiple URLs with flags
- URL-only invocation (no flags)
- --file flag compatibility

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

- [ ] All 6 new tests pass
- [ ] Existing scrape tests pass
- [ ] Manual CLI testing confirms flags work correctly
- [ ] `--output file.md` is not captured as URL
- [ ] `--concurrent 3` is not captured as URL
- [ ] Multiple flags can be combined
- [ ] Backward compatibility maintained (URL-only, --file flag)
- [ ] Documentation updated with examples
- [ ] No regression in other CLI commands

---

## Notes

**Key Learning:** Typer's variadic `Argument()` handling requires careful configuration to avoid capturing option flags as positional arguments.

**Resolution:** The fix depends on investigation results from Steps 20-27. Most likely solution is updating Typer configuration or restructuring command registration.

**Testing Strategy:** Created isolated test file (`test_scrape_flag_parsing.py`) to prevent interference with existing tests while validating all flag combinations.
