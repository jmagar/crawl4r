"""<module>
  <summary>Recursive crawler for code.claude.com using Crawl4AI HTTP API.</summary>
</module>"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse


def normalize_url(url: str) -> str | None:
    """<summary>Normalize a URL for deduplication.</summary>
    <param name="url">Input URL to normalize.</param>
    <returns>Normalized URL or None if invalid.</returns>"""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    if parsed.scheme not in {"http", "https"}:
        return None

    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            netloc,
            path,
            parsed.params,
            parsed.query,
            "",
        )
    )
    return normalized


ENGLISH_LOCALE_PREFIXES = {
    "en",
    "en-us",
    "en-gb",
    "en-ca",
    "en-au",
}

KNOWN_NON_ENGLISH_LOCALE_PREFIXES = {
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "pt-br",
    "pt-pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "th",
    "tr",
    "uk",
    "vi",
    "zh",
    "zh-cn",
    "zh-tw",
}

LOCALE_PREFIX_PATTERN = re.compile(r"^[a-z]{2}(-[a-z]{2})?$")


def is_non_english_locale_prefix(url: str) -> bool:
    """<summary>Detect non-English locale prefixes in URL paths.</summary>
    <param name="url">URL to inspect.</param>
    <returns>True when a non-English locale prefix is detected.</returns>"""
    parsed = urlparse(url)
    first_segment = parsed.path.strip("/").split("/", 1)[0].lower()
    if not first_segment:
        return False
    if first_segment in ENGLISH_LOCALE_PREFIXES:
        return False
    if first_segment in KNOWN_NON_ENGLISH_LOCALE_PREFIXES:
        return True
    if "-" in first_segment and LOCALE_PREFIX_PATTERN.match(first_segment):
        return True
    return False


def is_allowed_domain(url: str) -> bool:
    """<summary>Check if a URL is within the allowed claude.com domains.</summary>
    <param name="url">URL to validate.</param>
    <returns>True when the URL is in-scope.</returns>"""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host != "claude.com" and not host.endswith(".claude.com"):
        return False
    return not is_non_english_locale_prefix(url)


def get_filename_from_url(url: str) -> str:
    """<summary>Derive a filename-like label from a URL path.</summary>
    <param name="url">URL to derive a name from.</param>
    <returns>Filename-like string.</returns>"""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return "index"
    return path.split("/")[-1] or "index"


def iter_candidate_links(base_url: str, links: Iterable[str]) -> Iterable[str]:
    """<summary>Yield normalized, in-scope candidate links.</summary>
    <param name="base_url">Base URL for resolving relative links.</param>
    <param name="links">Links to normalize and filter.</param>
    <returns>Iterable of normalized URLs.</returns>"""
    for link in links:
        link_value: str | None = None
        if isinstance(link, str):
            link_value = link
        elif isinstance(link, dict):
            link_value = link.get("url") or link.get("href")

        if not link_value:
            continue

        absolute = urljoin(base_url, link_value)
        normalized = normalize_url(absolute)
        if normalized and is_allowed_domain(normalized):
            yield normalized


@lru_cache(maxsize=1)
def get_tokenizer():
    """<summary>Load the Qwen3 tokenizer (Qwen2TokenizerFast).</summary>
    <returns>Tokenizer instance.</returns>"""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for token-accurate chunking. "
            "Install it before running this script."
        ) from exc

    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B", use_fast=True
    )


def tokenize_with_offsets(text: str) -> list[tuple[int, int]]:
    """<summary>Return token character offsets for text.</summary>
    <param name="text">Text to tokenize.</param>
    <returns>List of (start, end) offsets for each token.</returns>"""
    tokenizer = get_tokenizer()
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    return list(encoded["offset_mapping"])


def split_markdown_sections(
    markdown: str, fallback_name: str
) -> list[dict[str, int | str]]:
    """<summary>Split markdown into heading-based sections.</summary>
    <param name="markdown">Markdown content.</param>
    <param name="fallback_name">Fallback section name when no headings exist.</param>
    <returns>List of section records with path and heading level.</returns>"""
    lines = markdown.splitlines()
    sections: list[dict[str, int | str]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_level = 0
    current_path = fallback_name
    saw_heading = False
    in_fence = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        if not in_fence and stripped.startswith("#"):
            heading_level = len(stripped) - len(stripped.lstrip("#"))
            if stripped[heading_level :].strip():
                saw_heading = True
                if current_lines:
                    sections.append(
                        {
                            "section_path": current_path,
                            "heading_level": current_level,
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                while stack and stack[-1][0] >= heading_level:
                    stack.pop()
                title = stripped[heading_level :].strip()
                stack.append((heading_level, title))
                current_level = heading_level
                current_path = " > ".join([item[1] for item in stack])
                current_lines = [line]
                continue
        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section_path": current_path,
                "heading_level": current_level,
                "text": "\n".join(current_lines).strip(),
            }
        )

    if saw_heading:
        return sections

    return split_paragraph_sections(markdown, fallback_name)


def split_paragraph_sections(
    markdown: str, fallback_name: str
) -> list[dict[str, int | str]]:
    """<summary>Split markdown into paragraph sections when no headings exist.</summary>
    <param name="markdown">Markdown content.</param>
    <param name="fallback_name">Fallback section name.</param>
    <returns>List of paragraph sections.</returns>"""
    sections: list[dict[str, int | str]] = []
    buffer: list[str] = []
    for line in markdown.splitlines():
        if not line.strip():
            if buffer:
                sections.append(
                    {
                        "section_path": fallback_name,
                        "heading_level": 0,
                        "text": "\n".join(buffer).strip(),
                    }
                )
                buffer = []
            continue
        buffer.append(line)

    if buffer:
        sections.append(
            {
                "section_path": fallback_name,
                "heading_level": 0,
                "text": "\n".join(buffer).strip(),
            }
        )

    return sections


def chunk_markdown(
    markdown: str, chunk_size: int, chunk_overlap: int, fallback_name: str
) -> list[dict[str, int | str]]:
    """<summary>Chunk markdown by headings, then by tokenizer tokens with overlap.</summary>
    <param name="markdown">Markdown text to chunk.</param>
    <param name="chunk_size">Target chunk size in tokenizer tokens.</param>
    <param name="chunk_overlap">Tokenizer-token overlap between chunks.</param>
    <param name="fallback_name">Fallback section name when no headings exist.</param>
    <returns>List of chunk records.</returns>"""
    sections = split_markdown_sections(markdown, fallback_name)
    if not sections:
        return []

    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[dict[str, int | str]] = []
    chunk_index = 0
    for section in sections:
        section_text = str(section["text"])
        offsets = tokenize_with_offsets(section_text)
        total_tokens = len(offsets)
        if total_tokens == 0:
            continue

        for start_token in range(0, total_tokens, step):
            end_token = min(start_token + chunk_size, total_tokens)
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]
            chunk_text = section_text[char_start:char_end].strip()
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "token_start": start_token,
                    "token_end": end_token,
                    "token_count": end_token - start_token,
                    "chunk_text": chunk_text,
                    "section_path": section["section_path"],
                    "heading_level": section["heading_level"],
                }
            )
            chunk_index += 1
            if end_token >= total_tokens:
                break
    return chunks


def post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    """<summary>POST JSON payload and return parsed response.</summary>
    <param name="url">Endpoint URL.</param>
    <param name="payload">JSON payload.</param>
    <returns>Parsed JSON response.</returns>"""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def extract_markdown(markdown_field: object) -> str:
    """<summary>Extract raw markdown string from Crawl4AI result.</summary>
    <param name="markdown_field">Markdown field from Crawl4AI response.</param>
    <returns>Raw markdown string.</returns>"""
    if isinstance(markdown_field, dict):
        return str(markdown_field.get("raw_markdown") or "")
    if isinstance(markdown_field, str):
        return markdown_field
    return ""


def crawl_site(
    start_url: str,
    output_dir: Path,
    max_depth: int,
    max_pages: int,
    max_urls_per_request: int,
    api_base_url: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """<summary>Crawl a site recursively to a max depth and page count.</summary>
    <param name="start_url">Seed URL for the crawl.</param>
    <param name="output_dir">Directory for markdown and JSONL output.</param>
    <param name="max_depth">Maximum crawl depth (inclusive).</param>
    <param name="max_pages">Maximum number of pages to fetch.</param>
    <param name="max_concurrent">Maximum concurrent requests.</param>
    <returns>None.</returns>"""
    markdown_dir = output_dir / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    pages_jsonl_path = output_dir / "pages.jsonl"
    chunks_jsonl_path = output_dir / "chunks.jsonl"
    summary_path = output_dir / "summary.json"

    visited: set[str] = set()
    next_level: list[str] = []

    start_normalized = normalize_url(start_url)
    if not start_normalized:
        raise ValueError(f"Invalid start URL: {start_url}")

    current_level = [start_normalized]
    total_fetched = 0
    started_at = int(time.time())

    for depth in range(max_depth + 1):
        if not current_level or total_fetched >= max_pages:
            break

        batch = [url for url in current_level if url not in visited]
        if not batch:
            current_level = []
            continue

        remaining = max_pages - total_fetched
        batch = batch[:remaining]

        level_links: list[str] = []
        for offset in range(0, len(batch), max_urls_per_request):
            batch_slice = batch[offset : offset + max_urls_per_request]
            payload = {
                "urls": batch_slice,
                "browser_config": {"headless": True, "user_agent": "crawl4r-bot/0.1"},
                "crawler_config": {"page_timeout": 30000, "remove_overlay_elements": True},
            }
            response = post_json(f"{api_base_url}/crawl", payload)
            results = response.get("results", [])

            with pages_jsonl_path.open("a", encoding="utf-8") as pages_file, chunks_jsonl_path.open(
                "a", encoding="utf-8"
            ) as chunks_file:
                for result in results:
                    url = normalize_url(result.get("url", "")) or result.get("url", "")
                    visited.add(url)
                    total_fetched += 1

                    success = bool(result.get("success"))
                    markdown_path = markdown_dir / f"{total_fetched:05d}.md"
                    markdown = extract_markdown(result.get("markdown"))
                    if success and markdown:
                        markdown_path.write_text(markdown, encoding="utf-8")

                    record = {
                        "url": url,
                        "depth": depth,
                        "success": success,
                        "title": (result.get("metadata") or {}).get("title"),
                        "description": (result.get("metadata") or {}).get("description"),
                        "markdown_path": str(markdown_path) if success else None,
                        "links": {
                            "internal": (result.get("links") or {}).get("internal", []),
                            "external": (result.get("links") or {}).get("external", []),
                        },
                        "status_code": result.get("status_code"),
                        "timestamp": int(time.time()),
                    }
                    pages_file.write(json.dumps(record) + "\n")

                    if success and markdown:
                        fallback_name = get_filename_from_url(url)
                        chunks = chunk_markdown(
                            markdown, chunk_size, chunk_overlap, fallback_name
                        )
                        for chunk in chunks:
                            chunk_record = {
                                "url": url,
                                "depth": depth,
                                "chunk_index": chunk["chunk_index"],
                                "token_start": chunk["token_start"],
                                "token_end": chunk["token_end"],
                                "token_count": chunk["token_count"],
                                "chunk_text": chunk["chunk_text"],
                                "section_path": chunk["section_path"],
                                "heading_level": chunk["heading_level"],
                                "markdown_path": str(markdown_path),
                                "timestamp": int(time.time()),
                            }
                            chunks_file.write(json.dumps(chunk_record) + "\n")

                    if not success or depth >= max_depth:
                        continue

                    combined_links = (result.get("links") or {}).get(
                        "internal", []
                    ) + (result.get("links") or {}).get("external", [])
                    for link in iter_candidate_links(url, combined_links):
                        if link not in visited:
                            level_links.append(link)

        next_level = list(dict.fromkeys(level_links))
        current_level = next_level

    summary = {
        "start_url": start_normalized,
        "max_depth": max_depth,
        "max_pages": max_pages,
        "pages_fetched": total_fetched,
        "unique_urls": len(visited),
        "started_at": started_at,
        "finished_at": int(time.time()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """<summary>Parse CLI arguments.</summary>
    <returns>Parsed arguments namespace.</returns>"""
    parser = argparse.ArgumentParser(description="Recursive crawl for code.claude.com")
    parser.add_argument("start_url", help="Seed URL for the crawl")
    parser.add_argument("--output-dir", default="data/crawl-code-claude")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--max-urls-per-request", type=int, default=100)
    parser.add_argument("--api-base-url", default="http://localhost:52004")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=77)
    return parser.parse_args()


def main() -> None:
    """<summary>Entry point for the crawler script.</summary>
    <returns>None.</returns>"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    try:
        get_tokenizer()
    except RuntimeError as exc:
        raise SystemExit(
            "Missing dependency: transformers.\n"
            "Install: pip install transformers\n"
            "Then re-run this script."
        ) from exc
    crawl_site(
        start_url=args.start_url,
        output_dir=output_dir,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        max_urls_per_request=args.max_urls_per_request,
        api_base_url=args.api_base_url.rstrip("/"),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
