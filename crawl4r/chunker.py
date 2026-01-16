"""Markdown document chunking with heading-based splitting.

This module provides the MarkdownChunker class for splitting markdown documents
into semantic chunks based on heading structure while preserving formatting.

Examples:
    Basic usage with defaults (512 tokens, 15% overlap):

        chunker = MarkdownChunker()
        chunks = chunker.chunk(markdown_text, filename="doc.md")

    Custom configuration:

        chunker = MarkdownChunker(chunk_size_tokens=1024, chunk_overlap_percent=20)
        chunks = chunker.chunk(markdown_text, filename="doc.md")
"""

import re
from typing import Any, TypedDict

import yaml


class SectionDict(TypedDict):
    """Dictionary structure for markdown sections during parsing.

    Attributes:
        text: The text content of the section
        section_path: Heading hierarchy with '>' separator, or filename if no headings
        heading_level: Heading level 1-6 for #-######, 0 for no heading
    """

    text: str
    section_path: str
    heading_level: int


class ChunkDict(TypedDict):
    """Dictionary structure for chunk metadata.

    Attributes:
        chunk_text: The text content of the chunk
        chunk_index: Sequential 0-based index of the chunk
        section_path: Heading hierarchy with '>' separator, or filename if no headings
        heading_level: Heading level 1-6 for #-######, 0 for no heading
        tags: List of tags from frontmatter, empty list if no frontmatter
    """

    chunk_text: str
    chunk_index: int
    section_path: str
    heading_level: int
    tags: list[str] | None


class MarkdownChunker:
    """Chunks markdown documents by headings while preserving structure.

    Splits markdown text into chunks targeting a specific token count with
    configurable overlap. Preserves heading hierarchy, formatting (code blocks,
    lists, inline styles), and generates metadata for each chunk.

    Attributes:
        chunk_size_tokens: Target chunk size in tokens (default 512)
        chunk_overlap_percent: Overlap percentage 0-50 (default 15)

    Examples:
        Basic chunking with defaults:

            chunker = MarkdownChunker()
            chunks = chunker.chunk("# Title\\n\\nContent", filename="doc.md")

        Custom configuration:

            chunker = MarkdownChunker(
                chunk_size_tokens=1024,
                chunk_overlap_percent=20
            )
            chunks = chunker.chunk(markdown_text, filename="doc.md")

    Notes:
        - Token estimation uses 1 token ≈ 4 characters
        - Chunks preserve markdown formatting (code blocks, lists, inline styles)
        - Section paths use '>' separator (e.g., "Guide > Installation > Prerequisites")
        - Files without headings use filename as section_path, heading_level=0
    """

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_percent: int = 15,
    ) -> None:
        """Initialize the markdown chunker.

        Args:
            chunk_size_tokens: Target chunk size in tokens (must be positive)
            chunk_overlap_percent: Overlap percentage between chunks (0-50)

        Raises:
            ValueError: If chunk_size_tokens <= 0
            ValueError: If chunk_overlap_percent not in range 0-50

        Examples:
            Default configuration:

                chunker = MarkdownChunker()

            Custom configuration:

                chunker = MarkdownChunker(
                    chunk_size_tokens=1024,
                    chunk_overlap_percent=20
                )
        """
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")

        if not 0 <= chunk_overlap_percent <= 50:
            raise ValueError("chunk_overlap_percent must be between 0 and 50")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_percent = chunk_overlap_percent

    def parse_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown text.

        Extracts YAML frontmatter enclosed in --- delimiters at the start of
        the document. Returns the parsed frontmatter as a dictionary and the
        remaining content without the frontmatter section.

        Args:
            text: Markdown text potentially containing frontmatter

        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)
            - frontmatter_dict: Parsed YAML as dict, empty dict if
                no/invalid frontmatter
            - content_without_frontmatter: Markdown content after
                frontmatter removed

        Examples:
            With valid frontmatter:

                text = '''---
                title: Example
                tags:
                  - python
                  - testing
                ---

                # Content
                '''
                fm, content = chunker.parse_frontmatter(text)
                # fm == {"title": "Example", "tags": ["python", "testing"]}
                # content == "\\n# Content\\n"

            Without frontmatter:

                text = "# Just Content"
                fm, content = chunker.parse_frontmatter(text)
                # fm == {}
                # content == "# Just Content"

        Notes:
            - Frontmatter must be at document start (no leading whitespace/content)
            - Invalid YAML returns empty dict and original content
            - Empty frontmatter (---\\n---) returns empty dict
        """
        # Check if text starts with frontmatter delimiter
        if not text.strip().startswith("---"):
            return {}, text

        # Find the closing delimiter
        # Split by lines to find second occurrence of ---
        lines = text.split("\n")
        if len(lines) < 3:  # Need at least: ---, content, ---
            return {}, text

        # Find closing --- delimiter (must be on its own line)
        closing_index = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                closing_index = i
                break

        # If no closing delimiter found, no valid frontmatter
        if closing_index == -1:
            return {}, text

        # Extract YAML content between delimiters
        yaml_content = "\n".join(lines[1:closing_index])

        # Parse YAML with custom loader that preserves dates as strings
        try:
            # Create a SafeLoader that doesn't auto-convert dates
            class NoDatesSafeLoader(yaml.SafeLoader):
                pass

            # Remove the timestamp constructor to keep dates as strings
            NoDatesSafeLoader.yaml_implicit_resolvers = {
                k: [
                    r
                    for r in v
                    if r[0]
                    not in (
                        "tag:yaml.org,2002:timestamp",
                        "tag:yaml.org,2002:python/object/apply:datetime.date",
                    )
                ]
                for k, v in NoDatesSafeLoader.yaml_implicit_resolvers.items()
            }

            frontmatter = yaml.load(yaml_content, Loader=NoDatesSafeLoader)
            # Handle empty frontmatter
            if frontmatter is None:
                frontmatter = {}
        except yaml.YAMLError:
            # Invalid YAML, return empty dict and original content
            return {}, text

        # Extract content after frontmatter
        # Join remaining lines after closing delimiter
        content = "\n".join(lines[closing_index + 1 :])

        return frontmatter, content

    def chunk(self, text: str, filename: str) -> list[ChunkDict]:
        """Chunk markdown text into semantic sections.

        Splits markdown by headings, creating chunks that target the configured
        token size with overlap. Preserves heading hierarchy and formatting.

        Args:
            text: Markdown text to chunk
            filename: Name of the source file (used for section_path when no headings)

        Returns:
            List of chunk dictionaries with metadata

        Examples:
            Chunk document with headings:

                chunks = chunker.chunk("# Title\\n\\nContent", filename="doc.md")
                # chunks[0]["section_path"] == "Title"
                # chunks[0]["heading_level"] == 1

            Document without headings:

                chunks = chunker.chunk("Plain text", filename="doc.md")
                # chunks[0]["section_path"] == "doc.md"
                # chunks[0]["heading_level"] == 0

        Notes:
            - Empty/whitespace-only text may return empty list or single chunk
            - Very short documents may produce single chunk regardless of settings
            - Formatting (code blocks, lists, inline styles) is preserved
        """
        # Handle empty/whitespace-only text
        if not text or not text.strip():
            return []

        # Parse frontmatter and extract tags
        frontmatter, content_without_frontmatter = self.parse_frontmatter(text)
        tags = frontmatter.get("tags", [])
        # Ensure tags is a list (or None if not present/invalid)
        if not isinstance(tags, list):
            tags = []

        # Split by headings to preserve structure (use content without frontmatter)
        sections = self._split_by_headings(content_without_frontmatter, filename)

        # Create chunks from sections
        chunks = []
        chunk_index = 0

        for section in sections:
            # Split large sections into smaller chunks
            section_chunks = self._split_section(section)

            for chunk_text in section_chunks:
                chunks.append(
                    ChunkDict(
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                        section_path=section["section_path"],
                        heading_level=section["heading_level"],
                        tags=tags if tags else None,
                    )
                )
                chunk_index += 1

        return chunks

    def _split_by_headings(self, text: str, filename: str) -> list[SectionDict]:
        """Split markdown text into sections by headings.

        Parses markdown headings (#-######) to create sections with hierarchy.
        Each section includes the text content, a section path showing the
        heading hierarchy (e.g., "Guide > Installation"), and the heading level.

        Args:
            text: Markdown text to split
            filename: Fallback name if no headings found

        Returns:
            List of section dictionaries, each containing:
                - text: Section content including heading line
                - section_path: Heading hierarchy with ' > ' separator
                - heading_level: 1-6 for #-######, 0 for files without headings

        Examples:
            Document with headings:

                sections = chunker._split_by_headings(
                    "# Title\\n\\n## Subtitle\\n\\nContent",
                    "doc.md"
                )
                # sections[0]["section_path"] == "Title"
                # sections[1]["section_path"] == "Title > Subtitle"

            Document without headings:

                sections = chunker._split_by_headings("Plain text", "doc.md")
                # sections[0]["section_path"] == "doc.md"
                # sections[0]["heading_level"] == 0
        """
        # Pattern to match markdown headings (# through ######)
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        # Find all headings with their positions
        headings = []
        for match in heading_pattern.finditer(text):
            level = len(match.group(1))  # Count # symbols
            title = match.group(2).strip()
            start_pos = match.start()
            headings.append({"level": level, "title": title, "start_pos": start_pos})

        # If no headings, return entire text as single section
        if not headings:
            return [
                SectionDict(
                    text=text,
                    section_path=filename,
                    heading_level=0,
                )
            ]

        # Build sections from headings
        sections = []
        heading_stack: list[str] = []  # Track heading hierarchy

        for i, heading in enumerate(headings):
            # Update heading stack for hierarchy
            # Remove headings at same or deeper level
            while heading_stack and len(heading_stack) >= heading["level"]:
                heading_stack.pop()

            # Add current heading to stack
            heading_stack.append(heading["title"])

            # Build section path from heading stack
            section_path = " > ".join(heading_stack)

            # Extract text for this section (until next heading or end)
            start_pos = heading["start_pos"]
            if i + 1 < len(headings):
                end_pos = headings[i + 1]["start_pos"]
            else:
                end_pos = len(text)

            section_text = text[start_pos:end_pos].strip()

            sections.append(
                SectionDict(
                    text=section_text,
                    section_path=section_path,
                    heading_level=heading["level"],
                )
            )

        return sections

    def _split_section(self, section: SectionDict) -> list[str]:
        """Split a section into chunks based on token size target.

        Splits long sections into smaller chunks targeting the configured
        token size with overlap. Uses paragraph boundaries (double newlines)
        for clean splits when possible.

        Args:
            section: Section dictionary with text, section_path, heading_level

        Returns:
            List of chunk text strings, each targeting chunk_size_tokens

        Examples:
            Short section (fits in one chunk):

                chunks = chunker._split_section(
                    SectionDict(
                        text="# Title\\n\\nShort content",
                        section_path="Title",
                        heading_level=1
                    )
                )
                # len(chunks) == 1

            Long section (split into multiple chunks):

                chunks = chunker._split_section(
                    SectionDict(
                        text="# Title\\n\\n" + "paragraph\\n\\n" * 100,
                        section_path="Title",
                        heading_level=1
                    )
                )
                # len(chunks) > 1, each ~512 tokens with 15% overlap

        Notes:
            - Token estimation uses 1 token ≈ 4 characters
            - Attempts to break at paragraph boundaries in last 20% of chunk
            - Overlap calculated as chunk_overlap_percent of chunk_size_tokens
        """
        text = section["text"]

        # Token estimation: 1 token ≈ 4 characters
        chars_per_token = 4
        target_chars = self.chunk_size_tokens * chars_per_token
        overlap_chars = int(target_chars * (self.chunk_overlap_percent / 100))

        # If section fits in one chunk, return as-is
        if len(text) <= target_chars:
            return [text]

        # Split into chunks with overlap
        chunks = []
        start = 0

        while start < len(text):
            # Extract chunk
            end = start + target_chars
            chunk_text = text[start:end]

            # Try to break at paragraph boundary (double newline)
            if end < len(text):
                # Look for paragraph break in last 20% of chunk
                search_start = max(0, len(chunk_text) - target_chars // 5)
                paragraph_break = chunk_text.rfind("\n\n", search_start)

                if paragraph_break > 0:
                    chunk_text = chunk_text[:paragraph_break].strip()
                    end = start + paragraph_break

            chunks.append(chunk_text)

            # Move to next chunk with overlap
            start = end - overlap_chars
            if start <= 0 or start >= len(text):
                break

        return chunks
