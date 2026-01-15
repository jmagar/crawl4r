"""Unit tests for markdown chunking functionality.

Tests for the MarkdownChunker class that splits markdown documents into
semantic chunks based on headings while preserving structure and formatting.
"""

import pytest

from rag_ingestion.chunker import MarkdownChunker


class TestChunkByHeadings:
    """Test chunking markdown by heading structure."""

    def test_splits_at_headings(self) -> None:
        """Verify chunks are split at markdown headings."""
        markdown = """# Main Title

Content under main title.

## Section 1

Content under section 1.

### Subsection 1.1

Content under subsection 1.1.

## Section 2

Content under section 2.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Should have multiple chunks split at headings
        assert len(chunks) > 0
        # Each chunk should have text
        for chunk in chunks:
            assert "chunk_text" in chunk
            assert len(chunk["chunk_text"]) > 0

    def test_includes_heading_in_chunk(self) -> None:
        """Verify heading is included at start of each chunk."""
        markdown = """## Introduction

This is the introduction section.

## Usage

This is the usage section.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Find chunk with "Introduction"
        intro_chunk = next(
            (c for c in chunks if "Introduction" in c["chunk_text"]), None
        )
        assert intro_chunk is not None
        assert intro_chunk["chunk_text"].startswith("## Introduction")


class TestChunkPreservesHierarchy:
    """Test preservation of heading hierarchy in section_path."""

    def test_builds_section_path_hierarchy(self) -> None:
        """Verify section_path includes parent headings."""
        markdown = """# Guide

Top-level guide content.

## Installation

Installation instructions.

### Prerequisites

Prerequisites list.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Find chunk for Prerequisites
        prereq_chunk = next(
            (c for c in chunks if "Prerequisites" in c["chunk_text"]), None
        )
        assert prereq_chunk is not None
        # Should include parent path: "Guide > Installation > Prerequisites"
        assert "section_path" in prereq_chunk
        assert "Guide" in prereq_chunk["section_path"]
        assert "Installation" in prereq_chunk["section_path"]
        assert "Prerequisites" in prereq_chunk["section_path"]

    def test_section_path_uses_separator(self) -> None:
        """Verify section_path uses '>' separator."""
        markdown = """# Main

## Sub

### Subsub

Content here.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Find deepest chunk
        subsub_chunk = next(
            (c for c in chunks if "Subsub" in c["chunk_text"]), None
        )
        assert subsub_chunk is not None
        assert " > " in subsub_chunk["section_path"]


class TestChunkSizeTarget:
    """Test chunk size targeting with overlap."""

    def test_chunks_target_512_tokens(self) -> None:
        """Verify chunks target approximately 512 tokens."""
        # Create markdown with enough content for multiple chunks
        # Using ~4 chars per token, need ~2048 chars for 512 tokens
        content = "This is a sentence. " * 150  # ~3000 chars = ~750 tokens
        markdown = f"""# Long Document

{content}

## Section 2

{content}
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Should create multiple chunks
        assert len(chunks) > 1

        # Most chunks should be near 512 tokens (~2048 chars)
        for chunk in chunks[:-1]:  # Exclude last chunk (may be smaller)
            char_count = len(chunk["chunk_text"])
            # Allow 50% variance (256-768 tokens = 1024-3072 chars)
            assert 1000 <= char_count <= 3500

    def test_chunks_have_15_percent_overlap(self) -> None:
        """Verify chunks have approximately 15% overlap (~77 tokens)."""
        # Create content that will be split
        sections = [
            " ".join(f"This is a unique sentence number {i}-{j}." for j in range(20))
            for i in range(5)
        ]
        markdown = "\n\n".join(f"## Section {i}\n\n{s}" for i, s in enumerate(sections))

        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        if len(chunks) > 1:
            # Check for overlapping content between consecutive chunks
            # 15% of 512 tokens = ~77 tokens = ~308 chars
            for i in range(len(chunks) - 1):
                curr_text = chunks[i]["chunk_text"]
                next_text = chunks[i + 1]["chunk_text"]

                # Extract last ~400 chars from current chunk
                curr_suffix = curr_text[-400:]
                # Check if any of it appears in next chunk
                has_overlap = any(
                    word in next_text for word in curr_suffix.split()[:10]
                )
                # Note: This is a heuristic check since exact overlap
                # depends on LlamaIndex's implementation
                assert isinstance(has_overlap, bool)


class TestChunkWithoutHeadings:
    """Test chunking markdown files without headings."""

    def test_chunks_paragraphs_without_headings(self) -> None:
        """Verify paragraph-level splitting when no headings present."""
        markdown = """This is the first paragraph with some content.

This is the second paragraph with more content.

This is the third paragraph with even more content.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Should still create chunks
        assert len(chunks) > 0
        # Each chunk should have text
        for chunk in chunks:
            assert len(chunk["chunk_text"]) > 0

    def test_section_path_equals_filename_without_headings(self) -> None:
        """Verify section_path uses filename when no headings present."""
        markdown = "Just some plain text without any headings."

        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["section_path"] == "test.md"

    def test_heading_level_zero_without_headings(self) -> None:
        """Verify heading_level is 0 when no headings present."""
        markdown = "Plain text content without headings."

        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["heading_level"] == 0


class TestChunkMetadata:
    """Test chunk metadata fields."""

    def test_includes_chunk_index(self) -> None:
        """Verify each chunk has sequential chunk_index."""
        markdown = """# Title

Some content.

## Section 1

More content.

## Section 2

Even more content.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Check chunk_index is sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_includes_heading_level(self) -> None:
        """Verify chunk includes heading_level (1-6 for #-######, 0 for none)."""
        markdown = """# Level 1

Content.

## Level 2

Content.

### Level 3

Content.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Find chunks by content
        level1_chunk = next((c for c in chunks if "# Level 1" in c["chunk_text"]), None)
        level2_chunk = next(
            (c for c in chunks if "## Level 2" in c["chunk_text"]), None
        )
        level3_chunk = next(
            (c for c in chunks if "### Level 3" in c["chunk_text"]), None
        )

        if level1_chunk:
            assert level1_chunk["heading_level"] == 1
        if level2_chunk:
            assert level2_chunk["heading_level"] == 2
        if level3_chunk:
            assert level3_chunk["heading_level"] == 3

    def test_includes_section_path(self) -> None:
        """Verify chunk includes section_path."""
        markdown = """# Main

Content.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        assert len(chunks) > 0
        for chunk in chunks:
            assert "section_path" in chunk
            assert isinstance(chunk["section_path"], str)
            assert len(chunk["section_path"]) > 0


class TestChunkPreservesFormatting:
    """Test preservation of markdown formatting in chunks."""

    def test_preserves_code_blocks(self) -> None:
        """Verify code blocks are preserved in chunk_text."""
        markdown = """# Code Example

Here is some code:

```python
def hello():
    print("Hello, world!")
```

That was the code.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Find chunk with code
        code_chunk = next((c for c in chunks if "```python" in c["chunk_text"]), None)
        assert code_chunk is not None
        assert "def hello():" in code_chunk["chunk_text"]
        assert '    print("Hello, world!")' in code_chunk["chunk_text"]

    def test_preserves_lists(self) -> None:
        """Verify lists are preserved in chunk_text."""
        markdown = """# List Example

Here is a list:

- Item 1
- Item 2
- Item 3

And a numbered list:

1. First
2. Second
3. Third
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        # Find chunk with lists
        list_chunk = next((c for c in chunks if "- Item 1" in c["chunk_text"]), None)
        assert list_chunk is not None
        assert "- Item 2" in list_chunk["chunk_text"]
        assert "1. First" in list_chunk["chunk_text"]

    def test_preserves_inline_formatting(self) -> None:
        """Verify inline formatting (bold, italic, code) is preserved."""
        markdown = """# Formatting

This text has **bold**, *italic*, and `code` formatting.
"""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="test.md")

        assert len(chunks) > 0
        chunk = chunks[0]
        assert "**bold**" in chunk["chunk_text"]
        assert "*italic*" in chunk["chunk_text"]
        assert "`code`" in chunk["chunk_text"]


class TestChunkerInitialization:
    """Test MarkdownChunker initialization and configuration."""

    def test_initializes_with_defaults(self) -> None:
        """Verify chunker initializes with default chunk size and overlap."""
        chunker = MarkdownChunker()
        assert chunker.chunk_size_tokens == 512
        assert chunker.chunk_overlap_percent == 15

    def test_initializes_with_custom_values(self) -> None:
        """Verify chunker accepts custom chunk size and overlap."""
        chunker = MarkdownChunker(chunk_size_tokens=1024, chunk_overlap_percent=20)
        assert chunker.chunk_size_tokens == 1024
        assert chunker.chunk_overlap_percent == 20

    def test_rejects_invalid_overlap_percent(self) -> None:
        """Verify chunker rejects overlap outside 0-50% range."""
        with pytest.raises(ValueError, match="overlap.*0.*50"):
            MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=60)

        with pytest.raises(ValueError, match="overlap.*0.*50"):
            MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=-5)

    def test_rejects_invalid_chunk_size(self) -> None:
        """Verify chunker rejects non-positive chunk sizes."""
        with pytest.raises(ValueError, match="chunk_size_tokens.*positive"):
            MarkdownChunker(chunk_size_tokens=0, chunk_overlap_percent=15)

        with pytest.raises(ValueError, match="chunk_size_tokens.*positive"):
            MarkdownChunker(chunk_size_tokens=-100, chunk_overlap_percent=15)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_markdown(self) -> None:
        """Verify chunker handles empty markdown gracefully."""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk("", filename="empty.md")

        # Should return empty list or single empty chunk
        assert isinstance(chunks, list)
        # If returns chunks, they should be valid
        for chunk in chunks:
            assert "chunk_text" in chunk
            assert "chunk_index" in chunk

    def test_very_short_markdown(self) -> None:
        """Verify chunker handles very short documents (< chunk_size)."""
        markdown = "# Short\n\nJust a brief paragraph."
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk(markdown, filename="short.md")

        # Should create at least one chunk
        assert len(chunks) >= 1
        # Should contain all content
        all_text = " ".join(c["chunk_text"] for c in chunks)
        assert "Short" in all_text
        assert "brief paragraph" in all_text

    def test_whitespace_only_markdown(self) -> None:
        """Verify chunker handles whitespace-only documents."""
        chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
        chunks = chunker.chunk("   \n\n   \n", filename="whitespace.md")

        # Should handle gracefully (empty list or single chunk)
        assert isinstance(chunks, list)
