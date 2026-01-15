"""Load testing fixtures for generating synthetic markdown documents.

Provides MarkdownGenerator class for creating realistic test data with
configurable size distributions and content patterns.

Used by performance tests to generate 100-1000 files for throughput
and memory testing.

Example:
    >>> from tests.performance.load_test_fixtures import MarkdownGenerator
    >>>
    >>> generator = MarkdownGenerator()
    >>> files = generator.generate_batch(
    ...     output_dir=Path("/tmp/docs"),
    ...     count=100,
    ...     distribution={"small": 30, "medium": 50, "large": 20}
    ... )
"""

from pathlib import Path
from typing import Literal


class MarkdownGenerator:
    """Generate synthetic markdown documents for load testing.

    Creates realistic markdown files with headings, sections, lists, and
    varied content lengths. Supports size distribution configuration.

    Size categories:
    - small: ~500 words (1-2 KB)
    - medium: ~2000 words (5-10 KB)
    - large: ~5000 words (15-20 KB)

    Example:
        >>> generator = MarkdownGenerator()
        >>>
        >>> # Generate 100 files with default distribution
        >>> files = generator.generate_batch(
        ...     output_dir=Path("/tmp/test"),
        ...     count=100
        ... )
        >>>
        >>> # Custom distribution: 40% small, 40% medium, 20% large
        >>> files = generator.generate_batch(
        ...     output_dir=Path("/tmp/test"),
        ...     count=100,
        ...     distribution={"small": 40, "medium": 40, "large": 20}
        ... )
    """

    # Size ranges (word counts)
    SIZES = {
        "small": (400, 600),  # ~500 words
        "medium": (1800, 2200),  # ~2000 words
        "large": (4500, 5500),  # ~5000 words
    }

    def generate_batch(
        self,
        output_dir: Path,
        count: int,
        distribution: dict[str, int] | None = None,
    ) -> list[Path]:
        """Generate batch of markdown files with size distribution.

        Args:
            output_dir: Directory to create files in
            count: Total number of files to generate
            distribution: Dict mapping size ("small", "medium", "large")
                to percentage (e.g., {"small": 30, "medium": 50, "large": 20})
                Defaults to 30% small, 50% medium, 20% large

        Returns:
            List of created file paths

        Raises:
            ValueError: If distribution percentages don't sum to 100

        Example:
            >>> generator = MarkdownGenerator()
            >>> files = generator.generate_batch(
            ...     output_dir=Path("/tmp/docs"),
            ...     count=100,
            ...     distribution={"small": 30, "medium": 50, "large": 20}
            ... )
            >>> len(files)
            100
        """
        # Use default distribution if none provided
        if distribution is None:
            distribution = {"small": 30, "medium": 50, "large": 20}

        # Validate distribution
        total_pct = sum(distribution.values())
        if total_pct != 100:
            raise ValueError(
                f"Distribution percentages must sum to 100, got {total_pct}"
            )

        # Calculate counts per size
        counts = {
            size: int(count * pct / 100) for size, pct in distribution.items()
        }

        # Handle rounding errors by adding remainder to medium
        total_allocated = sum(counts.values())
        if total_allocated < count:
            counts["medium"] += count - total_allocated

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate files
        files = []
        file_index = 0

        for size, size_count in counts.items():
            for _ in range(size_count):
                file_path = output_dir / f"doc_{file_index:04d}.md"
                self._generate_file(file_path, size)  # type: ignore[arg-type]
                files.append(file_path)
                file_index += 1

        return files

    def _generate_file(
        self, file_path: Path, size: Literal["small", "medium", "large"]
    ) -> None:
        """Generate single markdown file with specified size.

        Args:
            file_path: Path to create file at
            size: Size category ("small", "medium", "large")

        Example:
            >>> generator = MarkdownGenerator()
            >>> generator._generate_file(Path("/tmp/test.md"), "medium")
        """
        import random

        # Get word count range for size
        min_words, max_words = self.SIZES[size]
        word_count = random.randint(min_words, max_words)

        # Generate content
        content = self._generate_content(word_count, file_path.stem)

        # Write file
        file_path.write_text(content)

    def _generate_content(self, word_count: int, title: str) -> str:
        """Generate realistic markdown content with specified word count.

        Args:
            word_count: Target word count
            title: Document title

        Returns:
            Markdown content string

        Example:
            >>> generator = MarkdownGenerator()
            >>> content = generator._generate_content(500, "Test Document")
            >>> len(content.split())
            ~500
        """
        import random

        # Word pool for generating realistic text
        words = [
            "system",
            "document",
            "process",
            "data",
            "pipeline",
            "vector",
            "embedding",
            "storage",
            "query",
            "search",
            "index",
            "metadata",
            "chunk",
            "processing",
            "analysis",
            "performance",
            "optimization",
            "configuration",
            "implementation",
            "integration",
            "monitoring",
            "validation",
            "testing",
            "deployment",
            "architecture",
            "component",
            "service",
            "endpoint",
            "client",
            "server",
            "database",
            "collection",
            "point",
            "dimension",
            "similarity",
            "distance",
            "metric",
            "algorithm",
            "model",
            "inference",
            "training",
            "evaluation",
            "benchmark",
            "quality",
            "accuracy",
            "precision",
            "recall",
            "throughput",
            "latency",
            "scalability",
            "reliability",
        ]

        # Calculate sections needed (aim for ~200 words per section)
        section_count = max(3, word_count // 200)

        # Build content
        lines = [f"# {title}\n"]

        words_written = 2  # Title words
        remaining_words = word_count - words_written

        for section_num in range(section_count):
            # Add section heading
            lines.append(f"\n## Section {section_num + 1}\n")
            words_written += 3

            # Calculate words for this section
            if section_num == section_count - 1:
                # Last section gets remaining words
                section_words = remaining_words
            else:
                # Other sections get roughly equal distribution
                section_words = remaining_words // (section_count - section_num)

            # Generate section content
            section_content = []
            for _ in range(section_words):
                section_content.append(random.choice(words))

            # Add as paragraph (one sentence per 15 words)
            paragraph = []
            for i, word in enumerate(section_content):
                paragraph.append(word)
                if (i + 1) % 15 == 0:
                    # End sentence
                    paragraph[-1] = paragraph[-1] + "."
                    lines.append(" ".join(paragraph) + "\n")
                    paragraph = []

            # Add any remaining words
            if paragraph:
                paragraph[-1] = paragraph[-1] + "."
                lines.append(" ".join(paragraph) + "\n")

            words_written += section_words
            remaining_words -= section_words

        return "\n".join(lines)
