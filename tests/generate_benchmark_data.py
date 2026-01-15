#!/usr/bin/env python3
"""Generate benchmark dataset with 100 markdown files of varying sizes (500-3000 tokens)."""

import random
from pathlib import Path

# Sample content blocks to construct documents
HEADINGS = [
    "Introduction", "Overview", "Background", "Context", "Summary",
    "Details", "Analysis", "Implementation", "Results", "Conclusion",
    "Methods", "Approach", "Strategy", "Framework", "Architecture",
    "Design", "Development", "Testing", "Deployment", "Maintenance"
]

PARAGRAPHS = [
    "This section explores the fundamental concepts and principles underlying the system. "
    "We examine various approaches and methodologies that have been developed over time. "
    "The analysis considers multiple perspectives and evaluates different strategies. "
    "Key factors include performance, scalability, maintainability, and reliability. ",

    "The implementation follows industry best practices and design patterns. "
    "We leverage modern technologies and frameworks to build robust solutions. "
    "Special attention is paid to code quality, testing, and documentation. "
    "The architecture supports extensibility and future enhancements. ",

    "Performance optimization is critical for delivering excellent user experience. "
    "We apply caching strategies, efficient algorithms, and resource management. "
    "Benchmarking and profiling help identify bottlenecks and areas for improvement. "
    "Continuous monitoring ensures the system meets performance targets. ",

    "Security considerations are integrated throughout the development lifecycle. "
    "We implement authentication, authorization, encryption, and input validation. "
    "Regular security audits and penetration testing identify vulnerabilities. "
    "Compliance with industry standards and regulations is maintained. ",

    "The testing strategy encompasses unit, integration, and end-to-end tests. "
    "Automated testing pipelines ensure code quality and prevent regressions. "
    "Test coverage metrics guide testing efforts and identify gaps. "
    "Continuous integration and deployment streamline the release process. ",
]

def generate_content(target_tokens: int) -> str:
    """Generate markdown content with approximately target_tokens tokens."""
    content = f"# Document {random.randint(1000, 9999)}\n\n"
    current_tokens = 10  # Approximate for title

    while current_tokens < target_tokens:
        # Add heading (level 2 or 3)
        level = random.choice(["##", "###"])
        heading = random.choice(HEADINGS)
        content += f"\n{level} {heading}\n\n"
        current_tokens += 5

        # Add 1-3 paragraphs
        num_paragraphs = random.randint(1, 3)
        for _ in range(num_paragraphs):
            paragraph = random.choice(PARAGRAPHS)
            # Repeat sentences to reach target length
            repeats = max(1, (target_tokens - current_tokens) // 50)
            repeats = min(repeats, 5)  # Cap repetitions
            content += paragraph * repeats + "\n\n"
            current_tokens += len(paragraph.split()) * repeats

            if current_tokens >= target_tokens:
                break

    return content

def main():
    """Generate 100 markdown files with varying sizes."""
    output_dir = Path("tests/benchmark_data")
    output_dir.mkdir(exist_ok=True)

    # Generate files with sizes evenly distributed between 500-3000 tokens
    token_sizes = []
    for i in range(100):
        # Linear distribution from 500 to 3000
        target_tokens = 500 + (2500 * i // 100)
        token_sizes.append(target_tokens)

    # Shuffle to randomize order
    random.shuffle(token_sizes)

    print(f"Generating 100 markdown files in {output_dir}/")
    for i, target_tokens in enumerate(token_sizes, 1):
        content = generate_content(target_tokens)
        filepath = output_dir / f"doc_{i:03d}.md"
        filepath.write_text(content)
        actual_tokens = len(content.split())
        print(f"  {filepath.name}: ~{actual_tokens} tokens (target: {target_tokens})")

    print(f"\n✓ Generated {len(token_sizes)} files")
    print(f"✓ Token range: {min(token_sizes)}-{max(token_sizes)}")
    print(f"✓ Average: {sum(token_sizes) // len(token_sizes)} tokens")

if __name__ == "__main__":
    main()
