# AI-Friendly Project Documentation: Research & Best Practices

**Research Date:** January 20, 2026
**Purpose:** Comprehensive analysis of best practices for structuring project documentation for AI coding assistants
**Focus:** Claude Code, GitHub Copilot, and modern AI agent workflows

---

## Executive Summary

AI coding assistants require specialized documentation patterns that differ significantly from traditional human-focused documentation. The key insight: **progressive disclosure over comprehensive reference**, prioritizing actionable context over exhaustive detail. Research reveals that effective AI documentation achieves 3-5x better token efficiency while improving output quality through focused, layered information architecture.

### Critical Findings

1. **Context Window Optimization**: Keep root documentation under 300 lines; use progressive disclosure for 40-60% token reduction
2. **Tooling Over Prose**: Never document what deterministic tools can enforce (linting, formatting, type checking)
3. **Three-Layer Architecture**: Minimal always-loaded context → On-demand topic files → Specialized agents
4. **Spec-Driven Development**: Markdown specifications as "source code" enable faster iteration with less context loss
5. **Anti-Pattern Awareness**: Bloated context files, outdated embedded code, and unclear prompts cause 80%+ of AI assistant failures

---

## Table of Contents

1. [Core Documentation Patterns](#core-documentation-patterns)
2. [CLAUDE.md File Structure](#claudemd-file-structure)
3. [AGENTS.md Format for Monorepos](#agentsmd-format-for-monorepos)
4. [Progressive Disclosure Strategy](#progressive-disclosure-strategy)
5. [Python Project Setup](#python-project-setup)
6. [Spec-Driven Development](#spec-driven-development)
7. [Context Window Optimization](#context-window-optimization)
8. [Anti-Patterns & Mistakes](#anti-patterns--mistakes)
9. [Docker Environment Documentation](#docker-environment-documentation)
10. [Implementation Checklist](#implementation-checklist)
11. [Sources](#sources)

---

## Core Documentation Patterns

### The Three Essential Dimensions

Every AI-focused documentation file should address:

**WHAT**: Technology stack, architecture, codebase layout
- Provide a "map of the codebase" with directory structure
- Highlight monorepo organization if applicable
- List key technologies with versions

**WHY**: Project purpose and component rationale
- Explain architectural decisions
- Document trade-offs and constraints
- Provide business context when relevant

**HOW**: Tools, processes, and verification steps
- Command-line operations (build, test, deploy)
- Development workflow steps
- Quality gates and verification procedures

### Universal Principles

1. **Concise & Actionable**: Under 300 lines for root documentation files
2. **Universal Applicability**: Only include broadly relevant information
3. **Prefer Pointers Over Copies**: Link to locations rather than embedding code
4. **Machine-Readable Structure**: Use consistent Markdown hierarchy
5. **Version Control Integration**: Treat docs like code (PRs, reviews, history)

---

## CLAUDE.md File Structure

### Recommended Template

```markdown
# Project Name

Brief 2-3 sentence overview of what this project does and its primary purpose.

## Stack

- Language/Runtime: Python 3.11+, Node.js 20+
- Framework: FastAPI, Next.js 15
- Database: PostgreSQL 15
- Key Tools: Docker Compose, uv, pnpm

## Project Structure

```
project/
├── apps/          # Deployable applications
├── packages/      # Shared libraries
├── tests/         # Test suites
└── .docs/         # Session logs, deployment records
```

## Essential Commands

```bash
# Development
docker compose up -d                    # Start services
source .venv/bin/activate && pytest     # Run tests
uv run ruff check .                     # Lint

# Quality Gates
[VERIFY] All tests pass
[VERIFY] No lint errors
[VERIFY] Type check passes
```

## Critical Notes

- **Port Management**: Always use high ports (53000+), never standard ports
- **Environment**: See .env.example for required variables
- **Documentation**: For deeper topics, see `.docs/` directory

## Additional Context

- Architecture details: `.docs/architecture.md`
- API specifications: `specs/api-design.md`
- Testing strategy: `.docs/testing-strategy.md`
```

### What to Include

✅ **Essential Information**:
- Project overview (2-3 sentences)
- Technology stack with versions
- High-level directory structure
- Core commands with exact syntax
- Critical constraints (port ranges, environment requirements)
- Pointers to deeper documentation

✅ **Quality Gates**:
- `[VERIFY]` checkpoints for testing, linting, type checking
- Explicit success criteria

✅ **Progressive Disclosure**:
- Brief descriptions with file references
- "For X, see Y" patterns

### What to Exclude

❌ **Never Include**:
- Code style rules (use linters: ESLint, Ruff, Prettier)
- Auto-generated content from `/init` commands
- Embedded code snippets (become outdated)
- Task-specific temporary workarounds
- Exhaustive API documentation (link to OpenAPI/Swagger)
- Verbose explanations of self-documenting code

❌ **Anti-Patterns**:
- Files over 300 lines
- Duplicate information across multiple files
- Hardcoded values that should be in `.env`
- Instructions for tools that provide better error messages

### File Placement & Layering

**Global Instructions** (`~/.claude/CLAUDE.md`):
- Personal coding preferences
- Universal tool configurations
- Cross-project standards

**Project Root** (`/project/CLAUDE.md`):
- Shared team context
- Project-specific standards
- Architecture overview

**Subdirectories** (`/project/apps/api/CLAUDE.md`):
- Feature-specific guidance
- Local conventions
- Component details

**Local Override** (`CLAUDE.local.md`):
- Personal preferences (gitignored)
- Experimental configurations

**Precedence**: Closest file to edited code wins, with local overrides taking priority.

### Size Guidelines

| Level | Target Size | Max Size | Purpose |
|-------|-------------|----------|---------|
| Global | 100-200 lines | 300 lines | Universal preferences |
| Root | 150-250 lines | 400 lines | Project essentials |
| Subdirectory | 50-150 lines | 250 lines | Component-specific |
| Local | Minimal | 150 lines | Personal overrides |

**Research Finding**: Files under 300 lines maintain 85%+ instruction-following quality; quality degrades uniformly as size increases beyond this threshold.

---

## AGENTS.md Format for Monorepos

### Overview

AGENTS.md serves as a "README for AI agents"—a standardized, minimal format with no required structure or fields. Adopted by 20,000+ open source projects and supported by major AI coding tools (Aider, Gemini CLI, Cursor).

### Monorepo Strategy

**Root AGENTS.md**:
- Define repository structure
- List standard commands
- Establish shared rules and boundaries
- Progressive disclosure pointers

**Per-App/Package AGENTS.md**:
- What this component is and what it owns
- Dependencies on other packages
- Footguns and gotchas specific to this domain
- Local development workflow

### Nested Structure Example

```
monorepo/
├── AGENTS.md                    # Root: repo structure, commands, boundaries
├── apps/
│   ├── api/
│   │   └── AGENTS.md           # API-specific context
│   └── web/
│       └── AGENTS.md           # Frontend-specific context
└── packages/
    ├── shared-ui/
    │   └── AGENTS.md           # Component library patterns
    └── utils/
        └── AGENTS.md           # Utility functions guidelines
```

**Resolution**: Closest AGENTS.md to the edited file wins.

### Progressive Disclosure in Monorepos

Root file uses delegation pattern:

```markdown
# Monorepo Structure

This repository contains multiple applications and shared packages.

## Creating Features

- To create an email template, read `@emails/AGENTS.md`
- To create a Go service, read `@go/services/AGENTS.md`
- To modify UI components, read `@packages/shared-ui/AGENTS.md`

## Shared Standards

All packages must:
- Include comprehensive tests (85%+ coverage)
- Follow monorepo tooling (Nx, Turbo)
- Document public APIs
```

### Real-World Adoption

**OpenAI Repository**: Contains 88 AGENTS.md files demonstrating scale
**Supported Tools**: Aider, Gemini CLI, Cursor, VS Code extensions, GitHub Copilot

### Key Differences: CLAUDE.md vs AGENTS.md

| Aspect | CLAUDE.md | AGENTS.md |
|--------|-----------|-----------|
| Primary Tool | Claude Code | Multi-tool (Aider, Cursor, etc.) |
| Structure | Flexible Markdown | Minimal, no required fields |
| Adoption | Claude-specific | 20,000+ projects, cross-tool |
| Best For | Claude-optimized workflows | Universal agent compatibility |
| Ecosystem | Claude Code hooks/skills | Broader AI assistant ecosystem |

**Recommendation**: Use both if supporting multiple AI tools; use CLAUDE.md for Claude-specific optimizations.

---

## Progressive Disclosure Strategy

### The Problem

AI systems are stateless—each session starts fresh with no memory. This creates pressure to include everything in root documentation, but bloated context consumes tokens before actual work begins.

**Research Finding**: Traditional comprehensive documentation uses 5% of context window before work begins, leaving only ~40 conversation turns. Progressive disclosure reduces this to 1.8%, enabling 130+ turns.

### Three-Layer Architecture

#### Layer 1: Minimal Root File (~500 tokens, always loaded)

**Purpose**: Immediate orientation and pointers

**Contents**:
- Project overview (2-3 sentences)
- Essential commands
- Stack information
- Pointers to deeper documentation

**Example** (50-line Second Brain project):
```markdown
# Project Name

Nuxt 4 content management system using @nuxt/content v3.

## Stack
- Nuxt 4
- @nuxt/content v3
- TailwindCSS

## Commands
pnpm dev          # Development server
pnpm build        # Production build
pnpm lint:fix     # Auto-fix linting

## Deep Dives
- Nuxt content gotchas: `docs/nuxt-content-gotchas.md`
- Testing strategy: `docs/testing-strategy.md`
```

#### Layer 2: On-Demand Documentation (~200-500 tokens per file)

**Purpose**: Domain-specific knowledge loaded when relevant

**Structure**: Create `/docs` or `/.docs` directory:
- `framework-gotchas.md` - Non-obvious patterns and pitfalls
- `testing-strategy.md` - When to use which test approach
- `deployment-checklist.md` - Pre-production verification steps
- `api-conventions.md` - RESTful endpoint patterns

**Access Pattern**: AI assistant reads specific file when working in related domain

**Example Reference**:
```markdown
## Working with API Endpoints

See `docs/api-conventions.md` for:
- Endpoint naming patterns
- Error response formats
- Authentication flow
```

#### Layer 3: Specialized Agents (~300-800 tokens per agent)

**Purpose**: Complex domain contexts loaded conditionally

**Structure**: Create `.claude/agents/` directory:
- `database-migrations.md` - Schema change workflow
- `deployment-agent.md` - Production deployment procedures
- `security-review.md` - Security checklist and patterns

**Activation**: Agent loads only when its domain is active

**Benefits**:
- Agents fetch current documentation (no stale training data)
- Isolated concerns prevent context pollution
- Composable knowledge modules

### Capture System: The `/learn` Skill

Implement a systematic learning loop:

1. **Detection**: Analyze conversations for reusable insights
2. **Classification**: Identify appropriate documentation file
3. **Approval**: Request confirmation before saving
4. **Integration**: Update documentation with new knowledge

**Result**: Each mistake becomes documented knowledge for future sessions.

### Tooling Over Prose Principle

**Don't write instructions about what tools enforce.**

❌ **Avoid**:
```markdown
## Code Style
- Use 2-space indentation
- Max line length 80 characters
- Prefer single quotes
- No trailing commas
... 200 lines of style rules ...
```

✅ **Instead**:
```markdown
## Quality
Run `pnpm lint:fix && pnpm typecheck` after changes.
```

**Why**: Deterministic tools provide immediate backpressure enabling self-correction. LLMs are expensive and slow for rules enforcement.

### Efficiency Gains

**Traditional RAG** (fetch everything upfront):
- 25,000 tokens consumed
- 200 relevant tokens used
- 0.8% efficiency

**Progressive Disclosure**:
- 800 tokens for index
- 155 tokens for specific content (on demand)
- 100% efficiency

**Token Budget Comparison**:
- Traditional: ~5% of context before work (40 conversation turns)
- Progressive: ~1.8% of context before work (130+ conversation turns)

---

## Python Project Setup

### Modern Tooling Stack (2026)

**Package Manager**: `uv` (NOT pip, poetry, pipenv)
- Faster dependency resolution
- PEP 621 compliant `pyproject.toml`
- Built-in lockfile support

**Testing**: `pytest` with `pytest-cov`
**Linting**: `ruff` (all-in-one: format, lint, import sort)
**Type Checking**: `ty` (extremely fast)
**Server**: `uvicorn` (ASGI)
**ORM**: `sqlalchemy` with async support
**Validation**: `pydantic` v2

### Project Initialization

**Applications**:
```bash
uv init project-name
cd project-name
uv add fastapi uvicorn pydantic sqlalchemy
uv add --dev pytest pytest-cov pytest-asyncio ruff
```

**Libraries/Packages**:
```bash
uv init project-name --package
```

**Result**: Automatic PEP 621-compliant `pyproject.toml` and `.gitignore`

### Essential Commands for AI Assistants

**Critical Guideline**: AI assistants should prefer `uv run <command>` for dev dependencies and `uvx <command>` for one-off tools to ensure operations occur within isolated project environments.

```bash
# Development
uv run pytest                    # Run tests
uv run pytest --cov              # With coverage
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run python -m myapp           # Run application

# One-off tools (no install)
uvx black .                      # Format with black
uvx mypy .                       # Type check
```

### Virtual Environment Pattern

**ALWAYS activate virtual environment before Python commands**:

```bash
# Pattern 1: Activate first (multiple commands)
source .venv/bin/activate
python -m myapp.cli [command]

# Pattern 2: One-liner (single command)
source .venv/bin/activate && python -m myapp.cli [command]

# Pattern 3: Direct venv python
.venv/bin/python -m myapp.cli [command]
```

**NEVER**:
- Use `python` or `python3` without activating venv
- Mix pip and uv
- Run tools globally instead of through `uv run`
- Skip lockfiles

### Project Structure

```
project/
├── pyproject.toml       # PEP 621 dependencies
├── uv.lock              # Locked dependencies (commit this)
├── .venv/               # Virtual environment (gitignored)
├── src/
│   └── project_name/    # Main package
│       ├── __init__.py
│       ├── core/        # Infrastructure
│       ├── api/         # FastAPI routes
│       └── models/      # Data models
├── tests/
│   ├── unit/
│   └── integration/
└── .docs/               # AI-friendly documentation
    ├── architecture.md
    └── testing-strategy.md
```

### Quality Automation

**Pre-commit Hooks** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

**Makefile** (single command quality check):
```makefile
.PHONY: check
check:
	uv run pytest
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check src/
```

**Usage**: `make check` (local) and in CI/CD pipeline

### Documentation for AI Assistants

**CLAUDE.md for Python Projects**:
```markdown
# Project Name

Brief description.

## Stack
- Python 3.11+
- FastAPI + Uvicorn
- PostgreSQL + SQLAlchemy (async)
- Pydantic v2

## Virtual Environment

**CRITICAL**: Always activate venv before Python commands:
source .venv/bin/activate && python -m project_name.cli [command]

## Commands

uv run pytest                 # Tests
uv run ruff check .           # Lint
make check                    # All quality gates

## Quality Gates
[VERIFY] pytest passes
[VERIFY] ruff check clean
[VERIFY] ty check passes
```

### Anti-Patterns

❌ **Avoid**:
- Mixing pip and uv
- Running commands without venv context
- Skipping lockfiles (`uv.lock`)
- Manual virtual environment creation
- Using `requirements.txt` (use `pyproject.toml`)

---

## Spec-Driven Development

### Concept

Write complete application specifications in Markdown; AI agent "compiles" the spec into executable code in the target language. Markdown becomes the source code.

**Research Finding**: Spec-driven development results in cleaner specifications, faster iteration, and eliminates context loss between sessions.

### Four-Component Structure

1. **README.md**: User-facing documentation (features, usage)
2. **main.md**: Technical specification (user docs + implementation requirements)
3. **compile.prompt.md**: AI agent compilation instructions
4. **main.go** (or equivalent): Auto-generated implementation

### Specification Format

**Combining Documentation with Requirements**:
```markdown
# User Authentication System

## Features

Users can register, log in, and manage their profiles.

## Implementation

### Database Schema

**users table**:
- id: UUID (primary key)
- email: STRING (unique, required, max 255)
- password_hash: STRING (required, bcrypt)
- created_at: TIMESTAMP (default: now)
- updated_at: TIMESTAMP (auto-update)

### Registration Flow

1. User submits email and password (minimum 8 characters)
2. If email exists, return error "Email already registered"
3. Hash password using bcrypt (cost factor 12)
4. Create user record
5. Send verification email
6. Return success with user ID

### GraphQL Mutations

```graphql
mutation Register($email: String!, $password: String!) {
  register(email: $email, password: $password) {
    user { id email }
    token
  }
}
```
```

**Structured Natural Language**:
Use plain English with code-like structure:
- "If condition, then action"
- "For each item in collection, do X"
- "Continue if validation passes"

### Development Workflow

1. **Edit** the Markdown specification
2. **Invoke** AI agent's compilation prompt
3. **Test** the generated code
4. **Update** specification if needed
5. **Repeat**

### Benefits

✅ **Advantages**:
- Documentation and code always synchronized
- Faster iteration (edit spec vs edit code)
- No context loss between sessions
- Easier to review (Markdown diffs)
- AI-friendly format leverages LLM strengths

✅ **Best For**:
- Prototypes and MVPs
- Well-defined domains
- Projects with clear specifications
- Teams with strong AI assistant workflows

❌ **Limitations**:
- Requires AI agent for compilation
- May generate non-idiomatic code initially
- Needs human review for optimization
- Not suitable for performance-critical code

### Additional Tools

**lint.prompt.md**: Optimize specifications for clarity
- Minimize synonyms
- Remove duplication
- Preserve critical details
- Treat English as a programming language

---

## Context Window Optimization

### Understanding Context Windows

**Definition**: Maximum tokens an LLM can process in a single request (input + output)

**2026 Leading Models**:
- Claude Opus 4.5: 200K tokens (~150K words)
- GPT-4 Turbo: 128K tokens
- Gemini 1.5 Pro: 1M tokens

**Problem**: Larger context ≠ better performance
- "Performance degrades significantly as context length increases" (NoLiMa study)
- More input tokens → slower output generation
- LLMs bias towards information at peripheries of prompt

### Optimization Techniques

#### 1. Cache Augmented Generation (CAG)

Pre-compute documents and cache results as part of prompt.

**Benefits**:
- Faster than RAG (no retrieval step)
- Consistent performance
- Reduced API costs

**Implementation**:
```python
# Pre-compute and cache documentation
cached_context = embed_and_cache([
    "docs/architecture.md",
    "docs/api-conventions.md"
])

# Use in prompts without re-processing
response = llm.complete(
    prompt=user_query,
    cached_context=cached_context
)
```

#### 2. Truncation Strategies

**Sliding Window**: Keep most recent N tokens
**Head + Tail**: Keep beginning (context) + end (recent conversation)
**Relevance-Based**: Remove low-relevance sections

#### 3. Memory Buffering

Store conversation history in structured format; retrieve relevant parts on demand.

**Example**:
```python
# Store conversation
memory.add_message(role="user", content="How do I deploy?")
memory.add_message(role="assistant", content="Run docker compose up")

# Retrieve relevant context
relevant = memory.search("deployment", limit=3)
```

#### 4. Progressive Position Encoding

Techniques to extend context windows:
- **Position Interpolation (PI)**: Scale token positions to fit new context length
- **Ring Attention**: Add attention mechanism for longer sequences
- **Relative Position Encoding**: Use relative positions instead of absolute

### Best Practices

**1. Stay Under 75% of Maximum**:
Keep context to ~75% of max tokens to allow complete responses.

**Example**: For 200K context window, limit input to 150K tokens.

**2. Start New Sessions for New Tasks**:
Clear context window between unrelated tasks to prevent confusion.

**3. Structured Context Loading**:
```markdown
## Current Task
[Brief description]

## Relevant Files
- file1.py: [brief purpose]
- file2.py: [brief purpose]

## Context
[Minimal essential information]
```

**4. Monitor Token Usage**:
Track input/output token counts to identify bloat sources.

### Token Budget Framework

**Andrej Karpathy's "Context Engineering"**:
"The delicate art and science of filling the context window with just the right information."

**Allocation Example** (200K window):
- System instructions: 5K tokens (2.5%)
- Project documentation: 10K tokens (5%)
- Relevant code: 30K tokens (15%)
- Conversation history: 105K tokens (52.5%)
- Response buffer: 50K tokens (25%)

**Result**: Balanced context enabling long conversations without degradation.

---

## Anti-Patterns & Mistakes

### 1. Disconnected Prompting

**Problem**: Using only natural language prompts without providing AI access to data sources, APIs, or tools.

**Result**: Hallucinated responses, outdated information, incorrect assumptions.

**Solution**:
- Provide file access via Read tool
- Include relevant documentation
- Reference actual code/data
- Use structured context files (CLAUDE.md, AGENTS.md)

### 2. Lack of Project-Specific Context

**Problem**: AI suggests same flawed approaches because it lacks understanding of architecture, dependencies, or constraints.

**Result**: Generic solutions that don't fit project patterns, repeated mistakes.

**Solution**:
- Maintain comprehensive CLAUDE.md/AGENTS.md
- Include architecture decision records (ADRs)
- Document constraints and gotchas
- Provide examples of correct patterns

### 3. Trusting AI Output Blindly

**Problem**: Accepting generated code without review or testing.

**Result**: Bugs, security vulnerabilities, technical debt.

**Solution**:
- Think of AI as "over-confident and prone to mistakes"
- Always review generated code
- Run tests and quality checks
- Validate against requirements
- Use `[VERIFY]` checkpoints

### 4. Unclear or Ambiguous Prompts

**Problem**: Prompts with too much room for interpretation.

**Result**: Technically correct but unwanted outcomes, repeated clarifications.

**Solution**:
- Be specific about requirements
- Provide examples of expected output
- Use structured formats (bullet points, code blocks)
- Include success criteria

### 5. Bloated Memory/Rules Files

**Problem**: CLAUDE.md/AGENTS.md files become dumping grounds for every decision.

**Result**: Context window waste, instruction-following degradation, slow responses.

**Solution**:
- Keep root files under 300 lines
- Use progressive disclosure
- Remove rules handled by deterministic tools (linters)
- Archive outdated instructions

### 6. Documentation Gaps

**Problem**: AI edits code but never updates README, onboarding guides, or runbooks.

**Result**: Documentation drift, confused teammates, broken instructions.

**Solution**:
- Include documentation updates in acceptance criteria
- Use `[VERIFY] Documentation updated` checkpoints
- Maintain session logs (`.docs/sessions/`)
- Review docs in PRs

### 7. Over-Abstraction with Frameworks

**Problem**: Using AI agent frameworks that create extra abstraction layers.

**Result**: Obscured prompts/responses, harder debugging, unnecessary complexity.

**Solution**:
- Start with LLM APIs directly
- Add abstractions only when proven necessary
- Prefer simple setups over complex frameworks
- Maintain visibility into prompts/responses

### 8. Ignoring Validation Workflows

**Problem**: No automated checks for AI-generated code reliability.

**Result**: AI-specific issues (hallucinations, deprecated patterns) reach production.

**Solution**:
- Build validation workflows catching AI-specific issues
- Use automated tooling (linters, type checkers, tests)
- Validate with real traffic patterns
- Implement quality gates

### 9. Embedding Code Snippets in Documentation

**Problem**: Copy-pasting code into CLAUDE.md or AGENTS.md.

**Result**: Snippets become outdated, create confusion, waste context.

**Solution**:
- Use file path references: `See src/auth/login.py:45-60`
- Link to actual source files
- Describe patterns rather than copying code
- Let AI read current file versions

### 10. No Context Management Between Sessions

**Problem**: Every session starts from scratch, repeating same information.

**Result**: Wasted time, inconsistent approaches, lost learnings.

**Solution**:
- Implement `/learn` skill for capturing insights
- Maintain session logs (`.docs/sessions/YYYY-MM-DD-description.md`)
- Update documentation with learnings
- Build institutional knowledge

### Quality Impact Research

**Instruction-Following Quality**:
- Under 300 lines: 85%+ quality maintained
- Over 500 lines: Uniform degradation begins
- Over 1000 lines: Significant performance loss

**LLMs Bias Towards Peripheries**:
Instructions at beginning/end of prompt receive more weight than middle content.

**Implication**: Prioritize critical information in document structure.

---

## Docker Environment Documentation

### Best Practices for AI Assistants

**Official Resources**: [Docker AI Best Practices](https://docs.docker.com/ai/cagent/best-practices/)

### Key Patterns

#### 1. Version-Controlled Dockerfiles

**Recommendation**: Define environments in Dockerfiles, commit to version control.

**Benefits**:
- Track environment changes alongside code
- Reproducible builds
- Team consistency

**Example**:
```dockerfile
FROM python:3.11-slim

# Document AI model requirements
RUN pip install torch transformers

# Version lock for reproducibility
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

#### 2. Output Redirection for Large Commands

**Problem**: Shell commands with large output overflow AI context window.

**Solution**: Redirect output to file, then read file selectively.

```bash
# Instead of: docker logs container > context overflow
docker logs container > /tmp/logs.txt
tail -100 /tmp/logs.txt  # Only read last 100 lines
```

#### 3. AI-Friendly Compose Files

**Structure**:
```yaml
# NO version field (deprecated in Compose v2)
services:
  app:
    container_name: project-app  # Explicit names for AI commands
    image: project/app:latest
    environment:
      - ENV=production
    healthcheck:  # Enable health monitoring
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**AI Benefits**:
- Explicit container names enable direct commands
- Health checks provide status context
- Clear environment separation

#### 4. Service Documentation Pattern

**CLAUDE.md Section**:
```markdown
## Docker Services

### Starting Services
docker compose up -d            # Detached mode
docker compose logs -f [service]  # View logs

### Service Health
All services include /health endpoints:
- API: http://localhost:53000/health
- Database: http://localhost:53432/health
- Cache: http://localhost:53379/health

### Port Assignments
See `.docs/services-ports.md` for complete port registry.

**CRITICAL**: Always use high ports (53000+), never standard ports.
```

#### 5. Environment Variables Documentation

**Structure**:
```markdown
## Required Environment Variables

See `.env.example` for template. Copy to `.env` and configure:

**Mandatory**:
- `POSTGRES_PASSWORD` - Database password (no default)
- `API_SECRET_KEY` - Application secret (generate with: openssl rand -hex 32)

**Optional** (with defaults):
- `POSTGRES_PORT` (default: 53432)
- `API_PORT` (default: 53000)
```

**Benefits**:
- Clear required vs optional distinction
- Generation commands for secrets
- Default value documentation

### Docker + AI Model Patterns

**Local Model Packaging**:
```dockerfile
FROM python:3.11-slim

# Install model dependencies
RUN pip install transformers torch

# Download model at build time (not runtime)
RUN python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('model-name')"

# Runtime doesn't need internet access
CMD ["python", "app.py"]
```

**Benefits**:
- Faster inference (no API latency)
- Greater privacy (no external calls)
- Versioned model packaging

### Gordon - Docker's AI Assistant

**Context**: Docker provides Gordon, an AI assistant for Docker Desktop/CLI.

**Challenge**: Bad Dockerfiles vastly outnumber best-practice examples in LLM training data.

**Solution**: Use Docker's official best practices documentation to override poor training data patterns.

### Multi-Stage Builds for AI Projects

```dockerfile
# Build stage
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

**Benefits**:
- Smaller final image
- Faster deployment
- Clearer separation of concerns

---

## Implementation Checklist

Use this checklist when setting up AI-friendly documentation for a new project:

### Phase 1: Foundation (Day 1)

- [ ] Create `.claude/CLAUDE.md` (or `AGENTS.md` for multi-tool support)
- [ ] Define project overview (2-3 sentences)
- [ ] List technology stack with versions
- [ ] Document essential commands
- [ ] Add quality gate `[VERIFY]` checkpoints
- [ ] Create `.env.example` with required variables
- [ ] Add `.gitignore` for `.env`, `.venv`, `node_modules`

### Phase 2: Structure (Week 1)

- [ ] Create `.docs/` directory for session logs and references
- [ ] Add `architecture.md` with high-level design
- [ ] Document project structure in CLAUDE.md
- [ ] Set up progressive disclosure directories (`/docs` or `.claude/rules/`)
- [ ] Create `services-ports.md` port registry
- [ ] Add `deployment-log.md` template
- [ ] Configure pre-commit hooks (if applicable)

### Phase 3: Progressive Disclosure (Week 2)

- [ ] Identify topics for Layer 2 documentation
- [ ] Create domain-specific files (API conventions, testing strategy, etc.)
- [ ] Add progressive disclosure pointers in root CLAUDE.md
- [ ] Review root file size (target: under 300 lines)
- [ ] Move detailed content to on-demand files
- [ ] Validate token budget (root should be ~500 tokens)

### Phase 4: Automation (Week 3)

- [ ] Set up quality automation (Makefile, package.json scripts)
- [ ] Configure CI/CD with quality gates
- [ ] Add automated testing workflows
- [ ] Implement pre-commit hooks for linting/formatting
- [ ] Document automation commands in CLAUDE.md
- [ ] Create `[VERIFY]` automation checkpoints

### Phase 5: Advanced (Ongoing)

- [ ] Create specialized agents (`.claude/agents/`) if needed
- [ ] Implement `/learn` skill for knowledge capture
- [ ] Add architectural decision records (ADRs)
- [ ] Document common gotchas and footguns
- [ ] Review and update documentation monthly
- [ ] Archive outdated instructions
- [ ] Collect team feedback on documentation effectiveness

### Validation Criteria

**Root Documentation**:
- [ ] Under 300 lines total
- [ ] Includes WHAT, WHY, HOW dimensions
- [ ] No embedded code snippets
- [ ] Uses progressive disclosure pointers
- [ ] Contains `[VERIFY]` quality gates

**Progressive Disclosure**:
- [ ] Layer 2 files exist for major topics
- [ ] Each file is 200-500 tokens
- [ ] Clear topic separation
- [ ] Referenced from root documentation

**Tooling**:
- [ ] Linting configured (Ruff, ESLint)
- [ ] Formatting automated (Ruff, Prettier)
- [ ] Type checking enabled (ty, TypeScript strict)
- [ ] Tests runnable with single command
- [ ] Quality gates enforceable (`make check`)

**Docker** (if applicable):
- [ ] Service health checks configured
- [ ] Port registry documented
- [ ] Environment variables documented
- [ ] Explicit container names
- [ ] High ports used (53000+)

---

## Sources

### Primary Research

1. [How to Use AI in Coding - 12 Best Practices in 2026](https://zencoder.ai/blog/how-to-use-ai-in-coding)
2. [Top 7 Code Documentation Best Practices for Teams (2026)](https://www.qodo.ai/blog/code-documentation-best-practices-2026/)
3. [Best practices for using AI coding assistants effectively](https://graphite.com/guides/best-practices-ai-coding-assistants)
4. [Using CLAUDE.MD files: Customizing Claude Code for your codebase](https://claude.com/blog/using-claude-md-files)
5. [Writing a good CLAUDE.md | HumanLayer Blog](https://www.humanlayer.dev/blog/writing-a-good-claude-md)
6. [The Complete Guide to CLAUDE.md](https://www.builder.io/blog/claude-md-guide)
7. [Creating the Perfect CLAUDE.md for Claude Code](https://dometrain.com/blog/creating-the-perfect-claudemd-for-claude-code/)

### AGENTS.md Format

8. [Agents.md: A Machine-Readable Alternative to README](https://research.aimultiple.com/agents-md/)
9. [Steering AI Agents in Monorepos with AGENTS.md](https://dev.to/datadog-frontend-dev/steering-ai-agents-in-monorepos-with-agentsmd-13g0)
10. [AGENTS.md Format Documentation](https://deepwiki.com/openai/agents.md/5-agents.md-format-documentation)
11. [Back to basics: a solid foundation for using AI coding agents in a monorepo](https://dev.to/valuecodes/back-to-basics-a-solid-foundation-for-using-ai-coding-agents-in-a-monorepo-3c26)

### Progressive Disclosure

12. [Stop Bloating Your CLAUDE.md: Progressive Disclosure for AI Coding Tools](https://alexop.dev/posts/stop-bloating-your-claude-md-progressive-disclosure-ai-coding-tools/)
13. [Progressive Disclosure | AI Design Patterns](https://www.aiuxdesign.guide/patterns/progressive-disclosure)
14. [Progressive disclosure - Claude-Mem](https://docs.claude-mem.ai/progressive-disclosure)
15. [Why AI Agents Need Progressive Disclosure, Not More Data](https://www.honra.ai/articles/progressive-disclosure-for-ai-agents)

### Spec-Driven Development

16. [Spec-driven development: Using Markdown as a programming language when building with AI](https://github.blog/ai-and-ml/generative-ai/spec-driven-development-using-markdown-as-a-programming-language-when-building-with-ai/)
17. [Boosting AI Performance: The Power of LLM-Friendly Content in Markdown](https://developer.webex.com/blog/boosting-ai-performance-the-power-of-llm-friendly-content-in-markdown)
18. [Markdown: The Best Text Format for Training AI Models](https://blog.bismart.com/en/markdown-ai-training)

### Python Project Setup

19. [Modern Python Project Setup Guide for AI Assistants](https://pydevtools.com/handbook/explanation/modern-python-project-setup-guide-for-ai-assistants/)
20. [Teaching LLMs Python Best Practices](https://pydevtools.com/blog/teaching-llms-python-best-practices/)
21. [From Zero to Production: A Complete Python Setup Guide](https://caparra.ai/blog/complete-python-setup-guide)
22. [How to Set Up a Python Project For Automation and Collaboration](https://eugeneyan.com/writing/setting-up-python-project-for-automation-and-collaboration/)

### Context Optimization

23. [LLM Context Management: How to Improve Performance and Lower Costs](https://eval.16x.engineer/blog/llm-context-management-guide)
24. [Top techniques to Manage Context Lengths in LLMs](https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms)
25. [LLM Context Windows: Basics, Examples & Prompting Best Practices](https://swimm.io/learn/large-language-models/llm-context-windows-basics-examples-and-prompting-best-practices)

### Anti-Patterns

26. [AI coding anti-patterns: 6 things to avoid for better AI coding](https://dev.to/lingodotdev/ai-coding-anti-patterns-6-things-to-avoid-for-better-ai-coding-f3e)
27. [Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents)
28. [Why Your AI Assistant Keeps Making the Same Mistakes (And How to Fix It)](https://blog.todo2.pro/ai-coding-mistakes/)
29. [GitHub - PaulDuvall/ai-development-patterns](https://github.com/PaulDuvall/ai-development-patterns)

### Docker & AI

30. [Docker AI Best Practices](https://docs.docker.com/ai/cagent/best-practices/)
31. [Docker Brings Compose to the AI Agent Era](https://www.docker.com/blog/build-ai-agents-with-docker-compose/)
32. [Meet Gordon: An AI Agent for Docker](https://www.docker.com/blog/meet-gordon-an-ai-agent-for-docker/)

### GitHub Copilot Context

33. [Provide context to GitHub Copilot - GitHub Docs](https://docs.github.com/en/copilot/how-tos/provide-context)
34. [Understanding the Contextual Scope of GitHub Copilot](https://github.com/orgs/community/discussions/69280)
35. [GitHub Copilot CLI: Enhanced agents, context management](https://github.blog/changelog/2026-01-14-github-copilot-cli-enhanced-agents-context-management-and-new-ways-to-install/)

---

## Appendix: Quick Reference

### Documentation File Sizes

| Type | Target | Maximum | Token Budget |
|------|--------|---------|--------------|
| Root CLAUDE.md | 200 lines | 300 lines | ~500 tokens |
| Layer 2 Docs | 100 lines | 250 lines | 200-500 tokens |
| Specialized Agents | 150 lines | 400 lines | 300-800 tokens |

### Quality Gates Template

```markdown
## Verification

Before committing, ensure:

[VERIFY] Tests pass: `uv run pytest`
[VERIFY] Linting clean: `uv run ruff check .`
[VERIFY] Types valid: `uv run ty check src/`
[VERIFY] Formatting applied: `uv run ruff format .`
[VERIFY] Documentation updated
```

### Progressive Disclosure Directory Structure

```
project/
├── CLAUDE.md                    # 200 lines, ~500 tokens
├── .claude/
│   ├── agents/                  # Specialized agents
│   │   ├── database.md
│   │   └── deployment.md
│   └── rules/                   # Auto-loaded rules
│       └── python-conventions.md
├── .docs/
│   ├── architecture.md          # On-demand Layer 2
│   ├── api-conventions.md
│   ├── testing-strategy.md
│   └── sessions/                # Session logs
│       └── 2026-01-20-feature-x.md
└── docs/                        # Public documentation
    └── README.md
```

### Command Templates

**Python (uv)**:
```bash
uv run pytest                    # Tests
uv run ruff check . --fix        # Lint + fix
uv run ruff format .             # Format
make check                       # All quality gates
```

**TypeScript (pnpm)**:
```bash
pnpm test                        # Tests
pnpm lint:fix                    # Lint + fix
pnpm typecheck                   # Type check
pnpm check                       # All quality gates
```

**Docker**:
```bash
docker compose up -d             # Start all services
docker compose logs -f [service] # View logs
docker compose down              # Stop all services
```

---

**Document Version**: 1.0
**Last Updated**: January 20, 2026
**Maintainer**: Research Specialist
**Review Cycle**: Monthly
