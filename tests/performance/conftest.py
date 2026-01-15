"""Performance test configuration and fixtures.

Provides fixtures specific to performance and load testing.
Imports from parent conftest.py for shared fixtures.

Example:
    Run performance tests:
    $ pytest tests/performance/ -v -m performance
"""

# Import from parent conftest for shared fixtures
# Performance-specific fixtures are defined in parent conftest.py:
# - memory_tracker()
# - performance_timer()
# - generate_n_files()
