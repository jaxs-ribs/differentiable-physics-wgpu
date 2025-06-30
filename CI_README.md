# CI System Overview

## Simple and Clean Test Structure

The CI system is now a simple Python dispatcher that runs tests from organized directories.

### Running Tests

```bash
# Default: Run unit + integration tests
python ci.py

# Run only unit tests
python ci.py --unit

# Run only integration tests  
python ci.py --integration

# Run benchmarks
python ci.py --benchmarks

# Run everything
python ci.py --all

# Quick mode (skip slow tests)
python ci.py --quick
```

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for system behavior
├── benchmarks/     # Performance benchmarks
└── debugging/      # Debug scripts (not run by CI)
```

### Key Principles

1. **CI is just a dispatcher** - It doesn't contain tests, just runs them
2. **All tests use pytest** - Consistent test discovery and execution
3. **Clean separation** - Unit, integration, and benchmarks are distinct
4. **Simple flags** - Easy to run subsets of tests
5. **Debugging stays separate** - Debug tools available but not in CI

### Writing New Tests

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- All tests should be pytest-compatible
- Tests can also have `if __name__ == "__main__"` for standalone execution