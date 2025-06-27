# Proposed Directory Structure

Current issues:
- Too many files at root level
- Test results accumulating
- Mixed documentation and code
- Unclear hierarchy

## Proposed Structure:

```
physics_core/
├── README.md                    # Main project readme
├── .gitignore                   # Git ignore
├── pyproject.toml              # Project config (if needed)
│
├── physics/                     # Core physics engine
│   ├── __init__.py
│   ├── engine.py
│   ├── types.py
│   └── ... (existing files)
│
├── custom_ops/                  # Custom C operations
│   ├── README.md
│   ├── src/                    # C source
│   ├── python/                 # Python bindings
│   ├── examples/               # Usage examples
│   └── build/                  # Compiled libraries
│
├── tests/                       # All tests
│   ├── run_ci.py              # Main CI runner
│   ├── run_all_tests.py       # Interactive test runner
│   ├── unit/
│   ├── integration/
│   ├── debugging/
│   └── benchmarks/
│
├── docs/                        # All documentation
│   ├── AGENTS.md               # Project vision
│   ├── BUG_FIXES.md           # Known issues
│   ├── TEST_HELL.md           # Test runner docs
│   ├── TEST_SUMMARY.md        # Test results
│   └── architecture/          # Design docs
│
├── scripts/                     # Utility scripts
│   ├── run_physics.py         # Physics runner
│   ├── clean_test_results.py  # Cleanup script
│   └── setup_env.py           # Environment setup
│
├── artifacts/                   # Simulation outputs
│   └── .gitignore             # Ignore all artifacts
│
└── external/                    # External dependencies
    ├── tinygrad/               # Submodule
    └── tinygrad-notes/         # Reference docs
```

## Benefits:
1. **Clear separation** of concerns (code, tests, docs, scripts)
2. **No clutter** at root level - just README and config
3. **Easy navigation** - know where to find things
4. **Better git management** - test results in artifacts/ with .gitignore
5. **Scalable** - easy to add new components

## Migration Plan:
1. Create new directories
2. Move files to appropriate locations
3. Update all imports
4. Update documentation references
5. Add proper .gitignore entries