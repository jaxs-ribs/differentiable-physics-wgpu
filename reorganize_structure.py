#!/usr/bin/env python3
"""Reorganize the physics_core directory structure for better ergonomics."""

import os
import shutil
from pathlib import Path

def create_directories():
    """Create the new directory structure."""
    dirs = [
        "docs",
        "docs/architecture", 
        "scripts",
        "external"
    ]
    
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"✓ Created {d}/")

def move_documentation():
    """Move documentation files to docs/."""
    doc_files = [
        ("AGENTS.md", "docs/AGENTS.md"),
        ("BUG_FIXES.md", "docs/BUG_FIXES.md"),
        ("TEST_HELL_README.md", "docs/TEST_HELL.md"),
        ("TEST_SUMMARY.md", "docs/TEST_SUMMARY.md"),
        ("PROPOSED_STRUCTURE.md", "docs/PROPOSED_STRUCTURE.md")
    ]
    
    for src, dst in doc_files:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✓ Moved {src} → {dst}")

def move_scripts():
    """Move utility scripts to scripts/."""
    script_files = [
        ("run_physics.py", "scripts/run_physics.py"),
        ("run_all_tests.py", "scripts/run_test_hell.py")  # Rename for clarity
    ]
    
    for src, dst in script_files:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✓ Moved {src} → {dst}")

def move_external():
    """Move external dependencies to external/."""
    external_dirs = [
        ("tinygrad", "external/tinygrad"),
        ("tinygrad-notes", "external/tinygrad-notes")
    ]
    
    for src, dst in external_dirs:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"✓ Moved {src} → {dst}")

def clean_test_results():
    """Remove test result files."""
    import glob
    test_results = glob.glob("test_results_*.txt")
    for f in test_results:
        os.remove(f)
        print(f"✓ Removed {f}")

def update_gitignore():
    """Update .gitignore for new structure."""
    gitignore_additions = """
# Test results
test_results_*.txt

# Artifacts
artifacts/*.npy
artifacts/*.pkl
artifacts/*.txt

# Build files
custom_ops/build/*.o
custom_ops/build/*.so
custom_ops/build/*.dylib

# Python cache
**/__pycache__/
**/*.pyc
"""
    
    with open(".gitignore", "a") as f:
        f.write("\n# === Added by reorganize_structure.py ===\n")
        f.write(gitignore_additions)
    print("✓ Updated .gitignore")

def update_readme():
    """Update README with new structure."""
    readme_update = """

## Project Structure

```
physics_core/
├── physics/         # Core physics engine modules
├── custom_ops/      # Custom C operations for TinyGrad
├── tests/           # Comprehensive test suite
├── docs/            # Documentation and guides
├── scripts/         # Utility scripts
├── artifacts/       # Simulation outputs
└── external/        # External dependencies
```

See `docs/PROPOSED_STRUCTURE.md` for detailed organization.
"""
    
    # Read current README
    with open("README.md", "r") as f:
        content = f.read()
    
    # Find where to insert (after installation or at end)
    if "## Project Structure" not in content:
        # Add before "## Usage" if it exists
        if "## Usage" in content:
            parts = content.split("## Usage")
            content = parts[0] + readme_update + "\n## Usage" + parts[1]
        else:
            content += readme_update
        
        with open("README.md", "w") as f:
            f.write(content)
        print("✓ Updated README.md")

def create_script_launchers():
    """Create convenient launcher scripts at root."""
    launchers = {
        "test": """#!/bin/bash
# Run the interactive test hell session
python3 scripts/run_test_hell.py "$@"
""",
        "physics": """#!/bin/bash
# Run physics simulation
python3 scripts/run_physics.py "$@"
""",
        "ci": """#!/bin/bash
# Run CI tests
python3 tests/run_ci.py "$@"
"""
    }
    
    for name, content in launchers.items():
        with open(name, "w") as f:
            f.write(content)
        os.chmod(name, 0o755)
        print(f"✓ Created launcher script: {name}")

def main():
    """Run the reorganization."""
    print("Reorganizing physics_core directory structure...")
    print("=" * 60)
    
    # Create new directories
    create_directories()
    
    # Move files
    move_documentation()
    move_scripts()
    move_external()
    
    # Clean up
    clean_test_results()
    
    # Update configurations
    update_gitignore()
    update_readme()
    
    # Create convenience scripts
    create_script_launchers()
    
    print("=" * 60)
    print("✓ Reorganization complete!")
    print("\nNew structure:")
    print("  - Documentation in docs/")
    print("  - Scripts in scripts/")
    print("  - External deps in external/")
    print("  - Launcher scripts: ./test, ./physics, ./ci")
    print("\nNote: You may need to update imports in moved files.")

if __name__ == "__main__":
    main()