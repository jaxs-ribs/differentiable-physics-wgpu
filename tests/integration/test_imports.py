"""Test that all core physics modules can be imported successfully.

WHAT: Validates that all physics engine modules can be imported without errors.

WHY: Import failures are often the first sign of:
     - Missing dependencies
     - Syntax errors in module code
     - Circular import issues
     - Incorrect module structure
     Catching these early prevents runtime failures.

HOW: Uses Python's importlib to dynamically import each core module.
     If any module fails to import, the test fails with the specific error.
"""
import pytest
import importlib

def test_core_module_imports():
    """Verify all core physics modules import without errors."""
    modules = [
        'physics.types',
        'physics.math_utils', 
        'physics.integration',
        'physics.broadphase_tensor',
        'physics.narrowphase',
        'physics.solver',
        'physics.engine',
        'physics.main'
    ]
    
    for module_name in modules:
        # This will raise ImportError if module can't be imported
        module = importlib.import_module(module_name)
        assert module is not None, f"Module {module_name} imported as None"

if __name__ == "__main__":
    test_core_module_imports()
    print("All imports successful!")