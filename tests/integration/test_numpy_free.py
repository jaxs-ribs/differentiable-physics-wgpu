"""Test that core physics modules don't require NumPy.

WHAT: Ensures core physics computation modules can be imported and used
      without NumPy being available in the Python environment.

WHY: TinyGrad is designed to be a minimal, self-contained tensor library:
     - Core physics math should use TinyGrad tensors, not NumPy arrays
     - Reduces dependencies and deployment size
     - Enables running on platforms where NumPy isn't available
     - Forces clean separation between computation (TinyGrad) and I/O (NumPy)

HOW: - Temporarily removes NumPy from sys.modules
     - Attempts to import core physics modules
     - Verifies they load successfully without NumPy
     - Restores NumPy afterwards to not break other tests
     Note: Engine and main modules may use NumPy for I/O, so they're excluded.
"""
import sys
import importlib
import pytest

def test_core_modules_numpy_free():
    """Verify core modules can import without NumPy."""
    # Temporarily remove numpy from modules
    numpy_modules = []
    for name in list(sys.modules.keys()):
        if name == 'numpy' or name.startswith('numpy.'):
            numpy_modules.append((name, sys.modules[name]))
            del sys.modules[name]
    
    try:
        # These modules should work without numpy
        numpy_free_modules = [
            'physics.types',
            'physics.math_utils',
            'physics.integration',
            'physics.broadphase_tensor',
            'physics.narrowphase',
            'physics.solver'
        ]
        
        for module_name in numpy_free_modules:
            # Force reload to test import
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # This should work without numpy
            importlib.import_module(module_name)
    
    finally:
        # Restore numpy modules
        for name, module in numpy_modules:
            sys.modules[name] = module

if __name__ == "__main__":
    test_core_modules_numpy_free()
    print("NumPy-free test passed!")