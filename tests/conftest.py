"""
PyTest configuration with smart test failure dumps.

This module implements automatic state-saving when parity tests fail,
creating persistent artifacts for the visual debugger.
"""

import pytest
import numpy as np
import os
from pathlib import Path


def pytest_runtest_makereport(item, call):
    """
    Hook that runs after each test phase (setup, call, teardown).
    
    When a test fails, this will automatically dump CPU and GPU states
    if they are available in the test's locals.
    """
    if call.when == "call" and call.excinfo is not None:
        # Test failed during execution
        test_name = item.name
        test_function = item.function
        
        # Try to access the test's local variables
        frame = call.excinfo.traceback[-1].frame
        locals_dict = frame.locals
        
        # Look for engine and gpu_harness in the test's locals
        cpu_state = None
        gpu_state = None
        
        # Try to get CPU state from PhysicsEngine
        if 'engine' in locals_dict:
            try:
                engine = locals_dict['engine']
                if hasattr(engine, 'get_state'):
                    cpu_state = engine.get_state()
            except Exception as e:
                print(f"Failed to get CPU state: {e}")
        
        # Try to get GPU state from GpuHarness
        if 'gpu_harness' in locals_dict:
            try:
                gpu_harness = locals_dict['gpu_harness']
                if hasattr(gpu_harness, 'get_state_as_numpy'):
                    gpu_state = gpu_harness.get_state_as_numpy()
            except Exception as e:
                print(f"Failed to get GPU state: {e}")
        
        # Also check for alternative variable names
        if cpu_state is None and 'cpu_engine' in locals_dict:
            try:
                cpu_engine = locals_dict['cpu_engine']
                if hasattr(cpu_engine, 'get_state'):
                    cpu_state = cpu_engine.get_state()
            except Exception:
                pass
        
        if gpu_state is None and 'harness' in locals_dict:
            try:
                harness = locals_dict['harness']
                if hasattr(harness, 'get_state_as_numpy'):
                    gpu_state = harness.get_state_as_numpy()
            except Exception:
                pass
        
        # Save the states if we got them
        if cpu_state is not None or gpu_state is not None:
            # Create failures directory
            failure_dir = Path("tests/failures") / test_name
            failure_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CPU state
            if cpu_state is not None:
                cpu_path = failure_dir / "cpu_state.npy"
                np.save(cpu_path, cpu_state)
                print(f"\n💾 Saved CPU state to: {cpu_path}")
            
            # Save GPU state
            if gpu_state is not None:
                gpu_path = failure_dir / "gpu_state.npy"
                np.save(gpu_path, gpu_state)
                print(f"💾 Saved GPU state to: {gpu_path}")
            
            # Save additional debug info
            debug_info_path = failure_dir / "debug_info.txt"
            with open(debug_info_path, 'w') as f:
                f.write(f"Test: {test_name}\n")
                f.write(f"Module: {item.module.__name__}\n")
                f.write(f"Function: {test_function.__name__}\n")
                f.write(f"Error: {call.excinfo.typename}: {call.excinfo.value}\n")
                f.write(f"\nTraceback:\n{call.excinfo.getrepr()}\n")
            
            print(f"📝 Saved debug info to: {debug_info_path}")
            print(f"\n🔍 To visualize the failure, run:")
            print(f"   cargo run --features viz --bin debug_viz -- --oracle {cpu_path} --gpu {gpu_path}\n")


@pytest.fixture(autouse=True)
def cleanup_old_failures(request):
    """
    Fixture that runs before each test to clean up old failure dumps.
    This prevents accumulation of old failure data.
    """
    if request.config.getoption("--keep-failures", default=False):
        return
    
    test_name = request.node.name
    failure_dir = Path("tests/failures") / test_name
    
    if failure_dir.exists():
        import shutil
        shutil.rmtree(failure_dir)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--keep-failures",
        action="store_true",
        default=False,
        help="Keep previous test failure dumps instead of cleaning them up"
    )