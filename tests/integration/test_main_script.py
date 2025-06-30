"""Test the main physics simulation script.

WHAT: Validates the command-line interface and output of the main 
      physics simulation script (physics/main.py).

WHY: The main script is the primary user interface:
     - Users run simulations via command line
     - Output files must be in the correct format for the renderer
     - CLI argument parsing must work correctly
     - Ensures the "happy path" works for end users

HOW: - Runs main.py as a subprocess with various arguments
     - Captures stdout/stderr to verify success
     - Checks that output .npy files are created
     - Validates the shape and format of saved data
     - Uses subprocess.run to simulate real command-line usage
"""
import os
import subprocess
import numpy as np
import pytest

def test_main_script_execution():
    """Test that main.py runs successfully."""
    # Run with minimal steps for testing
    env = os.environ.copy()
    env['CI'] = 'true'  # Use CI mode for reduced steps
    
    result = subprocess.run(
        ['python3', '-m', 'physics.main', '--steps', '10'],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert result.returncode == 0, f"Main script failed: {result.stderr}"
    # Check for successful output indicators
    assert "Simulation ran" in result.stdout or "Created artifacts" in result.stdout

def test_main_script_output():
    """Test that main.py produces valid output."""
    output_file = 'artifacts/test_main_output.npy'
    
    # Clean up any existing file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Run simulation
    env = os.environ.copy()
    env['CI'] = 'true'
    
    result = subprocess.run(
        ['python3', '-m', 'physics.main', '--steps', '10', '--output', output_file],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert result.returncode == 0
    assert os.path.exists(output_file), "Output file not created"
    
    # Verify output format
    data = np.load(output_file)
    # In JIT mode, only saves initial and final frames
    assert data.shape[0] >= 2, "Should have at least 2 frames"
    assert data.shape[1] == 36, "Should have 36 values per frame (2 bodies * 18 properties)"
    
    # Clean up
    os.remove(output_file)

if __name__ == "__main__":
    test_main_script_execution()
    test_main_script_output()
    print("Main script tests passed!")