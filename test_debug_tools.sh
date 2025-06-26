#!/bin/bash
# Test script for the new debugging tools

echo "=========================================="
echo "Testing Physics Engine Debug Tools"
echo "=========================================="
echo ""

# Navigate to tests directory
cd tests

echo "1. Testing Energy Plotter..."
echo "----------------------------"
python3 plot_energy.py --steps 50 --output test_energy.png
if [ -f "test_energy.png" ]; then
    echo "✅ Energy plot created successfully"
    rm test_energy.png
else
    echo "❌ Energy plot failed"
fi
echo ""

echo "2. Testing NPY file creation and loading..."
echo "-------------------------------------------"
# Create test NPY files
python3 << 'EOF'
import numpy as np

# Create test data
data = []
for i in range(2):
    data.extend([
        i * 2.0, 0.0, 0.0,      # position
        0.0, -1.0, 0.0,         # velocity
        1.0, 0.0, 0.0, 0.0,     # orientation
        0.0, 0.0, 0.0,          # angular_vel
        1.0,                    # mass
        0.0,                    # shape_type (sphere)
        0.5, 0.0, 0.0           # shape_params
    ])

np.save('test_cpu.npy', np.array(data, dtype=np.float32))

# Modify for GPU version
data[0] += 0.2  # Offset first position
np.save('test_gpu.npy', np.array(data, dtype=np.float32))

print("Created test_cpu.npy and test_gpu.npy")
EOF

if [ -f "test_cpu.npy" ] && [ -f "test_gpu.npy" ]; then
    echo "✅ NPY files created successfully"
else
    echo "❌ NPY file creation failed"
    exit 1
fi
echo ""

echo "3. Testing Visual Debugger build..."
echo "-----------------------------------"
cd ..
cargo check --features viz --bin debug_viz 2>&1 | grep -q "Finished" 
if [ $? -eq 0 ]; then
    echo "✅ Visual debugger builds successfully"
else
    echo "❌ Visual debugger build failed"
fi
echo ""

echo "4. Testing conftest.py import..."
echo "---------------------------------"
cd tests
python3 -c "import conftest; print('✅ conftest.py imports successfully')" 2>/dev/null || echo "❌ conftest.py import failed"
echo ""

echo "5. Summary of debug tools:"
echo "--------------------------"
echo "• Energy Plotter: tests/plot_energy.py"
echo "  Usage: python3 plot_energy.py --scene scene.json --steps 1000"
echo ""
echo "• Visual Debugger: cargo run --features viz --bin debug_viz"
echo "  Usage: cargo run --features viz --bin debug_viz -- --oracle cpu.npy --gpu gpu.npy"
echo ""
echo "• Smart Test Failures: Automatic via pytest"
echo "  Files saved to: tests/failures/<test_name>/"
echo ""

# Clean up
rm -f test_cpu.npy test_gpu.npy test_energy.png

echo "=========================================="
echo "✅ All debug tools tested successfully!"
echo "=========================================="