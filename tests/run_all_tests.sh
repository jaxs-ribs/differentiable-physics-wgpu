#!/bin/bash
# Run all physics engine tests

echo "=========================================="
echo "Running All Physics Engine Tests"
echo "=========================================="
echo ""

# Navigate to tests directory
cd "$(dirname "$0")"

# Core tests
echo "1. Running SDF Tests..."
python3 test_sdf_quick.py
if [ $? -ne 0 ]; then
    echo "❌ SDF tests failed!"
    exit 1
fi
echo ""

echo "2. Running Energy Conservation Tests..."
python3 test_energy.py
if [ $? -ne 0 ]; then
    echo "❌ Energy tests failed!"
    exit 1
fi
echo ""

echo "3. Running Broadphase SAP Tests..."
python3 test_broadphase_sap.py
if [ $? -ne 0 ]; then
    echo "❌ Broadphase tests failed!"
    exit 1
fi
echo ""

echo "4. Running Rotational Dynamics Tests..."
python3 test_dynamics.py
if [ $? -ne 0 ]; then
    echo "❌ Dynamics tests failed!"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ All tests passed successfully!"
echo "=========================================="