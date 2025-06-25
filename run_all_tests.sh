#!/bin/bash
# Run all physics engine tests

echo "========================================="
echo "Running Physics Engine Test Suite"
echo "========================================="
echo

# Run Rust unit tests
echo "1. Running Rust unit tests..."
cargo test --lib
if [ $? -ne 0 ]; then
    echo "❌ Rust unit tests failed!"
    exit 1
fi
echo "✅ Rust unit tests passed!"
echo

# Run Python tests
echo "2. Running Python reference tests..."

echo "   - Integrator tests..."
python3 tests/test_integrator.py
if [ $? -ne 0 ]; then
    echo "❌ Integrator tests failed!"
    exit 1
fi

echo "   - SDF tests..."
python3 tests/test_sdf.py
if [ $? -ne 0 ]; then
    echo "❌ SDF tests failed!"
    exit 1
fi

echo "   - Broadphase tests..."
python3 tests/test_broadphase.py
if [ $? -ne 0 ]; then
    echo "❌ Broadphase tests failed!"
    exit 1
fi

echo "✅ All Python tests passed!"
echo

# Run GPU integration tests
echo "3. Running GPU integration tests..."

echo "   - Physics integration test..."
cargo run --bin test_runner > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Physics integration test failed!"
    exit 1
fi
echo "   ✅ Physics integration test passed!"

echo "   - SDF GPU test..."
cargo run --bin test_sdf > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ SDF GPU test failed!"
    exit 1
fi
echo "   ✅ SDF GPU test passed!"

echo "   - Contact solver test..."
cargo run --bin test_contact_solver > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Contact solver test failed!"
    exit 1
fi
echo "   ✅ Contact solver test passed!"

echo "   - Broadphase grid test..."
cargo run --bin test_broadphase_grid > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Broadphase grid test failed!"
    exit 1
fi
echo "   ✅ Broadphase grid test passed!"

echo
echo "========================================="
echo "✅ All tests passed successfully!"
echo "========================================="
echo
echo "To run benchmarks:"
echo "  cargo run --release --bin benchmark"
echo "  cargo run --release --bin benchmark_full"
echo
echo "To run demo:"
echo "  cargo run --bin demo_simple"