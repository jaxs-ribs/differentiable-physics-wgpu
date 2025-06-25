#!/bin/bash
# Comprehensive test runner for Physics Core engine
# This script runs ALL tests including unit, integration, fuzz, and stress tests

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Track timing
START_TIME=$(date +%s)

echo "================================================"
echo "Physics Core Comprehensive Test Suite"
echo "================================================"
echo "Started at: $(date)"
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

# Run additional comprehensive tests
echo "4. Running comprehensive Python tests..."

echo "   - Energy conservation tests..."
python3 tests/test_energy.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Energy tests failed!${NC}"
    exit 1
fi
echo -e "   ${GREEN}✅ Energy conservation verified${NC}"

echo "   - SDF property fuzz tests..."
python3 tests/test_sdf_fuzz.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ SDF fuzz tests failed!${NC}"
    exit 1
fi
echo -e "   ${GREEN}✅ SDF properties verified (1000+ tests)${NC}"

echo "   - Stability stress tests..."
echo -e "   ${YELLOW}(This may take ~30 seconds...)${NC}"
python3 tests/test_stability_stress.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Stability tests failed!${NC}"
    exit 1
fi
echo -e "   ${GREEN}✅ Stability verified (5000 bodies, 30s)${NC}"

echo

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "================================================"
echo -e "${GREEN}✅ All tests passed successfully!${NC}"
echo "================================================"
echo "Total time: ${DURATION} seconds"
echo
echo "Test coverage:"
echo "  • Unit tests: ✓"
echo "  • Integration tests: ✓"
echo "  • GPU tests: ✓"
echo "  • Fuzz tests: ✓"
echo "  • Stress tests: ✓"
echo
echo "Next steps:"
echo "  Run benchmarks:  ./bench.sh"
echo "  Run demos:       ./demo.sh --list"
echo "  Check code:      ./dev.sh check"