#!/bin/bash
echo "=== Testing Debug Viz Modes ==="
echo ""

echo "1. Testing DEFAULT MODE (no arguments)..."
echo "Command: cargo run --features viz --bin debug_viz"
echo "Press Ctrl-C after seeing the window to continue to next test"
echo "Expected: Window with 3 spheres in white"
read -p "Press Enter to run..." 
cargo run --features viz --bin debug_viz

echo ""
echo "2. Testing INSPECT MODE (oracle only)..."
echo "Command: cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy"
echo "Press Ctrl-C after seeing the window to continue to next test"
echo "Expected: Window with 10 spheres in white"
read -p "Press Enter to run..."
cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy

echo ""
echo "3. Testing DIFF MODE (both files)..."
echo "Command: cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy --gpu tests/oracle_dump.npy"
echo "Press Ctrl-C after seeing the window"
echo "Expected: Window with 10 spheres in a single color (since both are identical)"
read -p "Press Enter to run..."
cargo run --features viz --bin debug_viz -- --oracle tests/oracle_dump.npy --gpu tests/oracle_dump.npy

echo ""
echo "All modes tested!"