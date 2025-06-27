#!/bin/bash
echo "Running capture test..."
cargo run --features viz --release --bin debug_viz -- \
    --oracle tests/oracle_dump.npy \
    --record test_capture.mp4 \
    --duration 1 \
    --fps 5

echo "Checking file size..."
ls -lh test_capture.mp4

echo "Cleaning up..."
rm -f test_capture.mp4