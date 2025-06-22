#!/bin/bash

# Script to build and run binary_zerocheck benchmark on Ubuntu ARM64 with SVE2 optimizations
# Run this script on your Ubuntu ARM64 machine

set -e

echo "Setting up environment for ARM64 SVE2 optimizations..."

# Detect available SVE features
sve_features=""
if cat /proc/cpuinfo | grep -q "sve"; then
    sve_features="+sve"
    echo "SVE detected"
fi

if cat /proc/cpuinfo | grep -q "sve2"; then
    sve_features="+sve2,+sve"
    echo "SVE2 detected"
fi

# Set comprehensive SVE2 target features with maximum optimization
export RUSTFLAGS="-C target-feature=${sve_features},+neon,+aes,+crc,+crypto,+dotprod,+fp16,+rcpc,+lse -C target-cpu=native -C opt-level=3"

# Enable nightly features for maximum optimization
export CARGO_CFG_TARGET_FEATURE="sve2,sve,neon,aes"

echo "Building benchmarks with comprehensive ARM optimizations..."
echo "RUSTFLAGS: $RUSTFLAGS"

# Clean previous builds to ensure fresh compilation with new flags
cargo clean

# Build with maximum optimization
RUSTFLAGS="$RUSTFLAGS" cargo build --release --bench binary_zerocheck

# Run the benchmark
echo "Running benchmark with SVE optimizations..."
RUSTFLAGS="$RUSTFLAGS" cargo bench --bench binary_zerocheck

echo ""
echo "Benchmark completed!"
echo ""
echo "To compare performance, you can also run:"
echo "1. With SVE2 disabled: RUSTFLAGS=\"-C target-feature=-sve2\" cargo bench --bench binary_zerocheck"
echo "2. With all SIMD disabled: RUSTFLAGS=\"-C target-feature=-sve2,-sve,-neon\" cargo bench --bench binary_zerocheck"
echo "3. With only NEON: RUSTFLAGS=\"-C target-feature=-sve2,-sve,+neon,+aes\" cargo bench --bench binary_zerocheck"

# Additional optimizations for production builds
echo ""
echo "For maximum production performance, consider:"
echo "export RUSTFLAGS=\"-C target-feature=${sve_features},+neon,+aes,+crc,+crypto,+dotprod,+fp16,+rcpc,+lse -C target-cpu=native -C opt-level=3 -C codegen-units=1 -C lto=fat\"" 