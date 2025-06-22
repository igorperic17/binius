// Copyright 2024-2025 Irreducible Inc.

//! SVE (Scalable Vector Extensions) optimized implementations for AArch64
//! 
//! This module provides highly optimized field arithmetic operations using ARM's SVE,
//! which offers scalable vector operations that can adapt to different CPU implementations
//! with vector lengths from 128-bit to 2048-bit.

use cfg_if::cfg_if;

pub mod sve_intrinsics;
pub mod sve_arithmetic;
pub mod memory_ops;

cfg_if! {
    if #[cfg(target_feature = "sve2")] {
        // SVE2 enhanced operations
        pub mod sve2_arithmetic;
        pub use sve2_arithmetic::*;
    }
}

// SVE vector types
pub mod m128;
pub mod m256;
pub mod m512;

// Packed field implementations
pub mod packed_128;
pub mod packed_256;
pub mod packed_512;

pub mod packed_aes_128;
pub mod packed_aes_256;
pub mod packed_aes_512;

pub mod packed_polyval_128;
pub mod packed_polyval_256;
pub mod packed_polyval_512;

// Re-exports for easier access
pub use m128::M128;
pub use m256::M256;
pub use m512::M512;

// Constants for SVE optimization
pub const SVE_MIN_VECTOR_LEN: usize = 128; // Minimum SVE vector length in bits
pub const SVE_MAX_VECTOR_LEN: usize = 2048; // Maximum SVE vector length in bits

/// Get the current SVE vector length in bits at runtime
#[inline]
pub fn get_sve_vector_length_bits() -> usize {
    unsafe {
        let mut vl: u64;
        core::arch::asm!(
            "rdvl {}, #1",
            out(reg) vl,
            options(nomem, nostack, preserves_flags)
        );
        (vl * 8) as usize // Convert from bytes to bits
    }
}

/// Get the current SVE vector length in bytes at runtime
#[inline]
pub fn get_sve_vector_length_bytes() -> usize {
    get_sve_vector_length_bits() / 8
}

/// Check if SVE2 is available at runtime
#[inline]
pub fn has_sve2() -> bool {
    cfg!(target_feature = "sve2")
}

/// SVE predicates for different element sizes
pub mod predicates {
    use std::arch::aarch64::*;

    /// Generate a predicate for all elements (ptrue)
    #[inline]
    pub unsafe fn ptrue_b8() -> svbool_t {
        svptrue_b8()
    }

    #[inline]
    pub unsafe fn ptrue_b16() -> svbool_t {
        svptrue_b16()
    }

    #[inline]
    pub unsafe fn ptrue_b32() -> svbool_t {
        svptrue_b32()
    }

    #[inline]
    pub unsafe fn ptrue_b64() -> svbool_t {
        svptrue_b64()
    }

    /// Generate a first-n predicate (whilelt)
    #[inline]
    pub unsafe fn whilelt_b8(start: u64, end: u64) -> svbool_t {
        svwhilelt_b8_u64(start, end)
    }

    #[inline]
    pub unsafe fn whilelt_b16(start: u64, end: u64) -> svbool_t {
        svwhilelt_b16_u64(start, end)
    }

    #[inline]
    pub unsafe fn whilelt_b32(start: u64, end: u64) -> svbool_t {
        svwhilelt_b32_u64(start, end)
    }

    #[inline]
    pub unsafe fn whilelt_b64(start: u64, end: u64) -> svbool_t {
        svwhilelt_b64_u64(start, end)
    }
} 