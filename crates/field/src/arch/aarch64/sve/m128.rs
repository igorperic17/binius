// Copyright 2024-2025 Irreducible Inc.

//! SVE-optimized 128-bit vector type for AArch64

use std::{
    arch::aarch64::*,
    fmt::{Debug, Formatter, Result as FmtResult},
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

use crate::underlier::{UnderlierType, UnderlierWithBitOps, WithUnderlier};

/// SVE-optimized 128-bit vector type
#[derive(Clone, Copy)]
pub struct M128(pub(super) svuint8_t);

impl M128 {
    /// Create a new M128 from raw SVE vector
    #[inline]
    pub fn new(data: svuint8_t) -> Self {
        Self(data)
    }

    /// Get the underlying SVE vector
    #[inline]
    pub fn inner(self) -> svuint8_t {
        self.0
    }

    /// Create M128 filled with zeros
    #[inline]
    pub fn zero() -> Self {
        unsafe { Self(svdup_n_u8(0)) }
    }

    /// Create M128 filled with ones
    #[inline]
    pub fn ones() -> Self {
        unsafe { Self(svdup_n_u8(0xFF)) }
    }

    /// Create M128 from a single byte value (broadcast)
    #[inline]
    pub fn splat_byte(value: u8) -> Self {
        unsafe { Self(svdup_n_u8(value)) }
    }

    /// Load M128 from memory
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        let pg = svptrue_b8();
        Self(svld1_u8(pg, ptr))
    }

    /// Store M128 to memory
    #[inline]
    pub unsafe fn store(self, ptr: *mut u8) {
        let pg = svptrue_b8();
        svst1_u8(pg, ptr, self.0);
    }

    /// Convert to byte array
    #[inline]
    pub fn to_bytes(self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        unsafe {
            let pg = svptrue_b8();
            svst1_u8(pg, bytes.as_mut_ptr(), self.0);
        }
        bytes
    }

    /// Create from byte array
    #[inline]
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        unsafe {
            let pg = svptrue_b8();
            Self(svld1_u8(pg, bytes.as_ptr()))
        }
    }

    /// Convert to u128
    #[inline]
    pub fn to_u128(self) -> u128 {
        u128::from_le_bytes(self.to_bytes())
    }

    /// Create from u128
    #[inline]
    pub fn from_u128(value: u128) -> Self {
        Self::from_bytes(value.to_le_bytes())
    }

    /// SVE-optimized table lookup
    #[inline]
    pub unsafe fn table_lookup(self, table: &[u8; 256]) -> Self {
        // Split table into 16 chunks for SVE table operations
        let pg = svptrue_b8();
        let mut result = svdup_n_u8(0);

        for chunk_idx in 0..16 {
            let chunk_start = chunk_idx * 16;
            let table_chunk = svld1_u8(svptrue_b8(), table.as_ptr().add(chunk_start));
            
            let high_bits = svlsr_n_u8_z(pg, self.0, 4);
            let chunk_mask = svcmpeq_n_u8(pg, high_bits, chunk_idx as u8);
            
            let low_bits = svand_n_u8_z(pg, self.0, 0x0F);
            let lookup_result = svtbl_u8(table_chunk, low_bits);
            
            result = svsel_u8(chunk_mask, lookup_result, result);
        }

        Self(result)
    }

    /// SVE-optimized horizontal reduction (XOR all bytes)
    #[inline]
    pub unsafe fn horizontal_xor(self) -> u8 {
        let pg = svptrue_b8();
        // SVE doesn't have direct horizontal XOR, so we implement it
        let mut result = 0u8;
        let bytes = self.to_bytes();
        for &byte in bytes.iter() {
            result ^= byte;
        }
        result
    }

    /// SVE-optimized byte shuffle
    #[inline]
    pub unsafe fn shuffle_bytes(self, mask: [u8; 16]) -> Self {
        let pg = svptrue_b8();
        let mask_vec = svld1_u8(pg, mask.as_ptr());
        Self(svtbl_u8(self.0, mask_vec))
    }

    /// Check if all bytes are zero
    #[inline]
    pub fn is_zero(self) -> bool {
        unsafe {
            let pg = svptrue_b8();
            let zero = svdup_n_u8(0);
            let cmp = svcmpeq_u8(pg, self.0, zero);
            svptest_any(pg, cmp) == false // If no differences, all are zero
        }
    }
}

// Implement basic traits
impl Debug for M128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let bytes = unsafe { self.to_bytes() };
        f.debug_tuple("M128")
            .field(&format_args!("{:02x?}", bytes))
            .finish()
    }
}

impl PartialEq for M128 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let pg = svptrue_b8();
            let cmp = svcmpeq_u8(pg, self.0, other.0);
            !svptest_any(pg, svnot_z(pg, cmp)) // All elements equal
        }
    }
}

impl Eq for M128 {}

// Bitwise operations using SVE
impl BitXor for M128 {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe {
            let pg = svptrue_b8();
            Self(sveor_u8_z(pg, self.0, rhs.0))
        }
    }
}

impl BitXorAssign for M128 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl BitAnd for M128 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe {
            let pg = svptrue_b8();
            Self(svand_u8_z(pg, self.0, rhs.0))
        }
    }
}

impl BitAndAssign for M128 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitOr for M128 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe {
            let pg = svptrue_b8();
            Self(svorr_u8_z(pg, self.0, rhs.0))
        }
    }
}

impl BitOrAssign for M128 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl Not for M128 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        unsafe {
            let pg = svptrue_b8();
            Self(svnot_u8_z(pg, self.0))
        }
    }
}

// Conversions
impl From<u128> for M128 {
    #[inline]
    fn from(value: u128) -> Self {
        Self::from_u128(value)
    }
}

impl From<M128> for u128 {
    #[inline]
    fn from(value: M128) -> Self {
        value.to_u128()
    }
}

impl From<[u8; 16]> for M128 {
    #[inline]
    fn from(bytes: [u8; 16]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl From<M128> for [u8; 16] {
    #[inline]
    fn from(value: M128) -> Self {
        value.to_bytes()
    }
}

// Implement UnderlierType trait
impl UnderlierType for M128 {
    const LOG_BITS: usize = 7; // log2(128)
}

// Implement basic underlier operations
impl UnderlierWithBitOps for M128 {
    const ZERO: Self = unsafe { Self(std::mem::transmute([0u8; 16])) };
    const ONE: Self = unsafe { Self(std::mem::transmute([1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])) };

    #[inline]
    fn fill_with_bit(bit: usize) -> Self {
        if bit == 0 {
            Self::zero()
        } else {
            Self::ones()
        }
    }
}

impl WithUnderlier for M128 {
    type Underlier = Self;

    #[inline]
    fn to_underlier(self) -> Self::Underlier {
        self
    }

    #[inline]
    fn to_underlier_ref(&self) -> &Self::Underlier {
        self
    }

    #[inline]
    fn to_underlier_ref_mut(&mut self) -> &mut Self::Underlier {
        self
    }

    #[inline]
    fn mutate_underlier<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce(&mut Self::Underlier) -> T,
    {
        f(self)
    }

    #[inline]
    fn from_underlier(underlier: Self::Underlier) -> Self {
        underlier
    }
} 