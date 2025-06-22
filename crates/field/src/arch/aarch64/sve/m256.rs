// Copyright 2024-2025 Irreducible Inc.

//! SVE-optimized 256-bit vector type for AArch64

use std::{
    arch::aarch64::*,
    fmt::{Debug, Formatter, Result as FmtResult},
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

use super::M128;
use crate::underlier::{UnderlierType, UnderlierWithBitOps, WithUnderlier, ScaledUnderlier};

/// SVE-optimized 256-bit vector type (two 128-bit SVE vectors)
#[derive(Clone, Copy)]
pub struct M256 {
    pub lo: M128,
    pub hi: M128,
}

impl M256 {
    /// Create a new M256 from two M128 vectors
    #[inline]
    pub fn new(lo: M128, hi: M128) -> Self {
        Self { lo, hi }
    }

    /// Create M256 filled with zeros
    #[inline]
    pub fn zero() -> Self {
        Self {
            lo: M128::zero(),
            hi: M128::zero(),
        }
    }

    /// Create M256 filled with ones
    #[inline]
    pub fn ones() -> Self {
        Self {
            lo: M128::ones(),
            hi: M128::ones(),
        }
    }

    /// Create M256 from a single byte value (broadcast)
    #[inline]
    pub fn splat_byte(value: u8) -> Self {
        let vec = M128::splat_byte(value);
        Self { lo: vec, hi: vec }
    }

    /// Load M256 from memory
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        Self {
            lo: M128::load(ptr),
            hi: M128::load(ptr.add(16)),
        }
    }

    /// Store M256 to memory
    #[inline]
    pub unsafe fn store(self, ptr: *mut u8) {
        self.lo.store(ptr);
        self.hi.store(ptr.add(16));
    }

    /// Convert to byte array
    #[inline]
    pub fn to_bytes(self) -> [u8; 32] {
        let lo_bytes = self.lo.to_bytes();
        let hi_bytes = self.hi.to_bytes();
        let mut result = [0u8; 32];
        result[..16].copy_from_slice(&lo_bytes);
        result[16..].copy_from_slice(&hi_bytes);
        result
    }

    /// Create from byte array
    #[inline]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let mut lo_bytes = [0u8; 16];
        let mut hi_bytes = [0u8; 16];
        lo_bytes.copy_from_slice(&bytes[..16]);
        hi_bytes.copy_from_slice(&bytes[16..]);
        Self {
            lo: M128::from_bytes(lo_bytes),
            hi: M128::from_bytes(hi_bytes),
        }
    }

    /// SVE-optimized table lookup for both halves
    #[inline]
    pub unsafe fn table_lookup(self, table: &[u8; 256]) -> Self {
        Self {
            lo: self.lo.table_lookup(table),
            hi: self.hi.table_lookup(table),
        }
    }

    /// SVE-optimized horizontal reduction (XOR all bytes)
    #[inline]
    pub unsafe fn horizontal_xor(self) -> u8 {
        let lo_xor = self.lo.horizontal_xor();
        let hi_xor = self.hi.horizontal_xor();
        lo_xor ^ hi_xor
    }

    /// Check if all bytes are zero
    #[inline]
    pub fn is_zero(self) -> bool {
        self.lo.is_zero() && self.hi.is_zero()
    }

    /// SVE-optimized parallel operations on both halves
    #[inline]
    pub unsafe fn sve_parallel_op<F>(self, other: Self, op: F) -> Self
    where
        F: Fn(M128, M128) -> M128,
    {
        Self {
            lo: op(self.lo, other.lo),
            hi: op(self.hi, other.hi),
        }
    }
}

// Implement basic traits
impl Debug for M256 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let bytes = self.to_bytes();
        f.debug_tuple("M256")
            .field(&format_args!("{:02x?}", &bytes[..16]))
            .field(&format_args!("{:02x?}", &bytes[16..]))
            .finish()
    }
}

impl PartialEq for M256 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }
}

impl Eq for M256 {}

// Bitwise operations
impl BitXor for M256 {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            lo: self.lo ^ rhs.lo,
            hi: self.hi ^ rhs.hi,
        }
    }
}

impl BitXorAssign for M256 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl BitAnd for M256 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            lo: self.lo & rhs.lo,
            hi: self.hi & rhs.hi,
        }
    }
}

impl BitAndAssign for M256 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitOr for M256 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            lo: self.lo | rhs.lo,
            hi: self.hi | rhs.hi,
        }
    }
}

impl BitOrAssign for M256 {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl Not for M256 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        Self {
            lo: !self.lo,
            hi: !self.hi,
        }
    }
}

// Conversions
impl From<[u8; 32]> for M256 {
    #[inline]
    fn from(bytes: [u8; 32]) -> Self {
        Self::from_bytes(bytes)
    }
}

impl From<M256> for [u8; 32] {
    #[inline]
    fn from(value: M256) -> Self {
        value.to_bytes()
    }
}

impl From<(M128, M128)> for M256 {
    #[inline]
    fn from((lo, hi): (M128, M128)) -> Self {
        Self::new(lo, hi)
    }
}

impl From<M256> for (M128, M128) {
    #[inline]
    fn from(value: M256) -> Self {
        (value.lo, value.hi)
    }
}

// Implement UnderlierType trait
impl UnderlierType for M256 {
    const LOG_BITS: usize = 8; // log2(256)
}

// Implement basic underlier operations
impl UnderlierWithBitOps for M256 {
    const ZERO: Self = Self {
        lo: M128::ZERO,
        hi: M128::ZERO,
    };
    const ONE: Self = Self {
        lo: M128::ONE,
        hi: M128::ZERO,
    };

    #[inline]
    fn fill_with_bit(bit: usize) -> Self {
        if bit == 0 {
            Self::zero()
        } else {
            Self::ones()
        }
    }
}

impl WithUnderlier for M256 {
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