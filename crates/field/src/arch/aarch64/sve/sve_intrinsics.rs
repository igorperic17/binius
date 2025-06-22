// Copyright 2024-2025 Irreducible Inc.

//! Safe wrappers around ARM SVE intrinsics for field arithmetic operations

use std::arch::aarch64::*;

/// SVE-optimized vector addition for 8-bit elements
#[inline]
pub unsafe fn sve_add_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svuint8_t {
    svadd_u8_z(pg, a, b)
}

/// SVE-optimized vector subtraction for 8-bit elements
#[inline]
pub unsafe fn sve_sub_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svuint8_t {
    svsub_u8_z(pg, a, b)
}

/// SVE-optimized vector XOR for 8-bit elements
#[inline]
pub unsafe fn sve_eor_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svuint8_t {
    sveor_u8_z(pg, a, b)
}

/// SVE-optimized vector AND for 8-bit elements
#[inline]
pub unsafe fn sve_and_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svuint8_t {
    svand_u8_z(pg, a, b)
}

/// SVE-optimized vector OR for 8-bit elements
#[inline]
pub unsafe fn sve_orr_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svuint8_t {
    svorr_u8_z(pg, a, b)
}

/// SVE-optimized vector compare equal for 8-bit elements
#[inline]
pub unsafe fn sve_cmpeq_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svbool_t {
    svcmpeq_u8(pg, a, b)
}

/// SVE-optimized table lookup for 8-bit elements
#[inline]
pub unsafe fn sve_tbl_u8(data: svuint8_t, indices: svuint8_t) -> svuint8_t {
    svtbl_u8(data, indices)
}

/// SVE-optimized gather load for 8-bit elements
#[inline]
pub unsafe fn sve_ld1_gather_u8(pg: svbool_t, base: *const u8, indices: svuint8_t) -> svuint8_t {
    svld1_gather_u8index_u8(pg, base, indices)
}

/// SVE-optimized scatter store for 8-bit elements
#[inline]
pub unsafe fn sve_st1_scatter_u8(pg: svbool_t, base: *mut u8, indices: svuint8_t, data: svuint8_t) {
    svst1_scatter_u8index_u8(pg, base, indices, data)
}

/// SVE-optimized contiguous load for 8-bit elements
#[inline]
pub unsafe fn sve_ld1_u8(pg: svbool_t, base: *const u8) -> svuint8_t {
    svld1_u8(pg, base)
}

/// SVE-optimized contiguous store for 8-bit elements
#[inline]
pub unsafe fn sve_st1_u8(pg: svbool_t, base: *mut u8, data: svuint8_t) {
    svst1_u8(pg, base, data)
}

/// SVE-optimized vector duplicate for 8-bit elements
#[inline]
pub unsafe fn sve_dup_u8(value: u8) -> svuint8_t {
    svdup_n_u8(value)
}

/// SVE-optimized vector select based on predicate
#[inline]
pub unsafe fn sve_sel_u8(pg: svbool_t, a: svuint8_t, b: svuint8_t) -> svuint8_t {
    svsel_u8(pg, a, b)
}

/// SVE-optimized vector multiply for 8-bit elements (where available)
#[cfg(target_feature = "sve2")]
#[inline]
pub unsafe fn sve_mul_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svuint8_t {
    svmul_u8_z(pg, a, b)
}

/// SVE-optimized vector multiply-add for 8-bit elements (where available)
#[cfg(target_feature = "sve2")]
#[inline]
pub unsafe fn sve_mla_u8(a: svuint8_t, b: svuint8_t, c: svuint8_t, pg: svbool_t) -> svuint8_t {
    svmla_u8_z(pg, a, b, c)
}

/// SVE-optimized polynomial multiplication (where available)
#[cfg(target_feature = "sve2")]
#[inline]
pub unsafe fn sve_pmul_u8(a: svuint8_t, b: svuint8_t, pg: svbool_t) -> svuint16_t {
    svpmul_u8(a, b)
}

/// Higher-level operations for field arithmetic

/// SVE-optimized batch XOR operation
#[inline]
pub unsafe fn sve_batch_xor_u8(
    inputs: &[u8],
    outputs: &mut [u8],
    other: &[u8],
) {
    let len = inputs.len().min(outputs.len()).min(other.len());
    let mut i = 0;
    
    while i < len {
        let pg = svwhilelt_b8_u64(i as u64, len as u64);
        let a = svld1_u8(pg, inputs.as_ptr().add(i));
        let b = svld1_u8(pg, other.as_ptr().add(i));
        let result = sveor_u8_z(pg, a, b);
        svst1_u8(pg, outputs.as_mut_ptr().add(i), result);
        i += svcntb() as usize; // Increment by vector length in bytes
    }
}

/// SVE-optimized batch table lookup operation
#[inline]
pub unsafe fn sve_batch_table_lookup_u8(
    inputs: &[u8],
    outputs: &mut [u8],
    table: &[u8; 256],
) {
    let len = inputs.len().min(outputs.len());
    let mut i = 0;
    
    // Create lookup tables as SVE vectors
    let table_chunks: Vec<svuint8_t> = (0..16)
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * 16;
            svld1_u8(svptrue_b8(), table.as_ptr().add(chunk_start))
        })
        .collect();
    
    while i < len {
        let pg = svwhilelt_b8_u64(i as u64, len as u64);
        let indices = svld1_u8(pg, inputs.as_ptr().add(i));
        
        // Perform table lookup using SVE table operations
        // This is a simplified version - real implementation would handle full 256-entry table
        let high_bits = svlsr_n_u8_z(pg, indices, 4);
        let low_bits = svand_n_u8_z(pg, indices, 0x0F);
        
        // Use the appropriate table chunk based on high bits
        let mut result = svdup_n_u8(0);
        for (chunk_idx, &table_chunk) in table_chunks.iter().enumerate() {
            let mask = svcmpeq_n_u8(pg, high_bits, chunk_idx as u8);
            let lookup_result = svtbl_u8(table_chunk, low_bits);
            result = svsel_u8(mask, lookup_result, result);
        }
        
        svst1_u8(pg, outputs.as_mut_ptr().add(i), result);
        i += svcntb() as usize; // Increment by vector length in bytes
    }
}

/// SVE-optimized memory copy operation
#[inline]
pub unsafe fn sve_memcpy(src: *const u8, dst: *mut u8, len: usize) {
    let mut i = 0;
    
    while i < len {
        let pg = svwhilelt_b8_u64(i as u64, len as u64);
        let data = svld1_u8(pg, src.add(i));
        svst1_u8(pg, dst.add(i), data);
        i += svcntb() as usize; // Increment by vector length in bytes
    }
}

/// SVE-optimized memory set operation
#[inline]
pub unsafe fn sve_memset(dst: *mut u8, value: u8, len: usize) {
    let mut i = 0;
    let fill_value = svdup_n_u8(value);
    
    while i < len {
        let pg = svwhilelt_b8_u64(i as u64, len as u64);
        svst1_u8(pg, dst.add(i), fill_value);
        i += svcntb() as usize; // Increment by vector length in bytes
    }
} 