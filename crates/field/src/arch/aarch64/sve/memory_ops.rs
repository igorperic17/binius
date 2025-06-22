// Copyright 2024-2025 Irreducible Inc.

//! SVE-optimized memory operations for high-performance data movement

use std::arch::aarch64::*;

/// SVE-optimized memory copy with automatic vector length adaptation
#[inline]
pub unsafe fn sve_memcpy_adaptive(src: *const u8, dst: *mut u8, len: usize) {
    let mut offset = 0;
    
    // Use SVE's scalable nature to process as much data as possible per iteration
    while offset < len {
        let remaining = len - offset;
        let pg = svwhilelt_b8_u64(0, remaining as u64);
        
        // Load data using predicated load
        let data = svld1_u8(pg, src.add(offset));
        
        // Store data using predicated store
        svst1_u8(pg, dst.add(offset), data);
        
        // Advance by the actual vector length
        offset += svcntb() as usize;
    }
}

/// SVE-optimized memory set with pattern filling
#[inline]
pub unsafe fn sve_memset_pattern(dst: *mut u8, pattern: &[u8], len: usize) {
    if pattern.is_empty() {
        return;
    }
    
    let mut offset = 0;
    let pattern_len = pattern.len();
    
    // Create pattern vector by repeating the pattern
    let mut pattern_buf = [0u8; 256]; // Max SVE width / 8
    let vl_bytes = svcntb() as usize;
    
    // Fill pattern buffer with repeated pattern
    for i in 0..vl_bytes {
        pattern_buf[i] = pattern[i % pattern_len];
    }
    
    let pg_full = svptrue_b8();
    let pattern_vec = svld1_u8(pg_full, pattern_buf.as_ptr());
    
    while offset < len {
        let remaining = len - offset;
        let pg = svwhilelt_b8_u64(0, remaining as u64);
        
        svst1_u8(pg, dst.add(offset), pattern_vec);
        offset += svcntb() as usize;
    }
}

/// SVE-optimized memory compare operation
#[inline]
pub unsafe fn sve_memcmp(a: *const u8, b: *const u8, len: usize) -> bool {
    let mut offset = 0;
    
    while offset < len {
        let remaining = len - offset;
        let pg = svwhilelt_b8_u64(0, remaining as u64);
        
        let data_a = svld1_u8(pg, a.add(offset));
        let data_b = svld1_u8(pg, b.add(offset));
        
        let cmp = svcmpeq_u8(pg, data_a, data_b);
        
        // If any elements are not equal, return false
        if svptest_any(pg, svnot_z(pg, cmp)) {
            return false;
        }
        
        offset += svcntb() as usize;
    }
    
    true
}

/// SVE-optimized batch XOR operation for cryptographic operations
#[inline]
pub unsafe fn sve_batch_xor(
    inputs: &[*const u8],
    output: *mut u8,
    len: usize,
) {
    if inputs.is_empty() {
        return;
    }
    
    let mut offset = 0;
    
    while offset < len {
        let remaining = len - offset;
        let pg = svwhilelt_b8_u64(0, remaining as u64);
        
        // Load first input
        let mut result = svld1_u8(pg, inputs[0].add(offset));
        
        // XOR with remaining inputs
        for &input_ptr in inputs.iter().skip(1) {
            let input_data = svld1_u8(pg, input_ptr.add(offset));
            result = sveor_u8_z(pg, result, input_data);
        }
        
        // Store result
        svst1_u8(pg, output.add(offset), result);
        offset += svcntb() as usize;
    }
}

/// SVE-optimized gather operation for non-contiguous memory access
#[inline]
pub unsafe fn sve_gather_u8(
    base: *const u8,
    indices: &[usize],
    output: *mut u8,
    count: usize,
) {
    let mut processed = 0;
    
    while processed < count {
        let remaining = count - processed;
        let vl = svcntb() as usize;
        let chunk_size = remaining.min(vl);
        
        let pg = svwhilelt_b8_u64(0, chunk_size as u64);
        
        // Create index vector
        let mut idx_buf = [0u64; 32]; // Max possible SVE elements for 64-bit indices
        for i in 0..chunk_size {
            idx_buf[i] = indices[processed + i] as u64;
        }
        
        let pg64 = svwhilelt_b64_u64(0, chunk_size as u64);
        let idx_vec = svld1_u64(pg64, idx_buf.as_ptr());
        
        // Perform gather load
        let gathered = svld1_gather_u64index_u8(pg, base, idx_vec);
        
        // Store gathered data
        svst1_u8(pg, output.add(processed), gathered);
        
        processed += chunk_size;
    }
}

/// SVE-optimized scatter operation for non-contiguous memory access
#[inline]
pub unsafe fn sve_scatter_u8(
    base: *mut u8,
    indices: &[usize],
    data: *const u8,
    count: usize,
) {
    let mut processed = 0;
    
    while processed < count {
        let remaining = count - processed;
        let vl = svcntb() as usize;
        let chunk_size = remaining.min(vl);
        
        let pg = svwhilelt_b8_u64(0, chunk_size as u64);
        
        // Create index vector
        let mut idx_buf = [0u64; 32]; // Max possible SVE elements for 64-bit indices
        for i in 0..chunk_size {
            idx_buf[i] = indices[processed + i] as u64;
        }
        
        let pg64 = svwhilelt_b64_u64(0, chunk_size as u64);
        let idx_vec = svld1_u64(pg64, idx_buf.as_ptr());
        
        // Load data to scatter
        let scatter_data = svld1_u8(pg, data.add(processed));
        
        // Perform scatter store
        svst1_scatter_u64index_u8(pg, base, idx_vec, scatter_data);
        
        processed += chunk_size;
    }
}

/// SVE-optimized transpose operation for matrix data
#[inline]
pub unsafe fn sve_transpose_8x8(input: &[[u8; 8]; 8], output: &mut [[u8; 8]; 8]) {
    let pg = svptrue_b8();
    
    // Load 8 rows as vectors
    let mut rows = [svdup_n_u8(0); 8];
    for i in 0..8 {
        rows[i] = svld1_u8(pg, input[i].as_ptr());
    }
    
    // Perform transpose using SVE operations
    // This is a simplified version - real implementation would use
    // more sophisticated SVE transpose patterns
    for i in 0..8 {
        let mut col_data = [0u8; 8];
        for j in 0..8 {
            // Extract element from row j, column i
            let element = svlastb_u8(svwhilelt_b8_u64(i as u64, (i + 1) as u64), rows[j]);
            col_data[j] = element;
        }
        
        // Store transposed column as row
        let col_vec = svld1_u8(pg, col_data.as_ptr());
        svst1_u8(pg, output[i].as_mut_ptr(), col_vec);
    }
}

/// SVE-optimized bit reverse operation
#[inline]
pub unsafe fn sve_reverse_bits(input: *const u8, output: *mut u8, len: usize) {
    let mut offset = 0;
    
    // Bit reverse lookup table
    const BIT_REVERSE_TABLE: [u8; 256] = [
        0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
        0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
        0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
        0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
        0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
        0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
        0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
        0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
        0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
        0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
        0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
        0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
        0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
        0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
        0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
        0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
        0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
        0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
        0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
        0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
        0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
        0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
        0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
        0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
        0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
        0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
        0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
        0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
        0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
        0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
        0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
        0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
    ];
    
    while offset < len {
        let remaining = len - offset;
        let pg = svwhilelt_b8_u64(0, remaining as u64);
        
        let data = svld1_u8(pg, input.add(offset));
        
        // Use table lookup for bit reversal
        let reversed = sve_table_lookup_u8(data, &BIT_REVERSE_TABLE);
        
        svst1_u8(pg, output.add(offset), reversed);
        offset += svcntb() as usize;
    }
}

/// SVE table lookup helper function
#[inline]
unsafe fn sve_table_lookup_u8(indices: svuint8_t, table: &[u8; 256]) -> svuint8_t {
    let pg = svptrue_b8();
    let mut result = svdup_n_u8(0);

    // Split the 256-entry table into 16 chunks of 16 entries each
    for chunk_idx in 0..16 {
        let chunk_start = chunk_idx * 16;
        let table_chunk = svld1_u8(svptrue_b8(), table.as_ptr().add(chunk_start));
        
        let high_bits = svlsr_n_u8_z(pg, indices, 4);
        let chunk_mask = svcmpeq_n_u8(pg, high_bits, chunk_idx as u8);
        
        let low_bits = svand_n_u8_z(pg, indices, 0x0F);
        let lookup_result = svtbl_u8(table_chunk, low_bits);
        
        result = svsel_u8(chunk_mask, lookup_result, result);
    }

    result
} 