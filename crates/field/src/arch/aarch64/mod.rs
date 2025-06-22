// Copyright 2024-2025 Irreducible Inc.

use cfg_if::cfg_if;

cfg_if! {
	if #[cfg(all(target_feature = "sve2", target_feature = "aes"))] {
		// SVE2 provides the best ARM performance with enhanced vector operations
		pub mod sve;
		pub(super) mod m128;
		pub mod simd_arithmetic;

		pub use sve::{packed_128, packed_256, packed_512, packed_aes_128, packed_aes_256, packed_aes_512, packed_polyval_128, packed_polyval_256, packed_polyval_512};
		mod packed_macros;
	} else if #[cfg(all(target_feature = "sve", target_feature = "aes"))] {
		// SVE (base) still provides significant benefits over NEON
		pub mod sve;
		pub(super) mod m128;
		pub mod simd_arithmetic;

		pub use sve::{packed_128, packed_256, packed_aes_128, packed_aes_256, packed_polyval_128, packed_polyval_256};
		pub use super::portable::{packed_512, packed_aes_512, packed_polyval_512};
		mod packed_macros;
	} else if #[cfg(all(target_feature = "neon", target_feature = "aes"))] {
		// Fallback to NEON when SVE is not available
		pub(super) mod m128;
		pub mod simd_arithmetic;

		pub mod packed_128;
		pub mod packed_aes_128;
		pub mod packed_polyval_128;
		mod packed_macros;
	} else {
		// Pure portable implementation
		pub use super::portable::packed_128;
		pub use super::portable::packed_aes_128;
		pub use super::portable::packed_polyval_128;
	}
}
