// Copyright 2024-2025 Irreducible Inc.

use std::sync::Arc;

use binius_field::{ExtensionField, PackedExtension, PackedField, TowerField};
use binius_hal::{make_portable_backend, CpuBackend};
use binius_math::{
	BinarySubspace, EvaluationDomain, EvaluationOrder, IsomorphicEvaluationDomainFactory,
	MLEDirectAdapter, MultilinearPoly,
};
use binius_utils::{bail, sorting::is_sorted_ascending};

use crate::{
	fiat_shamir::{CanSample, Challenger},
	protocols::sumcheck::{
		immediate_switchover_heuristic,
		prove::{
			front_loaded, logging::FoldLowDimensionsData, RegularSumcheckProver, SumcheckProver,
		},
		zerocheck::{
			lagrange_evals_multilinear_extension, univariatizing_reduction_claim,
			BatchZerocheckOutput, ZerocheckRoundEvals,
		},
		BatchSumcheckOutput, Error,
	},
	transcript::ProverTranscript,
};

/// A zerocheck prover interface.
///
/// The primary reason for providing this logic via a trait is the ability to type erase univariate
/// round small fields, which may differ between the provers, and to decouple the batch prover
/// implementation from the relatively complex type signatures of the individual provers.
///
/// The batch prover must obey a specific sequence of calls: [`Self::execute_univariate_round`]
/// should be followed by [`Self::fold_univariate_round`], and then
/// [`Self::project_to_skipped_variables`]. Getters [`Self::n_vars`] and [`Self::domain_size`] are
/// used for alignment and maximal domain size calculation required by the Lagrange representation
/// of the univariate round polynomial. Folding univariate round results in a [`SumcheckProver`]
/// instance that can be driven to completion to prove the remaining multilinear rounds.
///
/// This trait is object-safe.
pub trait ZerocheckProver<'a, P: PackedField> {
	/// The number of variables in the multivariate polynomial.
	fn n_vars(&self) -> usize;

	/// Maximal required Lagrange domain size among compositions in this prover.
	///
	/// Returns `None` if the current prover state doesn't contain information about the domain
	/// size.
	fn domain_size(&self, skip_rounds: usize) -> Option<usize>;

	/// Computes the prover message for the univariate round as a univariate polynomial.
	///
	/// The prover message mixes the univariate polynomials of the underlying composites using
	/// the same approach as [`SumcheckProver::execute`].
	///
	/// Unlike multilinear rounds, the returned univariate is not in monomial basis but in
	/// Lagrange basis.
	fn execute_univariate_round(
		&mut self,
		skip_rounds: usize,
		max_domain_size: usize,
		batch_coeff: P::Scalar,
	) -> Result<ZerocheckRoundEvals<P::Scalar>, Error>;

	/// Folds into a regular multilinear prover for the remaining rounds.
	fn fold_univariate_round(
		&mut self,
		challenge: P::Scalar,
	) -> Result<Box<dyn SumcheckProver<P::Scalar> + 'a>, Error>;

	/// Projects witness onto the "skipped" variables for the univariatizing reduction.
	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[P::Scalar],
	) -> Result<Vec<Arc<dyn MultilinearPoly<P> + Send + Sync>>, Error>;
}

// NB: auto_impl does not currently handle ?Sized bound on Box<Self> receivers correctly.
impl<'a, P: PackedField, Prover: ZerocheckProver<'a, P> + ?Sized> ZerocheckProver<'a, P>
	for Box<Prover>
{
	fn n_vars(&self) -> usize {
		(**self).n_vars()
	}

	fn domain_size(&self, skip_rounds: usize) -> Option<usize> {
		(**self).domain_size(skip_rounds)
	}

	fn execute_univariate_round(
		&mut self,
		skip_rounds: usize,
		max_domain_size: usize,
		batch_coeff: P::Scalar,
	) -> Result<ZerocheckRoundEvals<P::Scalar>, Error> {
		(**self).execute_univariate_round(skip_rounds, max_domain_size, batch_coeff)
	}

	fn fold_univariate_round(
		&mut self,
		challenge: P::Scalar,
	) -> Result<Box<dyn SumcheckProver<P::Scalar> + 'a>, Error> {
		(**self).fold_univariate_round(challenge)
	}

	fn project_to_skipped_variables(
		self: Box<Self>,
		challenges: &[P::Scalar],
	) -> Result<Vec<Arc<dyn MultilinearPoly<P> + Send + Sync>>, Error> {
		(*self).project_to_skipped_variables(challenges)
	}
}

fn univariatizing_reduction_prover<F, FDomain, P>(
	mut projected_multilinears: Vec<Arc<dyn MultilinearPoly<P> + Send + Sync>>,
	skip_rounds: usize,
	univariatized_multilinear_evals: Vec<Vec<F>>,
	univariate_challenge: F,
	backend: &'_ CpuBackend,
) -> Result<impl SumcheckProver<F> + '_, Error>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: TowerField,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
{
	let sumcheck_claim =
		univariatizing_reduction_claim(skip_rounds, &univariatized_multilinear_evals)?;

	let subspace =
		BinarySubspace::<FDomain::Canonical>::with_dim(skip_rounds)?.isomorphic::<FDomain>();
	let ntt_domain = EvaluationDomain::from_points(subspace.iter().collect::<Vec<_>>(), false)?;

	projected_multilinears.push(
		MLEDirectAdapter::from(lagrange_evals_multilinear_extension(
			&ntt_domain,
			univariate_challenge,
		)?)
		.upcast_arc_dyn(),
	);

	// REVIEW: all multilins are large field, we could benefit from "no switchover" constructor, but
	// this sumcheck         is very small anyway.
	let prover = RegularSumcheckProver::<FDomain, P, _, _, _>::new(
		EvaluationOrder::HighToLow,
		projected_multilinears,
		sumcheck_claim.composite_sums().iter().cloned(),
		IsomorphicEvaluationDomainFactory::<FDomain::Canonical>::default(),
		immediate_switchover_heuristic,
		backend,
	)?;

	Ok(prover)
}

use binius_field::AESTowerField128b;
/// Prove a batched zerocheck protocol execution.
///
/// See the [`batch_verify_zerocheck`](`super::super::batch_verify_zerocheck`) docstring for
/// a detailed description of the zerocheck reduction stages. The `provers` in this invocation
/// should be provided in the same order as the corresponding claims during verification.
///
/// Zerocheck challenges (`max_n_vars - skip_rounds` of them) are to be sampled right before this
/// call and used for [`ZerocheckProver`] instances creation (most likely via calls to
/// [`ZerocheckProverImpl::new`](`super::zerocheck::ZerocheckProverImpl::new`))
#[allow(clippy::type_complexity)]
pub fn batch_prove<'a, F, FDomain, P, Prover, Challenger_>(
	mut provers: Vec<Prover>,
	skip_rounds: usize,
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<BatchZerocheckOutput<P::Scalar>, Error>
where
	F: TowerField + ExtensionField<FDomain>,
	FDomain: TowerField,
	P: PackedField<Scalar = F> + PackedExtension<F, PackedSubfield = P> + PackedExtension<FDomain>,
	Prover: ZerocheckProver<'a, P>,
	Challenger_: Challenger,
{
	let evals: ZerocheckRoundEvals<AESTowerField128b> = ZerocheckRoundEvals {
		evals: vec![
			AESTowerField128b::new(0xf91d259896a897bfc936879eb7ef13e1),
			AESTowerField128b::new(0x4d6bd90477caf7c7862afde60f98bfe5),
			AESTowerField128b::new(0x225dc345d46053808bfb184484b5e636),
			AESTowerField128b::new(0xbb8bbcdb77b6e2ad59fe4b0da9b7d914),
			AESTowerField128b::new(0x8ad78dac55311ad05c68f3cc6a7b7069),
			AESTowerField128b::new(0x453d366d8c150feb3b373ba84ee8caf6),
			AESTowerField128b::new(0xe97414fecfdcf77b3d46c2275c5ddb3d),
			AESTowerField128b::new(0x0f918162ee576ee7ed552c1a250cdcb3),
			AESTowerField128b::new(0xc8ffb721b52c6b9256e139939e265286),
			AESTowerField128b::new(0x99199e63751e6bdd7eb17929ac5d8948),
			AESTowerField128b::new(0xaf55e59aae93da760f0113992bd2c856),
			AESTowerField128b::new(0x6bc234950a2979b0915b8ae6c316bbeb),
			AESTowerField128b::new(0xb4ea7e6ded90e4bd00148b5dbfef4a31),
			AESTowerField128b::new(0x32f2d4137fffd4d05c9579b6fa7990ff),
			AESTowerField128b::new(0xaf0e49287f631a063f7eb8595fa0e9cd),
			AESTowerField128b::new(0xe7f3f4ade3e0548f3396b89853f2cbce),
			AESTowerField128b::new(0x59c7a46a92a3a3686bd3d581c7d82f36),
			AESTowerField128b::new(0x7db9ab5c9adce3bb32191ec5dd330aef),
			AESTowerField128b::new(0xffaec1a4f09bedfd57253ee7a64610e4),
			AESTowerField128b::new(0x3c1b3e367e2350969ec610e1383ec649),
			AESTowerField128b::new(0x67cc2f1ced5a2591c67c50f2625ec962),
			AESTowerField128b::new(0xbdbfd72987ff3c4fa321d65e7096db56),
			AESTowerField128b::new(0x3462ffede24ed9ddd32b7e14611c6847),
			AESTowerField128b::new(0x09c69e8bac76596eb13504f44a9ab541),
			AESTowerField128b::new(0xc2bd4fde3e2b40f91a78604d5d8f8624),
			AESTowerField128b::new(0x9efefa76cc96bb96bdff65c33970066f),
			AESTowerField128b::new(0x38cd6714ddc1565f9a376afba6d2cf05),
			AESTowerField128b::new(0x015d6c6e173b45867a684f15c488efe1),
			AESTowerField128b::new(0x6e761074b8c45a1f6c1b58ff017d24f0),
			AESTowerField128b::new(0x373c0b7482370c314e1edfa278d13bd6),
			AESTowerField128b::new(0x6c25dfdb9004a2ef1dcbb42432a52310),
			AESTowerField128b::new(0x8afe86ddf30b3698ff4430aa85083780),
			AESTowerField128b::new(0xc5d6bb3d2f3578dc1608ae2f1600267d),
			AESTowerField128b::new(0xa7498cc8c2a2b2de78242494104c4f74),
			AESTowerField128b::new(0xbe40b78e3f3704058ea2bdc5235f6eb8),
			AESTowerField128b::new(0x1a321583740c8e0a8bfe0494687653b0),
			AESTowerField128b::new(0xf9bc3479b0ce72c9b276551b237b7c2c),
			AESTowerField128b::new(0x0ac3bf6c6a3bbbbadc0cf9b5a3255c86),
			AESTowerField128b::new(0xd0b2e6a88ad9f7856653cf7705c8221c),
			AESTowerField128b::new(0x386783eceabd794286c41163e1518df3),
			AESTowerField128b::new(0x93c0b9d29eb2ffbcbee93ba2ec00c361),
			AESTowerField128b::new(0xec6fe07d8ad3931ae78a4ae47071d7c1),
			AESTowerField128b::new(0x8e67942cfebe085bf08e1235be378369),
			AESTowerField128b::new(0x6c8c9f1df0e6c6dcfbe470108d3355d8),
			AESTowerField128b::new(0xc8a7eaf2c4d331a7e985cc7be3bd7321),
			AESTowerField128b::new(0xe28f7be7e988e5a99142577f914944bb),
			AESTowerField128b::new(0xcbc77d2a7c11bdcad84aefdbb660352c),
			AESTowerField128b::new(0x611c1e2dddfa680e3e4c11c3df4d9403),
			AESTowerField128b::new(0xd5a3c91a8e251ca362c2f568c1d2c4fb),
			AESTowerField128b::new(0x5c246c3ca7d34becee1853a5d870dd66),
			AESTowerField128b::new(0x6967300d9b4cbb25fe6d4fc128a91049),
			AESTowerField128b::new(0x367eb63d7c143eeeb9470b14fefc2139),
			AESTowerField128b::new(0x79e3fc2c1aa79753d1ccbcf57ac6a2e0),
			AESTowerField128b::new(0x945a4cec001d6fd48d36bfdb49673a9f),
			AESTowerField128b::new(0xadc0ad6849978b6016763432987ceca9),
			AESTowerField128b::new(0x650a26c103ef4d7b1d2986818334551c),
			AESTowerField128b::new(0x941bb90e649a3d7d5226b87476fa2225),
			AESTowerField128b::new(0xf8ea81ea87132e0a8af0497440999e41),
			AESTowerField128b::new(0x4329101ab9afb9ef5cbabcaa8d060d66),
			AESTowerField128b::new(0x2767a3e96e331b1aac0b278ba847ed79),
			AESTowerField128b::new(0xb1c6b79234e24961d8107ff4dfa35db0),
			AESTowerField128b::new(0x649a563f0f587ae723dca8da86b8e9a2),
			AESTowerField128b::new(0x965ab56a446039f6f0fe5402da018cf5),
			AESTowerField128b::new(0x8048fc96c64ee92b1a688244807da7cd),
			AESTowerField128b::new(0x9616737c2c47ca3be12b2b7cb28a02f1),
			AESTowerField128b::new(0xcac30edbfaefe41b4945e3321db86501),
			AESTowerField128b::new(0xb03a4b8ffa6337c100da14a32d8351f0),
			AESTowerField128b::new(0x7cb7588101df92015ae049fe88de501b),
			AESTowerField128b::new(0x666090b84e9635bf70e581b029f029be),
			AESTowerField128b::new(0x6b650a0412312f78cc4e33d593c03dfd),
			AESTowerField128b::new(0x65ad9910ac205aa75c3d0bb32639f386),
			AESTowerField128b::new(0xd9169854729300aff65aa3a4f288b062),
			AESTowerField128b::new(0x81b576f20f113f43b516abcbe6b6d0e1),
			AESTowerField128b::new(0x86a40238e758e1b5047186ecd56e42bb),
			AESTowerField128b::new(0xf8283b9307877077faa8cb5bb6005aac),
			AESTowerField128b::new(0xbd54d0cb71fc9a35f12caa6db673d12d),
			AESTowerField128b::new(0x746f7ad5c96d1c78b30d2b900eade37f),
			AESTowerField128b::new(0x3094baa444198309e5709f54a0975f4f),
			AESTowerField128b::new(0x2f472fcbbdae5be396923daf1d9bf413),
			AESTowerField128b::new(0x1006a06765968865358f84fec6716190),
			AESTowerField128b::new(0xbe72ef89ef0221e855986dd46c87a961),
			AESTowerField128b::new(0xf91f8f65f0e56700a23a918d7b252dd4),
			AESTowerField128b::new(0x0c2759631b6535d024ec97a276f9b80b),
			AESTowerField128b::new(0xaadf58598670d2b27cbf5f044584a9d5),
			AESTowerField128b::new(0xe7273b12a2e4488586c876506c2d2095),
			AESTowerField128b::new(0x681b7f90d71f709d6b94a55888438a62),
			AESTowerField128b::new(0xc8d3f0f77d59c655c99505aaeb16cc80),
			AESTowerField128b::new(0x6a46f7350f8e86dbe33d19fdcd81194d),
			AESTowerField128b::new(0x49e7d93a6053750c7d08a871aa56f8bb),
			AESTowerField128b::new(0x891daa335224919987b114d7ebfbe719),
			AESTowerField128b::new(0x05f14ac5a26216e0e89de2500e9e10c4),
			AESTowerField128b::new(0x19d86f5d1a16e7401a16e34380132fcc),
			AESTowerField128b::new(0x558eb0d9531b1d3250d80fcffd26a3cf),
			AESTowerField128b::new(0x39234fd5868baa6195e3ca6b98e64f33),
			AESTowerField128b::new(0xfe6cebd962a4998ed4e7161d4ee7da4c),
			AESTowerField128b::new(0x8e94b5f535679c12b0969f8f4558c51b),
			AESTowerField128b::new(0xe5b1a32329288507e6d3293e6d500473),
			AESTowerField128b::new(0xbeeea2894b706f35bb962486906ffefa),
			AESTowerField128b::new(0x3e30d24d1da8871bef499ac35b561a41),
			AESTowerField128b::new(0x6a74858210a80716aab58b7bf0d33bb1),
			AESTowerField128b::new(0xf10a93824759a7817dc46f099d00a669),
			AESTowerField128b::new(0xd529ea3896057b3cd08a833e897ca076),
			AESTowerField128b::new(0x3f03e589f4a7a48d77b11b8026a712df),
			AESTowerField128b::new(0x1626d2318c9ccfcfefb18823fa4728bd),
			AESTowerField128b::new(0x3e289a38eded7a532f9c5136a43eddc4),
			AESTowerField128b::new(0x654dca3a8b71a2f1b055b51b5e0ac482),
			AESTowerField128b::new(0x8a66e06721a72d6a91efa46346a2ff5a),
			AESTowerField128b::new(0xd555c5d26ab798afe97b5b0ff2af5516),
			AESTowerField128b::new(0xd5b03f45861f8f1e2d596aec8d60c22a),
			AESTowerField128b::new(0xa8058d63cf8d60e649467956f53e5301),
			AESTowerField128b::new(0xabff56d5acf3a2589fd034a6ca88f51d),
			AESTowerField128b::new(0x17d7165959d86c7277d727a19aef5e38),
			AESTowerField128b::new(0xc57c2efef966bb2d76761dea5be558fe),
			AESTowerField128b::new(0xd28c1d0edcbb744c469f469dac4ddb65),
			AESTowerField128b::new(0xae37180c2d9bc23b625c985d330db540),
			AESTowerField128b::new(0x007dd2058990e76c6d370708f06367b2),
			AESTowerField128b::new(0x177713dca7db6ec9716d8554ae9fdc58),
			AESTowerField128b::new(0x343691ea7e47eb27a61595ff750273df),
			AESTowerField128b::new(0x3806c76edbed8cbbe33f46826ea67d9b),
			AESTowerField128b::new(0x7799fd06a735601581ab5520da67a8de),
			AESTowerField128b::new(0x39c16715bf6c4ee83ced5fb9a1f94b6b),
			AESTowerField128b::new(0xb96d23202b5a5309a6235b74849d5463),
			AESTowerField128b::new(0x0d3be7f934a9d6b1e6e9c6547bb30b5d),
			AESTowerField128b::new(0xeee9f9de05760ccddf60f3a676b8994d),
			AESTowerField128b::new(0xd064318281c7f3a4b431374383c97b84),
			AESTowerField128b::new(0x053dc93391bf37639fae4f2cc3be949e),
			AESTowerField128b::new(0xe412a9fd17b7a910bc83a79b024c34b8),
			AESTowerField128b::new(0xb58e3262702a032cfac7ffa3c55367d4),
		],
	};

	// Check that the provers are in non-descending order by n_vars
	if !is_sorted_ascending(provers.iter().map(|prover| prover.n_vars())) {
		bail!(Error::ClaimsOutOfOrder);
	}

	let max_domain_size = provers
		.iter()
		.map(|prover| {
			prover
				.domain_size(skip_rounds)
				.expect("domain size must be known")
		})
		.max()
		.unwrap_or(0);

	// Sample batching coefficients while computing round polynomials per claim, then batch
	// those in Lagrange domain.
	let mut batch_coeffs = Vec::with_capacity(provers.len());
	let mut round_evals =
		ZerocheckRoundEvals::zeros(max_domain_size.saturating_sub(1 << skip_rounds));
	for prover in &mut provers {
		let next_batch_coeff = transcript.sample();
		batch_coeffs.push(next_batch_coeff);
		println! {"batch coeffs: {:?}", next_batch_coeff};

		let prover_round_evals = evals;

		//println!("{:?}", prover_round_evals);

		round_evals.add_assign_lagrange(&(prover_round_evals * next_batch_coeff))?;
	}

	// Sample univariate challenge
	transcript.message().write_scalar_slice(&round_evals.evals);
	let univariate_challenge = transcript.sample();

	// Prove reduced multilinear eq-ind sumchecks, high-to-low, with front-loaded batching
	let mut sumcheck_provers = Vec::with_capacity(provers.len());
	for prover in &mut provers {
		let sumcheck_prover = prover.fold_univariate_round(univariate_challenge)?;
		sumcheck_provers.push(sumcheck_prover);
	}

	let regular_sumcheck_prover =
		front_loaded::BatchProver::new_prebatched(batch_coeffs, sumcheck_provers)?;

	let BatchSumcheckOutput {
		challenges: mut unskipped_challenges,
		multilinear_evals: mut univariatized_multilinear_evals,
	} = regular_sumcheck_prover.run(transcript)?;

	// Reverse challenges since folding high-to-low
	unskipped_challenges.reverse();

	// Drop equality indicator evals prior to univariatizing reduction
	for evals in &mut univariatized_multilinear_evals {
		evals
			.pop()
			.expect("equality indicator evaluation at last position");
	}

	// Project witness multilinears to "skipped" variables
	let mut projected_multilinears = Vec::new();
	let dimensions_data = FoldLowDimensionsData::new(skip_rounds, &provers);
	let mle_fold_low_span = tracing::debug_span!(
		"[task] Initial MLE Fold Low",
		phase = "zerocheck",
		perfetto_category = "task.main",
		?dimensions_data,
	)
	.entered();
	for prover in provers {
		let claim_projected_multilinears =
			Box::new(prover).project_to_skipped_variables(&unskipped_challenges)?;

		projected_multilinears.extend(claim_projected_multilinears);
	}
	drop(mle_fold_low_span);

	// Prove univariatizing reduction sumcheck.
	// It's small (`skip_rounds` variables), so portable backend is likely fine.
	let backend = make_portable_backend();
	let reduction_prover = univariatizing_reduction_prover::<_, FDomain, _>(
		projected_multilinears,
		skip_rounds,
		univariatized_multilinear_evals,
		univariate_challenge,
		&backend,
	)?;

	let batch_reduction_prover =
		front_loaded::BatchProver::new(vec![reduction_prover], transcript)?;

	let BatchSumcheckOutput {
		challenges: mut skipped_challenges,
		multilinear_evals: mut concat_multilinear_evals,
	} = batch_reduction_prover.run(transcript)?;

	// Reverse challenges since folding high-to-low
	skipped_challenges.reverse();

	let mut concat_multilinear_evals = concat_multilinear_evals
		.pop()
		.expect("multilinear_evals.len() == 1");

	concat_multilinear_evals
		.pop()
		.expect("Lagrange coefficients MLE eval at last position");

	// Fin
	let output = BatchZerocheckOutput {
		skipped_challenges,
		unskipped_challenges,
		concat_multilinear_evals,
	};

	Ok(output)
}
