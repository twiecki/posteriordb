# PyMC (nutpie/numba) vs Stan (cmdstan) — Benchmark Analysis

## Setup

We benchmarked **101 posteriordb models** comparing PyMC v5 with the nutpie sampler (numba backend) against Stan via cmdstanpy (CmdStan), both using default NUTS settings:

- **1,000 warmup + 1,000 draws, 4 chains, 3 independent runs per model**
- PyMC: `pm.sample(nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "numba"})`
- Stan: `cmdstanpy.CmdStanModel` with default settings
- Hardware: Apple Silicon (ARM64), macOS
- 8 models skipped due to numba compilation OOM or hangs

## Key Results

| Metric | Value |
|--------|-------|
| Models compared | 101 |
| PyMC faster | **52 (51.5%)** |
| Stan faster | **49 (48.5%)** |
| Geometric mean speedup | **1.30x in favor of PyMC** |

PyMC wins slightly more models, but more importantly, its wins tend to be **much larger** than Stan's wins. The largest PyMC advantage is 82x (diamonds), while Stan's largest advantage is 8x (arma11).

## Where PyMC Dominates (>5x faster)

| Model | PyMC (s) | Stan (s) | Speedup |
|-------|----------|----------|---------|
| diamonds | 2.5 | 208.3 | 82x |
| radon_hierarchical_intercept_noncentered | 7.9 | 192.6 | 24x |
| radon_hierarchical_intercept_centered | 5.6 | 136.3 | 24x |
| logearn_logheight_male | 2.1 | 37.7 | 18x |
| radon_variable_intercept_noncentered | 7.1 | 120.3 | 17x |
| radon_county_intercept | 5.0 | 83.2 | 17x |
| radon_partially_pooled_noncentered | 5.1 | 74.8 | 15x |
| radon_variable_slope_centered | 5.3 | 76.0 | 14x |
| radon_variable_intercept_centered | 5.0 | 67.7 | 14x |
| grsm_latent_reg_irt | 24.6 | 255.7 | 10x |
| normal_mixture_k | 39.1 | 373.0 | 10x |
| election88_full | 165.1 | 1507.4 | 9x |
| logearn_interaction | 2.7 | 24.9 | 9x |
| logearn_height_male | 1.4 | 10.6 | 8x |
| nes | 2.6 | 16.6 | 6x |

**Pattern**: Hierarchical models (especially the radon family) and models with large datasets show the biggest PyMC/nutpie advantage. nutpie's Rust-based NUTS implementation with numba-compiled gradients scales very well with model complexity.

## Where Stan Dominates (>3x faster)

| Model | PyMC (s) | Stan (s) | Speedup |
|-------|----------|----------|---------|
| arma11 | 3.6 | 0.4 | 8x |
| garch11 | 6.4 | 0.9 | 7x |
| gp_regr | 2.7 | 0.4 | 6x |
| GLMM_Poisson_model | 5.5 | 0.9 | 6x |
| hmm_example | 13.2 | 2.9 | 4x |
| GLM_Poisson_model | 1.8 | 0.4 | 4x |
| Rate_4_model | 0.9 | 0.3 | 4x |
| eight_schools_noncentered | 1.4 | 0.4 | 4x |
| dugongs_model | 2.0 | 0.6 | 3x |
| losscurve_sislob | 2.2 | 0.6 | 3x |
| GLM_Binomial_model | 1.3 | 0.4 | 3x |
| surgical_model | 1.3 | 0.4 | 3x |

**Pattern**: Stan wins on small, fast-sampling models where the absolute sampling times are under 1 second. Here, nutpie's per-iteration overhead (Python/numba interop, adaptive tuning) matters more relative to total runtime. Time-series models (arma11, garch11) are also a Stan strength.

## The Asymmetry

The speedup distribution is **heavily skewed in PyMC's favor**:
- When PyMC wins, the median speedup is **2.5x**, with a long tail up to 82x
- When Stan wins, the median speedup is **2.3x**, with a maximum of 8x

This means that switching from Stan to PyMC/nutpie would yield large gains on the models where it matters most (complex hierarchical models with long sampling times), while the losses on simple models are small in absolute terms (often <1 second difference).

## Compilation Time

- **PyMC (numba JIT)**: 0.5s – 35s for most models, with outliers up to 157s (ldaK5)
- **Stan (C++ compilation)**: 4s – 19s, very consistent

PyMC's numba compilation is generally faster than Stan's C++ compilation for simple models but can be much slower for complex model graphs. However, numba compilation happens at first run and is cached, similar to Stan's compiled binaries.

8 models were excluded because numba compilation either ran out of memory or hung indefinitely — models with very large computational graphs (hundreds of Potentials, neural networks, large LDA). Stan compiled all of these successfully.

## Diagnostics

Both samplers produce comparable diagnostics (Rhat, ESS, divergences) on most models. A few notable differences:
- **surgical_model**: PyMC shows severe convergence issues (Rhat=4.03, 3615 divergences) while Stan converges fine — likely a transpilation issue
- **mixture models** (normal_mixture, low_dim_gauss_mix_collapse): Both struggle with label switching, as expected
- **ldaK5**: Both take ~9 hours with poor convergence — this is an inherently difficult model

## Limitations

1. **Single machine**: Results may differ on x86 or with different core counts
2. **Default settings only**: Neither sampler was tuned (e.g., adapt_delta, target_accept)
3. **Transpiled models**: PyMC models were auto-transpiled from Stan, not hand-written. Native PyMC models might perform differently
4. **numba backend only**: PyMC/nutpie also supports a JAX backend which could be faster on GPU
5. **No parallel chains**: Stan runs chains in parallel by default; nutpie does too, but the parallelization strategies differ

## Conclusion

PyMC with nutpie/numba is competitive with Stan out of the box, winning on 52% of models with a geometric mean speedup of 1.30x. The advantage is most pronounced on hierarchical models, where nutpie can be 10-80x faster than Stan. Stan maintains an edge on small, simple models where its optimized C++ code has lower per-iteration overhead. For practitioners working primarily with hierarchical Bayesian models, PyMC/nutpie offers a significant performance advantage.
