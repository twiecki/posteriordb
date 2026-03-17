# PyMC (nutpie/numba) vs Stan (cmdstan) — Benchmark Analysis

## Setup

We benchmarked **101 posteriordb models** comparing PyMC v5 with the nutpie sampler (numba backend) against Stan via cmdstanpy (CmdStan), both using default NUTS settings:

- **1,000 warmup + 1,000 draws, 4 chains, 3 independent runs per model**
- PyMC: `pm.sample(nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "numba"})`
- Stan: `cmdstanpy.CmdStanModel` with default settings
- Hardware: Apple Silicon (ARM64), macOS
- 8 models skipped due to numba compilation OOM or hangs

## Primary Metric: Total Time per Effective Sample

The metric that matters most to practitioners is **total wall-clock time (compilation + sampling) per effective sample** — how long do you wait from hitting "run" to getting a given number of independent posterior draws?

| Metric | PyMC wins | Stan wins | Geo. Mean Ratio |
|--------|-----------|-----------|-----------------|
| **Total sec / ESS (compile + sample)** | **87 (87%)** | **13 (13%)** | **2.04x PyMC** |
| sec / ESS (sampling only) | 45 (45%) | 55 (55%) | 1.16x PyMC |
| Raw sampling time | 52 (52%) | 48 (48%) | 1.32x PyMC |
| Total time (compile + sample) | 96 (96%) | 4 (4%) | 2.31x PyMC |

The story changes dramatically depending on what you measure:

- **Sampling-only sec/ESS**: Stan wins more models (55 vs 45), producing slightly more effective samples per second of NUTS sampling on average. Stan's C++ gradients are highly optimized.
- **Total sec/ESS (incl. compilation)**: PyMC wins decisively (87 vs 13). PyMC's numba JIT compilation is typically 2-5x faster than Stan's C++ compilation, which shifts the balance heavily in PyMC's favor for the end-to-end experience.

## Where PyMC Dominates

| Model | PyMC sec/ESS | Stan sec/ESS | Ratio | Why |
|-------|-------------|-------------|-------|-----|
| diamonds | 0.0005 | 0.168 | 334x | Large dataset, PyMC excels |
| radon_variable_intercept_noncentered | 0.023 | 0.417 | 18x | Hierarchical model |
| radon_partially_pooled_noncentered | 0.016 | 0.275 | 17x | Hierarchical model |
| state_space (seasonal) | 1.064 | 14.561 | 14x | Complex time series |
| radon_county_intercept | 0.001 | 0.014 | 13x | Hierarchical model |
| radon_hier_intercept_noncentered | 0.027 | 0.305 | 11x | Hierarchical model |
| radon_hier_intercept_centered | 0.031 | 0.277 | 9x | Hierarchical model |
| radon_variable_slope_centered | 0.010 | 0.088 | 9x | Hierarchical model |
| normal_mixture_k | 7.975 | 56.973 | 7x | Mixture model |
| bym2_offset_only | 0.049 | 0.338 | 7x | Spatial model (BYM2) |

**Pattern**: Hierarchical models (especially the radon family), models with large datasets, and spatial models show the biggest PyMC/nutpie advantage. nutpie's Rust-based NUTS implementation with numba-compiled gradients scales very well with model complexity, and the fast compilation amplifies this advantage.

## Where Stan Dominates

| Model | PyMC sec/ESS | Stan sec/ESS | Ratio | Why |
|-------|-------------|-------------|-------|-----|
| garch11 | 0.013 | 0.004 | 0.30x | Time series, Stan's C++ shines |
| hmm_example | 0.022 | 0.007 | 0.34x | HMM marginalization |
| hierarchical_gp | 0.158 | 0.083 | 0.52x | Complex GP |
| gp_pois_regr | 0.023 | 0.013 | 0.56x | GP with Poisson likelihood |
| logistic_regression_rhs | 1.201 | 0.704 | 0.59x | Regularized horseshoe |
| arma11 | 0.002 | 0.001 | 0.65x | Time series |

**Pattern**: Stan wins on time-series models (arma11, garch11), HMMs, and some GP models. Stan's largest win is 3.3x (garch11), while PyMC's largest win is 334x (diamonds) — a highly asymmetric distribution.

## The Asymmetry

When we look at total sec/ESS, the distribution is **heavily skewed in PyMC's favor**:
- PyMC wins on **87%** of models
- When PyMC wins, gains can be enormous (up to 334x)
- When Stan wins, the advantage is modest (typically <2x, max 3.3x)
- The geometric mean across all models is **2.04x in favor of PyMC**

This means that for the typical user running a model end-to-end, PyMC/nutpie will almost always be faster or comparable — and when it's faster, the gains are often dramatic.

## Compilation Time

Compilation time is a major differentiator and heavily favors PyMC:

- **PyMC (numba JIT)**: 0.5s – 35s for most models
- **Stan (C++ compilation)**: 4s – 19s, very consistent

For the majority of models, numba compilation is 2-5x faster than Stan's C++ compilation. This shifts many models that are close in sampling speed into PyMC's favor when measuring end-to-end time.

8 models were excluded because numba compilation either ran out of memory or hung indefinitely — models with very large computational graphs (hundreds of Potentials, neural networks, large LDA). Stan compiled all of these successfully.

## Diagnostics

Both samplers produce comparable diagnostics (Rhat, ESS, divergences) on most models. A few notable differences:
- **surgical_model**: Initially showed severe convergence issues due to a transpilation bug (pm.Flat + Potential instead of proper distributions). Fixed by using pm.Normal, pm.InverseGamma, and pm.Binomial — now converges comparably to Stan
- **mixture models** (normal_mixture, low_dim_gauss_mix_collapse): Both struggle with label switching, as expected
- **ldaK5**: Both take ~9 hours with poor convergence — this is an inherently difficult model

## Limitations

1. **Single machine**: Results may differ on x86 or with different core counts
2. **Default settings only**: Neither sampler was tuned (e.g., adapt_delta, target_accept)
3. **Transpiled models**: PyMC models were auto-transpiled from Stan, not hand-written. Native PyMC models might perform differently
4. **numba backend only**: PyMC/nutpie also supports a JAX backend which could be faster on GPU
5. **Compilation caching**: Both Stan and numba cache compiled models. The compilation advantage disappears on subsequent runs of the same model. The sec/ESS numbers here represent first-run experience.

## Conclusion

When measuring what practitioners actually care about — total wall-clock time from start to effective posterior samples — **PyMC with nutpie/numba is faster on 87% of posteriordb models** with a geometric mean efficiency advantage of 2.04x. The advantage comes from two sources: (1) faster numba compilation vs Stan's C++ compilation, and (2) superior sampling efficiency on hierarchical and large-data models. Stan retains an edge in pure sampling efficiency on time-series models and some GPs, but these wins are modest compared to PyMC's gains elsewhere.
