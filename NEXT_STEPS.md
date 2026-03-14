# Porting the Remaining 16 posteriordb Models to PyMC

## Context

We have successfully transpiled 104/120 Stan models from posteriordb to PyMC v5.
All transpiled models are validated against BridgeStan for gradient equivalence.
The transpiler uses an agentic loop (Claude + BridgeStan validation) defined in
`/Users/twiecki/projects/transpailer/transpailer/stan_to_pymc.py`.

The remaining 16 models failed automatic transpilation because they require
PyMC-ecosystem-specific APIs that the transpiler doesn't know about yet.
This document describes how to port each one, grouped by the PyMC tool needed.

## Setup

```bash
# Posteriordb repo (on pymc-retranspile-batch2 branch)
cd /Users/twiecki/projects/posteriordb

# Transpailer (Stan→PyMC transpiler)
cd /Users/twiecki/projects/transpailer

# Run transpilation for a single model:
cd /Users/twiecki/projects/transpailer && BRIDGESTAN=$HOME/.bridgestan/bridgestan-2.7.0 \
  uv run python /Users/twiecki/projects/posteriordb/run_retranspile.py <model_name>

# All models follow: make_model(data: dict) -> pm.Model
# Output goes to: posterior_database/models/pymc/<model_name>.py
```

## Strategy

For each group below, the approach is:
1. Add the relevant PyMC API patterns to the transpailer skill (`transpailer/skills/stan_to_pymc.md`)
2. Re-run the transpiler for those models
3. If the transpiler still fails, port manually and validate with `tests/test_pymc_gradients.py`

---

## Group 1: ODE Models (3 models)

**Models:** `sir`, `covid19imperial_v2`, `covid19imperial_v3`
**Also possibly:** `soil_incubation` (if it uses ODEs internally)

### Tool: `sunode` (recommended) or `pymc.ode.DifferentialEquation` (fallback)

**sunode** wraps SUNDIALS solvers and is ~200x faster than the built-in `pymc.ode`:

```python
import sunode
import sunode.wrappers.as_pytensor

def sir_rhs(t, y, p):
    return {
        'S': -p.beta * y.S * y.I,
        'I': p.beta * y.S * y.I - p.gamma * y.I,
        'R': p.gamma * y.I,
    }

with pm.Model() as model:
    beta = pm.LogNormal("beta", mu=0, sigma=1)
    gamma = pm.LogNormal("gamma", mu=0, sigma=1)

    y_hat, _, problem, solver, _, _ = sunode.wrappers.as_pytensor.solve_ivp(
        y0={'S': (S0, ()), 'I': (I0, ()), 'R': (R0, ())},
        params={'beta': (beta, ()), 'gamma': (gamma, ())},
        rhs=sir_rhs,
        tvals=times,
        t0=t0,
    )
    # y_hat is a dict of tensors: y_hat['S'], y_hat['I'], y_hat['R']
    pm.Normal("obs", mu=y_hat['I'], sigma=sigma, observed=cases)
```

**Fallback** (`pymc.ode.DifferentialEquation`):
```python
from pymc.ode import DifferentialEquation

def sir_func(y, t, p):
    beta, gamma = p[0], p[1]
    S, I, R = y[0], y[1], y[2]
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

ode = DifferentialEquation(func=sir_func, times=times, n_states=3, n_theta=2, t0=0)
y_hat = ode(y0=[S0, I0, R0], theta=[beta, gamma])
```

### Experiment plan
1. Try `sunode` first for `sir` (simplest ODE model)
2. If sunode works, port `covid19imperial_v2/v3` (more complex SEIR with interventions)
3. For `soil_incubation`, check the Stan model to see if it's actually ODE-based
4. Note: The current `lotka_volterra` port uses a custom Op without gradients - consider
   re-porting it with sunode too for proper gradient support

### Key difference from current approach
The existing `one_comp_mm_elim_abs.py` and `lotka_volterra.py` use a custom PyTensor `Op`
with `scipy.solve_ivp` in `perform()`. This works for logp but has NO gradients, meaning
NUTS won't work efficiently. `sunode` provides proper adjoint gradients.

---

## Group 2: HMMs (3 models)

**Models:** `hmm_gaussian`, `hmm_drive_1`, `iohmm_reg`
**Already ported:** `hmm_example`, `hmm_drive_0` (manual forward algorithm)

### Tool: `pymc_extras.distributions.DiscreteMarkovChain` + `pymc_extras.marginalize()`

This implements the forward algorithm automatically:

```python
import pymc_extras as pmx
from pymc_extras.distributions import DiscreteMarkovChain

with pm.Model() as model:
    # Transition matrix
    P = pm.Dirichlet("P", a=np.ones(K), shape=(K, K))
    # Initial state distribution
    init_dist = pm.Categorical.dist(p=np.ones(K) / K)
    # Discrete Markov chain (will be marginalized out)
    states = DiscreteMarkovChain("states", P=P, init_dist=init_dist, steps=T-1)

    # Emission parameters
    mu = pm.Normal("mu", 0, 10, shape=K)       # or ordered for identifiability
    sigma = pm.HalfNormal("sigma", 5, shape=K)

    # Observations conditioned on hidden states
    y = pm.Normal("y", mu=mu[states], sigma=sigma[states], observed=data)

# Marginalize out the discrete states (forward algorithm)
marginalized_model = pmx.marginalize(model, ["states"])

# Sample from the marginalized model
with marginalized_model:
    idata = pm.sample()

# Recover posterior over discrete states
idata = pmx.recover_marginals(marginalized_model, idata)
```

### Per-model notes
- **`hmm_gaussian`**: Standard first-order Gaussian HMM. Direct fit with above pattern.
  Use `ordered[K] mu` for identifiability (Stan model does this).
- **`hmm_drive_1`**: Same structure as `hmm_drive_0` (which was ported successfully with
  manual forward algorithm). Try `DiscreteMarkovChain` approach instead.
- **`iohmm_reg`**: Input-Output HMM where transition probabilities depend on covariates.
  May need time-varying transition matrix. Check if `DiscreteMarkovChain` supports
  3D P tensors (P[t, i, j]). If not, may need manual forward algorithm with
  `pytensor.scan`.

### Validation note
The Stan HMM models implement the forward algorithm manually and marginalize over
discrete states. The logp values should match exactly since both approaches compute
the same marginal likelihood. Gradient comparison is the definitive test.

---

## Group 3: Gaussian Processes (2 models)

**Models:** `hierarchical_gp`, `kronecker_gp`

### Tool: `pm.gp.Marginal`, `pm.gp.MarginalKron`, `pm.gp.cov.*`

The transpailer already has GP skills (`skills/gp.md`, `skills/gp_accelerate.md`,
`skills/gp_cuda.md`) but those are for Rust compilation, NOT for PyMC GP API.

**Add to `stan_to_pymc.md` skill:**

```python
# Standard GP regression
with pm.Model() as model:
    ls = pm.InverseGamma("ls", 5, 5)
    eta = pm.HalfCauchy("eta", beta=2)
    cov = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ls)

    gp = pm.gp.Marginal(cov_func=cov)
    sigma = pm.HalfCauchy("sigma", beta=3)
    y_ = gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)

# Kronecker-structured GP (grid data)
with pm.Model() as model:
    cov1 = pm.gp.cov.ExpQuad(input_dim=1, ls=ls1)
    cov2 = pm.gp.cov.ExpQuad(input_dim=1, ls=ls2)
    gp = pm.gp.MarginalKron(cov_funcs=[cov1, cov2])
    sigma = pm.HalfNormal("sigma", 1)
    y_ = gp.marginal_likelihood("y", Xs=[X1, X2], y=y, sigma=sigma)
```

### Per-model notes
- **`kronecker_gp`**: Stan model manually implements Kronecker product eigendecomposition.
  `pm.gp.MarginalKron` does this automatically. Map Stan's custom functions
  (`kron_mvprod`, `calculate_eigenvalues`) to the built-in API.
- **`hierarchical_gp`**: Multi-level GP with shared hyperpriors across groups. Use
  multiple `pm.gp.Marginal` or `pm.gp.Latent` instances sharing lengthscale/amplitude
  priors. May also work with `pm.gp.HSGP` for scalability.

### Covariance function mapping (Stan → PyMC)
| Stan | PyMC |
|------|------|
| `cov_exp_quad(x, alpha, rho)` | `alpha**2 * pm.gp.cov.ExpQuad(D, ls=rho)` |
| `cov_matern32(x, alpha, rho)` | `alpha**2 * pm.gp.cov.Matern32(D, ls=rho)` |
| `cov_matern52(x, alpha, rho)` | `alpha**2 * pm.gp.cov.Matern52(D, ls=rho)` |
| `diag_matrix(rep_vector(sigma^2, N))` | `pm.gp.cov.WhiteNoise(sigma=sigma)` |

---

## Group 4: Topic Models / LDA (2 models)

**Models:** `ldaK2`, `ldaK5`

### Approach: Manual marginalization with `pt.logsumexp`

Stan marginalizes per-word topic assignments using `log_sum_exp`. PyMC equivalent:

```python
with pm.Model() as model:
    # Topic-word distributions: phi[k] is word distribution for topic k
    phi = pm.Dirichlet("phi", a=beta_prior, shape=(K, V))
    # Document-topic distributions: theta[m] is topic distribution for doc m
    theta = pm.Dirichlet("theta", a=alpha_prior, shape=(M, K))

    # Marginalized likelihood (log_sum_exp over topics for each word)
    # For word n in document doc[n]:
    #   p(w[n] | theta, phi) = sum_k theta[doc[n],k] * phi[k, w[n]]
    log_lik = pt.logsumexp(
        pt.log(theta[doc_idx, :]) + pt.log(phi[:, word_idx]).T,
        axis=1
    )
    pm.Potential("log_lik", pt.sum(log_lik))
```

This directly mirrors Stan's approach. Do NOT use `pmx.marginalize()` for LDA -
the number of discrete variables (one per word token) is far too large.

### Note on indexing
Stan's `doc` and `w` arrays are 1-based. Convert: `doc_idx = data['doc'] - 1`,
`word_idx = data['w'] - 1`.

---

## Group 5: Remaining Models (6 models)

### `dogs_log`, `dogs_nonhierarchical`
These are avoidance learning models with cumulative sums. The transpiler failed
on validation (logp mismatch after 30 attempts). The pattern is:
- Cumulative count of avoidances and shocks per dog
- `inv_logit(beta[1] * n_avoid + beta[2] * n_shock)`
- Bernoulli likelihood

The `dogs` model (without `_log`) was successfully ported. Check what differs.
The `_log` variant uses `log(p)` parameterization. Try increasing `max_turns`
or manually adjusting the transpiled code.

### `soil_incubation`
Check if this is ODE-based (see Group 1) or a simpler compartment model.
Read the Stan source to determine.

### `gpcm_latent_reg_irt`
Generalized Partial Credit Model. Similar to `grsm_latent_reg_irt` which was
successfully ported. Compare the two Stan models and adapt the working port.

### `nn_rbm1bJ10`, `nn_rbm1bJ100`
Restricted Boltzmann Machine models. These have discrete binary latent units
that need marginalization. For small J (J=10), `pmx.marginalize()` with
`pm.Bernoulli` might work. For J=100, manual marginalization is needed.

Check if the Stan model uses the standard RBM free energy:
`F(v) = -b'v - sum_j log(1 + exp(c_j + W_j v))`
This can be computed directly with `pt.log1pexp` without enumerating states.

---

## Adding Skills to the Transpailer

To teach the transpiler these patterns, add sections to
`/Users/twiecki/projects/transpailer/transpailer/skills/stan_to_pymc.md`:

1. **ODE Models** section with sunode pattern
2. **HMM / Discrete Marginalization** section with `pmx.DiscreteMarkovChain`
3. **GP Models** section with `pm.gp.*` API mapping
4. **Topic Models** section with `pt.logsumexp` marginalization pattern

Then re-run the transpiler. Models that still fail should be ported manually.

## Validation

All ports must pass:
```bash
cd /Users/twiecki/projects/transpailer && BRIDGESTAN=$HOME/.bridgestan/bridgestan-2.7.0 \
  uv run python -m pytest /Users/twiecki/projects/posteriordb/tests/test_pymc_gradients.py \
  -k "<model_name>" -v
```

For models using sunode or pmx.marginalize, gradient comparison against BridgeStan
is still the gold standard. The gradients should match within 1e-3 relative error.
