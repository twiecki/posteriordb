"""
Test PyMC model gradients against BridgeStan (Stan) reference gradients.

Gradients are invariant to normalization constants, making them a cleaner
comparison than log-density values which may differ by additive constants
between implementations.
"""

import importlib.util
import json
import os
import types
import zipfile

import bridgestan as bs
import numpy as np
import pymc as pm
import pytest

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
DB = os.path.join(os.path.dirname(__file__), "..", "posterior_database")
POSTERIORS_DIR = os.path.join(DB, "posteriors")
STAN_DIR = os.path.join(DB, "models", "stan")
PYMC_DIR = os.path.join(DB, "models", "pymc")
DATA_DIR = os.path.join(DB, "data", "data")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def load_data(data_name: str) -> dict:
    """Load data from a zipped JSON file."""
    path = os.path.join(DATA_DIR, f"{data_name}.json.zip")
    with zipfile.ZipFile(path) as zf:
        with zf.open(zf.namelist()[0]) as f:
            return json.load(f)


def load_pymc_model(model_name: str, data: dict) -> pm.Model:
    """Load a PyMC model file and call make_model(data)."""
    path = os.path.join(PYMC_DIR, f"{model_name}.py")
    # The model files use `pm.Model` in the type annotation at module level,
    # so we need to inject `pm` into the module namespace before exec.
    mod = types.ModuleType(model_name)
    mod.pm = pm
    with open(path) as f:
        exec(f.read(), mod.__dict__)
    return mod.make_model(data)


def build_stan_model(model_name: str, data: dict) -> bs.StanModel:
    """Compile and instantiate a BridgeStan model."""
    stan_file = os.path.join(STAN_DIR, f"{model_name}.stan")
    return bs.StanModel(stan_file, data=data)


def map_stan_to_pymc_point(
    stan_model: bs.StanModel,
    pymc_model: pm.Model,
    theta_unc: np.ndarray,
) -> dict:
    """Map a flat unconstrained Stan parameter vector to a PyMC point dict.

    Both frameworks use the same unconstrained parameterization for sampling
    (log for lower-bounded, logit for bounded, identity for unconstrained).
    We match parameters by iterating through PyMC's value_vars and slicing
    the Stan vector in the same order.

    This assumes the Stan and PyMC models declare parameters in the same order,
    which is the case for our transpiled models.
    """
    stan_names = stan_model.param_unc_names()
    point = {}
    offset = 0

    for value_var in pymc_model.value_vars:
        vname = value_var.name
        shape = tuple(value_var.type.shape)
        size = int(np.prod(shape)) if shape else 1

        point[vname] = theta_unc[offset : offset + size].reshape(shape) if shape else theta_unc[offset]
        offset += size

    assert offset == len(stan_names), (
        f"Parameter count mismatch: used {offset} of {len(stan_names)} Stan params. "
        f"Stan names: {stan_names}, PyMC value_vars: {[v.name for v in pymc_model.value_vars]}"
    )
    return point


# ──────────────────────────────────────────────────────────────────────
# Discover valid posteriors
# ──────────────────────────────────────────────────────────────────────
def discover_posteriors():
    """Find all posteriors that have both Stan and PyMC models + data."""
    pymc_models = {f[:-3] for f in os.listdir(PYMC_DIR) if f.endswith(".py")}
    posteriors = []

    for fname in sorted(os.listdir(POSTERIORS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(POSTERIORS_DIR, fname)) as f:
            p = json.load(f)
        model_name = p["model_name"]
        data_name = p["data_name"]
        if model_name not in pymc_models:
            continue
        if not os.path.exists(os.path.join(STAN_DIR, f"{model_name}.stan")):
            continue
        if not os.path.exists(os.path.join(DATA_DIR, f"{data_name}.json.zip")):
            continue
        posteriors.append((p["name"], model_name, data_name))

    return posteriors


# Models that are known to need special handling or are expected to fail
XFAIL_MODELS = {
    # Capture-recapture models use Potentials for manual discrete marginalization
    # which creates different computational graphs
    "Mth_model",
    "Mt_model",
    "M0_model",
    "Mb_model",
    "Mh_model",
    # Mixture models with custom log-sum-exp implementations
    "normal_mixture",
    "normal_mixture_k",
    "low_dim_gauss_mix",
    "low_dim_gauss_mix_collapse",
    # ODE models
    "one_comp_mm_elim_abs",
    # Models with ordered/simplex parameters that may have different transforms
    "bones_model",
    "irt_2pl",
}

# Models that are too slow to compile or run
SKIP_MODELS = set()

ALL_POSTERIORS = discover_posteriors()

# Deduplicate by model_name (same model with different data only needs one test)
_seen_models = set()
UNIQUE_MODEL_POSTERIORS = []
for name, model_name, data_name in ALL_POSTERIORS:
    if model_name not in _seen_models:
        _seen_models.add(model_name)
        UNIQUE_MODEL_POSTERIORS.append((name, model_name, data_name))


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "posterior_name, model_name, data_name",
    UNIQUE_MODEL_POSTERIORS,
    ids=[m for _, m, _ in UNIQUE_MODEL_POSTERIORS],
)
def test_gradient_matches(posterior_name, model_name, data_name):
    """Compare PyMC gradients to BridgeStan at a random unconstrained point."""
    if model_name in SKIP_MODELS:
        pytest.skip(f"Model {model_name} is in SKIP_MODELS")

    is_xfail = model_name in XFAIL_MODELS
    if is_xfail:
        pytest.xfail(f"Model {model_name} is expected to fail")

    data = load_data(data_name)

    # --- Stan side ---
    stan_model = build_stan_model(model_name, data)
    n_params = stan_model.param_unc_num()

    # Use a deterministic random point for reproducibility
    rng = np.random.default_rng(seed=42)
    theta_unc = rng.normal(0, 0.5, size=n_params)

    stan_logp, stan_grad = stan_model.log_density_gradient(
        theta_unc, propto=True, jacobian=True
    )

    # --- PyMC side ---
    pymc_model = load_pymc_model(model_name, data)

    # Verify parameter count matches
    pymc_n_params = sum(
        int(np.prod(v.type.shape)) if v.type.shape else 1
        for v in pymc_model.value_vars
    )
    assert pymc_n_params == n_params, (
        f"Parameter count mismatch: Stan has {n_params}, "
        f"PyMC has {pymc_n_params}. "
        f"Stan: {stan_model.param_unc_names()}, "
        f"PyMC: {[v.name for v in pymc_model.value_vars]}"
    )

    # Map Stan unconstrained vector to PyMC point dict
    pymc_point = map_stan_to_pymc_point(stan_model, pymc_model, theta_unc)

    # Compile and evaluate PyMC gradient
    dlogp_fn = pymc_model.compile_dlogp()
    pymc_grad = dlogp_fn(pymc_point)

    # --- Compare ---
    np.testing.assert_allclose(
        pymc_grad,
        stan_grad,
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            f"Gradient mismatch for {model_name}\n"
            f"Stan params: {stan_model.param_unc_names()}\n"
            f"PyMC value_vars: {[v.name for v in pymc_model.value_vars]}\n"
            f"Stan grad: {stan_grad}\n"
            f"PyMC grad: {pymc_grad}"
        ),
    )
