"""Test transpiled PyMC models against Stan originals via BridgeStan.

For each model that has both a Stan and PyMC version, this test:
1. Loads the Stan model via BridgeStan and computes reference logp values
2. Loads the transpiled PyMC model and evaluates logp at the same points
3. Checks that the logp values match (allowing for constant offsets from
   normalization constants, which don't affect the posterior)

Usage:
    cd bayes-ai-compiler && uv run python -m pytest ../posteriordb/test_transpiled_models.py -v
    cd bayes-ai-compiler && uv run python -m pytest ../posteriordb/test_transpiled_models.py -k "eight_schools" -v
"""
from __future__ import annotations

import importlib.util
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

POSTERIORDB = Path(__file__).resolve().parent / "posterior_database"
STAN_DIR = POSTERIORDB / "models" / "stan"
PYMC_DIR = POSTERIORDB / "models" / "pymc"
DATA_DIR = POSTERIORDB / "data" / "data"
POSTERIORS_DIR = POSTERIORDB / "posteriors"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_data_for_model(model_name: str) -> dict | None:
    """Find data for a model by scanning posteriors/*.json."""
    for posterior_file in POSTERIORS_DIR.glob("*.json"):
        info = json.loads(posterior_file.read_text())
        if info.get("model_name") == model_name:
            data_name = info.get("data_name")
            if data_name:
                zip_path = DATA_DIR / f"{data_name}.json.zip"
                if zip_path.exists():
                    with zipfile.ZipFile(zip_path) as zf:
                        with zf.open(zf.namelist()[0]) as f:
                            return json.load(f)
    return None


def load_pymc_model(model_name: str, data: dict):
    """Load a transpiled PyMC model and return (model, logp_fn)."""
    import pymc as pm
    import pytensor.tensor as pt

    pymc_path = PYMC_DIR / f"{model_name}.py"
    spec = importlib.util.spec_from_file_location(model_name, pymc_path)
    mod = importlib.util.module_from_spec(spec)
    # Inject common imports so type annotations like `-> pm.Model` work
    mod.__dict__["pm"] = pm
    mod.__dict__["pt"] = pt
    mod.__dict__["np"] = np
    spec.loader.exec_module(mod)
    model = mod.make_model(data)
    logp_fn = model.compile_logp()
    return model, logp_fn


def get_bridgestan_reference(stan_code: str, data: dict, n_extra: int = 3, seed: int = 123):
    """Compile Stan model via BridgeStan and return reference points.

    Returns (unc_param_names, points) where points is a list of
    (label, unc_point_array, ref_logp).
    """
    from transpailer.stan_exporter import StanModelExporter

    exporter = StanModelExporter(stan_code, data=data, n_extra_points=n_extra, seed=seed)
    ctx = exporter.context

    points = []
    points.append(("initial", np.array(ctx.initial_point.point), ctx.initial_point.logp))
    for i, ep in enumerate(ctx.extra_points):
        points.append((f"extra_{i}", np.array(ep.point), ep.logp))

    return ctx.unc_param_names, points


def map_unc_point_to_pymc(model, unc_point: np.ndarray, stan_param_names: list[str]) -> dict:
    """Map a BridgeStan unconstrained point to a PyMC point dict."""
    point_dict = {}
    pymc_vars = model.value_vars
    pymc_offset = 0

    # Group Stan params by base name
    stan_groups: dict[str, list[int]] = {}
    for i, name in enumerate(stan_param_names):
        base = re.sub(r"\.\d+$", "", name)
        if base not in stan_groups:
            stan_groups[base] = []
        stan_groups[base].append(i)

    for var in pymc_vars:
        var_name = var.name
        base_name = re.sub(
            r"_(log|logodds|interval|circular|ordered|simplex|zerosum)__$",
            "", var_name,
        )

        var_size = int(
            np.prod(var.type.shape)
            if hasattr(var.type, "shape") and var.type.shape
            else 1
        )

        matched = False
        for stan_base, indices in stan_groups.items():
            if stan_base == base_name or stan_base.replace(".", "_") == base_name:
                values = np.array([unc_point[j] for j in indices])
                point_dict[var_name] = values if len(values) > 1 else values[0]
                matched = True
                break

        if not matched:
            end = pymc_offset + var_size
            if end <= len(unc_point):
                values = unc_point[pymc_offset:end]
                point_dict[var_name] = values if var_size > 1 else float(values[0])
            else:
                point_dict[var_name] = np.zeros(var_size) if var_size > 1 else 0.0

        pymc_offset += var_size

    return point_dict


def count_half_distributions(model) -> int:
    """Count Half* distributions in a PyMC model."""
    _HALF_RV_OPS = {"HalfNormalRV", "HalfCauchyRV", "HalfStudentTRV", "HalfFlatRV"}
    return sum(
        1 for rv in model.free_RVs
        if type(rv.owner.op).__name__ in _HALF_RV_OPS
    )


def validate_logp_offset_tolerant(
    model, logp_fn, unc_param_names, reference_points, rel_tol=1e-2,
):
    """Validate PyMC logp against BridgeStan reference using offset-tolerant comparison.

    Returns (passed: bool, report: str).
    """
    n_half = count_half_distributions(model)
    half_correction = n_half * np.log(2)

    point_results = []  # (label, ref_logp, corrected_pymc_logp)

    for label, unc_point, ref_logp in reference_points:
        try:
            point_dict = map_unc_point_to_pymc(model, unc_point, unc_param_names)
        except Exception as e:
            return False, f"{label}: point mapping error: {e}"

        try:
            pymc_logp = float(logp_fn(point_dict))
        except Exception as e:
            return False, f"{label}: logp eval error: {e}"

        corrected = pymc_logp - half_correction
        point_results.append((label, ref_logp, corrected))

    if len(point_results) < 2:
        if not point_results:
            return False, "No points evaluated"
        label, ref, pymc = point_results[0]
        rel_err = abs(pymc - ref) / max(abs(ref), 1.0)
        if rel_err <= rel_tol:
            return True, f"{label}: rel_err={rel_err:.2e} OK"
        return False, f"{label}: rel_err={rel_err:.2e} MISMATCH"

    # Offset-tolerant: check if diffs are constant
    diffs = [ref - pymc for _, ref, pymc in point_results]
    mean_diff = sum(diffs) / len(diffs)

    report_lines = []
    errors = []
    for label, ref_logp, corrected in point_results:
        diff = ref_logp - corrected
        offset_dev = abs(diff - mean_diff)
        scale = max(abs(ref_logp), 1.0)
        rel_dev = offset_dev / scale
        rel_err = abs(corrected - ref_logp) / scale

        if rel_err <= rel_tol:
            status = "OK"
        elif rel_dev <= rel_tol:
            status = "OK (constant offset)"
        else:
            status = "MISMATCH"
            errors.append(
                f"{label}: BridgeStan={ref_logp:.6f} PyMC={corrected:.6f} "
                f"rel_err={rel_err:.2e} offset_dev={rel_dev:.2e}"
            )

        report_lines.append(
            f"{label}: BridgeStan={ref_logp:.6f} PyMC={corrected:.6f} "
            f"rel_err={rel_err:.2e} offset_dev={rel_dev:.2e} [{status}]"
        )

    if abs(mean_diff) > 1e-6:
        report_lines.append(f"constant offset: {mean_diff:.6f}")

    report = "\n".join(report_lines)
    if errors:
        return False, report + "\n\nErrors:\n" + "\n".join(errors)
    return True, report


# ---------------------------------------------------------------------------
# Discover all models that have both Stan and PyMC versions + data
# ---------------------------------------------------------------------------


def _discover_models() -> list[str]:
    """Find all models that have Stan, PyMC, and data available."""
    models = []
    for pymc_file in sorted(PYMC_DIR.glob("*.py")):
        model_name = pymc_file.stem
        stan_file = STAN_DIR / f"{model_name}.stan"
        if stan_file.exists():
            # Check data is available
            if find_data_for_model(model_name) is not None:
                models.append(model_name)
    return models


ALL_MODELS = _discover_models()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_logp_matches_bridgestan(model_name: str):
    """Test that the transpiled PyMC model matches BridgeStan logp."""
    # Load data
    data = find_data_for_model(model_name)
    assert data is not None, f"No data found for {model_name}"

    # Load Stan model via BridgeStan
    stan_code = (STAN_DIR / f"{model_name}.stan").read_text()
    unc_param_names, reference_points = get_bridgestan_reference(stan_code, data)

    # Load PyMC model
    model, logp_fn = load_pymc_model(model_name, data)

    # Compare logp values
    passed, report = validate_logp_offset_tolerant(
        model, logp_fn, unc_param_names, reference_points,
    )
    assert passed, f"logp mismatch for {model_name}:\n{report}"
