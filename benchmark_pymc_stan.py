"""Benchmark PyMC (nutpie/numba) vs Stan (cmdstan) on posteriordb models.

Usage:
    cd /Users/twiecki/projects/transpailer
    uv run python /Users/twiecki/projects/posteriordb/benchmark_pymc_stan.py

Measures:
    - Compilation time (separate from sampling)
    - Sampling time (wall clock)
    - Convergence diagnostics: Rhat, ESS (bulk & tail), divergences
    - Multiple runs per model for variance estimates
"""

import gc
import json
import importlib.util
import logging
import os
import sys
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import arviz as az
import cmdstanpy
import numpy as np
import pymc as pm

# Suppress noisy loggers
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("pymc").setLevel(logging.WARNING)
logging.getLogger("nutpie").setLevel(logging.WARNING)

# ── paths ──────────────────────────────────────────────────────────────────
POSTERIORDB = Path("/Users/twiecki/projects/posteriordb/posterior_database")
STAN_DIR = POSTERIORDB / "models" / "stan"
PYMC_DIR = POSTERIORDB / "models" / "pymc"
DATA_DIR = POSTERIORDB / "data" / "data"
POSTERIORS_DIR = POSTERIORDB / "posteriors"
RESULTS_DIR = Path("/Users/twiecki/projects/posteriordb/benchmark_results")

# ── benchmark configuration ───────────────────────────────────────────────
DRAWS = 1000
TUNE = 1000
CHAINS = 4
N_RUNS = 3
SEED = 42
PER_MODEL_TIMEOUT = 600  # skip model if a single run takes > 10 min

# Models known to OOM or take extremely long on numba compilation
SKIP_MODELS = {
    "Mth_model",        # 387 individual Potentials in for-loop, OOM
    "Mtbh_model",       # similar capture-recapture with huge graph
    "nn_rbm1bJ10",      # RBM neural network, huge graph
    "nn_rbm1bJ100",     # MNIST neural network, huge
    "Survey_model",     # loop creates massive pytensor graph, OOM
    "kronecker_gp",     # 10 loops, 3 potentials, complex GP
    "ldaK5",            # LDA K=5 on Pride&Prejudice, numba compilation hangs (>50min)
    "lotka_volterra",   # ODE model, numba object mode, OOM
}


def discover_models() -> list[tuple[str, str, str]]:
    """Auto-discover all models that have both Stan and PyMC implementations."""
    pymc_models = {p.stem for p in PYMC_DIR.glob("*.py")}
    stan_models = {p.stem for p in STAN_DIR.glob("*.stan")}
    both = sorted(pymc_models & stan_models)

    models = []
    for model_name in both:
        for pf in POSTERIORS_DIR.glob("*.json"):
            info = json.loads(pf.read_text())
            if info.get("model_name") == model_name:
                data_name = info.get("data_name")
                if data_name and (DATA_DIR / f"{data_name}.json.zip").exists():
                    models.append((pf.stem, model_name, data_name))
                    break
    return models


def load_data(data_name: str) -> dict:
    """Load data from posteriordb zip archive."""
    zip_path = DATA_DIR / f"{data_name}.json.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(zf.namelist()[0]) as f:
            return json.load(f)


def load_pymc_model(model_name: str, data: dict) -> pm.Model:
    """Load and instantiate a PyMC model from posteriordb."""
    model_path = PYMC_DIR / f"{model_name}.py"
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    mod = importlib.util.module_from_spec(spec)
    import pytensor.tensor as pt
    mod.pm = pm
    mod.pt = pt
    mod.np = np
    spec.loader.exec_module(mod)
    return mod.make_model(data)


def extract_diagnostics(idata) -> dict:
    """Extract convergence diagnostics from an InferenceData object."""
    summary = az.summary(idata, kind="diagnostics")
    return {
        "rhat_max": float(summary["r_hat"].max()),
        "rhat_mean": float(summary["r_hat"].mean()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_bulk_median": float(summary["ess_bulk"].median()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "ess_tail_median": float(summary["ess_tail"].median()),
        "n_divergences": int(idata.sample_stats["diverging"].sum().values),
    }


def write_stan_data_json(data: dict, path: Path):
    """Write data dict to JSON file for cmdstan."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    converted = {k: convert(v) for k, v in data.items()}
    path.write_text(json.dumps(converted))


# ── PyMC benchmark ─────────────────────────────────────────────────────────

_pymc_compiled: dict[str, float] = {}


def benchmark_pymc(model_name: str, data: dict, run_idx: int) -> dict:
    """Benchmark a single PyMC model run with nutpie/numba backend."""
    result = {"backend": "pymc_nutpie", "model": model_name, "run": run_idx}

    try:
        model = load_pymc_model(model_name, data)

        # compilation (first run only)
        if model_name not in _pymc_compiled:
            t0 = time.perf_counter()
            pm.sample(
                draws=10, tune=10, chains=1,
                nuts_sampler="nutpie",
                nuts_sampler_kwargs={"backend": "numba"},
                model=model, random_seed=SEED, progressbar=False,
            )
            compile_time = time.perf_counter() - t0
            _pymc_compiled[model_name] = compile_time
            result["compile_time_s"] = compile_time
            del model
            gc.collect()
            model = load_pymc_model(model_name, data)
        else:
            result["compile_time_s"] = 0.0

        # sampling
        t0 = time.perf_counter()
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            nuts_sampler="nutpie",
            nuts_sampler_kwargs={"backend": "numba"},
            model=model, random_seed=SEED + run_idx, progressbar=False,
        )
        sample_time = time.perf_counter() - t0
        result["sample_time_s"] = sample_time
        result["total_time_s"] = result["compile_time_s"] + sample_time

        diag = extract_diagnostics(idata)
        result.update(diag)
        result["draws_per_sec"] = (CHAINS * DRAWS) / sample_time
        result["success"] = True
        del idata, model

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    gc.collect()
    return result


# ── Stan benchmark ─────────────────────────────────────────────────────────

_stan_model_cache: dict[str, cmdstanpy.CmdStanModel] = {}


def benchmark_stan(model_name: str, data: dict, run_idx: int) -> dict:
    """Benchmark a single Stan model run with cmdstan."""
    result = {"backend": "stan_cmdstan", "model": model_name, "run": run_idx}

    stan_file = STAN_DIR / f"{model_name}.stan"
    data_json = RESULTS_DIR / f"{model_name}_data.json"

    try:
        write_stan_data_json(data, data_json)

        # compilation (only first run)
        if model_name not in _stan_model_cache:
            t0 = time.perf_counter()
            stan_model = cmdstanpy.CmdStanModel(
                stan_file=str(stan_file), force_compile=True
            )
            compile_time = time.perf_counter() - t0
            _stan_model_cache[model_name] = stan_model
            result["compile_time_s"] = compile_time
        else:
            stan_model = _stan_model_cache[model_name]
            result["compile_time_s"] = 0.0

        # sampling
        t0 = time.perf_counter()
        fit = stan_model.sample(
            data=str(data_json),
            iter_sampling=DRAWS, iter_warmup=TUNE,
            chains=CHAINS, seed=SEED + run_idx,
            show_progress=False,
        )
        sample_time = time.perf_counter() - t0
        result["sample_time_s"] = sample_time
        result["total_time_s"] = result["compile_time_s"] + sample_time

        idata = az.from_cmdstanpy(fit)
        diag = extract_diagnostics(idata)
        result.update(diag)
        result["draws_per_sec"] = (CHAINS * DRAWS) / sample_time
        result["n_divergences"] = int(sum(fit.divergences))

        # Stan's internal per-chain timing
        chain_config = fit.metadata.cmdstan_config
        if "time" in chain_config:
            t = chain_config["time"]
            result["stan_warmup_s"] = t.get("warmup", 0)
            result["stan_sampling_s"] = t.get("sampling", 0)

        result["success"] = True
        del idata, fit

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    gc.collect()
    return result


# ── reporting ──────────────────────────────────────────────────────────────


def print_summary(all_results: list[dict]):
    """Print summary table of benchmark results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    models_seen = []
    for r in all_results:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    header = (
        f"{'Model':<45} {'Backend':<15} {'Compile(s)':<12} "
        f"{'Sample(s)':<12} {'ESS_bulk':<10} {'Rhat':<8} {'Div':<5}"
    )
    print(header)
    print("-" * len(header))

    for model_name in models_seen:
        model_results = [r for r in all_results if r["model"] == model_name and r.get("success")]
        if not model_results:
            print(f"{model_name:<45} {'FAILED'}")
            continue

        for backend in ["pymc_nutpie", "stan_cmdstan"]:
            runs = [r for r in model_results if r["backend"] == backend]
            if not runs:
                continue

            compile_times = [r["compile_time_s"] for r in runs if r["compile_time_s"] > 0]
            sample_times = [r["sample_time_s"] for r in runs]

            avg_compile = np.mean(compile_times) if compile_times else 0
            avg_sample = np.mean(sample_times)
            std_sample = np.std(sample_times) if len(sample_times) > 1 else 0

            last = runs[-1]
            ess_bulk = last.get("ess_bulk_min", 0)
            rhat = last.get("rhat_max", 0)
            divs = last.get("n_divergences", 0)

            label = model_name if backend == "pymc_nutpie" else ""
            sample_str = f"{avg_sample:.1f}+/-{std_sample:.1f}"
            print(
                f"{label:<45} {backend:<15} {avg_compile:<12.1f} "
                f"{sample_str:<12} {ess_bulk:<10.0f} {rhat:<8.3f} {divs:<5}"
            )

    # Speedup summary
    print("\n" + "=" * 100)
    print("SPEEDUP SUMMARY (sampling time)")
    print("=" * 100)
    print(f"{'Model':<45} {'PyMC(s)':<12} {'Stan(s)':<12} {'Ratio':<10}")
    print("-" * 79)

    speedups = []
    for model_name in models_seen:
        model_results = [r for r in all_results if r["model"] == model_name and r.get("success")]
        pymc_runs = [r for r in model_results if r["backend"] == "pymc_nutpie"]
        stan_runs = [r for r in model_results if r["backend"] == "stan_cmdstan"]

        if pymc_runs and stan_runs:
            pymc_avg = np.mean([r["sample_time_s"] for r in pymc_runs])
            stan_avg = np.mean([r["sample_time_s"] for r in stan_runs])
            ratio = stan_avg / pymc_avg if pymc_avg > 0 else float("inf")
            faster = "PyMC" if ratio > 1 else "Stan"
            ratio_display = ratio if ratio > 1 else 1 / ratio
            speedups.append((model_name, ratio))
            print(
                f"{model_name:<45} {pymc_avg:<12.2f} {stan_avg:<12.2f} "
                f"{faster} {ratio_display:.2f}x faster"
            )

    if speedups:
        pymc_wins = sum(1 for _, r in speedups if r > 1)
        stan_wins = len(speedups) - pymc_wins
        geo_mean = np.exp(np.mean(np.log([r for _, r in speedups])))
        print(f"\nPyMC faster: {pymc_wins}/{len(speedups)},  Stan faster: {stan_wins}/{len(speedups)}")
        print(f"Geometric mean ratio (>1 = PyMC faster): {geo_mean:.2f}x")


# ── persistence ────────────────────────────────────────────────────────────

RESULTS_FILE = RESULTS_DIR / "benchmark_all.json"


def load_previous_results() -> list[dict]:
    """Load results from previous benchmark run."""
    if RESULTS_FILE.exists():
        print(f"Loading previous results from: {RESULTS_FILE.name}")
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(all_results: list[dict]):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def models_already_done(all_results: list[dict]) -> set[str]:
    """Return model names that already have N_RUNS of both backends."""
    pymc_counts: Counter = Counter()
    stan_counts: Counter = Counter()
    for r in all_results:
        if r.get("success") or r.get("error"):
            if r["backend"] == "pymc_nutpie":
                pymc_counts[r["model"]] += 1
            else:
                stan_counts[r["model"]] += 1
    return {
        m for m in pymc_counts
        if pymc_counts[m] >= N_RUNS and stan_counts.get(m, 0) >= N_RUNS
    }


# ── main ───────────────────────────────────────────────────────────────────


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_models = discover_models()
    all_results = load_previous_results()
    done = models_already_done(all_results)

    remaining = [
        (p, m, d) for p, m, d in all_models
        if m not in done and m not in SKIP_MODELS
    ]

    print("=" * 100)
    print("PyMC (nutpie/numba) vs Stan (cmdstan) — Posteriordb Benchmark")
    print(f"Config: {DRAWS} draws, {TUNE} tune, {CHAINS} chains, {N_RUNS} runs per model")
    print(f"Total models: {len(all_models)}, already done: {len(done)}, "
          f"skipped: {len(SKIP_MODELS)}, remaining: {len(remaining)}")
    print("=" * 100)
    sys.stdout.flush()

    for idx, (posterior_name, model_name, data_name) in enumerate(remaining):
        print(f"\n{'─' * 80}")
        print(f"[{idx+1}/{len(remaining)}] Model: {model_name} (data: {data_name})")
        print(f"{'─' * 80}")
        sys.stdout.flush()

        data = load_data(data_name)

        for run_idx in range(N_RUNS):
            print(f"\n  Run {run_idx + 1}/{N_RUNS}")

            # PyMC
            print(f"    PyMC (nutpie/numba)...", end=" ", flush=True)
            pymc_result = benchmark_pymc(model_name, data, run_idx)
            all_results.append(pymc_result)
            if pymc_result["success"]:
                print(
                    f"compile={pymc_result['compile_time_s']:.1f}s "
                    f"sample={pymc_result['sample_time_s']:.1f}s "
                    f"ESS={pymc_result['ess_bulk_min']:.0f} "
                    f"Rhat={pymc_result['rhat_max']:.3f} "
                    f"div={pymc_result['n_divergences']}"
                )
            else:
                print(f"FAILED: {pymc_result.get('error', '?')[:200]}")

            # Stan
            print(f"    Stan (cmdstan)...", end=" ", flush=True)
            stan_result = benchmark_stan(model_name, data, run_idx)
            all_results.append(stan_result)
            if stan_result["success"]:
                print(
                    f"compile={stan_result['compile_time_s']:.1f}s "
                    f"sample={stan_result['sample_time_s']:.1f}s "
                    f"ESS={stan_result['ess_bulk_min']:.0f} "
                    f"Rhat={stan_result['rhat_max']:.3f} "
                    f"div={stan_result['n_divergences']}"
                )
            else:
                print(f"FAILED: {stan_result.get('error', '?')[:200]}")

            sys.stdout.flush()

            # Timeout check
            pymc_time = pymc_result.get("sample_time_s", 0) or 0
            stan_time = stan_result.get("sample_time_s", 0) or 0
            if max(pymc_time, stan_time) > PER_MODEL_TIMEOUT:
                print(f"    TIMEOUT: >{PER_MODEL_TIMEOUT}s, skipping remaining runs")
                break

        gc.collect()
        save_results(all_results)

    print_summary(all_results)
    save_results(all_results)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
