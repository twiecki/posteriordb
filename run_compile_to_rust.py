"""Compile transpiled PyMC models to Rust via the transpailer agentic loop.

Usage: cd /Users/twiecki/projects/transpailer && uv run python /Users/twiecki/projects/posteriordb/run_compile_to_rust.py [model_name ...]

If no model names given, compiles all PyMC models that don't have a Rust port yet.

Each model's output is saved to:
  posterior_database/models/rust/<model_name>/
    ├── generated.rs      # The compiled Rust logp+gradient code
    └── results.tsv       # Optimization trace (for analysis/plotting)

Aggregate statistics are saved to:
  rust_compile_results.json   # Per-model stats (tokens, timing, success)
"""

import importlib.util
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

from transpailer.compiler import compile_model

POSTERIORDB = Path(__file__).resolve().parent / "posterior_database"
PYMC_DIR = POSTERIORDB / "models" / "pymc"
RUST_DIR = POSTERIORDB / "models" / "rust"
DATA_DIR = POSTERIORDB / "data" / "data"
POSTERIORS_DIR = POSTERIORDB / "posteriors"
RESULTS_JSON = Path(__file__).resolve().parent / "rust_compile_results.json"
ERROR_LOG = Path(__file__).resolve().parent / "rust_compile_errors.md"


def find_data_for_model(model_name: str) -> dict | None:
    """Find data for a model by scanning posteriors/*.json."""
    import zipfile

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
    """Load a transpiled PyMC model and return (model, source_code)."""
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt

    pymc_path = PYMC_DIR / f"{model_name}.py"
    source_code = pymc_path.read_text()

    spec = importlib.util.spec_from_file_location(model_name, pymc_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["pm"] = pm
    mod.__dict__["pt"] = pt
    mod.__dict__["np"] = np
    spec.loader.exec_module(mod)
    model = mod.make_model(data)
    return model, source_code


def get_available_models() -> list[str]:
    """List all PyMC models available for compilation."""
    return sorted(p.stem for p in PYMC_DIR.glob("*.py"))


def get_missing_rust_models() -> list[str]:
    """Find PyMC models without a Rust port."""
    pymc_models = {p.stem for p in PYMC_DIR.glob("*.py")}
    rust_models = set()
    if RUST_DIR.exists():
        rust_models = {
            p.name
            for p in RUST_DIR.iterdir()
            if p.is_dir() and (p / "generated.rs").exists()
        }
    return sorted(pymc_models - rust_models)


def log_error(model_name: str, error_msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(ERROR_LOG, "a") as f:
        f.write(f"\n### {model_name} ({timestamp})\n")
        f.write(f"```\n{error_msg}\n```\n")


def compile_single_model(model_name: str, max_turns: int = 30) -> dict:
    """Compile a single PyMC model to Rust. Returns stats dict."""
    import time

    pymc_path = PYMC_DIR / f"{model_name}.py"
    if not pymc_path.exists():
        msg = f"PyMC file not found: {pymc_path}"
        print(f"ERROR: {msg}")
        log_error(model_name, msg)
        return {"model": model_name, "success": False, "error": msg}

    data = find_data_for_model(model_name)
    if data is None:
        msg = f"No data found for {model_name}"
        print(f"WARNING: {msg}, skipping")
        log_error(model_name, msg)
        return {"model": model_name, "success": False, "error": msg}

    print(f"\n{'=' * 60}")
    print(f"Compiling to Rust: {model_name}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Load the PyMC model
    model, source_code = load_pymc_model(model_name, data)

    # Set up output directory
    output_dir = RUST_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compile to Rust
    result = compile_model(
        model=model,
        source_code=source_code,
        max_turns=max_turns,
        verbose=True,
    )

    elapsed = time.time() - t0

    # Save outputs
    stats = {
        "model": model_name,
        "success": result.success,
        "n_tool_calls": result.n_tool_calls,
        "conversation_turns": result.conversation_turns,
        "n_build_attempts": result.n_attempts,
        "token_usage": result.token_usage,
        "us_per_eval": result.us_per_eval,
        "wall_time_s": round(elapsed, 1),
        "validation_errors": result.validation_errors,
    }

    if result.success:
        # Save generated.rs
        rs_path = output_dir / "generated.rs"
        rs_path.write_text(result.rust_code)

        # Save optimization trace as TSV
        if result.optimization_log:
            tsv_path = output_dir / "results.tsv"
            result.write_results_tsv(tsv_path)
            stats["trace_file"] = str(tsv_path)

        # Copy the full build directory for reproducibility
        if result.build_dir:
            build_dest = output_dir / "build"
            if build_dest.exists():
                shutil.rmtree(build_dest)
            shutil.copytree(result.build_dir, build_dest)

        print(f"\n✓ SAVED: {output_dir}")
        print(f"  Tokens: {result.token_usage}")
        print(f"  us/eval: {result.us_per_eval}")
        print(f"  Wall time: {elapsed:.1f}s")
    else:
        # Still save the trace even for failures (useful for analysis)
        if result.optimization_log:
            tsv_path = output_dir / "results.tsv"
            result.write_results_tsv(tsv_path)
            stats["trace_file"] = str(tsv_path)

        # Save partial Rust code if any was generated
        if result.rust_code:
            rs_path = output_dir / "generated.rs.partial"
            rs_path.write_text(result.rust_code)

        error_msg = "; ".join(result.validation_errors)
        print(f"\n✗ FAILED: {model_name}")
        for err in result.validation_errors:
            print(f"  - {err}")
        log_error(model_name, f"Compilation failed: {error_msg}")

    return stats


def save_aggregate_results(all_stats: list[dict]):
    """Save aggregate results to JSON for later analysis."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(all_stats),
        "succeeded": sum(1 for s in all_stats if s["success"]),
        "failed": sum(1 for s in all_stats if not s["success"]),
        "total_tokens": sum(
            s.get("token_usage", {}).get("total_tokens", 0) for s in all_stats
        ),
        "models": all_stats,
    }

    # Merge with existing results if present
    if RESULTS_JSON.exists():
        existing = json.loads(RESULTS_JSON.read_text())
        existing_models = {m["model"]: m for m in existing.get("models", [])}
        for s in all_stats:
            existing_models[s["model"]] = s
        summary["models"] = list(existing_models.values())
        summary["total_models"] = len(summary["models"])
        summary["succeeded"] = sum(1 for m in summary["models"] if m["success"])
        summary["failed"] = sum(1 for m in summary["models"] if not m["success"])
        summary["total_tokens"] = sum(
            m.get("token_usage", {}).get("total_tokens", 0)
            for m in summary["models"]
        )

    RESULTS_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nAggregate results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    models = sys.argv[1:]
    if not models:
        models = get_missing_rust_models()
        if not models:
            print("All PyMC models already have Rust ports!")
            available = get_available_models()
            print(f"({len(available)} models available, use --force to recompile)")
            sys.exit(0)
        print(f"Found {len(models)} PyMC models without Rust ports:")
        for m in models:
            print(f"  - {m}")
        print()

    # Handle --force flag to recompile existing models
    if "--force" in models:
        models.remove("--force")
        if not models:
            models = get_available_models()

    # Initialize error log
    if not ERROR_LOG.exists():
        ERROR_LOG.write_text("# Rust Compilation Error Log\n\n")

    all_stats = []
    for model_name in models:
        try:
            stats = compile_single_model(model_name)
            all_stats.append(stats)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\nERROR compiling {model_name}: {e}")
            log_error(model_name, f"Exception: {e}\n{tb}")
            all_stats.append(
                {"model": model_name, "success": False, "error": str(e)}
            )

        # Save intermediate results after each model
        save_aggregate_results(all_stats)

    # Final summary
    print(f"\n\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for stats in all_stats:
        status = "✓" if stats["success"] else "✗"
        extra = ""
        if stats.get("us_per_eval"):
            extra = f" ({stats['us_per_eval']:.1f} us/eval)"
        elif stats.get("error"):
            extra = f" ({stats['error'][:50]})"
        print(f"  {status}  {stats['model']}{extra}")

    succeeded = sum(1 for s in all_stats if s["success"])
    print(f"\n{succeeded}/{len(all_stats)} succeeded")
