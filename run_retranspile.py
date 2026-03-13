"""Re-run transpilation for failed models using updated transpailer skills.

Usage: cd /Users/twiecki/projects/transpailer && uv run python /Users/twiecki/projects/posteriordb/run_retranspile.py [model_name ...]

If no model names given, runs all models that don't have a PyMC port yet.
"""
import json
import os
import sys
import zipfile
import traceback
from datetime import datetime
from pathlib import Path

# Set BridgeStan path before importing transpailer
os.environ.setdefault("BRIDGESTAN", os.path.expanduser("~/.bridgestan/bridgestan-2.7.0"))

from transpailer.stan_to_pymc import transpile_stan_to_pymc

POSTERIORDB = Path(__file__).resolve().parent / "posterior_database"
STAN_DIR = POSTERIORDB / "models" / "stan"
PYMC_DIR = POSTERIORDB / "models" / "pymc"
DATA_DIR = POSTERIORDB / "data" / "data"
POSTERIORS_DIR = POSTERIORDB / "posteriors"
ERROR_LOG = Path(__file__).resolve().parent / "retranspile_errors.md"


def find_data_for_model(model_name: str) -> dict | None:
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


def get_missing_models() -> list[str]:
    """Find Stan models without a PyMC port."""
    stan_models = {p.stem for p in STAN_DIR.glob("*.stan")}
    pymc_models = {p.stem for p in PYMC_DIR.glob("*.py")}
    missing = sorted(stan_models - pymc_models)
    return missing


def log_error(model_name: str, error_msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(ERROR_LOG, "a") as f:
        f.write(f"\n### {model_name} ({timestamp})\n")
        f.write(f"```\n{error_msg}\n```\n")


def transpile_model(model_name: str) -> bool:
    stan_path = STAN_DIR / f"{model_name}.stan"
    if not stan_path.exists():
        msg = f"Stan file not found: {stan_path}"
        print(f"ERROR: {msg}")
        log_error(model_name, msg)
        return False

    stan_code = stan_path.read_text()
    data = find_data_for_model(model_name)

    if data is None:
        msg = f"No data found for {model_name}"
        print(f"WARNING: {msg}, skipping")
        log_error(model_name, msg)
        return False

    print(f"\n{'='*60}")
    print(f"Transpiling: {model_name}")
    print(f"{'='*60}")

    result = transpile_stan_to_pymc(
        stan_code=stan_code,
        data=data,
        verbose=True,
        max_turns=30,  # Give more iterations for complex models
    )

    if result.success:
        PYMC_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PYMC_DIR / f"{model_name}.py"
        result.save(output_path)
        print(f"\n✓ SAVED: {output_path}")
        print(f"  Tokens: {result.token_usage}")
        return True
    else:
        error_msg = "; ".join(result.validation_errors)
        print(f"\n✗ FAILED: {model_name}")
        for err in result.validation_errors:
            print(f"  - {err}")
        log_error(model_name, f"Validation failed: {error_msg}")
        return False


if __name__ == "__main__":
    models = sys.argv[1:]
    if not models:
        models = get_missing_models()
        print(f"Found {len(models)} models without PyMC ports:")
        for m in models:
            print(f"  - {m}")
        print()

    # Initialize error log
    if not ERROR_LOG.exists():
        ERROR_LOG.write_text("# Retranspilation Error Log\n\n")

    results = {}
    for model_name in models:
        try:
            results[model_name] = transpile_model(model_name)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\nERROR transpiling {model_name}: {e}")
            log_error(model_name, f"Exception: {e}\n{tb}")
            results[model_name] = False

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    print(f"\n{sum(results.values())}/{len(results)} succeeded")
