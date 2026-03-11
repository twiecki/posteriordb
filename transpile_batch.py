"""Batch transpile Stan models from posteriordb to PyMC v5."""

import json
import sys
import zipfile
from pathlib import Path

# Add pymc-rust-ai-compiler to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pymc-rust-ai-compiler"))

from pymc_rust_compiler.stan_to_pymc import transpile_stan_to_pymc

POSTERIORDB = Path(__file__).resolve().parent / "posterior_database"
STAN_DIR = POSTERIORDB / "models" / "stan"
PYMC_DIR = POSTERIORDB / "models" / "pymc"
DATA_DIR = POSTERIORDB / "data" / "data"
POSTERIORS_DIR = POSTERIORDB / "posteriors"


def find_data_for_model(model_name: str) -> dict | None:
    """Find data for a model by looking at posteriors that reference it."""
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


def transpile_model(model_name: str) -> bool:
    """Transpile a single Stan model to PyMC."""
    stan_path = STAN_DIR / f"{model_name}.stan"
    if not stan_path.exists():
        print(f"ERROR: Stan file not found: {stan_path}")
        return False

    stan_code = stan_path.read_text()
    data = find_data_for_model(model_name)

    if data is None:
        print(f"WARNING: No data found for {model_name}, skipping")
        return False

    print(f"\n{'='*60}")
    print(f"Transpiling: {model_name}")
    print(f"{'='*60}")

    result = transpile_stan_to_pymc(
        stan_code=stan_code,
        data=data,
        verbose=True,
    )

    if result.success:
        # Save to posteriordb pymc directory
        PYMC_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PYMC_DIR / f"{model_name}.py"
        result.save(output_path)
        print(f"\nSAVED: {output_path}")
        print(f"Tokens: {result.token_usage}")
        return True
    else:
        print(f"\nFAILED: {model_name}")
        for err in result.validation_errors:
            print(f"  - {err}")
        return False


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else [
        "eight_schools_centered",
        "eight_schools_noncentered",
        "blr",
        "earn_height",
        "wells_dist",
        "radon_pooled",
    ]

    results = {}
    for model_name in models:
        try:
            results[model_name] = transpile_model(model_name)
        except Exception as e:
            print(f"\nERROR transpiling {model_name}: {e}")
            results[model_name] = False

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {'OK' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(results.values())}/{len(results)} succeeded")
