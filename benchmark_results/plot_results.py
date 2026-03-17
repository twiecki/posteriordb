"""Plot benchmark results: PyMC (nutpie/numba) vs Stan (cmdstan).

Primary metric: seconds per effective sample (sample_time / ESS_bulk_min).
Secondary plots show sampling-only and total time (incl. compilation).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
RESULTS_FILE = RESULTS_DIR / "benchmark_all.json"

with open(RESULTS_FILE) as f:
    results = json.load(f)

# ── Aggregate per model ──
models = {}
for entry in results:
    if not entry.get("success", False):
        continue
    model = entry["model"]
    backend = entry["backend"]
    key = "pymc" if backend == "pymc_nutpie" else "stan"

    if model not in models:
        models[model] = {
            "pymc": [], "stan": [],
            "pymc_compile": [], "stan_compile": [],
            "pymc_ess": [], "stan_ess": [],
            "pymc_total": [], "stan_total": [],
        }

    models[model][key].append(entry["sample_time_s"])
    models[model][f"{key}_compile"].append(entry.get("compile_time_s", 0))
    models[model][f"{key}_ess"].append(entry.get("ess_bulk_min", 0))
    models[model][f"{key}_total"].append(entry.get("total_time_s", entry["sample_time_s"]))


def compute_ratios(data_list):
    """Compute summary stats from a list of model dicts with 'ratio' field."""
    n_pymc = sum(1 for d in data_list if d["ratio"] > 1)
    n_stan = sum(1 for d in data_list if d["ratio"] <= 1)
    ratios = [d["ratio"] for d in data_list if d["ratio"] > 0 and np.isfinite(d["ratio"])]
    geo_mean = np.exp(np.mean(np.log(ratios))) if ratios else 1.0
    return n_pymc, n_stan, geo_mean


def make_log2_hbar(data_list, title, xlabel, filename, summary_label=""):
    """Create a horizontal bar chart with log2 ratio."""
    data_list = sorted(data_list, key=lambda x: x["ratio"])
    names = [d["model"] for d in data_list]
    log_ratios = [np.log2(max(d["ratio"], 1e-10)) for d in data_list]

    fig, ax = plt.subplots(figsize=(12, max(20, len(data_list) * 0.28)))
    colors = ["#e74c3c" if lr < 0 else "#2ecc71" for lr in log_ratios]
    ax.barh(names, log_ratios, color=colors, edgecolor="none", height=0.7)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

    # Custom ticks
    max_abs = max(abs(min(log_ratios)), abs(max(log_ratios)))
    tick_range = int(np.ceil(max_abs)) + 1
    ticks = list(range(-tick_range, tick_range + 1))
    tick_labels = []
    for t in ticks:
        if t == 0:
            tick_labels.append("1x")
        elif t > 0:
            tick_labels.append(f"{2**t:.0f}x PyMC")
        else:
            tick_labels.append(f"{2**(-t):.0f}x Stan")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)

    legend_elements = [
        Patch(facecolor="#2ecc71", label="PyMC faster"),
        Patch(facecolor="#e74c3c", label="Stan faster"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

    n_pymc, n_stan, geo_mean = compute_ratios(data_list)
    ax.text(0.02, 0.98,
            f"PyMC faster: {n_pymc}/{len(data_list)}  |  Stan faster: {n_stan}/{len(data_list)}  |  Geo. mean: {geo_mean:.2f}x"
            + (f"  ({summary_label})" if summary_label else ""),
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.savefig(RESULTS_DIR / f"{filename}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}.png / .pdf  (PyMC: {n_pymc}, Stan: {n_stan}, geo mean: {geo_mean:.2f}x)")


# ── Build comparison data for different metrics ──
ess_data = []       # sec / ESS (sampling only)
ess_total_data = [] # sec / ESS (including compilation)
sample_data = []    # raw sampling time
total_data = []     # total time (compile + sample)

for model, t in models.items():
    if not (t["pymc"] and t["stan"]):
        continue

    pymc_sample = np.mean(t["pymc"])
    stan_sample = np.mean(t["stan"])
    pymc_compile = max(t["pymc_compile"])  # first-run compilation
    stan_compile = max(t["stan_compile"])
    pymc_ess = np.mean(t["pymc_ess"]) if t["pymc_ess"] else 1
    stan_ess = np.mean(t["stan_ess"]) if t["stan_ess"] else 1

    # Avoid division by zero
    pymc_ess = max(pymc_ess, 1)
    stan_ess = max(stan_ess, 1)

    # sec/ESS: lower is better → ratio = (stan sec/ESS) / (pymc sec/ESS) → >1 means PyMC better
    pymc_sec_per_ess = pymc_sample / pymc_ess
    stan_sec_per_ess = stan_sample / stan_ess
    ess_ratio = stan_sec_per_ess / pymc_sec_per_ess if pymc_sec_per_ess > 0 else 1

    # sec/ESS with compilation
    pymc_sec_per_ess_total = (pymc_sample + pymc_compile) / pymc_ess
    stan_sec_per_ess_total = (stan_sample + stan_compile) / stan_ess
    ess_total_ratio = stan_sec_per_ess_total / pymc_sec_per_ess_total if pymc_sec_per_ess_total > 0 else 1

    base = {"model": model, "pymc_sample": pymc_sample, "stan_sample": stan_sample}

    ess_data.append({**base, "ratio": ess_ratio,
                     "pymc_val": pymc_sec_per_ess, "stan_val": stan_sec_per_ess})
    ess_total_data.append({**base, "ratio": ess_total_ratio,
                           "pymc_val": pymc_sec_per_ess_total, "stan_val": stan_sec_per_ess_total})
    sample_data.append({**base, "ratio": stan_sample / pymc_sample if pymc_sample > 0 else 1})
    total_data.append({**base, "ratio": (stan_sample + stan_compile) / (pymc_sample + pymc_compile)
                       if (pymc_sample + pymc_compile) > 0 else 1})


# ── Plot 1: sec/ESS (sampling only) — THE primary metric ──
make_log2_hbar(
    ess_data,
    "PyMC vs Stan — Efficiency: seconds / effective sample (sampling only)",
    "log₂(Stan sec/ESS ÷ PyMC sec/ESS)  —  positive = PyMC more efficient",
    "efficiency_sec_per_ess",
    summary_label="sec/ESS, sampling only",
)

# ── Plot 2: sec/ESS (including compilation) ──
make_log2_hbar(
    ess_total_data,
    "PyMC vs Stan — Efficiency: seconds / effective sample (incl. compilation)",
    "log₂(Stan sec/ESS ÷ PyMC sec/ESS)  —  positive = PyMC more efficient",
    "efficiency_sec_per_ess_total",
    summary_label="sec/ESS, incl. compilation",
)

# ── Plot 3: Raw sampling time (for reference) ──
make_log2_hbar(
    sample_data,
    "PyMC vs Stan — Sampling Time Ratio (log₂)",
    "log₂(Stan time / PyMC time)  —  positive = PyMC faster",
    "speedup_log2_hbar",
    summary_label="sampling time only",
)

# ── Plot 4: Total time including compilation ──
make_log2_hbar(
    total_data,
    "PyMC vs Stan — Total Time Ratio incl. Compilation (log₂)",
    "log₂(Stan total / PyMC total)  —  positive = PyMC faster",
    "speedup_total_log2_hbar",
    summary_label="incl. compilation",
)

# ── Plot 5: Compilation time scatter ──
comp_data = []
for model, t in models.items():
    if t["pymc_compile"] and t["stan_compile"]:
        pymc_comp = max(t["pymc_compile"])
        stan_comp = max(t["stan_compile"])
        if pymc_comp > 0 and stan_comp > 0:
            comp_data.append({"model": model, "pymc": pymc_comp, "stan": stan_comp})

if comp_data:
    fig, ax = plt.subplots(figsize=(10, 8))
    pymc_c = [d["pymc"] for d in comp_data]
    stan_c = [d["stan"] for d in comp_data]

    ax.scatter(stan_c, pymc_c, alpha=0.6, s=40, c="#3498db", edgecolors="white", linewidth=0.5)
    max_val = max(max(pymc_c), max(stan_c)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal")
    ax.set_xlabel("Stan compilation time (s)", fontsize=12)
    ax.set_ylabel("PyMC numba compilation time (s)", fontsize=12)
    ax.set_title("Compilation Time: PyMC (numba) vs Stan (C++)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "compilation_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved compilation_scatter.png")

# ── Summary ──
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
n_p, n_s, gm = compute_ratios(ess_data)
print(f"sec/ESS (sampling):     PyMC {n_p}/{len(ess_data)}, Stan {n_s}/{len(ess_data)}, geo mean {gm:.2f}x")
n_p, n_s, gm = compute_ratios(ess_total_data)
print(f"sec/ESS (total):        PyMC {n_p}/{len(ess_total_data)}, Stan {n_s}/{len(ess_total_data)}, geo mean {gm:.2f}x")
n_p, n_s, gm = compute_ratios(sample_data)
print(f"Sampling time:          PyMC {n_p}/{len(sample_data)}, Stan {n_s}/{len(sample_data)}, geo mean {gm:.2f}x")
n_p, n_s, gm = compute_ratios(total_data)
print(f"Total time (w/ compile): PyMC {n_p}/{len(total_data)}, Stan {n_s}/{len(total_data)}, geo mean {gm:.2f}x")
