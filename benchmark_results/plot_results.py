"""Plot benchmark results: PyMC (nutpie/numba) vs Stan (cmdstan)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "benchmark_all.json"

with open(RESULTS_FILE) as f:
    results = json.load(f)

# Aggregate: for each model, get mean sampling time for pymc and stan
models = {}
for entry in results:
    if not entry.get("success", False):
        continue
    model = entry["model"]
    backend = entry["backend"]
    sample_time = entry["sample_time_s"]

    if model not in models:
        models[model] = {"pymc": [], "stan": [], "pymc_compile": [], "stan_compile": []}

    if backend == "pymc_nutpie":
        models[model]["pymc"].append(sample_time)
        models[model]["pymc_compile"].append(entry.get("compile_time_s", 0))
    elif backend == "stan_cmdstan":
        models[model]["stan"].append(sample_time)
        models[model]["stan_compile"].append(entry.get("compile_time_s", 0))

# Compute speedup for models where both ran
data = []
for model, times in models.items():
    if times["pymc"] and times["stan"]:
        pymc_mean = np.mean(times["pymc"])
        stan_mean = np.mean(times["stan"])
        # Positive = PyMC faster, negative = Stan faster
        pct_speedup = (stan_mean - pymc_mean) / stan_mean * 100
        ratio = stan_mean / pymc_mean if pymc_mean > 0 else float("inf")
        data.append({
            "model": model,
            "pymc_mean": pymc_mean,
            "stan_mean": stan_mean,
            "pct_speedup": pct_speedup,
            "ratio": ratio,
        })

# Sort by speedup
data.sort(key=lambda x: x["pct_speedup"])

model_names = [d["model"] for d in data]
pct_speedups = [d["pct_speedup"] for d in data]

# ── Plot 1: Horizontal bar chart of % speedup ──
fig, ax = plt.subplots(figsize=(12, max(20, len(data) * 0.28)))

colors = ["#e74c3c" if s < 0 else "#2ecc71" for s in pct_speedups]
bars = ax.barh(model_names, pct_speedups, color=colors, edgecolor="none", height=0.7)

ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("% Speed-up (positive = PyMC faster, negative = Stan faster)", fontsize=12)
ax.set_title("PyMC (nutpie/numba) vs Stan (cmdstan) — Sampling Speed-up %", fontsize=14, fontweight="bold")
ax.tick_params(axis="y", labelsize=8)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", label="PyMC faster"),
    Patch(facecolor="#e74c3c", label="Stan faster"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

# Summary stats
n_pymc = sum(1 for s in pct_speedups if s > 0)
n_stan = sum(1 for s in pct_speedups if s < 0)
geo_mean = np.exp(np.mean(np.log([d["ratio"] for d in data])))
ax.text(0.02, 0.98, f"PyMC faster: {n_pymc}/{len(data)}  |  Stan faster: {n_stan}/{len(data)}  |  Geo. mean ratio: {geo_mean:.2f}x",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

plt.tight_layout()
plt.savefig(Path(__file__).parent / "speedup_pct_hbar.png", dpi=150, bbox_inches="tight")
plt.savefig(Path(__file__).parent / "speedup_pct_hbar.pdf", bbox_inches="tight")
print("Saved speedup_pct_hbar.png / .pdf")

# ── Plot 2: Log-scale ratio (more intuitive for large differences) ──
fig2, ax2 = plt.subplots(figsize=(12, max(20, len(data) * 0.28)))

# Sort by log ratio
data.sort(key=lambda x: x["ratio"])
model_names2 = [d["model"] for d in data]
log_ratios = [np.log2(d["ratio"]) for d in data]

colors2 = ["#e74c3c" if lr < 0 else "#2ecc71" for lr in log_ratios]
ax2.barh(model_names2, log_ratios, color=colors2, edgecolor="none", height=0.7)

ax2.axvline(x=0, color="black", linewidth=0.8)
ax2.set_xlabel("log₂(Stan time / PyMC time)  —  positive = PyMC faster", fontsize=12)
ax2.set_title("PyMC (nutpie/numba) vs Stan (cmdstan) — Sampling Time Ratio (log₂)", fontsize=14, fontweight="bold")
ax2.tick_params(axis="y", labelsize=8)

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
ax2.set_xticks(ticks)
ax2.set_xticklabels(tick_labels, fontsize=8, rotation=45)

ax2.legend(handles=legend_elements, loc="lower right", fontsize=11)
ax2.text(0.02, 0.98, f"PyMC faster: {n_pymc}/{len(data)}  |  Stan faster: {n_stan}/{len(data)}  |  Geo. mean ratio: {geo_mean:.2f}x",
        transform=ax2.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

plt.tight_layout()
plt.savefig(Path(__file__).parent / "speedup_log2_hbar.png", dpi=150, bbox_inches="tight")
plt.savefig(Path(__file__).parent / "speedup_log2_hbar.pdf", bbox_inches="tight")
print("Saved speedup_log2_hbar.png / .pdf")

# ── Plot 3: Compilation time comparison ──
# Compilation: use max compile time per model (first run has it)
comp_data = []
for model, times_dict in models.items():
    if times_dict["pymc_compile"] and times_dict["stan_compile"]:
        pymc_comp = max(times_dict["pymc_compile"])
        stan_comp = max(times_dict["stan_compile"])
        if pymc_comp > 0 and stan_comp > 0:
            comp_data.append({"model": model, "pymc": pymc_comp, "stan": stan_comp})

if comp_data:
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    pymc_comp_times = [d["pymc"] for d in comp_data]
    stan_comp_times = [d["stan"] for d in comp_data]

    ax3.scatter(stan_comp_times, pymc_comp_times, alpha=0.6, s=40, c="#3498db", edgecolors="white", linewidth=0.5)

    max_val = max(max(pymc_comp_times), max(stan_comp_times)) * 1.1
    ax3.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal")
    ax3.set_xlabel("Stan compilation time (s)", fontsize=12)
    ax3.set_ylabel("PyMC numba compilation time (s)", fontsize=12)
    ax3.set_title("Compilation Time: PyMC (numba) vs Stan (C++)", fontsize=14, fontweight="bold")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "compilation_scatter.png", dpi=150, bbox_inches="tight")
    print("Saved compilation_scatter.png")

print(f"\nTotal models compared: {len(data)}")
print(f"PyMC faster: {n_pymc}, Stan faster: {n_stan}")
print(f"Geometric mean speedup ratio: {geo_mean:.2f}x (>1 = PyMC faster)")
