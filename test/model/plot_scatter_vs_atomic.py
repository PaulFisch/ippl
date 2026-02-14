import pandas as pd
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="out.csv")
ap.add_argument("--filter_dim", type=int, default=None)
ap.add_argument("--filter_kernel", default=None)
args = ap.parse_args()

df = pd.read_csv(args.csv)

if args.filter_dim is not None:
    df = df[df["dim"] == args.filter_dim]
if args.filter_kernel is not None:
    df = df[df["kernel"] == args.filter_kernel]

# One common x-axis choice: particle count
# You can also use grid, method, etc.
df = df.sort_values(["nParticles", "method", "sort"])

# Compute GUpdates/s and baseline GOp/s
df["scatter_GUpdates_s"] = df["scatter_updates_per_s"] / 1e9
df["atomicNoCont_GOps_s"] = df["atomic_nocont_ops_per_s"] / 1e9

# Plot scatter throughput and a “baseline line” for atomic nocont (same y-units is not strictly identical),
# so we also plot utilization on a second axis.
fig, ax1 = plt.subplots()

for (method, sort), g in df.groupby(["method", "sort"]):
    ax1.plot(g["nParticles"], g["scatter_GUpdates_s"], marker="o", label=f"Scatter {method} sort={sort}")

ax1.set_xlabel("nParticles (per rank)")
ax1.set_ylabel("Scatter throughput (GUpdates/s)")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(df["nParticles"], df["util"], linestyle="--", marker="x", label="Utilization vs atomic_nocont")
ax2.set_ylabel("util = scatter_updates_per_s / atomic_nocont_ops_per_s")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.show()
