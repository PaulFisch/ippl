#!/usr/bin/env python3
"""
Plot NUFFT density sweep benchmark results.

Compares native implementations vs cuFINUFFT across particle densities.

Expected CSV columns:
  density, n_particles, n_modes, type, method, time_ms, throughput_mpts

Usage:
    python plot_density_sweep.py -i density_sweep_results.csv -o fig_density
    python plot_density_sweep.py -i density_sweep_results.csv -o fig_density --speedup
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'mathtext.fontset': 'stix',

        # Axes
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.axisbelow': True,

        # Ticks
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Legend
        'legend.fontsize': 7,
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,

        # Figure
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # Grid
        'grid.linewidth': 0.3,
        'grid.alpha': 0.5,

        # Lines
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })


# Okabe-Ito colorblind-friendly palette
COLORS = {
    # Type 1 (scatter) methods
    'Native_OutputFocused': '#0072B2',  # Blue
    'Native_Tiled': '#009E73',          # Bluish green
    'Native_Atomic': '#E69F00',         # Orange
    # Type 2 (gather) methods
    'Native_AtomicSort': '#56B4E9',     # Sky blue
    # Reference
    'cuFINUFFT': '#D55E00',             # Vermillion
}

MARKERS = {
    'Native_OutputFocused': 'o',
    'Native_Tiled': '^',
    'Native_Atomic': 's',
    'Native_AtomicSort': 'D',
    'cuFINUFFT': 'X',
}

LABELS = {
    'Native_OutputFocused': 'Grid-Parallel',
    'Native_Tiled': 'Tiled',
    'Native_Atomic': 'Atomic',
    'Native_AtomicSort': 'Sorted',
    'cuFINUFFT': 'cuFINUFFT',
}


def load_data(filepath):
    """Load and preprocess benchmark CSV."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


def plot_throughput_vs_density(df, output_prefix):
    """Create line plot of throughput vs particle density."""
    setup_style()

    type1_df = df[df['type'] == 1]
    type2_df = df[df['type'] == 2]

    type1_order = ['Native_OutputFocused', 'Native_Tiled', 'Native_Atomic', 'cuFINUFFT']
    type2_order = ['Native_Atomic', 'Native_AtomicSort', 'cuFINUFFT']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Type 1 (spreading) ---
    ax = axes[0]
    for method in type1_order:
        mdf = type1_df[type1_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue
        ax.plot(mdf['density'], mdf['throughput_mpts'],
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#666666'),
                label=LABELS.get(method, method),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-1 (spreading)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    # --- Type 2 (interpolation) ---
    ax = axes[1]
    for method in type2_order:
        mdf = type2_df[type2_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue
        ax.plot(mdf['density'], mdf['throughput_mpts'],
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#666666'),
                label=LABELS.get(method, method),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-2 (interpolation)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0)

    plt.savefig(f'{output_prefix}_throughput.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_throughput.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}_throughput.pdf, {output_prefix}_throughput.png")


def plot_time_vs_density(df, output_prefix):
    """Create line plot of execution time vs particle density."""
    setup_style()

    type1_df = df[df['type'] == 1]
    type2_df = df[df['type'] == 2]

    type1_order = ['Native_OutputFocused', 'Native_Tiled', 'Native_Atomic', 'cuFINUFFT']
    type2_order = ['Native_Atomic', 'Native_AtomicSort', 'cuFINUFFT']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Type 1 ---
    ax = axes[0]
    for method in type1_order:
        mdf = type1_df[type1_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue
        ax.plot(mdf['density'], mdf['time_ms'],
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#666666'),
                label=LABELS.get(method, method),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Type-1 (spreading)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    # --- Type 2 ---
    ax = axes[1]
    for method in type2_order:
        mdf = type2_df[type2_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue
        ax.plot(mdf['density'], mdf['time_ms'],
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#666666'),
                label=LABELS.get(method, method),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Type-2 (interpolation)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0)

    plt.savefig(f'{output_prefix}_time.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_time.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}_time.pdf, {output_prefix}_time.png")


def plot_speedup_vs_density(df, output_prefix):
    """Create line plot of speedup over cuFINUFFT vs particle density."""
    setup_style()

    type1_df = df[df['type'] == 1]
    type2_df = df[df['type'] == 2]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Type 1 ---
    ax = axes[0]
    cufinufft_t1 = type1_df[type1_df['method'] == 'cuFINUFFT'].set_index('density')['time_ms']

    for method in ['Native_OutputFocused', 'Native_Tiled', 'Native_Atomic']:
        mdf = type1_df[type1_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue

        speedups = []
        densities = []
        for _, row in mdf.iterrows():
            if row['density'] in cufinufft_t1.index:
                ref_time = cufinufft_t1.loc[row['density']]
                speedups.append(ref_time / row['time_ms'])
                densities.append(row['density'])

        if speedups:
            ax.plot(densities, speedups,
                    marker=MARKERS.get(method, 'o'),
                    color=COLORS.get(method, '#666666'),
                    label=LABELS.get(method, method),
                    markeredgecolor='black', markeredgewidth=0.3)

    ax.axhline(y=1.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.7, label='cuFINUFFT')
    ax.set_xscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Speedup vs cuFINUFFT')
    ax.set_title('Type-1 (spreading)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    # --- Type 2 ---
    ax = axes[1]
    cufinufft_t2 = type2_df[type2_df['method'] == 'cuFINUFFT'].set_index('density')['time_ms']

    for method in ['Native_Atomic', 'Native_AtomicSort']:
        mdf = type2_df[type2_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue

        speedups = []
        densities = []
        for _, row in mdf.iterrows():
            if row['density'] in cufinufft_t2.index:
                ref_time = cufinufft_t2.loc[row['density']]
                speedups.append(ref_time / row['time_ms'])
                densities.append(row['density'])

        if speedups:
            ax.plot(densities, speedups,
                    marker=MARKERS.get(method, 'o'),
                    color=COLORS.get(method, '#666666'),
                    label=LABELS.get(method, method),
                    markeredgecolor='black', markeredgewidth=0.3)

    ax.axhline(y=1.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.7, label='cuFINUFFT')
    ax.set_xscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Speedup vs cuFINUFFT')
    ax.set_title('Type-2 (interpolation)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0)

    plt.savefig(f'{output_prefix}_speedup.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_speedup.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}_speedup.pdf, {output_prefix}_speedup.png")


def plot_combined(df, output_prefix):
    """Create a 2x2 combined plot with throughput and speedup."""
    setup_style()

    type1_df = df[df['type'] == 1]
    type2_df = df[df['type'] == 2]

    type1_order = ['Native_OutputFocused', 'Native_Tiled', 'Native_Atomic', 'cuFINUFFT']
    type2_order = ['Native_Atomic', 'Native_AtomicSort', 'cuFINUFFT']

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5))

    # --- Row 1: Throughput ---
    # Type 1
    ax = axes[0, 0]
    for method in type1_order:
        mdf = type1_df[type1_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue
        ax.plot(mdf['density'], mdf['throughput_mpts'],
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#666666'),
                label=LABELS.get(method, method),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-1 (spreading)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='upper left', fontsize=6)

    # Type 2
    ax = axes[0, 1]
    for method in type2_order:
        mdf = type2_df[type2_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue
        ax.plot(mdf['density'], mdf['throughput_mpts'],
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, '#666666'),
                label=LABELS.get(method, method),
                markeredgecolor='black', markeredgewidth=0.3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-2 (interpolation)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='upper left', fontsize=6)

    # --- Row 2: Speedup ---
    # Type 1
    ax = axes[1, 0]
    cufinufft_t1 = type1_df[type1_df['method'] == 'cuFINUFFT'].set_index('density')['time_ms']

    for method in ['Native_OutputFocused', 'Native_Tiled', 'Native_Atomic']:
        mdf = type1_df[type1_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue

        speedups, densities = [], []
        for _, row in mdf.iterrows():
            if row['density'] in cufinufft_t1.index:
                speedups.append(cufinufft_t1.loc[row['density']] / row['time_ms'])
                densities.append(row['density'])

        if speedups:
            ax.plot(densities, speedups,
                    marker=MARKERS.get(method, 'o'),
                    color=COLORS.get(method, '#666666'),
                    label=LABELS.get(method, method),
                    markeredgecolor='black', markeredgewidth=0.3)

    ax.axhline(y=1.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Speedup vs cuFINUFFT')
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best', fontsize=6)

    # Type 2
    ax = axes[1, 1]
    cufinufft_t2 = type2_df[type2_df['method'] == 'cuFINUFFT'].set_index('density')['time_ms']

    for method in ['Native_Atomic', 'Native_AtomicSort']:
        mdf = type2_df[type2_df['method'] == method].sort_values('density')
        if mdf.empty:
            continue

        speedups, densities = [], []
        for _, row in mdf.iterrows():
            if row['density'] in cufinufft_t2.index:
                speedups.append(cufinufft_t2.loc[row['density']] / row['time_ms'])
                densities.append(row['density'])

        if speedups:
            ax.plot(densities, speedups,
                    marker=MARKERS.get(method, 'o'),
                    color=COLORS.get(method, '#666666'),
                    label=LABELS.get(method, method),
                    markeredgecolor='black', markeredgewidth=0.3)

    ax.axhline(y=1.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Density (particles/mode)')
    ax.set_ylabel('Speedup vs cuFINUFFT')
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best', fontsize=6)

    plt.tight_layout(h_pad=2.5, w_pad=2.5)

    plt.savefig(f'{output_prefix}_combined.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_combined.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}_combined.pdf, {output_prefix}_combined.png")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("DENSITY SWEEP BENCHMARK SUMMARY")
    print("=" * 70)

    n_modes = df['n_modes'].iloc[0]
    grid_size = int(round(n_modes ** (1/3)))

    print(f"\nGrid: {grid_size}³ = {n_modes:,} modes")
    print(f"Density range: {df['density'].min():.0e} - {df['density'].max():.0e} particles/mode")

    for nufft_type in [1, 2]:
        type_df = df[df['type'] == nufft_type]
        type_name = "Type-1 (spreading)" if nufft_type == 1 else "Type-2 (interpolation)"

        print(f"\n--- {type_name} ---")

        methods = type_df['method'].unique()
        cufinufft_df = type_df[type_df['method'] == 'cuFINUFFT'].set_index('density')

        for method in sorted(methods):
            if method == 'cuFINUFFT':
                continue

            mdf = type_df[type_df['method'] == method]
            label = LABELS.get(method, method)

            # Compute average speedup
            speedups = []
            for _, row in mdf.iterrows():
                if row['density'] in cufinufft_df.index:
                    ref_time = cufinufft_df.loc[row['density'], 'time_ms']
                    speedups.append(ref_time / row['time_ms'])

            if speedups:
                avg_speedup = np.mean(speedups)
                min_speedup = np.min(speedups)
                max_speedup = np.max(speedups)
                print(f"  {label:20s}: avg {avg_speedup:.2f}x  (range: {min_speedup:.2f}x - {max_speedup:.2f}x)")


def print_latex_table(df):
    """Print results as LaTeX table for selected densities."""
    n_modes = df['n_modes'].iloc[0]
    grid_size = int(round(n_modes ** (1/3)))

    # Select representative densities
    selected_densities = [1e-3, 0.1, 1.0, 10.0, 100.0, 1000.0]
    available = df['density'].unique()
    selected_densities = [d for d in selected_densities if d in available]

    print("\n% LaTeX table")
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{Performance across particle densities. Grid $N={grid_size}^3$.}}")
    print("\\label{tab:density-sweep}")
    print("\\small")
    print("\\begin{tabular}{lrr" + "r" * len(selected_densities) + "}")
    print("\\toprule")

    # Header
    header = "Type & Method & "
    header += " & ".join([f"$\\rho={d:.0e}$" for d in selected_densities])
    header += " \\\\"
    print(header)
    print("\\midrule")

    for nufft_type, type_label, methods in [
        (1, "Type-1", ['Native_OutputFocused', 'Native_Tiled', 'Native_Atomic', 'cuFINUFFT']),
        (2, "Type-2", ['Native_Atomic', 'Native_AtomicSort', 'cuFINUFFT'])
    ]:
        type_df = df[df['type'] == nufft_type]
        first = True

        for method in methods:
            mdf = type_df[type_df['method'] == method]
            if mdf.empty:
                continue

            label = LABELS.get(method, method)
            type_col = type_label if first else ""
            first = False

            row_str = f"{type_col:7s} & {label:15s}"

            for density in selected_densities:
                ddf = mdf[np.isclose(mdf['density'], density, rtol=0.1)]
                if not ddf.empty:
                    tput = ddf['throughput_mpts'].values[0]
                    row_str += f" & {tput:6.0f}"
                else:
                    row_str += " & --"

            row_str += " \\\\"
            print(row_str)

        if nufft_type == 1:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot NUFFT density sweep benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_density_sweep.py -i density_sweep_results.csv -o fig_density
  python plot_density_sweep.py -i density_sweep_results.csv -o fig_density --speedup
  python plot_density_sweep.py -i density_sweep_results.csv -o fig_density --combined
  python plot_density_sweep.py -i density_sweep_results.csv --latex
        """
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file')
    parser.add_argument('--output', '-o', default='density_sweep',
                        help='Output file prefix')
    parser.add_argument('--speedup', action='store_true',
                        help='Plot speedup vs cuFINUFFT')
    parser.add_argument('--time', action='store_true',
                        help='Plot execution time')
    parser.add_argument('--combined', action='store_true',
                        help='Create combined 2x2 plot')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX table')
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = load_data(args.input)

    print(f"Loaded {len(df)} rows")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Densities: {sorted(df['density'].unique())}")
    print(f"Types: {sorted(df['type'].unique())}")

    print_summary(df)

    if args.latex:
        print_latex_table(df)

    # Generate plots
    plot_throughput_vs_density(df, args.output)

    if args.time:
        plot_time_vs_density(df, args.output)

    if args.speedup:
        plot_speedup_vs_density(df, args.output)

    if args.combined:
        plot_combined(df, args.output)


if __name__ == '__main__':
    main()