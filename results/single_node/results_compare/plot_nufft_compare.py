#!/usr/bin/env python3
"""
Plot single-GPU NUFFT benchmark results comparing IPPL implementations vs cuFINUFFT.

Expected input format (text files):
Configuration: Grid=256^3, Particles=167772160 (10 per point)
================================================================================
  IPPL OutputFocused | Type 1 | Grid: 256^3 | Particles:  167772160 | Time:    335.935 ms | Throughput:   499.42 Mpts/s
  ...

Usage:
    python plot_nufft_benchmark.py [--input-dir DIR] [--output PREFIX]
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import argparse


def parse_benchmark_file(filepath):
    """Parse a single benchmark results file."""
    results = []
    current_grid = None
    current_particles = None
    cluster_name = filepath.stem  # Use filename as cluster identifier

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse configuration header
            config_match = re.match(
                r'Configuration: Grid=(\d+)\^3, Particles=(\d+)',
                line
            )
            if config_match:
                current_grid = int(config_match.group(1))
                current_particles = int(config_match.group(2))
                continue

            # Parse result line
            result_match = re.match(
                r'^\s*(.+?)\s*\|\s*Type\s+(\d+)\s*\|\s*Grid:\s*(\d+)\^3\s*\|'
                r'\s*Particles:\s*(\d+)\s*\|\s*Time:\s*([\d.]+)\s*ms\s*\|'
                r'\s*Throughput:\s*([\d.]+)\s*Mpts/s',
                line
            )
            if result_match:
                method = result_match.group(1).strip()
                nufft_type = int(result_match.group(2))
                grid_size = int(result_match.group(3))
                particles = int(result_match.group(4))
                time_ms = float(result_match.group(5))
                throughput = float(result_match.group(6))

                results.append({
                    'cluster': cluster_name,
                    'method': method,
                    'type': nufft_type,
                    'grid_size': grid_size,
                    'particles': particles,
                    'time_ms': time_ms,
                    'throughput_mpts': throughput
                })

    return results


def load_all_results(input_dir):
    """Load results from all .txt files in directory."""
    all_results = []
    input_path = Path(input_dir)

    for filepath in input_path.glob('*.out'):
        results = parse_benchmark_file(filepath)
        all_results.extend(results)

    if not all_results:
        for filepath in input_path.iterdir():
            if filepath.is_file() and not filepath.suffix:
                results = parse_benchmark_file(filepath)
                all_results.extend(results)

    return pd.DataFrame(all_results)


def filter_methods(df):
    """Filter out methods we don't want to plot."""
    exclude_methods = ['kokkos_nufft', 'IPPL Native']
    return df[~df['method'].isin(exclude_methods)]


def get_method_label(method):
    """Create clean labels for plotting."""
    label_map = {
        'IPPL OutputFocused': 'Grid-Parallel',
        'IPPL Tiled': 'Tiled',
        'IPPL Atomic': 'Atomic',
        'IPPL Warp': 'Warp-Parallel',
        'IPPL Naive': 'Naive',
        'IPPL Atomic Sort': 'Sorted',
        'cuFINUFFT': 'cuFINUFFT'
    }
    return label_map.get(method, method)


def setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        # Font settings - use serif for publication
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
    })


# Professional colorblind-friendly palette (Okabe-Ito)
COLORS = {
    'IPPL OutputFocused': '#0072B2',  # Blue
    'IPPL Warp': '#009E73',           # Bluish green
    'IPPL Tiled': '#009E73',          # Bluish green
    'IPPL Atomic': '#E69F00',         # Orange
    'IPPL Naive': '#E69F00',          # Orange
    'IPPL Atomic Sort': '#56B4E9',    # Sky blue
    'cuFINUFFT': '#D55E00',           # Vermillion
}

# Hatching for additional distinction (useful for B&W printing)
HATCHES = {
    'IPPL OutputFocused': None,
    'IPPL Tiled': None,
    'IPPL Naive': None,
    'IPPL Warp': None,
    'IPPL Atomic': None,
    'IPPL Atomic Sort': None,
    'cuFINUFFT': '///',
}


def plot_single_cluster(df, cluster_name, output_prefix):
    """Create top-tier publication plot with dual y-axes for single cluster."""
    setup_style()

    # Additional refinements for top-tier appearance
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    df = filter_methods(df)
    cluster_df = df[df['cluster'] == cluster_name] if 'cluster' in df.columns else df

    # Method order for each type
    type1_methods = ['IPPL Atomic', 'IPPL Tiled', 'IPPL OutputFocused', 'cuFINUFFT']
    type2_methods = ['IPPL Naive', 'IPPL Atomic Sort', 'IPPL Warp', 'cuFINUFFT']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4),
                                   gridspec_kw={'width_ratios': [4, 3], 'wspace': 0.35})

    type1_df = cluster_df[cluster_df['type'] == 1]
    type2_df = cluster_df[cluster_df['type'] == 2]

    type1_present = [m for m in type1_methods if m in type1_df['method'].values]
    type2_present = [m for m in type2_methods if m in type2_df['method'].values]

    bar_width = 0.65

    # Track methods for legend
    legend_handles = {}

    # ============ TYPE-1 PANEL (LEFT) ============
    x_type1 = np.arange(len(type1_present))

    for i, method in enumerate(type1_present):
        method_data = type1_df[type1_df['method'] == method]
        if not method_data.empty:
            throughput = method_data['throughput_mpts'].values[0]
            bar = ax1.bar(x_type1[i], throughput, bar_width,
                          color=COLORS.get(method, '#666666'),
                          edgecolor='black',
                          linewidth=0.4,
                          hatch=HATCHES.get(method),
                          zorder=3)

            if method not in legend_handles:
                legend_handles[method] = bar[0]

            # Value label
            ax1.annotate(f'{throughput:.0f}',
                         xy=(x_type1[i], throughput),
                         xytext=(0, 2),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=6.5,
                         color='#333333')

    ax1.set_xticks(x_type1)
    ax1.set_xticklabels([get_method_label(m) for m in type1_present],
                        rotation=30, ha='right', rotation_mode='anchor')
    ax1.set_ylabel('Throughput (Mpts/s)')
    ax1.set_title('Type-1 (spreading)', fontsize=10, fontweight='semibold', pad=8)

    type1_max = type1_df['throughput_mpts'].max()
    ax1.set_ylim(0, type1_max * 1.18)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax1.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # ============ TYPE-2 PANEL (RIGHT) ============
    x_type2 = np.arange(len(type2_present))

    for i, method in enumerate(type2_present):
        method_data = type2_df[type2_df['method'] == method]
        if not method_data.empty:
            throughput = method_data['throughput_mpts'].values[0]
            bar = ax2.bar(x_type2[i], throughput, bar_width,
                          color=COLORS.get(method, '#666666'),
                          edgecolor='black',
                          linewidth=0.4,
                          hatch=HATCHES.get(method),
                          zorder=3)

            if method not in legend_handles:
                legend_handles[method] = bar[0]

            # Value label
            ax2.annotate(f'{throughput:.0f}',
                         xy=(x_type2[i], throughput),
                         xytext=(0, 2),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=6.5,
                         color='#333333')

    ax2.set_xticks(x_type2)
    ax2.set_xticklabels([get_method_label(m) for m in type2_present],
                        rotation=30, ha='right', rotation_mode='anchor')
    ax2.set_ylabel('Throughput (Mpts/s)')
    ax2.set_title('Type-2 (interpolation)', fontsize=10, fontweight='semibold', pad=8)

    type2_max = type2_df['throughput_mpts'].max()
    ax2.set_ylim(0, type2_max * 1.18)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax2.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    # ============ UNIFIED LEGEND ============
    legend_order = ['IPPL Atomic', 'IPPL Tiled', 'IPPL OutputFocused',
                    'IPPL Naive', 'IPPL Atomic Sort', 'IPPL Warp', 'cuFINUFFT']
    ordered_handles = []
    ordered_labels = []
    for method in legend_order:
        if method in legend_handles:
            ordered_handles.append(legend_handles[method])
            ordered_labels.append(get_method_label(method))

    fig.legend(ordered_handles, ordered_labels,
               loc='upper center',
               ncol=len(ordered_labels),
               fontsize=7,
               frameon=False,
               bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.0,
               handletextpad=0.4,
               handlelength=1.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.20, left=0.10, right=0.97)

    plt.savefig(f'{output_prefix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}.pdf, {output_prefix}.png")


def plot_multi_cluster(df, output_prefix):
    """Create top-tier publication plot with dual y-axes for different scales."""
    setup_style()

    # Additional refinements for top-tier appearance
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })

    df = filter_methods(df)
    clusters = sorted(df['cluster'].unique())
    n_clusters = len(clusters)

    # Refined color palette - professional and distinguishable
    cluster_colors = {
        'Alps (Nvidia H100)': '#2166AC',    # Strong blue
        'Juwels (Nvidia A100)': '#D6604D',  # Muted red-orange
        'Lumi (AMD MI250X)': '#4DAF4A',     # Green
    }
    # Fallback colors if cluster names don't match
    fallback_colors = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3', '#FF7F00']

    type1_methods = ['IPPL Atomic', 'IPPL Tiled', 'IPPL OutputFocused', 'cuFINUFFT']
    type2_methods = ['IPPL Naive', 'IPPL Atomic Sort', 'IPPL Warp', 'cuFINUFFT']

    # Create figure with two subplots sharing nothing - true dual panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.4),
                                   gridspec_kw={'width_ratios': [4, 3], 'wspace': 0.35})

    # Get methods present for each type
    type1_df = df[df['type'] == 1]
    type2_df = df[df['type'] == 2]

    type1_present = [m for m in type1_methods if m in type1_df['method'].values]
    type2_present = [m for m in type2_methods if m in type2_df['method'].values]

    n_type1 = len(type1_present)
    n_type2 = len(type2_present)

    total_width = 0.82
    bar_width = total_width / n_clusters

    # Track cluster handles for legend
    cluster_handles = {}

    def get_color(cluster, idx):
        return cluster_colors.get(cluster, fallback_colors[idx % len(fallback_colors)])

    # ============ TYPE-1 PANEL (LEFT) ============
    x_type1 = np.arange(n_type1)

    for i, cluster in enumerate(clusters):
        cluster_data = type1_df[type1_df['cluster'] == cluster]
        throughputs = []
        for method in type1_present:
            method_data = cluster_data[cluster_data['method'] == method]
            throughputs.append(method_data['throughput_mpts'].values[0]
                               if not method_data.empty else 0)

        offset = (i - n_clusters / 2 + 0.5) * bar_width
        color = get_color(cluster, i)

        for j, (x, val, method) in enumerate(zip(x_type1, throughputs, type1_present)):
            bar = ax1.bar(x + offset, val, bar_width * 0.9,
                          color=color,
                          edgecolor='black',
                          linewidth=0.4,
                          hatch=HATCHES.get(method),
                          zorder=3)
            if cluster not in cluster_handles:
                cluster_handles[cluster] = bar[0]

    ax1.set_xticks(x_type1)
    ax1.set_xticklabels([get_method_label(m) for m in type1_present],
                        rotation=30, ha='right', rotation_mode='anchor')
    ax1.set_ylabel('Throughput (Mpts/s)')
    ax1.set_title('Type-1 (spreading)', fontsize=10, fontweight='semibold', pad=8)

    # Set y-axis for Type-1
    type1_max = type1_df['throughput_mpts'].max()
    ax1.set_ylim(0, type1_max * 1.12)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))

    # Subtle grid
    ax1.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # ============ TYPE-2 PANEL (RIGHT) ============
    x_type2 = np.arange(n_type2)

    for i, cluster in enumerate(clusters):
        cluster_data = type2_df[type2_df['cluster'] == cluster]
        throughputs = []
        for method in type2_present:
            method_data = cluster_data[cluster_data['method'] == method]
            throughputs.append(method_data['throughput_mpts'].values[0]
                               if not method_data.empty else 0)

        offset = (i - n_clusters / 2 + 0.5) * bar_width
        color = get_color(cluster, i)

        for j, (x, val, method) in enumerate(zip(x_type2, throughputs, type2_present)):
            bar = ax2.bar(x + offset, val, bar_width * 0.9,
                          color=color,
                          edgecolor='black',
                          linewidth=0.4,
                          hatch=HATCHES.get(method),
                          zorder=3)

    ax2.set_xticks(x_type2)
    ax2.set_xticklabels([get_method_label(m) for m in type2_present],
                        rotation=30, ha='right', rotation_mode='anchor')
    ax2.set_ylabel('Throughput (Mpts/s)')
    ax2.set_title('Type-2 (interpolation)', fontsize=10, fontweight='semibold', pad=8)

    # Set y-axis for Type-2 (different scale)
    type2_max = type2_df['throughput_mpts'].max()
    ax2.set_ylim(0, type2_max * 1.12)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))

    # Subtle grid
    ax2.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    # ============ UNIFIED LEGEND ============
    legend_handles_list = [cluster_handles[c] for c in clusters]
    legend_labels = list(clusters)

    # Add hatching indicator for cuFINUFFT
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='black',
                                 hatch='///', linewidth=0.5)
    legend_handles_list.append(hatch_patch)
    legend_labels.append('cuFINUFFT')

    # Place legend at top, spanning both panels
    fig.legend(legend_handles_list, legend_labels,
               loc='upper center',
               ncol=len(legend_labels),
               fontsize=7.5,
               frameon=False,
               bbox_to_anchor=(0.5, 1.04),
               columnspacing=1.2,
               handletextpad=0.5,
               handlelength=1.8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.20, left=0.08, right=0.97)

    plt.savefig(f'{output_prefix}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}.png', dpi=300, facecolor='white')
    plt.show()

    print(f"Saved: {output_prefix}.pdf, {output_prefix}.png")


def print_latex_table(df):
    """Print results as a LaTeX table."""
    df = filter_methods(df)

    print("\n% LaTeX table")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Single-GPU kernel throughput (Mpts/s). Grid size $256^3$, "
          "$\\rho = 10$ particles per grid point.}")
    print("\\label{tab:single-gpu}")
    print("\\small")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{Type-1 (spreading)} & "
          "\\multicolumn{2}{c}{Type-2 (interpolation)} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}")
    print("Method & Time (ms) & Mpts/s & Time (ms) & Mpts/s \\\\")
    print("\\midrule")

    methods_order = ['IPPL Atomic', 'IPPL Atomic Sort', 'IPPL Tiled',
                     'IPPL OutputFocused', 'cuFINUFFT']

    for method in methods_order:
        method_df = df[df['method'] == method]
        if method_df.empty:
            continue

        label = get_method_label(method)

        t1 = method_df[method_df['type'] == 1]
        t2 = method_df[method_df['type'] == 2]

        t1_time = f"{t1['time_ms'].values[0]:.1f}" if not t1.empty else "---"
        t1_tput = f"{t1['throughput_mpts'].values[0]:.0f}" if not t1.empty else "---"
        t2_time = f"{t2['time_ms'].values[0]:.1f}" if not t2.empty else "---"
        t2_tput = f"{t2['throughput_mpts'].values[0]:.0f}" if not t2.empty else "---"

        print(f"{label:15s} & {t1_time:>8s} & {t1_tput:>6s} & "
              f"{t2_time:>8s} & {t2_tput:>6s} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_summary(df):
    """Print summary statistics."""
    df = filter_methods(df)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for cluster in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster]
        grid_size = cluster_df['grid_size'].iloc[0]
        particles = cluster_df['particles'].iloc[0]

        print(f"\n{cluster} (Grid: {grid_size}³, Particles: {particles:,})")
        print("-" * 60)

        for nufft_type in [1, 2]:
            type_df = cluster_df[cluster_df['type'] == nufft_type]
            if type_df.empty:
                continue

            type_name = "Type-1 (spreading)" if nufft_type == 1 else "Type-2 (interpolation)"
            print(f"\n  {type_name}:")

            type_df = type_df.sort_values('throughput_mpts', ascending=False)
            best = type_df['throughput_mpts'].max()

            for _, row in type_df.iterrows():
                rel = row['throughput_mpts'] / best
                label = get_method_label(row['method'])
                print(f"    {label:15s}  {row['throughput_mpts']:7.0f} Mpts/s  "
                      f"({row['time_ms']:7.1f} ms)  [{rel:.2f}x]")


def main():
    parser = argparse.ArgumentParser(
        description='Plot NUFFT benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_nufft_benchmark.py -i ./results -o fig_nufft
  python plot_nufft_benchmark.py -i ./results -o fig_nufft --latex
        """
    )
    parser.add_argument('--input-dir', '-i', default='.',
                        help='Directory containing benchmark .txt files')
    parser.add_argument('--output', '-o', default='nufft_benchmark',
                        help='Output file prefix')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX table')
    args = parser.parse_args()

    print(f"Loading results from {args.input_dir}...")
    df = load_all_results(args.input_dir)

    if df.empty:
        print("No results found. Check input directory and file format.")
        return

    print(f"Loaded {len(df)} results from {df['cluster'].nunique()} file(s)")

    print_summary(df)

    if args.latex:
        print_latex_table(df)

    # Plot
    clusters = df['cluster'].unique()
    if len(clusters) == 1:
        plot_single_cluster(df, clusters[0], args.output)
    else:
        plot_multi_cluster(df, args.output)


if __name__ == '__main__':
    main()