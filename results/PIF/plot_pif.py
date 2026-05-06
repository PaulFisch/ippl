#!/usr/bin/env python3
"""
Plot strong scaling results for solver comparison across multiple clusters.

Reads timing data from files like:
  solver_comparison_paper_scalings/Alps/512_512_512_Pc_8_tol_1em4/nodes_16/timing.dat

Usage:
    python plot_solver_scaling.py -d solver_comparison_paper_scalings -o fig_solver_scaling
    python plot_solver_scaling.py -d solver_comparison_paper_scalings -o fig_solver_scaling --all
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re
from collections import defaultdict


def setup_style():
    """Set up publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'mathtext.fontset': 'stix',
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


# Colorblind-friendly palette (Okabe-Ito)
CLUSTER_COLORS = {
    'alps': '#2166AC',      # Blue
    'booster': '#D6604D',   # Red-orange
    'lumi': '#4DAF4A',      # Green
}

CLUSTER_MARKERS = {
    'alps': 'o',
    'booster': 's',
    'lumi': '^',
}

CLUSTER_LABELS = {
    'alps': 'Alps (GH200)',
    'booster': 'JUWELS Booster (A100)',
    'lumi': 'LUMI (MI250X)',
}

# Component colors for breakdown plots
COMPONENT_COLORS = {
    'ScatterPIFNUFFT': '#0072B2',     # Blue
    'GatherPIFNUFFT': '#E69F00',      # Orange
    'updateParticle': '#56B4E9',      # Sky blue
    'dumpData': '#009E73',            # Bluish green
    'particlesCreation': '#F0E442',   # Yellow
    'kick': '#CC79A7',                # Pink
    'drift': '#D55E00',               # Vermillion
    'particleBC': '#999999',          # Gray
    'mainTimer': '#000000',           # Black
}

# Key components to plot (in order)
KEY_COMPONENTS = [
    'ScatterPIFNUFFT',
    'GatherPIFNUFFT',
    'updateParticle',
    'dumpData',
]

# Display labels for components (customize these as needed)
COMPONENT_LABELS = {
    'ScatterPIFNUFFT': 'Scatter (Type-1)',
    'GatherPIFNUFFT': 'Gather (Type-2)',
    'updateParticle': 'Particle Update',
    'dumpData': 'I/O',
    'particlesCreation': 'Particle Creation',
    'kick': 'Kick',
    'drift': 'Drift',
    'particleBC': 'Boundary Conditions',
    'mainTimer': 'Total',
}

# Sub-components for detailed breakdown
SCATTER_SUBCOMPONENTS = ['scatterKernel', 'getBuffer']
GATHER_SUBCOMPONENTS = ['gather', 'fillHalo', 'gatherKernel', 'gatherKernelSort']


def parse_timing_file(filepath):
    """
    Parse a timing.dat file and return a dictionary of timer values.

    Returns dict with structure:
    {
        'ranks': int,
        'wall_tot': float,
        'timers': {
            'timer_name': {'max': float, 'min': float, 'avg': float},
            ...
        }
    }
    """
    result = {
        'ranks': None,
        'wall_tot': None,
        'timers': {}
    }

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

    in_detailed_section = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and separators
        if not line or line.startswith('='):
            continue

        # Check for header line indicating detailed section
        if 'Wall max' in line and 'Wall min' in line:
            in_detailed_section = True
            continue

        # Parse the summary section header
        if 'ranks' in line and 'Wall tot' in line and 'Wall max' not in line:
            continue

        # Parse data lines
        # Format: "timerName........    ranks   value1   value2   ..."
        # First, separate the timer name (with dots) from the values
        # Timer names end with a series of dots before the numbers

        # Find where the timer name ends (look for pattern of multiple dots followed by space)
        match = re.match(r'^(\S+?)\.{2,}\s+(.+)$', line)
        if not match:
            # Try without dots (in case timer name has no trailing dots)
            match = re.match(r'^(\S+)\s+(\d.*)$', line)

        if not match:
            continue

        timer_name = match.group(1)
        values_str = match.group(2)

        # Split the values
        parts = values_str.split()

        if len(parts) < 2:
            continue

        try:
            if not in_detailed_section:
                # Summary section: ranks, wall_tot
                if len(parts) >= 2:
                    ranks = int(parts[0])
                    wall_tot = float(parts[1])
                    if timer_name == 'mainTimer':
                        result['ranks'] = ranks
                        result['wall_tot'] = wall_tot
            else:
                # Detailed section: ranks, max, min, avg
                if len(parts) >= 4:
                    ranks = int(parts[0])
                    wall_max = float(parts[1])
                    wall_min = float(parts[2])
                    wall_avg = float(parts[3])

                    result['timers'][timer_name] = {
                        'max': wall_max,
                        'min': wall_min,
                        'avg': wall_avg
                    }

                    if timer_name == 'mainTimer':
                        result['ranks'] = ranks
        except (ValueError, IndexError) as e:
            continue

    return result


def load_all_data(base_dir):
    """
    Load all timing data from the directory structure.

    Returns a nested dict:
    {
        cluster: {
            config: {
                nodes: timing_data
            }
        }
    }
    """
    base_path = Path(base_dir)
    all_data = defaultdict(lambda: defaultdict(dict))

    # Find all clusters
    for cluster_dir in base_path.iterdir():
        if not cluster_dir.is_dir():
            continue

        cluster_name = cluster_dir.name.lower()

        # Find all configurations
        for config_dir in cluster_dir.iterdir():
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name

            # Find all node counts
            for nodes_dir in config_dir.iterdir():
                if not nodes_dir.is_dir():
                    continue

                # Extract node count from directory name
                match = re.search(r'nodes_(\d+)', nodes_dir.name)
                if not match:
                    continue

                num_nodes = int(match.group(1))

                # Look for timing.dat
                timing_file = nodes_dir / 'timing.dat'
                if timing_file.exists():
                    timing_data = parse_timing_file(timing_file)
                    if timing_data and timing_data['ranks'] is not None:
                        all_data[cluster_name][config_name][num_nodes] = timing_data
                        print(f"Loaded: {cluster_name}/{config_name}/nodes_{num_nodes}")

    return all_data


def extract_tolerance(config_name):
    """Extract tolerance from config name like '512_512_512_Pc_8_tol_1em4'."""
    if '1em4' in config_name:
        return '1e-4'
    elif '1em7' in config_name:
        return '1e-7'
    else:
        return 'unknown'


def extract_grid_size(config_name):
    """Extract grid size from config name like '512_512_512_Pc_8_tol_1em4'."""
    match = re.match(r'(\d+)_(\d+)_(\d+)', config_name)
    if match:
        return f"{match.group(1)}³"
    return config_name


def plot_strong_scaling(all_data, output_prefix, grid_filter='512'):
    """
    Plot strong scaling comparison across clusters.
    Left panel: epsilon = 1e-4
    Right panel: epsilon = 1e-7
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    tolerances = ['1e-4', '1e-7']

    for ax, tol in zip(axes, tolerances):
        tol_suffix = '1em4' if tol == '1e-4' else '1em7'

        for cluster in sorted(all_data.keys()):
            cluster_configs = all_data[cluster]

            # Find config matching this tolerance and grid size
            matching_config = None
            for config_name in cluster_configs:
                if tol_suffix in config_name and config_name.startswith(grid_filter):
                    matching_config = config_name
                    break

            if not matching_config:
                continue

            config_data = cluster_configs[matching_config]

            # Extract data points
            nodes = sorted(config_data.keys())
            times = [config_data[n]['wall_tot'] for n in nodes]

            color = CLUSTER_COLORS.get(cluster, '#999999')
            marker = CLUSTER_MARKERS.get(cluster, 'o')
            label = CLUSTER_LABELS.get(cluster, cluster)

            ax.plot(nodes, times, marker=marker, color=color,
                    markersize=4, linewidth=1, label=label)

        # Add ideal scaling line
        all_nodes = []
        all_times = []
        for cluster in all_data:
            for config_name in all_data[cluster]:
                if tol_suffix in config_name and config_name.startswith(grid_filter):
                    config_data = all_data[cluster][config_name]
                    all_nodes.extend(config_data.keys())
                    all_times.extend([config_data[n]['wall_tot'] for n in config_data])

        if all_nodes:
            min_nodes = min(all_nodes)
            max_nodes = max(all_nodes)
            # Get average time at min_nodes for reference
            ref_times = []
            for cluster in all_data:
                for config_name in all_data[cluster]:
                    if tol_suffix in config_name and config_name.startswith(grid_filter):
                        config_data = all_data[cluster][config_name]
                        if min_nodes in config_data:
                            ref_times.append(config_data[min_nodes]['wall_tot'])

            if ref_times:
                ref_time = np.mean(ref_times)
                ideal_nodes = sorted(set(all_nodes))
                ideal_times = [ref_time * min_nodes / n for n in ideal_nodes]
                ax.plot(ideal_nodes, ideal_times, 'k:', linewidth=0.8, alpha=0.5, label='Ideal')

        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Wall Time (s)')
        ax.set_title(f'ε = {tol}', fontsize=9)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=6)

        # Set x-ticks to actual node counts
        if all_nodes:
            unique_nodes = sorted(set(all_nodes))
            ax.set_xticks(unique_nodes)
            ax.set_xticklabels([str(n) for n in unique_nodes], rotation=45)

    # plt.suptitle(f'Strong Scaling ({grid_filter}³ grid)', fontsize=9, y=1.02)
    plt.tight_layout()

    plt.savefig(f'{output_prefix}_strong_scaling.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_strong_scaling.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_strong_scaling.pdf, {output_prefix}_strong_scaling.png")


def plot_efficiency(all_data, output_prefix, grid_filter='512'):
    """Plot parallel efficiency comparison."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    tolerances = ['1e-4', '1e-7']

    for ax, tol in zip(axes, tolerances):
        tol_suffix = '1em4' if tol == '1e-4' else '1em7'

        for cluster in sorted(all_data.keys()):
            cluster_configs = all_data[cluster]

            # Find config matching this tolerance
            matching_config = None
            for config_name in cluster_configs:
                if tol_suffix in config_name and config_name.startswith(grid_filter):
                    matching_config = config_name
                    break

            if not matching_config:
                continue

            config_data = cluster_configs[matching_config]

            # Extract data points and compute efficiency
            nodes = sorted(config_data.keys())
            times = [config_data[n]['wall_tot'] for n in nodes]

            # Compute efficiency relative to smallest node count
            min_nodes = min(nodes)
            ref_time = config_data[min_nodes]['wall_tot']
            efficiency = [(ref_time * min_nodes) / (t * n) * 100 if t > 0 else 0
                          for n, t in zip(nodes, times)]

            color = CLUSTER_COLORS.get(cluster, '#999999')
            marker = CLUSTER_MARKERS.get(cluster, 'o')
            label = CLUSTER_LABELS.get(cluster, cluster)

            ax.plot(nodes, efficiency, marker=marker, color=color,
                    markersize=4, linewidth=1, label=label)

        # Add 100% reference line
        ax.axhline(y=100, color='k', linestyle=':', linewidth=0.8, alpha=0.5)

        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Parallel Efficiency (%)')
        ax.set_title(f'ε = {tol}', fontsize=9)
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 110)
        ax.legend(loc='best', fontsize=6)

        # Set x-ticks to actual node counts
        all_nodes = []
        for cluster in all_data:
            for config_name in all_data[cluster]:
                if tol_suffix in config_name and config_name.startswith(grid_filter):
                    all_nodes.extend(all_data[cluster][config_name].keys())
        if all_nodes:
            unique_nodes = sorted(set(all_nodes))
            ax.set_xticks(unique_nodes)
            ax.set_xticklabels([str(n) for n in unique_nodes], rotation=45)

    plt.suptitle(f'Parallel Efficiency ({grid_filter}³ grid)', fontsize=9, y=1.02)
    plt.tight_layout()

    plt.savefig(f'{output_prefix}_efficiency.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_efficiency.pdf, {output_prefix}_efficiency.png")


def plot_component_scaling(all_data, output_prefix, cluster, config_name):
    """
    Plot component timing breakdown as line plot for a specific configuration.
    """
    setup_style()

    if cluster not in all_data or config_name not in all_data[cluster]:
        print(f"Warning: No data for {cluster}/{config_name}")
        return

    config_data = all_data[cluster][config_name]
    nodes = sorted(config_data.keys())

    if not nodes:
        return

    # Get all available timers from first entry
    sample_timers = config_data[nodes[0]]['timers']

    # Define components to plot
    components_to_plot = []
    for comp in KEY_COMPONENTS:
        if comp in sample_timers:
            components_to_plot.append(comp)

    if not components_to_plot:
        print(f"Warning: No key components found in {cluster}/{config_name}")
        return

    fig, ax = plt.subplots(figsize=(4, 3))

    for comp in components_to_plot:
        times = []
        valid_nodes = []
        for n in nodes:
            if comp in config_data[n]['timers']:
                times.append(config_data[n]['timers'][comp]['avg'])
                valid_nodes.append(n)

        if times:
            color = COMPONENT_COLORS.get(comp, '#999999')
            label = COMPONENT_LABELS.get(comp, comp)  # Use display label
            ax.plot(valid_nodes, times, marker='o', color=color,
                    markersize=4, linewidth=1, label=label)

    # Also plot total time
    # total_times = [config_data[n]['wall_tot'] for n in nodes]
    # ax.plot(nodes, total_times, marker='s', color='black',
    #         markersize=4, linewidth=1.5, label='Total', linestyle='--')

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time (s)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(n) for n in nodes], rotation=45)
    ax.legend(loc='best', fontsize=6)

    # Title
    tol = extract_tolerance(config_name)
    grid = extract_grid_size(config_name)
    cluster_label = CLUSTER_LABELS.get(cluster, cluster)
    ax.set_title(f'{cluster_label}\n{grid}, ε = {tol}', fontsize=8)

    plt.tight_layout()

    # Safe filename
    safe_config = config_name.replace('_', '-')
    plt.savefig(f'{output_prefix}_components_{cluster}_{safe_config}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components_{cluster}_{safe_config}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components_{cluster}_{safe_config}.pdf")


def plot_component_breakdown_stacked(all_data, output_prefix, cluster, config_name):
    """
    Plot stacked area chart of component times vs node count.
    """
    setup_style()

    if cluster not in all_data or config_name not in all_data[cluster]:
        print(f"Warning: No data for {cluster}/{config_name}")
        return

    config_data = all_data[cluster][config_name]
    nodes = sorted(config_data.keys())

    if not nodes:
        return

    # Components for stacking (order matters - bottom to top)
    stack_components = ['ScatterPIFNUFFT', 'GatherPIFNUFFT', 'updateParticle', 'dumpData']

    fig, ax = plt.subplots(figsize=(4, 3))

    # Build data arrays
    y_data = []
    labels = []
    colors = []

    for comp in stack_components:
        times = []
        for n in nodes:
            if comp in config_data[n]['timers']:
                times.append(config_data[n]['timers'][comp]['avg'])
            else:
                times.append(0)

        if any(t > 0 for t in times):
            y_data.append(times)
            labels.append(COMPONENT_LABELS.get(comp, comp))  # Use display label
            colors.append(COMPONENT_COLORS.get(comp, '#999999'))

    if y_data:
        ax.stackplot(nodes, y_data, labels=labels, colors=colors,
                     edgecolor='black', linewidth=0.3, alpha=0.8)

    # Plot total time as line
    # total_times = [config_data[n]['wall_tot'] for n in nodes]
    # ax.plot(nodes, total_times, 'k-', marker='o', markersize=3,
    #         linewidth=1.5, label='Total')

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Time (s)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(n) for n in nodes], rotation=45)
    ax.legend(loc='upper right', fontsize=5)

    # Title
    tol = extract_tolerance(config_name)
    grid = extract_grid_size(config_name)
    cluster_label = CLUSTER_LABELS.get(cluster, cluster)
    ax.set_title(f'{cluster_label}\n{grid}, ε = {tol}', fontsize=8)

    plt.tight_layout()

    safe_config = config_name.replace('_', '-')
    plt.savefig(f'{output_prefix}_stacked_{cluster}_{safe_config}.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_stacked_{cluster}_{safe_config}.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_stacked_{cluster}_{safe_config}.pdf")


def plot_all_components_grid(all_data, output_prefix, grid_filter='512'):
    """
    Create a grid of component plots for all cluster/tolerance combinations.
    """
    setup_style()

    # Collect all valid configurations
    configs = []
    for cluster in sorted(all_data.keys()):
        for config_name in sorted(all_data[cluster].keys()):
            if config_name.startswith(grid_filter):
                configs.append((cluster, config_name))

    if not configs:
        print("No configurations found for component grid plot")
        return

    # Determine grid layout
    n_configs = len(configs)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows))

    if n_configs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (cluster, config_name) in enumerate(configs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        config_data = all_data[cluster][config_name]
        nodes = sorted(config_data.keys())

        # Plot key components
        for comp in KEY_COMPONENTS:
            times = []
            valid_nodes = []
            for n in nodes:
                if comp in config_data[n]['timers']:
                    times.append(config_data[n]['timers'][comp]['avg'])
                    valid_nodes.append(n)

            if times:
                color = COMPONENT_COLORS.get(comp, '#999999')
                label = COMPONENT_LABELS.get(comp, comp)  # Use display label
                ax.plot(valid_nodes, times, marker='o', color=color,
                        markersize=3, linewidth=1, label=label)

        # Plot total
        total_times = [config_data[n]['wall_tot'] for n in nodes]
        ax.plot(nodes, total_times, marker='s', color='black',
                markersize=3, linewidth=1.5, label='Total', linestyle='--')

        ax.set_xlabel('Nodes')
        ax.set_ylabel('Time (s)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes], rotation=45, fontsize=6)

        tol = extract_tolerance(config_name)
        cluster_label = CLUSTER_LABELS.get(cluster, cluster)
        ax.set_title(f'{cluster_label}, ε={tol}', fontsize=8)

        if idx == 0:
            ax.legend(loc='best', fontsize=5)

    # Hide unused subplots
    for idx in range(n_configs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    plt.savefig(f'{output_prefix}_components_grid.pdf', facecolor='white')
    plt.savefig(f'{output_prefix}_components_grid.png', dpi=300, facecolor='white')
    print(f"Saved: {output_prefix}_components_grid.pdf")


def print_summary(all_data):
    """Print summary of loaded data."""
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    for cluster in sorted(all_data.keys()):
        print(f"\n{CLUSTER_LABELS.get(cluster, cluster)}:")
        print("-" * 60)

        for config_name in sorted(all_data[cluster].keys()):
            config_data = all_data[cluster][config_name]
            nodes = sorted(config_data.keys())

            tol = extract_tolerance(config_name)
            grid = extract_grid_size(config_name)

            print(f"\n  {grid}, ε = {tol}:")
            print(f"  {'Nodes':>8} {'Wall Time (s)':>14} {'Ranks':>8}")
            print(f"  {'-'*8} {'-'*14} {'-'*8}")

            for n in nodes:
                wall_time = config_data[n]['wall_tot']
                ranks = config_data[n]['ranks']
                print(f"  {n:>8} {wall_time:>14.3f} {ranks:>8}")


def main():
    parser = argparse.ArgumentParser(description='Plot solver comparison scaling results')
    parser.add_argument('--dir', '-d', default='solver_comparison_paper_scalings',
                        help='Base directory containing benchmark results')
    parser.add_argument('--output', '-o', default='solver_scaling',
                        help='Output file prefix')
    parser.add_argument('--grid', '-g', default='512',
                        help='Grid size filter (default: 512)')
    parser.add_argument('--efficiency', action='store_true',
                        help='Generate efficiency plot')
    parser.add_argument('--components', action='store_true',
                        help='Generate individual component plots')
    parser.add_argument('--components-grid', action='store_true',
                        help='Generate component grid plot')
    parser.add_argument('--stacked', action='store_true',
                        help='Generate stacked component plots')
    parser.add_argument('--all', action='store_true',
                        help='Generate all plot types')
    args = parser.parse_args()

    # Load all data
    print(f"Loading data from {args.dir}...")
    all_data = load_all_data(args.dir)

    if not all_data:
        print("No data found!")
        return

    # Print summary
    print_summary(all_data)

    # Generate plots
    if args.all:
        args.efficiency = True
        args.components = True
        args.components_grid = True
        args.stacked = True

    # Always generate strong scaling plot
    plot_strong_scaling(all_data, args.output, grid_filter=args.grid)

    if args.efficiency:
        plot_efficiency(all_data, args.output, grid_filter=args.grid)

    if args.components_grid:
        plot_all_components_grid(all_data, args.output, grid_filter=args.grid)

    if args.components or args.stacked:
        # Generate individual component plots for each configuration
        for cluster in all_data:
            for config_name in all_data[cluster]:
                if config_name.startswith(args.grid):
                    if args.components:
                        plot_component_scaling(all_data, args.output, cluster, config_name)
                    if args.stacked:
                        plot_component_breakdown_stacked(all_data, args.output, cluster, config_name)


if __name__ == '__main__':
    main()