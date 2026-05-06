#!/usr/bin/env python3
"""
Unified plotting script for the single-node benchmark section.

Reads the CSV output of SingleNodeBenchmark (bench_kernels.csv, bench_breakdown.csv)
from up to three clusters and produces all figures for the paper.

Expected CSV formats:
  _kernels.csv:    grid,rho,tolerance,method,type,time_ms,throughput_mpts
  _breakdown.csv:  grid,rho,tolerance,type,component,time_ms,fraction

Usage:
  python plot_single_node.py \
      --alps   alps_kernels.csv \
      --juwels juwels_kernels.csv \
      --lumi   lumi_kernels.csv \
      --breakdown alps_breakdown.csv \
      --output-dir plots

Outputs (all PDF + PNG):
  1. kernel_throughput_{cluster}_rho{R}.pdf
        — throughput vs tolerance line plots, left=Type1 right=Type2
        — one file per (cluster, rho) combination
        — includes cuFINUFFT where available
  2. nufft_breakdown.pdf
        — stacked bar breakdown vs grid size
  3. nufft_compare_rho{R}.pdf
        — cross-architecture grouped bar comparison at --compare-tol
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════
# Style
# ════════════════════════════════════════════════════════════════════════

def setup_style():
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
        'axes.axisbelow': True,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 7,
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.5,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
    })


# Okabe-Ito colorblind-friendly palette
# Type-1 (spreading) methods
COLORS_T1 = {
    'Atomic':        '#E69F00',  # Orange
    'Tiled':         '#009E73',  # Bluish green
    'Grid-Parallel': '#0072B2',  # Blue
    'cuFINUFFT':     '#D55E00',  # Vermillion
    'FINUFFT':       '#D55E00',
}
MARKERS_T1 = {
    'Atomic': 's', 'Tiled': '^', 'Grid-Parallel': 'o',
    'cuFINUFFT': 'X', 'FINUFFT': 'X',
}
T1_ORDER = ['Atomic', 'Tiled', 'Grid-Parallel', 'cuFINUFFT', 'FINUFFT']

# Type-2 (interpolation) methods
COLORS_T2 = {
    'Direct':    '#E69F00',  # Orange
    'Sorted':    '#56B4E9',  # Sky blue
    'cuFINUFFT': '#D55E00',  # Vermillion
    'FINUFFT':   '#D55E00',
}
MARKERS_T2 = {
    'Direct': 's', 'Sorted': 'D',
    'cuFINUFFT': 'X', 'FINUFFT': 'X',
}
T2_ORDER = ['Direct', 'Sorted', 'cuFINUFFT', 'FINUFFT']

# Breakdown colors — deliberately distinct
# Reviewer: "use different colors for Halo and Spreading/Interpolation"
BREAKDOWN_COLORS = {
    'Spreading':     '#0072B2',  # Blue
    'Interpolation': '#0072B2',  # Blue (same role as spreading)
    'FFT':           '#CC79A7',  # Pink/mauve
    'Halo':          '#E69F00',  # Orange — clearly distinct from blue
    'Deconvolution': '#009E73',  # Bluish green
    'Precorrection': '#009E73',  # Bluish green (same role as deconv)
}
BREAKDOWN_ORDER_T1 = ['Spreading', 'FFT', 'Deconvolution', 'Halo']
BREAKDOWN_ORDER_T2 = ['Interpolation', 'FFT', 'Precorrection', 'Halo']

# Cluster styling for cross-architecture comparison
CLUSTER_COLORS = {
    'alps':   '#2166AC',  # Strong blue
    'juwels': '#D6604D',  # Muted red-orange
    'lumi':   '#4DAF4A',  # Green
}
CLUSTER_LABELS = {
    'alps':   'Alps (GH200)',
    'juwels': 'JUWELS (A100)',
    'lumi':   'LUMI (MI250X)',
}


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════

def load_kernels(path):
    """Load _kernels.csv, return DataFrame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
    df['rho'] = pd.to_numeric(df['rho'], errors='coerce')
    df['type'] = df['type'].astype(str)
    return df


def load_breakdown(path):
    """Load _breakdown.csv, return DataFrame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
    df['rho'] = pd.to_numeric(df['rho'], errors='coerce')
    df['type'] = df['type'].astype(str)
    return df


def _save(fig, path_stem):
    fig.savefig(f'{path_stem}.pdf', facecolor='white')
    fig.savefig(f'{path_stem}.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'  -> {path_stem}.pdf')


# ════════════════════════════════════════════════════════════════════════
# 1. Kernel throughput vs tolerance
# ════════════════════════════════════════════════════════════════════════

def plot_kernel_throughput(df, cluster_name, rho, output_dir):
    """Line plot: throughput vs tolerance for one cluster and one rho."""
    setup_style()

    rdf = df[np.isclose(df['rho'], rho, rtol=0.05)]
    if rdf.empty:
        print(f'  [skip] no data for {cluster_name} rho={rho}')
        return

    t1 = rdf[rdf['type'] == '1']
    t2 = rdf[rdf['type'] == '2']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Type 1 ---
    ax = axes[0]
    for method in T1_ORDER:
        mdf = t1[t1['method'] == method].sort_values('tolerance', ascending=False)
        if mdf.empty:
            continue
        ax.plot(mdf['tolerance'], mdf['throughput_mpts'],
                marker=MARKERS_T1.get(method, 'o'),
                color=COLORS_T1.get(method, '#666'),
                label=method,
                markeredgecolor='black', markeredgewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'Tolerance $\varepsilon$')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type\u20111 (spreading)', fontsize=9, pad=6)
    ax.invert_xaxis()
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best')

    # --- Type 2 ---
    ax = axes[1]
    for method in T2_ORDER:
        mdf = t2[t2['method'] == method].sort_values('tolerance', ascending=False)
        if mdf.empty:
            continue
        ax.plot(mdf['tolerance'], mdf['throughput_mpts'],
                marker=MARKERS_T2.get(method, 'o'),
                color=COLORS_T2.get(method, '#666'),
                label=method,
                markeredgecolor='black', markeredgewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'Tolerance $\varepsilon$')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type\u20112 (interpolation)', fontsize=9, pad=6)
    ax.invert_xaxis()
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0)

    # Format rho for filename: 0.1 -> "0p1", 10 -> "10", 100 -> "100"
    rho_str = f'{rho:g}'.replace('.', 'p')
    _save(fig, os.path.join(output_dir, f'kernel_throughput_{cluster_name}_rho{rho_str}'))


# ════════════════════════════════════════════════════════════════════════
# 2. NUFFT timing breakdown
# ════════════════════════════════════════════════════════════════════════

def plot_breakdown(df, output_dir, rho=None, tol=None):
    """Stacked bar breakdown vs grid size."""
    setup_style()

    if rho is not None:
        df = df[np.isclose(df['rho'], rho, rtol=0.05)]
    if tol is not None:
        df = df[np.isclose(df['tolerance'], tol, rtol=0.5)]

    if df.empty:
        print('  [skip] no breakdown data after filtering')
        return

    grid_sizes = sorted(df['grid'].unique())

    t1 = df[df['type'] == '1']
    t2 = df[df['type'] == '2']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    x = np.arange(len(grid_sizes))
    bar_width = 0.65

    # --- Type 1 ---
    ax = axes[0]
    for gi, gs in enumerate(grid_sizes):
        gdf = t1[t1['grid'] == gs]
        bottom = 0.0
        for ci, comp in enumerate(BREAKDOWN_ORDER_T1):
            cdf = gdf[gdf['component'] == comp]
            if cdf.empty:
                continue
            h = cdf['time_ms'].values[0]
            lbl = comp if gi == 0 else None
            ax.bar(x[gi], h, bar_width, bottom=bottom,
                   color=BREAKDOWN_COLORS.get(comp, '#999'),
                   edgecolor='black', linewidth=0.3, label=lbl)
            bottom += h
        ax.annotate(f'{bottom:.0f}', xy=(x[gi], bottom),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'${g}^3$' for g in grid_sizes])
    ax.set_xlabel('Grid size $N$')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Type\u20111 (spreading)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=6)

    # --- Type 2 ---
    ax = axes[1]
    for gi, gs in enumerate(grid_sizes):
        gdf = t2[t2['grid'] == gs]
        bottom = 0.0
        for ci, comp in enumerate(BREAKDOWN_ORDER_T2):
            cdf = gdf[gdf['component'] == comp]
            if cdf.empty:
                continue
            h = cdf['time_ms'].values[0]
            lbl = comp if gi == 0 else None
            ax.bar(x[gi], h, bar_width, bottom=bottom,
                   color=BREAKDOWN_COLORS.get(comp, '#999'),
                   edgecolor='black', linewidth=0.3, label=lbl)
            bottom += h
        ax.annotate(f'{bottom:.0f}', xy=(x[gi], bottom),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'${g}^3$' for g in grid_sizes])
    ax.set_xlabel('Grid size $N$')
    ax.set_title('Type\u20112 (interpolation)', fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=6)

    plt.tight_layout()

    rho_str = f'_rho{rho:g}'.replace('.', 'p') if rho is not None else ''
    tol_str = f'_tol{tol:.0e}' if tol is not None else ''
    _save(fig, os.path.join(output_dir, f'nufft_breakdown{rho_str}{tol_str}'))


# ════════════════════════════════════════════════════════════════════════
# 3. Cross-architecture comparison (grouped bars at fixed tolerance)
# ════════════════════════════════════════════════════════════════════════

def plot_comparison(cluster_data, rho, tol, output_dir):
    """Grouped bar chart: throughput per method, grouped by cluster.

    cluster_data: dict  cluster_name -> DataFrame (from _kernels.csv)
    """
    setup_style()
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    })

    clusters = list(cluster_data.keys())
    n_clusters = len(clusters)
    if n_clusters == 0:
        return

    # Filter each cluster to rho and tolerance
    filtered = {}
    for c, df in cluster_data.items():
        fdf = df[np.isclose(df['rho'], rho, rtol=0.05)
                 & np.isclose(df['tolerance'], tol, rtol=0.5)]
        if not fdf.empty:
            filtered[c] = fdf

    if not filtered:
        print(f'  [skip] no comparison data for rho={rho} tol={tol}')
        return

    clusters = list(filtered.keys())
    n_clusters = len(clusters)

    # Determine methods present
    t1_methods = []
    t2_methods = []
    for df in filtered.values():
        for m in T1_ORDER:
            if m in df[df['type'] == '1']['method'].values and m not in t1_methods:
                t1_methods.append(m)
        for m in T2_ORDER:
            if m in df[df['type'] == '2']['method'].values and m not in t2_methods:
                t2_methods.append(m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6),
                                    gridspec_kw={'width_ratios': [len(t1_methods),
                                                                   len(t2_methods)],
                                                 'wspace': 0.35})

    total_width = 0.82
    bar_width = total_width / n_clusters
    cluster_handles = {}

    def _draw_panel(ax, methods, colors_map, type_str, title):
        x = np.arange(len(methods))
        for ci, cluster in enumerate(clusters):
            cdf = filtered[cluster]
            cdf = cdf[cdf['type'] == type_str]
            offset = (ci - n_clusters / 2 + 0.5) * bar_width
            color = CLUSTER_COLORS.get(cluster, '#666')

            for mi, method in enumerate(methods):
                mdf = cdf[cdf['method'] == method]
                val = mdf['throughput_mpts'].values[0] if not mdf.empty else 0
                is_finufft = method in ('cuFINUFFT', 'FINUFFT')
                bar = ax.bar(x[mi] + offset, val, bar_width * 0.9,
                             color=color, edgecolor='black', linewidth=0.4,
                             hatch='///' if is_finufft else None, zorder=3)
                if cluster not in cluster_handles:
                    cluster_handles[cluster] = bar[0]

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_ylabel('Throughput (Mpts/s)')
        ax.set_title(title, fontsize=10, fontweight='semibold', pad=8)
        all_vals = []
        for cdf in filtered.values():
            t = cdf[cdf['type'] == type_str]
            all_vals.extend(t['throughput_mpts'].values)
        if all_vals:
            ax.set_ylim(0, max(all_vals) * 1.15)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
        ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)

    _draw_panel(ax1, t1_methods, COLORS_T1, '1', 'Type\u20111 (spreading)')
    _draw_panel(ax2, t2_methods, COLORS_T2, '2', 'Type\u20112 (interpolation)')

    # Legend
    handles = [cluster_handles[c] for c in clusters if c in cluster_handles]
    labels = [CLUSTER_LABELS.get(c, c) for c in clusters if c in cluster_handles]

    # Add cuFINUFFT hatch indicator if cuFINUFFT data exists
    any_finufft = any(
        ('cuFINUFFT' in df['method'].values or 'FINUFFT' in df['method'].values)
        for df in filtered.values()
    )
    if any_finufft:
        handles.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                       hatch='///', linewidth=0.5))
        labels.append('cuFINUFFT')

    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.04),
               columnspacing=1.2, handletextpad=0.5, handlelength=1.8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.20, left=0.08, right=0.97)

    rho_str = f'{rho:g}'.replace('.', 'p')
    tol_str = f'{tol:.0e}'
    _save(fig, os.path.join(output_dir, f'nufft_compare_rho{rho_str}_tol{tol_str}'))


# ════════════════════════════════════════════════════════════════════════
# 4. Density sweep (throughput vs rho at fixed tolerance)
# ════════════════════════════════════════════════════════════════════════

def plot_density_sweep(df, cluster_name, output_dir, tol=None):
    """Line plot: throughput vs particle density for one cluster.

    df: DataFrame with columns grid,rho,tolerance,method,type,time_ms,throughput_mpts
    If tol is given, filter to that tolerance; otherwise use the most common one.
    """
    setup_style()

    if tol is not None:
        df = df[np.isclose(df['tolerance'], tol, rtol=0.5)]
    else:
        # Pick most frequent tolerance
        tol = df['tolerance'].mode().iloc[0]
        df = df[np.isclose(df['tolerance'], tol, rtol=0.5)]

    if df.empty:
        print(f'  [skip] no density data for {cluster_name}')
        return

    t1 = df[df['type'] == '1']
    t2 = df[df['type'] == '2']

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Type 1 (spreading) ---
    ax = axes[0]
    for method in T1_ORDER:
        mdf = t1[t1['method'] == method].sort_values('rho')
        if mdf.empty:
            continue
        ax.plot(mdf['rho'], mdf['throughput_mpts'],
                marker=MARKERS_T1.get(method, 'o'),
                color=COLORS_T1.get(method, '#666'),
                label=method,
                markeredgecolor='black', markeredgewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'Particle density $\rho$ (particles/cell)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type\u20111 (spreading)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    # --- Type 2 (interpolation) ---
    ax = axes[1]
    for method in T2_ORDER:
        mdf = t2[t2['method'] == method].sort_values('rho')
        if mdf.empty:
            continue
        ax.plot(mdf['rho'], mdf['throughput_mpts'],
                marker=MARKERS_T2.get(method, 'o'),
                color=COLORS_T2.get(method, '#666'),
                label=method,
                markeredgecolor='black', markeredgewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r'Particle density $\rho$ (particles/cell)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type\u20112 (interpolation)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0)

    tol_str = f'_tol{tol:.0e}' if tol is not None else ''
    _save(fig, os.path.join(output_dir, f'density_sweep_{cluster_name}{tol_str}'))


def plot_density_speedup(df, cluster_name, output_dir, tol=None):
    """Speedup vs cuFINUFFT/FINUFFT as a function of particle density."""
    setup_style()

    if tol is not None:
        df = df[np.isclose(df['tolerance'], tol, rtol=0.5)]
    else:
        tol = df['tolerance'].mode().iloc[0]
        df = df[np.isclose(df['tolerance'], tol, rtol=0.5)]

    if df.empty:
        print(f'  [skip] no density data for {cluster_name}')
        return

    t1 = df[df['type'] == '1']
    t2 = df[df['type'] == '2']

    # Determine reference library name
    ref_name = None
    for name in ['cuFINUFFT', 'FINUFFT']:
        if name in df['method'].values:
            ref_name = name
            break
    if ref_name is None:
        print(f'  [skip] no cuFINUFFT/FINUFFT data for speedup plot')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

    # --- Type 1 ---
    ax = axes[0]
    ref_t1 = t1[t1['method'] == ref_name].set_index('rho')['time_ms']

    for method in T1_ORDER:
        if method in ('cuFINUFFT', 'FINUFFT'):
            continue
        mdf = t1[t1['method'] == method].sort_values('rho')
        if mdf.empty:
            continue
        speedups, rhos = [], []
        for _, row in mdf.iterrows():
            closest = ref_t1.index[np.argmin(np.abs(ref_t1.index - row['rho']))]
            if np.isclose(closest, row['rho'], rtol=0.1):
                speedups.append(ref_t1.loc[closest] / row['time_ms'])
                rhos.append(row['rho'])
        if speedups:
            ax.plot(rhos, speedups,
                    marker=MARKERS_T1.get(method, 'o'),
                    color=COLORS_T1.get(method, '#666'),
                    label=method,
                    markeredgecolor='black', markeredgewidth=0.3)

    ax.axhline(y=1.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.7,
               label=ref_name)
    ax.set_xscale('log')
    ax.set_xlabel(r'Particle density $\rho$ (particles/cell)')
    ax.set_ylabel(f'Speedup vs {ref_name}')
    ax.set_title('Type\u20111 (spreading)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    # --- Type 2 ---
    ax = axes[1]
    ref_t2 = t2[t2['method'] == ref_name].set_index('rho')['time_ms']

    for method in T2_ORDER:
        if method in ('cuFINUFFT', 'FINUFFT'):
            continue
        mdf = t2[t2['method'] == method].sort_values('rho')
        if mdf.empty:
            continue
        speedups, rhos = [], []
        for _, row in mdf.iterrows():
            closest = ref_t2.index[np.argmin(np.abs(ref_t2.index - row['rho']))]
            if np.isclose(closest, row['rho'], rtol=0.1):
                speedups.append(ref_t2.loc[closest] / row['time_ms'])
                rhos.append(row['rho'])
        if speedups:
            ax.plot(rhos, speedups,
                    marker=MARKERS_T2.get(method, 'o'),
                    color=COLORS_T2.get(method, '#666'),
                    label=method,
                    markeredgecolor='black', markeredgewidth=0.3)

    ax.axhline(y=1.0, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.7,
               label=ref_name)
    ax.set_xscale('log')
    ax.set_xlabel(r'Particle density $\rho$ (particles/cell)')
    ax.set_ylabel(f'Speedup vs {ref_name}')
    ax.set_title('Type\u20112 (interpolation)', fontsize=9, pad=6)
    ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5, zorder=0)
    ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0)

    tol_str = f'_tol{tol:.0e}' if tol is not None else ''
    _save(fig, os.path.join(output_dir, f'density_speedup_{cluster_name}{tol_str}'))


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate all single-node benchmark figures for the paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_single_node.py --alps bench_kernels.csv
  python plot_single_node.py --alps alps.csv --lumi lumi.csv --juwels juwels.csv \\
         --breakdown alps_breakdown.csv --compare-rho 10 --compare-tol 1e-4
        """)
    parser.add_argument('--alps', help='Alps _kernels.csv')
    parser.add_argument('--juwels', help='JUWELS _kernels.csv')
    parser.add_argument('--lumi', help='LUMI _kernels.csv')
    parser.add_argument('--breakdown', help='Breakdown CSV (from any cluster)')
    parser.add_argument('--breakdown-rho', type=float, default=10,
                        help='Rho filter for breakdown plot (default: 10)')
    parser.add_argument('--breakdown-tol', type=float, default=1e-4,
                        help='Tolerance filter for breakdown plot (default: 1e-4)')
    parser.add_argument('--compare-rho', type=float, default=10,
                        help='Rho for cross-architecture comparison (default: 10)')
    parser.add_argument('--compare-tol', type=float, default=1e-4,
                        help='Tolerance for cross-architecture comparison (default: 1e-4)')
    parser.add_argument('--density', nargs='+', metavar='CLUSTER:CSV',
                        help='Density sweep CSVs, e.g. --density alps:alps_density.csv '
                             'lumi:lumi_density.csv (or just a single CSV path)')
    parser.add_argument('--density-tol', type=float, default=1e-4,
                        help='Tolerance filter for density plots (default: 1e-4)')
    parser.add_argument('--output-dir', '-o', default='plots',
                        help='Output directory (default: plots)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load cluster data
    cluster_data = {}
    for name, path in [('alps', args.alps), ('juwels', args.juwels), ('lumi', args.lumi)]:
        if path and Path(path).exists():
            print(f'Loading {name}: {path}')
            cluster_data[name] = load_kernels(path)

    if not cluster_data and not args.breakdown and not args.density:
        print('No input files specified. Use --alps, --juwels, --lumi, --breakdown, or --density.')
        sys.exit(1)

    # ── 1. Kernel throughput vs tolerance (per cluster, per rho) ──────
    for cname, df in cluster_data.items():
        rhos = sorted(df['rho'].unique())
        print(f'\n{cname}: rhos={[f"{r:g}" for r in rhos]}, '
              f'tols={sorted(df["tolerance"].unique(), reverse=True)[:3]}...')
        for rho in rhos:
            print(f'  Plotting kernel throughput {cname} rho={rho:g}')
            plot_kernel_throughput(df, cname, rho, args.output_dir)

    # ── 2. Breakdown ──────────────────────────────────────────────────
    if args.breakdown and Path(args.breakdown).exists():
        print(f'\nLoading breakdown: {args.breakdown}')
        bdf = load_breakdown(args.breakdown)
        print(f'  Plotting breakdown (rho={args.breakdown_rho}, tol={args.breakdown_tol})')
        plot_breakdown(bdf, args.output_dir,
                       rho=args.breakdown_rho, tol=args.breakdown_tol)

    # ── 3. Cross-architecture comparison ──────────────────────────────
    if len(cluster_data) >= 2:
        print(f'\nPlotting cross-architecture comparison '
              f'(rho={args.compare_rho}, tol={args.compare_tol})')
        plot_comparison(cluster_data, args.compare_rho, args.compare_tol,
                        args.output_dir)

    # ── 4. Density sweep ──────────────────────────────────────────────
    if args.density:
        density_data = {}
        for spec in args.density:
            if ':' in spec:
                cname, cpath = spec.split(':', 1)
            else:
                # Single CSV, guess cluster name from filename
                cpath = spec
                cname = Path(cpath).stem.split('_')[0]
            if Path(cpath).exists():
                print(f'\nLoading density {cname}: {cpath}')
                density_data[cname] = load_kernels(cpath)
            else:
                print(f'  [warn] file not found: {cpath}')

        for cname, ddf in density_data.items():
            rhos = sorted(ddf['rho'].unique())
            print(f'  {cname}: {len(rhos)} densities ({rhos[0]:g}..{rhos[-1]:g})')
            print(f'  Plotting density throughput {cname}')
            plot_density_sweep(ddf, cname, args.output_dir, tol=args.density_tol)
            print(f'  Plotting density speedup {cname}')
            plot_density_speedup(ddf, cname, args.output_dir, tol=args.density_tol)

    print('\nDone.')


if __name__ == '__main__':
    main()
