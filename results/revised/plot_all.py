#!/usr/bin/env python3
"""
Unified plotting script for the revised paper results section.

Generates all figures from the data in revised/{alps,juwels,lumi}/:
  1. Single-node kernel throughput vs tolerance (with cuFINUFFT on Alps)
  2. Single-node NUFFT breakdown (per cluster, stacked bars)
  3. Cross-architecture comparison (grouped bars)
  4. Density sweep (throughput vs rho)
  5. NUFFT strong scaling (time vs GPUs, all clusters)
  6. NUFFT component breakdown (stacked bars vs GPU count)
  7. Pruned vs Full FFT strong scaling
  8. PIF / Landau damping strong scaling (Full FFT and Pruned FFT)

Usage:
  python plot_all.py [--output-dir plots_paper]
"""

import argparse
import os
import re
import glob as globmod
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

# ════════════════════════════════════════════════════════════════════════
# Paths
# ════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parent
CLUSTERS = ['alps', 'juwels', 'lumi']
GPUS_PER_NODE = {'alps': 4, 'juwels': 4, 'lumi': 8}

FFT_DIR = BASE.parent / 'fft'
FFT_GRID_DIRS = {
    1024: FFT_DIR / 'benchmark_results_20251209_1024',
    2048: FFT_DIR / 'benchmark_results_20251209_2048',
}

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

# ── Color palettes (Okabe-Ito, colorblind-safe) ──────────────────────

# Type-1 (spreading) methods
COLORS_T1 = {
    'Atomic':        '#E69F00',
    'Tiled':         '#009E73',
    'Grid-Parallel': '#0072B2',
    'cuFINUFFT':     '#D55E00',
}
MARKERS_T1 = {
    'Atomic': 's', 'Tiled': '^', 'Grid-Parallel': 'o', 'cuFINUFFT': 'X',
}
T1_ORDER = ['Atomic', 'Tiled', 'Grid-Parallel', 'cuFINUFFT']

# Type-2 (interpolation) methods
COLORS_T2 = {
    'Direct':    '#E69F00',
    'Sorted':    '#56B4E9',
    'cuFINUFFT': '#D55E00',
}
MARKERS_T2 = {'Direct': 's', 'Sorted': 'D', 'cuFINUFFT': 'X'}
T2_ORDER = ['Direct', 'Sorted', 'cuFINUFFT']

# Breakdown / component colors — reviewer: make Halo clearly distinct from Spreading
COMP_COLORS = {
    'Spreading':     '#0072B2',   # Blue
    'Interpolation': '#56B4E9',   # Sky blue
    'FFT':           '#CC79A7',   # Pink/mauve
    'Halo':          '#E69F00',   # Orange — clearly distinct
    'Deconvolution': '#009E73',   # Green
    'Precorrection': '#009E73',   # Green
}
COMP_ORDER_T1 = ['Spreading', 'FFT', 'Deconvolution', 'Halo']
COMP_ORDER_T2 = ['Interpolation', 'FFT', 'Precorrection', 'Halo']

# Cluster colors and labels
CLUSTER_COLORS = {
    'alps':   '#2166AC',
    'juwels': '#D6604D',
    'lumi':   '#4DAF4A',
}
CLUSTER_MARKERS = {'alps': 'o', 'juwels': 's', 'lumi': '^'}
CLUSTER_LABELS = {
    'alps':   'Alps (GH200)',
    'juwels': 'JUWELS Booster (A100)',
    'lumi':   'LUMI (MI250X)',
}

# Timer name mappings for NUFFT components
TIMER_MAP_T1 = {
    'scatterTimerNUFFT1': 'Spreading',
    'FFTNUFFT1':          'FFT',
    'deconvolutionNUFFT1': 'Deconvolution',
}
TIMER_MAP_T2 = {
    'GatherNUFFT2':       'Interpolation',
    'FFTNUFFT2':          'FFT',
    'PrecorrectionNUFFT2': 'Precorrection',
}

# ════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════

def _save(fig, path_stem):
    fig.savefig(f'{path_stem}.pdf', facecolor='white')
    fig.savefig(f'{path_stem}.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'  -> {path_stem}.pdf')


def _find_nufft_csv(cluster, grid_key, nodes, kind):
    """Find nufft_scaling or nufft_components CSV.
    kind: 'scaling' or 'components'
    grid_key: 'current' (512) or '1024'
    """
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    folder = f'strong_nufft_{grid_key}'
    base_dir = BASE / cluster / folder / f'grid_{grid_size}' / f'nodes_{nodes}'

    # Try direct location
    pattern = f'nufft_{kind}_{gpus}.csv'
    path = base_dir / pattern
    if path.exists():
        return path
    # Try results/ subdirectory
    path = base_dir / 'results' / pattern
    if path.exists():
        return path
    return None


def _find_landau_csv(cluster, grid_key, nodes, pruned=False, tol=None):
    """Find LandauDampingPIF CSV.  tol=None -> default (1e-4), tol='1e8' -> tol_1e8 subdir."""
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    prefix = 'LandauDampingPIFPruned' if pruned else 'LandauDampingPIF'
    folder = f'strong_landau_{grid_key}'
    if tol is not None:
        base_dir = BASE / f'tol_{tol}' / cluster / folder / f'grid_{grid_size}' / f'nodes_{nodes}'
    else:
        base_dir = BASE / cluster / folder / f'grid_{grid_size}' / f'nodes_{nodes}'
    path = base_dir / f'{prefix}{gpus}.csv'
    if path.exists():
        return path
    return None


def load_landau_timings(path):
    """Load Landau PIF CSV, extract per-step timings for Scatter, Gather, updateParticle.

    Returns DataFrame with columns: step, scatter_s, gather_s, update_s, total_s
    Each timer has alternating large (real) and tiny (overhead) measurements.
    We filter by keeping only values above a threshold.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    timers = ['ScatterPIFNUFFT', 'GatherPIFNUFFT', 'updateParticle']
    result = {}

    for timer in timers:
        tdf = df[df['timer_name'] == timer]
        if tdf.empty:
            continue
        # Use rank 0 only
        tdf = tdf[tdf['rank'] == 0].copy()
        vals = tdf['duration_seconds'].values

        if timer in ('ScatterPIFNUFFT', 'GatherPIFNUFFT'):
            # Alternating: real, tiny, real, tiny, ...
            # Filter: keep values > 1ms (the tiny ones are < 0.001s)
            threshold = np.median(vals) * 0.1
            vals = vals[vals > threshold]

        result[timer] = vals

    if not result:
        return None

    # Align to shortest length
    min_len = min(len(v) for v in result.values())
    rows = []
    for i in range(min_len):
        s = result.get('ScatterPIFNUFFT', np.zeros(min_len))[i]
        g = result.get('GatherPIFNUFFT', np.zeros(min_len))[i]
        u = result.get('updateParticle', np.zeros(min_len))[i]
        rows.append({'step': i, 'scatter_s': s, 'gather_s': g, 'update_s': u,
                      'total_s': s + g + u})

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# 1. Kernel throughput vs tolerance (per cluster, with cuFINUFFT on Alps)
# ════════════════════════════════════════════════════════════════════════

def plot_kernel_throughput(output_dir):
    """Kernel throughput vs tolerance for each cluster. cuFINUFFT only on Alps."""
    setup_style()
    print('\n=== Kernel throughput vs tolerance ===')

    for cluster in CLUSTERS:
        path = BASE / cluster / 'single_node' / 'bench_kernels.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
        df['type'] = df['type'].astype(str)

        rho = 10.0
        rdf = df[np.isclose(df['rho'], rho, rtol=0.05)]
        if rdf.empty:
            continue

        t1 = rdf[rdf['type'] == '1']
        t2 = rdf[rdf['type'] == '2']

        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

        # Type 1
        ax = axes[0]
        for method in T1_ORDER:
            mdf = t1[t1['method'] == method].sort_values('tolerance', ascending=False)
            if mdf.empty:
                continue
            ax.plot(mdf['tolerance'], mdf['throughput_mpts'],
                    marker=MARKERS_T1[method], color=COLORS_T1[method],
                    label=method, markeredgecolor='black', markeredgewidth=0.3)
        ax.set_xscale('log')
        ax.set_xlabel(r'Tolerance $\varepsilon$')
        ax.set_ylabel('Throughput (Mpts/s)')
        ax.set_title('Type\u20111 (spreading)', fontsize=9, pad=6)
        ax.invert_xaxis()
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc='best')

        # Type 2
        ax = axes[1]
        for method in T2_ORDER:
            mdf = t2[t2['method'] == method].sort_values('tolerance', ascending=False)
            if mdf.empty:
                continue
            ax.plot(mdf['tolerance'], mdf['throughput_mpts'],
                    marker=MARKERS_T2[method], color=COLORS_T2[method],
                    label=method, markeredgecolor='black', markeredgewidth=0.3)
        ax.set_xscale('log')
        ax.set_xlabel(r'Tolerance $\varepsilon$')
        ax.set_ylabel('Throughput (Mpts/s)')
        ax.set_title('Type\u20112 (interpolation)', fontsize=9, pad=6)
        ax.invert_xaxis()
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc='best')

        plt.tight_layout(w_pad=3.0)
        _save(fig, os.path.join(output_dir, f'kernel_throughput_{cluster}_rho10'))


# ════════════════════════════════════════════════════════════════════════
# 2. Single-node NUFFT breakdown (stacked bars per cluster)
# ════════════════════════════════════════════════════════════════════════

def plot_breakdown_all(output_dir):
    """Stacked bar breakdown vs grid size, one figure per cluster. Caption mentions cluster."""
    setup_style()
    print('\n=== NUFFT breakdown ===')

    for cluster in CLUSTERS:
        path = BASE / cluster / 'single_node' / 'bench_breakdown.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df['type'] = df['type'].astype(str)

        grid_sizes = sorted(df['grid'].unique())
        t1 = df[df['type'] == '1']
        t2 = df[df['type'] == '2']

        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
        x = np.arange(len(grid_sizes))
        bw = 0.65

        for ax, tdf, comp_order, title in [
            (axes[0], t1, COMP_ORDER_T1, 'Type\u20111 (spreading)'),
            (axes[1], t2, COMP_ORDER_T2, 'Type\u20112 (interpolation)'),
        ]:
            for gi, gs in enumerate(grid_sizes):
                gdf = tdf[tdf['grid'] == gs]
                bottom = 0.0
                for comp in comp_order:
                    cdf = gdf[gdf['component'] == comp]
                    if cdf.empty:
                        continue
                    h = cdf['time_ms'].values[0]
                    lbl = comp if gi == 0 else None
                    ax.bar(x[gi], h, bw, bottom=bottom,
                           color=COMP_COLORS.get(comp, '#999'),
                           edgecolor='black', linewidth=0.3, label=lbl)
                    bottom += h
                ax.annotate(f'{bottom:.0f}', xy=(x[gi], bottom),
                            xytext=(0, 2), textcoords='offset points',
                            ha='center', va='bottom', fontsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels([f'${g}^3$' for g in grid_sizes])
            ax.set_xlabel('Grid size $N$')
            ax.set_ylabel('Time (ms)')
            ax.set_title(title, fontsize=9)
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper left', fontsize=6)

        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'nufft_breakdown_{cluster}'))


# ════════════════════════════════════════════════════════════════════════
# 3. Cross-architecture comparison (grouped bars)
# ════════════════════════════════════════════════════════════════════════

def plot_comparison(output_dir):
    """Grouped bar: throughput per method, grouped by cluster. Multiple tolerances."""
    setup_style()
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    })
    print('\n=== Cross-architecture comparison ===')

    cluster_data = {}
    for cluster in CLUSTERS:
        path = BASE / cluster / 'single_node' / 'bench_kernels.csv'
        if path.exists():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
            df['type'] = df['type'].astype(str)
            cluster_data[cluster] = df

    if len(cluster_data) < 2:
        return

    for tol in [1e-4]:
        rho = 10.0
        filtered = {}
        for c, df in cluster_data.items():
            fdf = df[np.isclose(df['rho'], rho, rtol=0.05)
                     & np.isclose(df['tolerance'], tol, rtol=0.5)]
            if not fdf.empty:
                filtered[c] = fdf

        if not filtered:
            continue

        clusters = list(filtered.keys())
        n_clusters = len(clusters)

        # Determine methods
        t1_methods, t2_methods = [], []
        for fdf in filtered.values():
            for m in T1_ORDER:
                if m in fdf[fdf['type'] == '1']['method'].values and m not in t1_methods:
                    t1_methods.append(m)
            for m in T2_ORDER:
                if m in fdf[fdf['type'] == '2']['method'].values and m not in t2_methods:
                    t2_methods.append(m)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.6),
                                        gridspec_kw={
                                            'width_ratios': [max(len(t1_methods), 1),
                                                             max(len(t2_methods), 1)],
                                            'wspace': 0.35})
        total_width = 0.82
        bar_width = total_width / n_clusters
        cluster_handles = {}

        def _draw(ax, methods, type_str, title):
            x = np.arange(len(methods))
            for ci, cl in enumerate(clusters):
                cdf = filtered[cl][filtered[cl]['type'] == type_str]
                offset = (ci - n_clusters / 2 + 0.5) * bar_width
                color = CLUSTER_COLORS[cl]
                for mi, method in enumerate(methods):
                    mdf = cdf[cdf['method'] == method]
                    val = mdf['throughput_mpts'].values[0] if not mdf.empty else 0
                    is_fin = method == 'cuFINUFFT'
                    bar = ax.bar(x[mi] + offset, val, bar_width * 0.9,
                                 color=color, edgecolor='black', linewidth=0.4,
                                 hatch='///' if is_fin else None, zorder=3)
                    if cl not in cluster_handles:
                        cluster_handles[cl] = bar[0]
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right', rotation_mode='anchor')
            ax.set_ylabel('Throughput (Mpts/s)')
            ax.set_title(title, fontsize=10, fontweight='semibold', pad=8)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
            ax.grid(True, axis='y', linewidth=0.3, alpha=0.4, zorder=0)

        _draw(ax1, t1_methods, '1', 'Type\u20111 (spreading)')
        _draw(ax2, t2_methods, '2', 'Type\u20112 (interpolation)')

        handles = [cluster_handles[c] for c in clusters if c in cluster_handles]
        labels = [CLUSTER_LABELS[c] for c in clusters if c in cluster_handles]
        # Add cuFINUFFT hatch indicator
        any_fin = any('cuFINUFFT' in fdf['method'].values for fdf in filtered.values())
        if any_fin:
            handles.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                           hatch='///', linewidth=0.5))
            labels.append('cuFINUFFT')
        fig.legend(handles, labels, loc='upper center', ncol=len(labels),
                   fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.04),
                   columnspacing=1.2, handletextpad=0.5, handlelength=1.8)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.20, left=0.08, right=0.97)

        tol_str = f'{tol:.0e}'
        _save(fig, os.path.join(output_dir, f'nufft_compare_rho10_tol{tol_str}'))


# ════════════════════════════════════════════════════════════════════════
# 4. Density sweep
# ════════════════════════════════════════════════════════════════════════

def plot_density_sweep(output_dir):
    """Throughput vs particle density at fixed tolerance, Alps only (has cuFINUFFT)."""
    setup_style()
    print('\n=== Density sweep ===')

    for cluster in CLUSTERS:
        path = BASE / cluster / 'single_node' / 'bench_density.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
        df['type'] = df['type'].astype(str)

        tol = 1e-4
        df = df[np.isclose(df['tolerance'], tol, rtol=0.5)]
        if df.empty:
            continue

        t1 = df[df['type'] == '1']
        t2 = df[df['type'] == '2']

        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))

        for ax, tdf, methods, colors, markers, title in [
            (axes[0], t1, T1_ORDER, COLORS_T1, MARKERS_T1, 'Type\u20111 (spreading)'),
            (axes[1], t2, T2_ORDER, COLORS_T2, MARKERS_T2, 'Type\u20112 (interpolation)'),
        ]:
            for method in methods:
                mdf = tdf[tdf['method'] == method].sort_values('rho')
                if mdf.empty:
                    continue
                ax.plot(mdf['rho'], mdf['throughput_mpts'],
                        marker=markers.get(method, 'o'), color=colors[method],
                        label=method, markeredgecolor='black', markeredgewidth=0.3)
            ax.set_xscale('log')
            ax.set_xlabel(r'Particle density $\rho$ (particles/cell)')
            ax.set_ylabel('Throughput (Mpts/s)')
            ax.set_title(title, fontsize=9, pad=6)
            ax.grid(True, which='both', linewidth=0.3, alpha=0.5, zorder=0)
            ax.legend(loc='best')

        plt.tight_layout(w_pad=3.0)
        _save(fig, os.path.join(output_dir, f'density_sweep_{cluster}'))


# ════════════════════════════════════════════════════════════════════════
# 5. NUFFT strong scaling (time vs GPUs)
# ════════════════════════════════════════════════════════════════════════

def _load_nufft_scaling_data(grid_key):
    """Load NUFFT scaling data for all clusters. Returns dict cluster -> DataFrame."""
    data = {}
    for cluster in CLUSTERS:
        rows = []
        grid_size = 512 if grid_key == 'current' else 1024
        grid_dir = BASE / cluster / f'strong_nufft_{grid_key}' / f'grid_{grid_size}'
        if not grid_dir.exists():
            continue
        for node_dir in sorted(grid_dir.iterdir()):
            if not node_dir.is_dir() or not node_dir.name.startswith('nodes_'):
                continue
            nodes = int(node_dir.name.split('_')[1])
            gpus = nodes * GPUS_PER_NODE[cluster]
            path = _find_nufft_csv(cluster, grid_key, nodes, 'scaling')
            if path is None:
                continue
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df['gpus'] = gpus
            df['cluster'] = cluster
            rows.append(df)
        if rows:
            data[cluster] = pd.concat(rows, ignore_index=True)
    return data


def _load_nufft_components_data(grid_key):
    """Load NUFFT component data for all clusters."""
    data = {}
    for cluster in CLUSTERS:
        rows = []
        grid_size = 512 if grid_key == 'current' else 1024
        grid_dir = BASE / cluster / f'strong_nufft_{grid_key}' / f'grid_{grid_size}'
        if not grid_dir.exists():
            continue
        for node_dir in sorted(grid_dir.iterdir()):
            if not node_dir.is_dir() or not node_dir.name.startswith('nodes_'):
                continue
            nodes = int(node_dir.name.split('_')[1])
            gpus = nodes * GPUS_PER_NODE[cluster]
            path = _find_nufft_csv(cluster, grid_key, nodes, 'components')
            if path is None:
                continue
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            df['gpus'] = gpus
            df['cluster'] = cluster
            rows.append(df)
        if rows:
            data[cluster] = pd.concat(rows, ignore_index=True)
    return data


def plot_nufft_scaling(output_dir, grid_key='current'):
    """Strong scaling: time vs GPUs for all clusters on one plot."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== NUFFT strong scaling (grid {grid_label}) ===')

    scaling_data = _load_nufft_scaling_data(grid_key)
    if not scaling_data:
        print('  [skip] no scaling data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=False)

    for ax, ttype, title in [
        (axes[0], 'type1', 'Type\u20111 (spreading)'),
        (axes[1], 'type2', 'Type\u20112 (interpolation)'),
    ]:
        all_gpus = set()
        for cluster, df in scaling_data.items():
            tdf = df[df['transform_type'] == ttype]
            if tdf.empty:
                continue
            stats = tdf.groupby('gpus')['time_ms'].agg(['mean', 'min', 'max']).reset_index()
            stats = stats.sort_values('gpus')
            all_gpus.update(stats['gpus'])

            ax.plot(stats['gpus'], stats['mean'],
                    marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                    label=CLUSTER_LABELS[cluster],
                    markeredgecolor='black', markeredgewidth=0.3)
            ax.fill_between(stats['gpus'], stats['min'], stats['max'],
                            color=CLUSTER_COLORS[cluster], alpha=0.15)

        # Ideal scaling line from smallest common GPU count
        if all_gpus:
            sorted_gpus = sorted(all_gpus)
            # Use first cluster's data at the lowest GPU count for reference
            ref_cluster = list(scaling_data.keys())[0]
            ref_df = scaling_data[ref_cluster]
            ref_tdf = ref_df[(ref_df['transform_type'] == ttype)]
            ref_min_gpus = ref_tdf['gpus'].min()
            ref_time = ref_tdf[ref_tdf['gpus'] == ref_min_gpus]['time_ms'].mean()
            ideal_gpus = np.array(sorted_gpus)
            ideal_time = ref_time * (ref_min_gpus / ideal_gpus)
            ax.plot(ideal_gpus, ideal_time, 'k--', linewidth=0.8, alpha=0.5,
                    label='Ideal', zorder=0)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(title, fontsize=9, pad=6)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=6)

    plt.tight_layout(w_pad=3.0)
    fft_note = '(pruned FFT)' if grid_key == 'current' else '(pruned FFT)'
    _save(fig, os.path.join(output_dir, f'nufft_scaling_time_{grid_label}'))


# ════════════════════════════════════════════════════════════════════════
# 6. NUFFT component breakdown (stacked bars vs GPU count)
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_components(output_dir, grid_key='current'):
    """Stacked bar: NUFFT component breakdown vs GPU count, one figure per cluster."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== NUFFT component breakdown (grid {grid_label}) ===')

    comp_data = _load_nufft_components_data(grid_key)
    if not comp_data:
        print('  [skip] no component data')
        return

    for cluster, df in comp_data.items():
        gpu_counts = sorted(df['gpus'].unique())

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
        x = np.arange(len(gpu_counts))
        bw = 0.7

        for ax, timer_map, comp_order, title in [
            (axes[0], TIMER_MAP_T1, COMP_ORDER_T1, 'Type\u20111 (spreading)'),
            (axes[1], TIMER_MAP_T2, COMP_ORDER_T2, 'Type\u20112 (interpolation)'),
        ]:
            for gi, gpus in enumerate(gpu_counts):
                gdf = df[df['gpus'] == gpus]
                bottom = 0.0
                for comp in comp_order:
                    # Find the timer name(s) for this component
                    timer_names = [k for k, v in timer_map.items() if v == comp]
                    total = 0.0
                    for tn in timer_names:
                        tdf = gdf[gdf['timer'] == tn]
                        if not tdf.empty:
                            total += tdf['time_s'].mean() * 1000  # to ms
                    if total == 0:
                        continue
                    lbl = comp if gi == 0 else None
                    ax.bar(x[gi], total, bw, bottom=bottom,
                           color=COMP_COLORS.get(comp, '#999'),
                           edgecolor='black', linewidth=0.3, label=lbl)
                    bottom += total

            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=6)
            ax.set_xlabel('Number of GPUs')
            ax.set_ylabel('Time (ms)')
            ax.set_title(title, fontsize=9)
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=6)

        plt.suptitle(f'{CLUSTER_LABELS[cluster]} — Grid ${grid_label}^3$ (pruned FFT)',
                      fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'nufft_components_{cluster}_{grid_label}'))


# ════════════════════════════════════════════════════════════════════════
# 7. Pruned vs Full FFT strong scaling (line plot, reviewer request)
# ════════════════════════════════════════════════════════════════════════

def plot_fft_pruned_vs_full(output_dir):
    """Strong scaling: pruned vs full FFT as line plot (reviewer: not violin)."""
    setup_style()
    print('\n=== Pruned vs Full FFT ===')

    all_data = []
    for grid_size, data_dir in FFT_GRID_DIRS.items():
        if not data_dir.exists():
            print(f'  [skip] {data_dir} not found')
            continue
        for csv_path in sorted(data_dir.glob('timings_gpus*_conc*.csv')):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df['grid_size'] = grid_size
            all_data.append(df)

    if not all_data:
        print('  [skip] no FFT data')
        return

    df = pd.concat(all_data, ignore_index=True)

    # Use forward direction, concurrency 4 as primary (best balance per paper)
    for conc in [4]:
        cdf = df[(df['num_concurrent'] == conc) & (df['direction'] == 'forward')]
        if cdf.empty:
            continue

        grid_sizes = sorted(cdf['grid_size'].unique())
        methods = ['full', 'pruned']
        # Shade by grid size: lighter for smaller, darker for larger
        _grid_method_colors = {
            (min(grid_sizes), 'full'): '#999999', (min(grid_sizes), 'pruned'): '#F0A050',
            (max(grid_sizes), 'full'): '#333333', (max(grid_sizes), 'pruned'): '#D55E00',
        }
        method_labels = {'full': 'Full FFT', 'pruned': 'Pruned FFT'}

        fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

        for gs in grid_sizes:
            orig_gs = gs // 2  # show original (non-upsampled) grid size
            for method in methods:
                mdf = cdf[(cdf['grid_size'] == gs) & (cdf['method'] == method)]
                if mdf.empty:
                    continue
                stats = mdf.groupby('num_gpus')['time_ms'].agg(
                    ['mean', 'min', 'max']).reset_index().sort_values('num_gpus')

                ls = '-' if gs == max(grid_sizes) else '--'
                marker = 'o' if method == 'pruned' else 's'
                color = _grid_method_colors[(gs, method)]
                label = f'{method_labels[method]} ({orig_gs}$^3$)'
                ax.plot(stats['num_gpus'], stats['mean'],
                        marker=marker, color=color,
                        linestyle=ls, label=label,
                        markeredgecolor='black', markeredgewidth=0.3)
                ax.fill_between(stats['num_gpus'], stats['min'], stats['max'],
                                color=color, alpha=0.1)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Forward FFT — Concurrency {conc} (Alps)', fontsize=9)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=6)

        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'pruned_fft_scaling_conc{conc}'))


def plot_fft_pruned_vs_full_all_conc(output_dir):
    """Show pruned vs full for concurrency 1, 4, 8 in subplots."""
    setup_style()
    print('\n=== Pruned vs Full FFT (all concurrencies) ===')

    all_data = []
    for grid_size, data_dir in FFT_GRID_DIRS.items():
        if not data_dir.exists():
            continue
        for csv_path in sorted(data_dir.glob('timings_gpus*_conc*.csv')):
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df['grid_size'] = grid_size
            all_data.append(df)

    if not all_data:
        return

    df = pd.concat(all_data, ignore_index=True)
    fwd = df[df['direction'] == 'forward']

    concs = [1, 4, 8]
    fig, axes = plt.subplots(1, len(concs), figsize=(7.0, 2.8), sharey=True)

    method_labels = {'full': 'Full FFT', 'pruned': 'Pruned FFT'}

    for ai, (ax, conc) in enumerate(zip(axes, concs)):
        cdf = fwd[fwd['num_concurrent'] == conc]
        if cdf.empty:
            continue

        grid_sizes = sorted(cdf['grid_size'].unique())
        # Shade by grid size: lighter for smaller, darker for larger
        _grid_method_colors = {
            (min(grid_sizes), 'full'): '#999999', (min(grid_sizes), 'pruned'): '#F0A050',
            (max(grid_sizes), 'full'): '#333333', (max(grid_sizes), 'pruned'): '#D55E00',
        }
        for gi, gs in enumerate(grid_sizes):
            orig_gs = gs // 2  # show original (non-upsampled) grid size
            ls = '-' if gi == len(grid_sizes) - 1 else '--'
            for method in ['full', 'pruned']:
                mdf = cdf[(cdf['grid_size'] == gs) & (cdf['method'] == method)]
                if mdf.empty:
                    continue
                stats = mdf.groupby('num_gpus')['time_ms'].agg(
                    ['mean', 'min', 'max']).reset_index().sort_values('num_gpus')

                marker = 'o' if method == 'pruned' else 's'
                color = _grid_method_colors[(gs, method)]
                label = f'{method_labels[method]} ({orig_gs}$^3$)' if ai == 0 else None
                ax.plot(stats['num_gpus'], stats['mean'],
                        marker=marker, color=color,
                        linestyle=ls, label=label, markersize=3.5,
                        markeredgecolor='black', markeredgewidth=0.3)
                ax.fill_between(stats['num_gpus'], stats['min'], stats['max'],
                                color=color, alpha=0.08)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        if ai == 0:
            ax.set_ylabel('Time (ms)')
        ax.set_title(f'$n_{{\\mathrm{{conc}}}} = {conc}$', fontsize=9)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)

    axes[0].legend(loc='best', fontsize=5.5)
    # plt.suptitle('Pruned vs Full FFT — Forward, Alps', fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'pruned_fft_scaling_all_conc'))


# ════════════════════════════════════════════════════════════════════════
# 8. PIF / Landau damping strong scaling
# ════════════════════════════════════════════════════════════════════════

def _load_landau_scaling(grid_key, pruned=False, tol=None):
    """Load Landau damping scaling data for all clusters.
    Returns dict: cluster -> DataFrame with cols [gpus, mean_s, min_s, max_s, q25, q75,
                                                   scatter_mean, gather_mean, update_mean]
    tol=None -> default (1e-4), tol='1e8' -> tol_1e8 subdir.
    """
    data = {}
    for cluster in CLUSTERS:
        grid_size = 512 if grid_key == 'current' else 1024
        if tol is not None:
            grid_dir = BASE / f'tol_{tol}' / cluster / f'strong_landau_{grid_key}' / f'grid_{grid_size}'
        else:
            grid_dir = BASE / cluster / f'strong_landau_{grid_key}' / f'grid_{grid_size}'
        if not grid_dir.exists():
            continue
        rows = []
        for node_dir in sorted(grid_dir.iterdir()):
            if not node_dir.is_dir() or not node_dir.name.startswith('nodes_'):
                continue
            nodes = int(node_dir.name.split('_')[1])
            gpus = nodes * GPUS_PER_NODE[cluster]
            path = _find_landau_csv(cluster, grid_key, nodes, pruned=pruned, tol=tol)
            if path is None:
                continue
            try:
                tdf = load_landau_timings(path)
            except Exception as e:
                print(f'  [warn] {path}: {e}')
                continue
            if tdf is None or tdf.empty:
                continue

            # Skip first 3 steps (warmup) if we have enough data
            if len(tdf) > 5:
                tdf = tdf.iloc[3:]

            totals = tdf['total_s'].values
            rows.append({
                'gpus': gpus,
                'mean_s': np.mean(totals),
                'min_s': np.min(totals),
                'max_s': np.max(totals),
                'q25': np.percentile(totals, 25),
                'q75': np.percentile(totals, 75),
                'scatter_mean': tdf['scatter_s'].mean(),
                'gather_mean': tdf['gather_s'].mean(),
                'update_mean': tdf['update_s'].mean(),
            })
        if rows:
            data[cluster] = pd.DataFrame(rows).sort_values('gpus').reset_index(drop=True)
    return data


def _tol_label(tol):
    """Human-readable tolerance label."""
    if tol is None:
        return r'$\varepsilon=10^{-4}$'
    return r'$\varepsilon=10^{-' + tol.lstrip('1e') + r'}$'


def _tol_suffix(tol):
    """File-name suffix for a tolerance."""
    return '' if tol is None else f'_tol{tol}'


def plot_pif_scaling(output_dir, grid_key='current', tol=None):
    """PIF strong scaling: all clusters, Full FFT, with asymmetric error bands."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    tl = _tol_label(tol)
    print(f'\n=== PIF scaling (grid {grid_label}, Full FFT, {tl}) ===')

    data = _load_landau_scaling(grid_key, pruned=False, tol=tol)
    if not data:
        print('  [skip] no PIF data')
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

    all_gpus = set()
    for cluster, df in data.items():
        all_gpus.update(df['gpus'])
        ax.plot(df['gpus'], df['mean_s'] * 1000,
                marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                label=CLUSTER_LABELS[cluster],
                markeredgecolor='black', markeredgewidth=0.3)
        ax.fill_between(df['gpus'], df['min_s'] * 1000, df['max_s'] * 1000,
                        color=CLUSTER_COLORS[cluster], alpha=0.15)

    if all_gpus:
        ref_cluster = list(data.keys())[0]
        ref_df = data[ref_cluster]
        ref_gpus = ref_df['gpus'].iloc[0]
        ref_time = ref_df['mean_s'].iloc[0]
        sorted_gpus = np.array(sorted(all_gpus))
        ideal = ref_time * 1000 * (ref_gpus / sorted_gpus)
        ax.plot(sorted_gpus, ideal, 'k--', linewidth=0.8, alpha=0.5,
                label='Ideal', zorder=0)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time per step (ms)')
    ax.set_title(f'PIF Grid ${grid_label}^3$ (full FFT, {tl})', fontsize=9, pad=6)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_scaling_{grid_label}{_tol_suffix(tol)}'))


def plot_pif_scaling_pruned(output_dir, grid_key='current', tol=None):
    """PIF strong scaling: all clusters that have Pruned data."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    tl = _tol_label(tol)
    print(f'\n=== PIF scaling (grid {grid_label}, Pruned FFT, {tl}) ===')

    data = _load_landau_scaling(grid_key, pruned=True, tol=tol)
    if not data:
        print('  [skip] no pruned PIF data')
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

    for cluster, df in data.items():
        ax.plot(df['gpus'], df['mean_s'] * 1000,
                marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                label=CLUSTER_LABELS[cluster],
                markeredgecolor='black', markeredgewidth=0.3)
        ax.fill_between(df['gpus'], df['min_s'] * 1000, df['max_s'] * 1000,
                        color=CLUSTER_COLORS[cluster], alpha=0.15)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time per step (ms)')
    ax.set_title(f'PIF Grid ${grid_label}^3$ (pruned FFT, {tl})', fontsize=9, pad=6)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_scaling_pruned_{grid_label}{_tol_suffix(tol)}'))


def _load_landau_scaling_robust(grid_key, pruned=False, tol=None):
    """Load Landau damping scaling, returning median and trimmed min/max (drop 2 worst).
    tol=None -> default (1e-4), tol='1e8' -> tol_1e8 subdir.
    """
    data = {}
    for cluster in CLUSTERS:
        grid_size = 512 if grid_key == 'current' else 1024
        if tol is not None:
            grid_dir = BASE / f'tol_{tol}' / cluster / f'strong_landau_{grid_key}' / f'grid_{grid_size}'
        else:
            grid_dir = BASE / cluster / f'strong_landau_{grid_key}' / f'grid_{grid_size}'
        if not grid_dir.exists():
            continue
        rows = []
        for node_dir in sorted(grid_dir.iterdir()):
            if not node_dir.is_dir() or not node_dir.name.startswith('nodes_'):
                continue
            nodes = int(node_dir.name.split('_')[1])
            gpus = nodes * GPUS_PER_NODE[cluster]
            path = _find_landau_csv(cluster, grid_key, nodes, pruned=pruned, tol=tol)
            if path is None:
                continue
            try:
                tdf = load_landau_timings(path)
            except Exception as e:
                print(f'  [warn] {path}: {e}')
                continue
            if tdf is None or tdf.empty:
                continue
            if len(tdf) > 5:
                tdf = tdf.iloc[3:]
            totals = np.sort(tdf['total_s'].values)
            # Remove 2 worst (largest) outliers if enough data
            if len(totals) > 4:
                totals = totals[:-2]
            rows.append({
                'gpus': gpus,
                'median_ms': np.median(totals) * 1000,
                'min_ms': np.min(totals) * 1000,
                'max_ms': np.max(totals) * 1000,
            })
        if rows:
            data[cluster] = pd.DataFrame(rows).sort_values('gpus').reset_index(drop=True)
    return data


def plot_pif_scaling_combined(output_dir, grid_key='current', tol=None):
    """Combined PIF scaling: all clusters, pruned (solid) vs full (dashed), median, no error bars."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    tl = _tol_label(tol)
    print(f'\n=== PIF combined pruned vs full (grid {grid_label}, {tl}) ===')

    data_pruned = _load_landau_scaling_robust(grid_key, pruned=True, tol=tol)
    data_full = _load_landau_scaling_robust(grid_key, pruned=False, tol=tol)

    if not data_pruned and not data_full:
        print('  [skip] no PIF data')
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

    all_gpus = set()
    for cluster in CLUSTERS:
        # Pruned (solid)
        if cluster in data_pruned:
            df = data_pruned[cluster]
            all_gpus.update(df['gpus'])
            ax.plot(df['gpus'], df['median_ms'],
                    marker=CLUSTER_MARKERS[cluster],
                    color=CLUSTER_COLORS[cluster], linestyle='-',
                    label=f'{CLUSTER_LABELS[cluster]} (pruned)',
                    markeredgecolor='black', markeredgewidth=0.3,
                    markersize=3.5)
            ax.fill_between(df['gpus'], df['min_ms'], df['max_ms'],
                            color=CLUSTER_COLORS[cluster], alpha=0.08)

        # Full / upsampled (dashed)
        if cluster in data_full:
            df = data_full[cluster]
            all_gpus.update(df['gpus'])
            ax.plot(df['gpus'], df['median_ms'],
                    marker=CLUSTER_MARKERS[cluster],
                    color=CLUSTER_COLORS[cluster], linestyle='--',
                    label=f'{CLUSTER_LABELS[cluster]} (full)',
                    markeredgecolor='black', markeredgewidth=0.3,
                    markersize=3.5, fillstyle='none')
            ax.fill_between(df['gpus'], df['min_ms'], df['max_ms'],
                            color=CLUSTER_COLORS[cluster], alpha=0.08)

    # Ideal scaling from first available cluster
    if all_gpus:
        first = next((c for c in CLUSTERS if c in data_full or c in data_pruned), None)
        if first:
            ref_df = data_full[first] if first in data_full else data_pruned[first]
            ref_gpus = ref_df['gpus'].iloc[0]
            ref_time = ref_df['median_ms'].iloc[0]
            sorted_gpus = np.array(sorted(all_gpus))
            ideal = ref_time * (ref_gpus / sorted_gpus)
            ax.plot(sorted_gpus, ideal, 'k--', linewidth=0.8, alpha=0.5,
                    label='Ideal', zorder=0)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time per step (ms)')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=5)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_scaling_combined_{grid_label}{_tol_suffix(tol)}'))


def plot_pif_tol_comparison(output_dir, grid_key='current', pruned=False):
    """Compare two tolerances (1e-4 vs 1e-8) on the same axes, one line per cluster.
    Left subplot: tol 1e-4, Right subplot: tol 1e-8, with shared y-axis for direct comparison.
    """
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    fft_label = 'pruned' if pruned else 'full'
    print(f'\n=== PIF tolerance comparison (grid {grid_label}, {fft_label} FFT) ===')

    data_1e4 = _load_landau_scaling_robust(grid_key, pruned=pruned, tol=None)
    data_1e8 = _load_landau_scaling_robust(grid_key, pruned=pruned, tol='1e8')

    if not data_1e4 and not data_1e8:
        print('  [skip] no data for comparison')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)

    for ax, data, tl in [
        (axes[0], data_1e4, r'$\varepsilon=10^{-4}$'),
        (axes[1], data_1e8, r'$\varepsilon=10^{-8}$'),
    ]:
        all_gpus = set()
        if data:
            for cluster in CLUSTERS:
                if cluster not in data:
                    continue
                df = data[cluster]
                all_gpus.update(df['gpus'])
                ax.plot(df['gpus'], df['median_ms'],
                        marker=CLUSTER_MARKERS[cluster],
                        color=CLUSTER_COLORS[cluster],
                        label=CLUSTER_LABELS[cluster],
                        markeredgecolor='black', markeredgewidth=0.3,
                        markersize=3.5)
                ax.fill_between(df['gpus'], df['min_ms'], df['max_ms'],
                                color=CLUSTER_COLORS[cluster], alpha=0.08)

        if all_gpus:
            first = next((c for c in CLUSTERS if c in data), None)
            if first:
                ref_df = data[first]
                ref_gpus = ref_df['gpus'].iloc[0]
                ref_time = ref_df['median_ms'].iloc[0]
                sorted_gpus = np.array(sorted(all_gpus))
                ideal = ref_time * (ref_gpus / sorted_gpus)
                ax.plot(sorted_gpus, ideal, 'k--', linewidth=0.8, alpha=0.5,
                        label='Ideal', zorder=0)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.set_title(tl, fontsize=9)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=5)

    axes[0].set_ylabel('Time per step (ms)')

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_tol_comparison_{grid_label}_{fft_label}'))


def plot_pif_tol_overlay(output_dir, grid_key='current', pruned=False):
    """Overlay tol 1e-4 (solid) vs tol 1e-8 (dashed) on one axis, all clusters."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    fft_label = 'pruned' if pruned else 'full'
    print(f'\n=== PIF tolerance overlay (grid {grid_label}, {fft_label} FFT) ===')

    data_1e4 = _load_landau_scaling_robust(grid_key, pruned=pruned, tol=None)
    data_1e8 = _load_landau_scaling_robust(grid_key, pruned=pruned, tol='1e8')

    if not data_1e4 and not data_1e8:
        print('  [skip] no data for overlay')
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

    all_gpus = set()
    for cluster in CLUSTERS:
        # tol 1e-4 (solid)
        if data_1e4 and cluster in data_1e4:
            df = data_1e4[cluster]
            all_gpus.update(df['gpus'])
            ax.plot(df['gpus'], df['median_ms'],
                    marker=CLUSTER_MARKERS[cluster],
                    color=CLUSTER_COLORS[cluster], linestyle='-',
                    label=f'{CLUSTER_LABELS[cluster]} ($10^{{-4}}$)',
                    markeredgecolor='black', markeredgewidth=0.3,
                    markersize=3.5)
            ax.fill_between(df['gpus'], df['min_ms'], df['max_ms'],
                            color=CLUSTER_COLORS[cluster], alpha=0.08)

        # tol 1e-8 (dashed)
        if data_1e8 and cluster in data_1e8:
            df = data_1e8[cluster]
            all_gpus.update(df['gpus'])
            ax.plot(df['gpus'], df['median_ms'],
                    marker=CLUSTER_MARKERS[cluster],
                    color=CLUSTER_COLORS[cluster], linestyle='--',
                    label=f'{CLUSTER_LABELS[cluster]} ($10^{{-8}}$)',
                    markeredgecolor='black', markeredgewidth=0.3,
                    markersize=3.5, fillstyle='none')
            ax.fill_between(df['gpus'], df['min_ms'], df['max_ms'],
                            color=CLUSTER_COLORS[cluster], alpha=0.08)

    if all_gpus:
        first = next((c for c in CLUSTERS
                       if (data_1e4 and c in data_1e4) or (data_1e8 and c in data_1e8)), None)
        if first:
            ref_df = (data_1e4[first] if data_1e4 and first in data_1e4
                      else data_1e8[first])
            ref_gpus = ref_df['gpus'].iloc[0]
            ref_time = ref_df['median_ms'].iloc[0]
            sorted_gpus = np.array(sorted(all_gpus))
            ideal = ref_time * (ref_gpus / sorted_gpus)
            ax.plot(sorted_gpus, ideal, 'k--', linewidth=0.8, alpha=0.5,
                    label='Ideal', zorder=0)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time per step (ms)')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=5)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_tol_overlay_{grid_label}_{fft_label}'))


def plot_pif_full_vs_pruned(output_dir, grid_key='current'):
    """Compare Full vs Pruned FFT in PIF scaling — separate subplot per cluster."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== PIF Full vs Pruned (grid {grid_label}) ===')

    full_data = _load_landau_scaling(grid_key, pruned=False)
    pruned_data = _load_landau_scaling(grid_key, pruned=True)

    # Only clusters that have both
    clusters_with_both = [c for c in CLUSTERS if c in full_data and c in pruned_data]
    if not clusters_with_both:
        print('  [skip] no clusters with both full and pruned')
        return

    n = len(clusters_with_both)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 2.8), sharey=True, squeeze=False)
    axes = axes[0]

    for ai, cluster in enumerate(clusters_with_both):
        ax = axes[ai]
        fdf = full_data[cluster]
        pdf = pruned_data[cluster]

        ax.plot(fdf['gpus'], fdf['mean_s'] * 1000,
                marker='o', color=CLUSTER_COLORS[cluster], linestyle='-',
                label='Full FFT', markeredgecolor='black', markeredgewidth=0.3)
        ax.fill_between(fdf['gpus'], fdf['min_s'] * 1000, fdf['max_s'] * 1000,
                        color=CLUSTER_COLORS[cluster], alpha=0.12)

        ax.plot(pdf['gpus'], pdf['mean_s'] * 1000,
                marker='s', color=CLUSTER_COLORS[cluster], linestyle='--',
                label='Pruned FFT', markeredgecolor='black', markeredgewidth=0.3)
        ax.fill_between(pdf['gpus'], pdf['min_s'] * 1000, pdf['max_s'] * 1000,
                        color=CLUSTER_COLORS[cluster], alpha=0.12)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        if ai == 0:
            ax.set_ylabel('Time per step (ms)')
        ax.set_title(CLUSTER_LABELS[cluster], fontsize=9)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=6)

    plt.suptitle(f'PIF Landau Damping — Full vs Pruned FFT, Grid ${grid_label}^3$',
                  fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_full_vs_pruned_{grid_label}'))


def plot_pif_components(output_dir, grid_key='current'):
    """PIF component breakdown (Scatter, Gather, Update) as stacked bars."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== PIF component breakdown (grid {grid_label}) ===')

    data = _load_landau_scaling(grid_key, pruned=False)
    if not data:
        print('  [skip] no PIF data')
        return

    comp_colors = {
        'Scatter (Type 1)': '#0072B2',
        'Gather (Type 2)':  '#56B4E9',
        'Particle update':  '#E69F00',
    }
    comp_order = ['Scatter (Type 1)', 'Gather (Type 2)', 'Particle update']

    clusters_with_data = [c for c in CLUSTERS if c in data]
    n = len(clusters_with_data)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 2.8), sharey=True, squeeze=False)
    axes = axes[0]

    for ai, cluster in enumerate(clusters_with_data):
        ax = axes[ai]
        df = data[cluster]
        gpu_counts = df['gpus'].values
        x = np.arange(len(gpu_counts))
        bw = 0.7

        for gi in range(len(gpu_counts)):
            row = df.iloc[gi]
            vals = {
                'Scatter (Type 1)': row['scatter_mean'] * 1000,
                'Gather (Type 2)': row['gather_mean'] * 1000,
                'Particle update': row['update_mean'] * 1000,
            }
            bottom = 0.0
            for comp in comp_order:
                v = vals[comp]
                lbl = comp if gi == 0 else None
                ax.bar(x[gi], v, bw, bottom=bottom,
                       color=comp_colors[comp], edgecolor='black',
                       linewidth=0.3, label=lbl)
                bottom += v

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=6)
        ax.set_xlabel('Number of GPUs')
        if ai == 0:
            ax.set_ylabel('Time per step (ms)')
        ax.set_title(CLUSTER_LABELS[cluster], fontsize=9)
        ax.set_ylim(bottom=0)
        if ai == 0:
            ax.legend(loc='upper right', fontsize=6)

    plt.suptitle(f'PIF Component Breakdown — Grid ${grid_label}^3$ (full FFT)',
                  fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_components_{grid_label}'))


# ════════════════════════════════════════════════════════════════════════
# 9. NUFFT strong scaling with LUMI (all three clusters, both grids)
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_scaling_combined(output_dir):
    """Combined NUFFT scaling for 512 and 1024 grids, all clusters."""
    setup_style()
    print('\n=== NUFFT scaling combined (512 + 1024) ===')

    data_512 = _load_nufft_scaling_data('current')
    data_1024 = _load_nufft_scaling_data('1024')

    if not data_512 and not data_1024:
        print('  [skip] no data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    for ax, ttype, title in [
        (axes[0], 'type1', 'Type\u20111 (spreading)'),
        (axes[1], 'type2', 'Type\u20112 (interpolation)'),
    ]:
        for grid_key, data, ls in [('512', data_512, '-'), ('1024', data_1024, '--')]:
            for cluster, df in data.items():
                tdf = df[df['transform_type'] == ttype]
                if tdf.empty:
                    continue
                stats = tdf.groupby('gpus')['time_ms'].agg(
                    ['mean', 'min', 'max']).reset_index().sort_values('gpus')

                label = f'{CLUSTER_LABELS[cluster]} ({grid_key}$^3$)'
                ax.plot(stats['gpus'], stats['mean'],
                        marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                        linestyle=ls, label=label, markersize=3.5,
                        markeredgecolor='black', markeredgewidth=0.3)
                ax.fill_between(stats['gpus'], stats['min'], stats['max'],
                                color=CLUSTER_COLORS[cluster], alpha=0.08)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(title, fontsize=9, pad=6)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=5)

    plt.suptitle('NUFFT Strong Scaling (pruned FFT)', fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'nufft_scaling_combined'))


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate all paper plots')
    parser.add_argument('--output-dir', '-o', default='plots_paper',
                        help='Output directory (default: plots_paper)')
    args = parser.parse_args()

    output_dir = str(BASE / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # ── Single-node plots ─────────────────────────────────────────────
    plot_kernel_throughput(output_dir)           # Fig 1: kernels vs tol
    plot_breakdown_all(output_dir)               # Fig 2: NUFFT breakdown
    plot_comparison(output_dir)                  # Fig 3: cross-arch comparison
    plot_density_sweep(output_dir)               # Fig 4: density sweep

    # ── Pruned vs Full FFT (before distributed, per reviewer) ─────────
    plot_fft_pruned_vs_full(output_dir)          # Fig 5a: pruned FFT scaling
    plot_fft_pruned_vs_full_all_conc(output_dir) # Fig 5b: all concurrencies

    # ── Distributed NUFFT ─────────────────────────────────────────────
    plot_nufft_scaling(output_dir, 'current')    # Fig 6a: 512 scaling
    plot_nufft_scaling(output_dir, '1024')       # Fig 6b: 1024 scaling
    plot_nufft_scaling_combined(output_dir)       # Fig 6c: combined
    plot_nufft_components(output_dir, 'current') # Fig 7a: 512 components
    plot_nufft_components(output_dir, '1024')    # Fig 7b: 1024 components

    # ── PIF / Landau damping ──────────────────────────────────────────
    plot_pif_scaling(output_dir, 'current')      # Fig 8a: PIF 512 full
    plot_pif_scaling(output_dir, '1024')         # Fig 8b: PIF 1024 full
    plot_pif_scaling_pruned(output_dir, 'current')  # Fig 8c: PIF 512 pruned
    plot_pif_scaling_pruned(output_dir, '1024')     # Fig 8d: PIF 1024 pruned
    plot_pif_scaling_combined(output_dir, 'current')  # Fig 9: combined pruned vs full 512
    plot_pif_scaling_combined(output_dir, '1024')    # Fig 9b: combined pruned vs full 1024
    plot_pif_full_vs_pruned(output_dir, 'current')  # Fig 9c: full vs pruned 512
    plot_pif_full_vs_pruned(output_dir, '1024')     # Fig 9d: full vs pruned 1024
    plot_pif_components(output_dir, 'current')   # Fig 10a: PIF components 512
    plot_pif_components(output_dir, '1024')      # Fig 10b: PIF components 1024

    # ── PIF / tol 1e-8 ────────────────────────────────────────────────
    plot_pif_scaling(output_dir, 'current', tol='1e8')
    plot_pif_scaling(output_dir, '1024', tol='1e8')
    plot_pif_scaling_pruned(output_dir, 'current', tol='1e8')
    plot_pif_scaling_pruned(output_dir, '1024', tol='1e8')
    plot_pif_scaling_combined(output_dir, 'current', tol='1e8')
    plot_pif_scaling_combined(output_dir, '1024', tol='1e8')

    # ── PIF / tolerance comparisons ───────────────────────────────────
    plot_pif_tol_comparison(output_dir, 'current', pruned=False)
    plot_pif_tol_comparison(output_dir, '1024', pruned=False)
    plot_pif_tol_comparison(output_dir, 'current', pruned=True)
    plot_pif_tol_comparison(output_dir, '1024', pruned=True)
    plot_pif_tol_overlay(output_dir, 'current', pruned=False)
    plot_pif_tol_overlay(output_dir, '1024', pruned=False)
    plot_pif_tol_overlay(output_dir, 'current', pruned=True)
    plot_pif_tol_overlay(output_dir, '1024', pruned=True)

    print('\nDone.')


if __name__ == '__main__':
    main()
