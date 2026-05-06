#!/usr/bin/env python3
"""
Plot NUFFT scaling from upsampled benchmark data, and compare with pruned-FFT results.

Generates:
  1. Upsampled-only scaling (time vs GPUs) for 512 and 1024 grids
  2. Upsampled-only component breakdown (stacked bars) per cluster
  3. Comparison: upsampled vs pruned scaling (overlay on same axes)
  4. Comparison: upsampled vs pruned component breakdown (side-by-side bars)

Data layout:
  revised/upsampled/{cluster}_nufft_upsampled/strong_nufft_{key}/grid_{N}/nodes_{n}/results/
      nufft_scaling_{gpus}_upsampled.csv
      nufft_components_{gpus}_upsampled.csv

Usage:
  python plot_upsampled.py [--output-dir plots_upsampled]
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

BASE = Path(__file__).resolve().parent
CLUSTERS = ['alps', 'juwels', 'lumi']
GPUS_PER_NODE = {'alps': 4, 'juwels': 4, 'lumi': 8}

# ═══════════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════════

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

CLUSTER_COLORS = {'alps': '#2166AC', 'juwels': '#D6604D', 'lumi': '#4DAF4A'}
CLUSTER_MARKERS = {'alps': 'o', 'juwels': 's', 'lumi': '^'}
CLUSTER_LABELS = {
    'alps':   'Alps (GH200)',
    'juwels': 'JUWELS Booster (A100)',
    'lumi':   'LUMI (MI250X)',
}

COMP_COLORS = {
    'Spreading':     '#0072B2',
    'Interpolation': '#56B4E9',
    'FFT':           '#CC79A7',
    'Halo':          '#E69F00',
    'Deconvolution': '#009E73',
    'Precorrection': '#009E73',
}
COMP_ORDER_T1 = ['Spreading', 'FFT', 'Deconvolution', 'Halo']
COMP_ORDER_T2 = ['Interpolation', 'FFT', 'Precorrection', 'Halo']

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


def _save(fig, path_stem):
    fig.savefig(f'{path_stem}.pdf', facecolor='white')
    fig.savefig(f'{path_stem}.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'  -> {path_stem}.pdf')


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def _find_upsampled_csv(cluster, grid_key, nodes, kind):
    """Find upsampled nufft_scaling or nufft_components CSV."""
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    base_dir = (BASE / 'upsampled' / f'{cluster}_nufft_upsampled'
                / f'strong_nufft_{grid_key}' / f'grid_{grid_size}' / f'nodes_{nodes}')
    pattern = f'nufft_{kind}_{gpus}_upsampled.csv'
    for p in [base_dir / pattern, base_dir / 'results' / pattern]:
        if p.exists():
            return p
    return None


def _find_pruned_csv(cluster, grid_key, nodes, kind):
    """Find original (pruned-FFT) nufft_scaling or nufft_components CSV."""
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    base_dir = BASE / cluster / f'strong_nufft_{grid_key}' / f'grid_{grid_size}' / f'nodes_{nodes}'
    pattern = f'nufft_{kind}_{gpus}.csv'
    for p in [base_dir / pattern, base_dir / 'results' / pattern]:
        if p.exists():
            return p
    return None


def _load_scaling_data(grid_key, upsampled=False):
    """Load NUFFT scaling data. Returns dict: cluster -> DataFrame."""
    finder = _find_upsampled_csv if upsampled else _find_pruned_csv
    data = {}
    for cluster in CLUSTERS:
        grid_size = 512 if grid_key == 'current' else 1024
        if upsampled:
            grid_dir = (BASE / 'upsampled' / f'{cluster}_nufft_upsampled'
                        / f'strong_nufft_{grid_key}' / f'grid_{grid_size}')
        else:
            grid_dir = BASE / cluster / f'strong_nufft_{grid_key}' / f'grid_{grid_size}'
        if not grid_dir.exists():
            continue
        rows = []
        for node_dir in sorted(grid_dir.iterdir()):
            if not node_dir.is_dir() or not node_dir.name.startswith('nodes_'):
                continue
            nodes = int(node_dir.name.split('_')[1])
            gpus = nodes * GPUS_PER_NODE[cluster]
            path = finder(cluster, grid_key, nodes, 'scaling')
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


def _load_components_data(grid_key, upsampled=False):
    """Load NUFFT component data. Returns dict: cluster -> DataFrame."""
    finder = _find_upsampled_csv if upsampled else _find_pruned_csv
    data = {}
    for cluster in CLUSTERS:
        grid_size = 512 if grid_key == 'current' else 1024
        if upsampled:
            grid_dir = (BASE / 'upsampled' / f'{cluster}_nufft_upsampled'
                        / f'strong_nufft_{grid_key}' / f'grid_{grid_size}')
        else:
            grid_dir = BASE / cluster / f'strong_nufft_{grid_key}' / f'grid_{grid_size}'
        if not grid_dir.exists():
            continue
        rows = []
        for node_dir in sorted(grid_dir.iterdir()):
            if not node_dir.is_dir() or not node_dir.name.startswith('nodes_'):
                continue
            nodes = int(node_dir.name.split('_')[1])
            gpus = nodes * GPUS_PER_NODE[cluster]
            path = finder(cluster, grid_key, nodes, 'components')
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


# ═══════════════════════════════════════════════════════════════════════
# 1. Upsampled NUFFT scaling (time vs GPUs)
# ═══════════════════════════════════════════════════════════════════════

def plot_upsampled_scaling(output_dir, grid_key='current'):
    """Strong scaling: time vs GPUs for upsampled NUFFT, all clusters."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== Upsampled NUFFT scaling (grid {grid_label}) ===')

    data = _load_scaling_data(grid_key, upsampled=True)
    if not data:
        print('  [skip] no upsampled scaling data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=False)

    for ax, ttype, title in [
        (axes[0], 'type1', 'Type\u20111 (spreading)'),
        (axes[1], 'type2', 'Type\u20112 (interpolation)'),
    ]:
        all_gpus = set()
        for cluster, df in data.items():
            tdf = df[df['transform_type'] == ttype]
            if tdf.empty:
                continue
            stats = tdf.groupby('gpus')['time_ms'].agg(
                ['mean', 'min', 'max']).reset_index().sort_values('gpus')
            all_gpus.update(stats['gpus'])

            ax.plot(stats['gpus'], stats['mean'],
                    marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                    label=CLUSTER_LABELS[cluster],
                    markeredgecolor='black', markeredgewidth=0.3)
            ax.fill_between(stats['gpus'], stats['min'], stats['max'],
                            color=CLUSTER_COLORS[cluster], alpha=0.15)

        if all_gpus:
            sorted_gpus = sorted(all_gpus)
            ref_cluster = list(data.keys())[0]
            ref_df = data[ref_cluster]
            ref_tdf = ref_df[ref_df['transform_type'] == ttype]
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
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(title, fontsize=9)
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=6)

    plt.suptitle(f'Upsampled NUFFT Strong Scaling — Grid ${grid_label}^3$',
                 fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'upsampled_scaling_time_{grid_label}'))


# ═══════════════════════════════════════════════════════════════════════
# 2. Upsampled NUFFT component breakdown
# ═══════════════════════════════════════════════════════════════════════

def plot_upsampled_components(output_dir, grid_key='current'):
    """Stacked bar: upsampled NUFFT component breakdown vs GPU count."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== Upsampled NUFFT components (grid {grid_label}) ===')

    comp_data = _load_components_data(grid_key, upsampled=True)
    if not comp_data:
        print('  [skip] no upsampled component data')
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
                    timer_names = [k for k, v in timer_map.items() if v == comp]
                    total = 0.0
                    for tn in timer_names:
                        tdf = gdf[gdf['timer'] == tn]
                        if not tdf.empty:
                            total += tdf['time_s'].mean() * 1000
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

        # plt.suptitle(f'{CLUSTER_LABELS[cluster]} — Upsampled NUFFT Components, Grid ${grid_label}^3$',
        #              fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'upsampled_components_{cluster}_{grid_label}'))


# ═══════════════════════════════════════════════════════════════════════
# 3. Comparison: upsampled vs pruned scaling (overlay)
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison_scaling(output_dir, grid_key='current'):
    """Overlay upsampled and pruned NUFFT scaling on the same axes."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== Upsampled vs Pruned scaling comparison (grid {grid_label}) ===')

    data_pruned = _load_scaling_data(grid_key, upsampled=False)
    data_upsampled = _load_scaling_data(grid_key, upsampled=True)

    if not data_pruned and not data_upsampled:
        print('  [skip] no data')
        return

    # One subplot per cluster that has both datasets
    clusters_any = [c for c in CLUSTERS
                    if c in data_pruned or c in data_upsampled]
    if not clusters_any:
        return

    for ttype, type_title, suffix in [
        ('type1', 'Type\u20111 (spreading)', 'type1'),
        ('type2', 'Type\u20112 (interpolation)', 'type2'),
    ]:
        n = len(clusters_any)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.8), sharey=True, squeeze=False)
        axes = axes[0]

        for ai, cluster in enumerate(clusters_any):
            ax = axes[ai]

            # Pruned
            if cluster in data_pruned:
                df = data_pruned[cluster]
                tdf = df[df['transform_type'] == ttype]
                if not tdf.empty:
                    stats = tdf.groupby('gpus')['time_ms'].agg(
                        ['mean', 'min', 'max']).reset_index().sort_values('gpus')
                    ax.plot(stats['gpus'], stats['mean'],
                            marker='o', color=CLUSTER_COLORS[cluster], linestyle='-',
                            label='Pruned FFT',
                            markeredgecolor='black', markeredgewidth=0.3)
                    ax.fill_between(stats['gpus'], stats['min'], stats['max'],
                                    color=CLUSTER_COLORS[cluster], alpha=0.12)

            # Upsampled
            if cluster in data_upsampled:
                df = data_upsampled[cluster]
                tdf = df[df['transform_type'] == ttype]
                if not tdf.empty:
                    stats = tdf.groupby('gpus')['time_ms'].agg(
                        ['mean', 'min', 'max']).reset_index().sort_values('gpus')
                    ax.plot(stats['gpus'], stats['mean'],
                            marker='s', color=CLUSTER_COLORS[cluster], linestyle='--',
                            label='Upsampled (full FFT)',
                            markeredgecolor='black', markeredgewidth=0.3)
                    ax.fill_between(stats['gpus'], stats['min'], stats['max'],
                                    color=CLUSTER_COLORS[cluster], alpha=0.12)

            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Number of GPUs')
            if ai == 0:
                ax.set_ylabel('Time (ms)')
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_title(CLUSTER_LABELS[cluster], fontsize=9)
            ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
            ax.legend(loc='best', fontsize=6)

        # plt.suptitle(f'Pruned vs Upsampled NUFFT — {type_title}, Grid ${grid_label}^3$',
        #              fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir,
              f'comparison_scaling_{grid_label}_{suffix}'))


def plot_comparison_scaling_combined(output_dir, grid_key='current', max_gpus=None):
    """All clusters on one plot, pruned (solid) vs upsampled (dashed)."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    suffix = f'_{max_gpus}' if max_gpus else ''
    print(f'\n=== Combined pruned vs upsampled scaling (grid {grid_label}{suffix}) ===')

    data_pruned = _load_scaling_data(grid_key, upsampled=False)
    data_upsampled = _load_scaling_data(grid_key, upsampled=True)

    if not data_pruned and not data_upsampled:
        print('  [skip] no data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    for ax, ttype, title in [
        (axes[0], 'type1', 'Type\u20111 (spreading)'),
        (axes[1], 'type2', 'Type\u20112 (interpolation)'),
    ]:
        for cluster in CLUSTERS:
            # Pruned (solid)
            if cluster in data_pruned:
                df = data_pruned[cluster]
                tdf = df[df['transform_type'] == ttype]
                if max_gpus:
                    tdf = tdf[tdf['gpus'] <= max_gpus]
                if not tdf.empty:
                    stats = tdf.groupby('gpus')['time_ms'].agg(
                        ['mean', 'min', 'max']).reset_index().sort_values('gpus')
                    ax.plot(stats['gpus'], stats['mean'],
                            marker=CLUSTER_MARKERS[cluster],
                            color=CLUSTER_COLORS[cluster], linestyle='-',
                            label=f'{CLUSTER_LABELS[cluster]} (pruned)',
                            markeredgecolor='black', markeredgewidth=0.3,
                            markersize=3.5)
                    ax.fill_between(stats['gpus'], stats['min'], stats['max'],
                                    color=CLUSTER_COLORS[cluster], alpha=0.08)

            # Upsampled (dashed)
            if cluster in data_upsampled:
                df = data_upsampled[cluster]
                tdf = df[df['transform_type'] == ttype]
                if max_gpus:
                    tdf = tdf[tdf['gpus'] <= max_gpus]
                if not tdf.empty:
                    stats = tdf.groupby('gpus')['time_ms'].agg(
                        ['mean', 'min', 'max']).reset_index().sort_values('gpus')
                    ax.plot(stats['gpus'], stats['mean'],
                            marker=CLUSTER_MARKERS[cluster],
                            color=CLUSTER_COLORS[cluster], linestyle='--',
                            label=f'{CLUSTER_LABELS[cluster]} (upsampled)',
                            markeredgecolor='black', markeredgewidth=0.3,
                            markersize=3.5, fillstyle='none')
                    ax.fill_between(stats['gpus'], stats['min'], stats['max'],
                                    color=CLUSTER_COLORS[cluster], alpha=0.08)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(title, fontsize=9)
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=5)

    # gpu_note = f', $\\leq${max_gpus} GPUs' if max_gpus else ''
    # plt.suptitle(f'Pruned vs Upsampled NUFFT — Grid ${grid_label}^3${gpu_note}',
    #              fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'comparison_scaling_combined_{grid_label}{suffix}'))


# ═══════════════════════════════════════════════════════════════════════
# 4. Comparison: upsampled vs pruned component breakdown (side-by-side)
# ═══════════════════════════════════════════════════════════════════════

def _get_component_ms(df, gpus, timer_map, comp):
    """Extract total ms for a component at a given GPU count."""
    gdf = df[df['gpus'] == gpus]
    timer_names = [k for k, v in timer_map.items() if v == comp]
    total = 0.0
    for tn in timer_names:
        tdf = gdf[gdf['timer'] == tn]
        if not tdf.empty:
            total += tdf['time_s'].mean() * 1000
    return total


def plot_comparison_components(output_dir, grid_key='current'):
    """Side-by-side stacked bars: pruned vs upsampled components per cluster."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== Pruned vs Upsampled component comparison (grid {grid_label}) ===')

    comp_pruned = _load_components_data(grid_key, upsampled=False)
    comp_upsampled = _load_components_data(grid_key, upsampled=True)

    if not comp_pruned and not comp_upsampled:
        print('  [skip] no component data')
        return

    clusters_both = [c for c in CLUSTERS
                     if c in comp_pruned and c in comp_upsampled]
    if not clusters_both:
        print('  [skip] no cluster has both pruned and upsampled data')
        return

    for cluster in clusters_both:
        df_p = comp_pruned[cluster]
        df_u = comp_upsampled[cluster]

        # Use GPU counts that appear in both datasets
        gpus_p = set(df_p['gpus'].unique())
        gpus_u = set(df_u['gpus'].unique())
        gpu_counts = sorted(gpus_p & gpus_u)
        if not gpu_counts:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        bw = 0.35

        for ax, timer_map, comp_order, title in [
            (axes[0], TIMER_MAP_T1, COMP_ORDER_T1, 'Type\u20111 (spreading)'),
            (axes[1], TIMER_MAP_T2, COMP_ORDER_T2, 'Type\u20112 (interpolation)'),
        ]:
            x = np.arange(len(gpu_counts))

            for offset, df, bar_label, hatch in [
                (-bw/2, df_p, 'Pruned', None),
                (+bw/2, df_u, 'Upsampled', '///'),
            ]:
                for gi, gpus in enumerate(gpu_counts):
                    bottom = 0.0
                    for ci, comp in enumerate(comp_order):
                        val = _get_component_ms(df, gpus, timer_map, comp)
                        if val == 0:
                            continue
                        lbl = None
                        # Label only for first GPU count, pruned gets comp name,
                        # upsampled gets "Upsampled" once
                        if gi == 0 and hatch is None:
                            lbl = comp
                        ax.bar(x[gi] + offset, val, bw, bottom=bottom,
                               color=COMP_COLORS.get(comp, '#999'),
                               edgecolor='black', linewidth=0.3,
                               hatch=hatch, label=lbl, alpha=0.85 if hatch else 1.0)
                        bottom += val

                    # Total annotation
                    if bottom > 0 and gi == 0:
                        pass  # skip crowded annotations

            # Add a legend entry for hatching
            import matplotlib.patches as mpatches
            pruned_patch = mpatches.Patch(facecolor='#cccccc', edgecolor='black',
                                          linewidth=0.3, label='Pruned')
            upsampled_patch = mpatches.Patch(facecolor='#cccccc', edgecolor='black',
                                             linewidth=0.3, hatch='///', label='Upsampled')

            handles, labels = ax.get_legend_handles_labels()
            handles.extend([pruned_patch, upsampled_patch])
            labels.extend(['Pruned', 'Upsampled'])

            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=6)
            ax.set_xlabel('Number of GPUs')
            ax.set_ylabel('Time (ms)')
            ax.set_title(title, fontsize=9)
            ax.set_ylim(bottom=0)
            ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=5)

        plt.suptitle(f'{CLUSTER_LABELS[cluster]} — Pruned vs Upsampled Components, Grid ${grid_label}^3$',
                     fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir,
              f'comparison_components_{cluster}_{grid_label}'))


# ═══════════════════════════════════════════════════════════════════════
# 5. Speedup ratio: pruned time / upsampled time
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison_ratio(output_dir, grid_key='current'):
    """Ratio plot: pruned_time / upsampled_time vs GPUs."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== Pruned / Upsampled ratio (grid {grid_label}) ===')

    data_pruned = _load_scaling_data(grid_key, upsampled=False)
    data_upsampled = _load_scaling_data(grid_key, upsampled=True)

    clusters_both = [c for c in CLUSTERS
                     if c in data_pruned and c in data_upsampled]
    if not clusters_both:
        print('  [skip] no overlapping data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    for ax, ttype, title in [
        (axes[0], 'type1', 'Type\u20111 (spreading)'),
        (axes[1], 'type2', 'Type\u20112 (interpolation)'),
    ]:
        for cluster in clusters_both:
            df_p = data_pruned[cluster]
            df_u = data_upsampled[cluster]

            tp = df_p[df_p['transform_type'] == ttype].groupby('gpus')['time_ms'].mean()
            tu = df_u[df_u['transform_type'] == ttype].groupby('gpus')['time_ms'].mean()

            common = sorted(set(tp.index) & set(tu.index))
            if not common:
                continue

            ratio = [tp[g] / tu[g] for g in common]
            ax.plot(common, ratio,
                    marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                    label=CLUSTER_LABELS[cluster],
                    markeredgecolor='black', markeredgewidth=0.3)

        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Pruned / Upsampled time ratio')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(title, fontsize=9)
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=6)

    plt.suptitle(f'Pruned / Upsampled Time Ratio — Grid ${grid_label}^3$',
                 fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'comparison_ratio_{grid_label}'))


# ═══════════════════════════════════════════════════════════════════════
# 6. FFT-only: pruned vs full (extracted from component CSVs)
# ═══════════════════════════════════════════════════════════════════════

def _extract_fft_times(grid_key, upsampled=False):
    """Extract FFTNUFFT1 / FFTNUFFT2 times from component CSVs.

    Returns dict: cluster -> DataFrame with columns
        gpus, fft1_mean_ms, fft1_min_ms, fft1_max_ms,
              fft2_mean_ms, fft2_min_ms, fft2_max_ms
    """
    comp_data = _load_components_data(grid_key, upsampled=upsampled)
    result = {}
    for cluster, df in comp_data.items():
        rows = []
        for gpus in sorted(df['gpus'].unique()):
            gdf = df[df['gpus'] == gpus]
            row = {'gpus': gpus}
            for label, timer in [('fft1', 'FFTNUFFT1'), ('fft2', 'FFTNUFFT2')]:
                tdf = gdf[gdf['timer'] == timer]
                if tdf.empty:
                    continue
                vals_ms = tdf['time_s'].values * 1000
                row[f'{label}_mean_ms'] = np.mean(vals_ms)
                row[f'{label}_min_ms'] = np.min(vals_ms)
                row[f'{label}_max_ms'] = np.max(vals_ms)
            if len(row) > 1:
                rows.append(row)
        if rows:
            result[cluster] = pd.DataFrame(rows).sort_values('gpus').reset_index(drop=True)
    return result


def plot_fft_pruned_vs_full(output_dir, grid_key='current'):
    """Pruned vs full FFT time per cluster (one subplot each), separate per type."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== FFT pruned vs full from NUFFT benchmarks (grid {grid_label}) ===')

    full_data = _extract_fft_times(grid_key, upsampled=True)
    pruned_data = _extract_fft_times(grid_key, upsampled=False)

    clusters_any = [c for c in CLUSTERS if c in full_data or c in pruned_data]
    if not clusters_any:
        print('  [skip] no FFT data')
        return

    for fft_timer, type_label, suffix in [
        ('fft1', 'Type\u20111 (forward)', 'type1'),
        ('fft2', 'Type\u20112 (backward)', 'type2'),
    ]:
        col_mean = f'{fft_timer}_mean_ms'
        col_min = f'{fft_timer}_min_ms'
        col_max = f'{fft_timer}_max_ms'

        n = len(clusters_any)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.8), sharey=True, squeeze=False)
        axes = axes[0]

        for ai, cluster in enumerate(clusters_any):
            ax = axes[ai]

            # Full FFT (from upsampled benchmarks)
            if cluster in full_data:
                fdf = full_data[cluster]
                if col_mean in fdf.columns:
                    cdf = fdf.dropna(subset=[col_mean])
                    if not cdf.empty:
                        ax.plot(cdf['gpus'], cdf[col_mean],
                                marker='s', color=CLUSTER_COLORS[cluster], linestyle='--',
                                label='Full FFT (upsampled)',
                                markeredgecolor='black', markeredgewidth=0.3)
                        ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                        color=CLUSTER_COLORS[cluster], alpha=0.12)

            # Pruned FFT (from standard benchmarks)
            if cluster in pruned_data:
                pdf = pruned_data[cluster]
                if col_mean in pdf.columns:
                    cdf = pdf.dropna(subset=[col_mean])
                    if not cdf.empty:
                        ax.plot(cdf['gpus'], cdf[col_mean],
                                marker='o', color=CLUSTER_COLORS[cluster], linestyle='-',
                                label='Pruned FFT',
                                markeredgecolor='black', markeredgewidth=0.3)
                        ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                        color=CLUSTER_COLORS[cluster], alpha=0.12)

            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Number of GPUs')
            if ai == 0:
                ax.set_ylabel('FFT time (ms)')
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_title(CLUSTER_LABELS[cluster], fontsize=9)
            ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
            ax.legend(loc='best', fontsize=6)

        plt.suptitle(f'FFT {type_label} — Pruned vs Full, Grid ${grid_label}^3$',
                     fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'fft_pruned_vs_full_{grid_label}_{suffix}'))


def plot_fft_pruned_vs_full_combined(output_dir, grid_key='current'):
    """All clusters on one plot: pruned (solid) vs full (dashed) FFT, Type 1 + Type 2."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== FFT pruned vs full combined (grid {grid_label}) ===')

    full_data = _extract_fft_times(grid_key, upsampled=True)
    pruned_data = _extract_fft_times(grid_key, upsampled=False)

    if not full_data and not pruned_data:
        print('  [skip] no FFT data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    for ax, fft_timer, type_label in [
        (axes[0], 'fft1', 'Type\u20111 (forward)'),
        (axes[1], 'fft2', 'Type\u20112 (backward)'),
    ]:
        col_mean = f'{fft_timer}_mean_ms'
        col_min = f'{fft_timer}_min_ms'
        col_max = f'{fft_timer}_max_ms'

        for cluster in CLUSTERS:
            # Pruned (solid)
            if cluster in pruned_data:
                pdf = pruned_data[cluster]
                if col_mean in pdf.columns:
                    cdf = pdf.dropna(subset=[col_mean])
                    if not cdf.empty:
                        ax.plot(cdf['gpus'], cdf[col_mean],
                                marker=CLUSTER_MARKERS[cluster],
                                color=CLUSTER_COLORS[cluster], linestyle='-',
                                label=f'{CLUSTER_LABELS[cluster]} (pruned)',
                                markeredgecolor='black', markeredgewidth=0.3,
                                markersize=3.5)
                        ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                        color=CLUSTER_COLORS[cluster], alpha=0.08)

            # Full (dashed, open markers)
            if cluster in full_data:
                fdf = full_data[cluster]
                if col_mean in fdf.columns:
                    cdf = fdf.dropna(subset=[col_mean])
                    if not cdf.empty:
                        ax.plot(cdf['gpus'], cdf[col_mean],
                                marker=CLUSTER_MARKERS[cluster],
                                color=CLUSTER_COLORS[cluster], linestyle='--',
                                label=f'{CLUSTER_LABELS[cluster]} (full)',
                                markeredgecolor='black', markeredgewidth=0.3,
                                markersize=3.5, fillstyle='none')
                        ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                        color=CLUSTER_COLORS[cluster], alpha=0.08)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('FFT time (ms)')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(type_label, fontsize=9)
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=5)

    plt.suptitle(f'FFT Pruned vs Full — Grid ${grid_label}^3$',
                 fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'fft_pruned_vs_full_combined_{grid_label}'))


def plot_fft_pruned_vs_full_type1_both_grids(output_dir):
    """Type 1 FFT pruned vs full, 512^3 on left, 1024^3 on right."""
    setup_style()
    print('\n=== FFT pruned vs full Type 1 (512 + 1024) ===')

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)

    for ax, grid_key in [(axes[0], 'current'), (axes[1], '1024')]:
        grid_label = '512' if grid_key == 'current' else '1024'
        full_data = _extract_fft_times(grid_key, upsampled=True)
        pruned_data = _extract_fft_times(grid_key, upsampled=False)

        col_mean, col_min, col_max = 'fft1_mean_ms', 'fft1_min_ms', 'fft1_max_ms'

        for cluster in CLUSTERS:
            if cluster in pruned_data:
                pdf = pruned_data[cluster]
                if col_mean in pdf.columns:
                    cdf = pdf.dropna(subset=[col_mean])
                    if not cdf.empty:
                        ax.plot(cdf['gpus'], cdf[col_mean],
                                marker=CLUSTER_MARKERS[cluster],
                                color=CLUSTER_COLORS[cluster], linestyle='-',
                                label=f'{CLUSTER_LABELS[cluster]} (pruned)',
                                markeredgecolor='black', markeredgewidth=0.3,
                                markersize=3.5)
                        ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                        color=CLUSTER_COLORS[cluster], alpha=0.08)

            if cluster in full_data:
                fdf = full_data[cluster]
                if col_mean in fdf.columns:
                    cdf = fdf.dropna(subset=[col_mean])
                    if not cdf.empty:
                        ax.plot(cdf['gpus'], cdf[col_mean],
                                marker=CLUSTER_MARKERS[cluster],
                                color=CLUSTER_COLORS[cluster], linestyle='--',
                                label=f'{CLUSTER_LABELS[cluster]} (full)',
                                markeredgecolor='black', markeredgewidth=0.3,
                                markersize=3.5, fillstyle='none')
                        ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                        color=CLUSTER_COLORS[cluster], alpha=0.08)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(f'Grid ${grid_label}^3$', fontsize=9)
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=5)

    axes[0].set_ylabel('FFT time (ms)')

    plt.suptitle('FFT Type\u20111 — Pruned vs Full', fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'fft_pruned_vs_full_type1'))


def plot_fft_ratio(output_dir, grid_key='current'):
    """Ratio plot: pruned FFT time / full FFT time vs GPUs, per cluster."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== FFT pruned / full ratio (grid {grid_label}) ===')

    full_data = _extract_fft_times(grid_key, upsampled=True)
    pruned_data = _extract_fft_times(grid_key, upsampled=False)

    clusters_both = [c for c in CLUSTERS if c in full_data and c in pruned_data]
    if not clusters_both:
        print('  [skip] no overlapping FFT data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    for ax, fft_timer, type_label in [
        (axes[0], 'fft1', 'Type\u20111 (forward)'),
        (axes[1], 'fft2', 'Type\u20112 (backward)'),
    ]:
        col_mean = f'{fft_timer}_mean_ms'

        for cluster in clusters_both:
            pdf = pruned_data[cluster]
            fdf = full_data[cluster]
            if col_mean not in pdf.columns or col_mean not in fdf.columns:
                continue

            p_by_gpu = pdf.set_index('gpus')[col_mean]
            f_by_gpu = fdf.set_index('gpus')[col_mean]
            common = sorted(set(p_by_gpu.index) & set(f_by_gpu.index))
            if not common:
                continue

            ratio = [p_by_gpu[g] / f_by_gpu[g] for g in common]
            ax.plot(common, ratio,
                    marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                    label=CLUSTER_LABELS[cluster],
                    markeredgecolor='black', markeredgewidth=0.3)

        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Pruned / Full FFT time ratio')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_title(type_label, fontsize=9)
        ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
        ax.legend(loc='best', fontsize=6)

    plt.suptitle(f'Pruned / Full FFT Time Ratio — Grid ${grid_label}^3$',
                 fontsize=9, y=1.02)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'fft_ratio_{grid_label}'))


# ═══════════════════════════════════════════════════════════════════════
# 7. JUWELS FFT scaling (like Alps plot in plot_all.py, from NUFFT data)
# ═══════════════════════════════════════════════════════════════════════

def plot_juwels_fft_scaling(output_dir):
    """Pruned vs full FFT on JUWELS, both grid sizes on one plot (like Alps in plot_all.py)."""
    setup_style()
    print('\n=== JUWELS FFT pruned vs full scaling ===')

    method_labels = {'full': 'Full FFT', 'pruned': 'Pruned FFT'}

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

    grid_keys = [('current', 512), ('1024', 1024)]
    # Shade by grid size: lighter for smaller, darker for larger
    _grid_method_colors = {
        (512,  'full'): '#999999', (512,  'pruned'): '#F0A050',
        (1024, 'full'): '#333333', (1024, 'pruned'): '#D55E00',
    }
    has_data = False

    for grid_key, grid_size in grid_keys:
        pruned = _extract_fft_times(grid_key, upsampled=False)
        full = _extract_fft_times(grid_key, upsampled=True)

        ls = '-' if grid_size == 1024 else '--'

        # Pruned
        if 'juwels' in pruned:
            pdf = pruned['juwels']
            col = 'fft1_mean_ms'
            if col in pdf.columns:
                cdf = pdf.dropna(subset=[col])
                if not cdf.empty:
                    has_data = True
                    color = _grid_method_colors[(grid_size, 'pruned')]
                    ax.plot(cdf['gpus'], cdf[col],
                            marker='o', color=color,
                            linestyle=ls,
                            label=f'{method_labels["pruned"]} ({grid_size}$^3$)',
                            markeredgecolor='black', markeredgewidth=0.3)
                    ax.fill_between(cdf['gpus'], cdf['fft1_min_ms'], cdf['fft1_max_ms'],
                                    color=color, alpha=0.1)

        # Full (from upsampled benchmarks)
        if 'juwels' in full:
            fdf = full['juwels']
            col = 'fft1_mean_ms'
            if col in fdf.columns:
                cdf = fdf.dropna(subset=[col])
                if not cdf.empty:
                    has_data = True
                    color = _grid_method_colors[(grid_size, 'full')]
                    ax.plot(cdf['gpus'], cdf[col],
                            marker='s', color=color,
                            linestyle=ls,
                            label=f'{method_labels["full"]} ({grid_size}$^3$)',
                            markeredgecolor='black', markeredgewidth=0.3)
                    ax.fill_between(cdf['gpus'], cdf['fft1_min_ms'], cdf['fft1_max_ms'],
                                    color=color, alpha=0.1)

    if not has_data:
        print('  [skip] no JUWELS FFT data')
        plt.close(fig)
        return

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    # ax.set_title('Forward FFT — JUWELS Booster (A100)', fontsize=9)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'juwels_fft_scaling'))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plot upsampled NUFFT scaling and comparison with pruned')
    parser.add_argument('--output-dir', '-o', default='plots_upsampled',
                        help='Output directory (default: plots_upsampled)')
    args = parser.parse_args()

    output_dir = str(BASE / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    # ── Upsampled-only plots ─────────────────────────────────────────
    plot_upsampled_scaling(output_dir, 'current')
    plot_upsampled_scaling(output_dir, '1024')
    plot_upsampled_components(output_dir, 'current')
    plot_upsampled_components(output_dir, '1024')

    # ── Comparison: pruned vs upsampled ──────────────────────────────
    plot_comparison_scaling(output_dir, 'current')
    plot_comparison_scaling(output_dir, '1024')
    plot_comparison_scaling_combined(output_dir, 'current')
    plot_comparison_scaling_combined(output_dir, 'current', max_gpus=256)
    plot_comparison_scaling_combined(output_dir, '1024')
    plot_comparison_components(output_dir, 'current')
    plot_comparison_components(output_dir, '1024')
    plot_comparison_ratio(output_dir, 'current')
    plot_comparison_ratio(output_dir, '1024')

    # ── FFT pruned vs full (extracted from component CSVs) ───────────
    plot_fft_pruned_vs_full(output_dir, 'current')
    plot_fft_pruned_vs_full(output_dir, '1024')
    plot_fft_pruned_vs_full_combined(output_dir, 'current')
    plot_fft_pruned_vs_full_combined(output_dir, '1024')
    plot_fft_pruned_vs_full_type1_both_grids(output_dir)
    plot_fft_ratio(output_dir, 'current')
    plot_fft_ratio(output_dir, '1024')

    # ── JUWELS FFT scaling (like Alps in plot_all.py) ────────────────
    plot_juwels_fft_scaling(output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
