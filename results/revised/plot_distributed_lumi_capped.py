#!/usr/bin/env python3
"""
Re-plot distributed scaling figures with LUMI capped at 256 GPUs
(node misconfigurations at higher counts).

Generates into plots_paper_lumi_capped/:
  - nufft_scaling_time_{512,1024}
  - nufft_scaling_combined
  - nufft_components_{cluster}_{512,1024}
  - pif_scaling_{512,1024}
  - pif_scaling_pruned_{512,1024}
  - pif_full_vs_pruned_{512,1024}
  - pif_components_{512,1024}

Usage:
  python plot_distributed_lumi_capped.py [--output-dir plots_paper_lumi_capped]
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

# ════════════════════════════════════════════════════════════════════════
# Paths & constants
# ════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parent
CLUSTERS = ['alps', 'juwels', 'lumi']
GPUS_PER_NODE = {'alps': 4, 'juwels': 4, 'lumi': 8}

LUMI_MAX_GPUS = 256

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

# ── Colors ────────────────────────────────────────────────────────────

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

CLUSTER_COLORS = {'alps': '#2166AC', 'juwels': '#D6604D', 'lumi': '#4DAF4A'}
CLUSTER_MARKERS = {'alps': 'o', 'juwels': 's', 'lumi': '^'}
CLUSTER_LABELS = {
    'alps':   'Alps (GH200)',
    'juwels': 'JUWELS Booster (A100)',
    'lumi':   'LUMI (MI250X)',
}

TIMER_MAP_T1 = {
    'scatterTimerNUFFT1':  'Spreading',
    'FFTNUFFT1':           'FFT',
    'deconvolutionNUFFT1': 'Deconvolution',
}
TIMER_MAP_T2 = {
    'GatherNUFFT2':        'Interpolation',
    'FFTNUFFT2':           'FFT',
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


def _cap_lumi(df, cluster):
    """Drop rows with gpus > LUMI_MAX_GPUS for LUMI."""
    if cluster == 'lumi':
        return df[df['gpus'] <= LUMI_MAX_GPUS].copy()
    return df


# ════════════════════════════════════════════════════════════════════════
# NUFFT data loaders (with LUMI cap)
# ════════════════════════════════════════════════════════════════════════

def _find_nufft_csv(cluster, grid_key, nodes, kind):
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    folder = f'strong_nufft_{grid_key}'
    base_dir = BASE / cluster / folder / f'grid_{grid_size}' / f'nodes_{nodes}'
    pattern = f'nufft_{kind}_{gpus}.csv'
    path = base_dir / pattern
    if path.exists():
        return path
    path = base_dir / 'results' / pattern
    if path.exists():
        return path
    return None


def _load_nufft_scaling_data(grid_key):
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
            data[cluster] = _cap_lumi(pd.concat(rows, ignore_index=True), cluster)
    return data


def _load_nufft_components_data(grid_key):
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
            data[cluster] = _cap_lumi(pd.concat(rows, ignore_index=True), cluster)
    return data


# ════════════════════════════════════════════════════════════════════════
# PIF / Landau data loaders (with LUMI cap)
# ════════════════════════════════════════════════════════════════════════

def _find_landau_csv(cluster, grid_key, nodes, pruned=False):
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    prefix = 'LandauDampingPIFPruned' if pruned else 'LandauDampingPIF'
    base_dir = BASE / cluster / f'strong_landau_{grid_key}' / f'grid_{grid_size}' / f'nodes_{nodes}'
    path = base_dir / f'{prefix}{gpus}.csv'
    return path if path.exists() else None


def load_landau_timings(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    timers = ['ScatterPIFNUFFT', 'GatherPIFNUFFT', 'updateParticle']
    result = {}
    for timer in timers:
        tdf = df[(df['timer_name'] == timer) & (df['rank'] == 0)]
        if tdf.empty:
            continue
        vals = tdf['duration_seconds'].values
        if timer in ('ScatterPIFNUFFT', 'GatherPIFNUFFT'):
            threshold = np.median(vals) * 0.1
            vals = vals[vals > threshold]
        result[timer] = vals
    if not result:
        return None
    min_len = min(len(v) for v in result.values())
    rows = []
    for i in range(min_len):
        s = result.get('ScatterPIFNUFFT', np.zeros(min_len))[i]
        g = result.get('GatherPIFNUFFT', np.zeros(min_len))[i]
        u = result.get('updateParticle', np.zeros(min_len))[i]
        rows.append({'step': i, 'scatter_s': s, 'gather_s': g, 'update_s': u,
                     'total_s': s + g + u})
    return pd.DataFrame(rows)


def _load_landau_scaling(grid_key, pruned=False):
    data = {}
    for cluster in CLUSTERS:
        grid_size = 512 if grid_key == 'current' else 1024
        grid_dir = BASE / cluster / f'strong_landau_{grid_key}' / f'grid_{grid_size}'
        if not grid_dir.exists():
            continue
        rows = []
        for node_dir in sorted(grid_dir.iterdir()):
            if not node_dir.is_dir() or not node_dir.name.startswith('nodes_'):
                continue
            nodes = int(node_dir.name.split('_')[1])
            gpus = nodes * GPUS_PER_NODE[cluster]
            # Apply LUMI cap at the loading stage
            if cluster == 'lumi' and gpus > LUMI_MAX_GPUS:
                continue
            path = _find_landau_csv(cluster, grid_key, nodes, pruned=pruned)
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


# ════════════════════════════════════════════════════════════════════════
# NUFFT scaling plot
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_scaling(output_dir, grid_key='current'):
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

        if all_gpus:
            ref_cluster = list(scaling_data.keys())[0]
            ref_df = scaling_data[ref_cluster]
            ref_tdf = ref_df[ref_df['transform_type'] == ttype]
            ref_min_gpus = ref_tdf['gpus'].min()
            ref_time = ref_tdf[ref_tdf['gpus'] == ref_min_gpus]['time_ms'].mean()
            ideal_gpus = np.array(sorted(all_gpus))
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
    _save(fig, os.path.join(output_dir, f'nufft_scaling_time_{grid_label}'))


# ════════════════════════════════════════════════════════════════════════
# NUFFT scaling combined (512 + 1024)
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_scaling_combined(output_dir):
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
# NUFFT component breakdown
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_components(output_dir, grid_key='current'):
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

        plt.suptitle(f'{CLUSTER_LABELS[cluster]} — Grid ${grid_label}^3$ (pruned FFT)',
                     fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'nufft_components_{cluster}_{grid_label}'))


# ════════════════════════════════════════════════════════════════════════
# PIF scaling (full FFT)
# ════════════════════════════════════════════════════════════════════════

def plot_pif_scaling(output_dir, grid_key='current'):
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== PIF scaling (grid {grid_label}, Full FFT) ===')

    data = _load_landau_scaling(grid_key, pruned=False)
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
    ax.set_title(f'PIF Landau Damping — Grid ${grid_label}^3$ (full FFT)', fontsize=9, pad=6)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=6)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_scaling_{grid_label}'))


# ════════════════════════════════════════════════════════════════════════
# PIF scaling (pruned FFT)
# ════════════════════════════════════════════════════════════════════════

def plot_pif_scaling_pruned(output_dir, grid_key='current'):
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== PIF scaling (grid {grid_label}, Pruned FFT) ===')

    data = _load_landau_scaling(grid_key, pruned=True)
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
    ax.set_title(f'PIF Landau Damping — Grid ${grid_label}^3$ (pruned FFT)', fontsize=9, pad=6)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=6)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, f'pif_scaling_pruned_{grid_label}'))


# ════════════════════════════════════════════════════════════════════════
# PIF full vs pruned
# ════════════════════════════════════════════════════════════════════════

def plot_pif_full_vs_pruned(output_dir, grid_key='current'):
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== PIF Full vs Pruned (grid {grid_label}) ===')

    full_data = _load_landau_scaling(grid_key, pruned=False)
    pruned_data = _load_landau_scaling(grid_key, pruned=True)

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


# ════════════════════════════════════════════════════════════════════════
# PIF component breakdown
# ════════════════════════════════════════════════════════════════════════

def plot_pif_components(output_dir, grid_key='current'):
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
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Re-plot distributed scalings with LUMI capped at 256 GPUs')
    parser.add_argument('--output-dir', '-o', default='plots_paper_lumi_capped',
                        help='Output directory (default: plots_paper_lumi_capped)')
    args = parser.parse_args()

    output_dir = str(BASE / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')
    print(f'LUMI capped at {LUMI_MAX_GPUS} GPUs')

    # NUFFT scaling
    plot_nufft_scaling(output_dir, 'current')
    plot_nufft_scaling(output_dir, '1024')
    plot_nufft_scaling_combined(output_dir)

    # NUFFT components
    plot_nufft_components(output_dir, 'current')
    plot_nufft_components(output_dir, '1024')

    # PIF scaling
    plot_pif_scaling(output_dir, 'current')
    plot_pif_scaling(output_dir, '1024')
    plot_pif_scaling_pruned(output_dir, 'current')
    plot_pif_scaling_pruned(output_dir, '1024')
    plot_pif_full_vs_pruned(output_dir, 'current')
    plot_pif_full_vs_pruned(output_dir, '1024')

    # PIF components
    plot_pif_components(output_dir, 'current')
    plot_pif_components(output_dir, '1024')

    print('\nDone.')


if __name__ == '__main__':
    main()
