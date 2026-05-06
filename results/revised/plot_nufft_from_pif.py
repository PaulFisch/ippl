#!/usr/bin/env python3
"""
Extract NUFFT-level scaling, full-vs-pruned comparison, and component breakdown
from the PIF Landau damping CSV files (LandauDampingPIF*.csv / LandauDampingPIFPruned*.csv).

These CSVs contain per-timer measurements including the internal NUFFT timers:
  Type-1: scatterTimerNUFFT1, FFTNUFFT1, deconvolutionNUFFT1, NativeNUFFT1
  Type-2: GatherNUFFT2, FFTNUFFT2, PrecorrectionNUFFT2, NativeNUFFT2

Generates:
  1. nufft_pif_scaling_{512,1024}.{pdf,png}
       — Strong scaling of the NUFFT (from PIF) for Type-1 and Type-2
  2. nufft_pif_full_vs_pruned_{512,1024}.{pdf,png}
       — Full vs Pruned FFT comparison at the NUFFT level
  3. nufft_pif_components_{cluster}_{512,1024}.{pdf,png}
       — NUFFT component breakdown (stacked bars) extracted from PIF runs
  4. nufft_pif_pruned_vs_full_components_{cluster}_{512,1024}.{pdf,png}
       — Pruned vs Full side-by-side: FFT, Spreading/Interpolation lines

Usage:
  python plot_nufft_from_pif.py [--output-dir plots_paper]
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
# Paths & constants (mirrored from plot_all.py)
# ════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parent
CLUSTERS = ['alps', 'juwels', 'lumi']
GPUS_PER_NODE = {'alps': 4, 'juwels': 4, 'lumi': 8}

# ════════════════════════════════════════════════════════════════════════
# Style (identical to plot_all.py)
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
    'Deconvolution': '#009E73',
    'Precorrection': '#009E73',
}
COMP_ORDER_T1 = ['Spreading', 'FFT', 'Deconvolution']
COMP_ORDER_T2 = ['Interpolation', 'FFT', 'Precorrection']

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

# Timer name -> component label
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


def _tol_label(tol):
    if tol is None:
        return r'$\varepsilon=10^{-4}$'
    return r'$\varepsilon=10^{-' + tol.lstrip('1e') + r'}$'


def _tol_suffix(tol):
    return '' if tol is None else f'_tol{tol}'


def _find_landau_csv(cluster, grid_key, nodes, pruned=False, tol=None):
    """Find LandauDampingPIF CSV.  tol=None -> default, tol='1e8' -> tol_1e8 subdir."""
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


def load_nufft_timers_from_landau(path):
    """Load a Landau CSV and extract per-measurement NUFFT timer values (rank 0).

    Returns dict: timer_name -> np.array of durations (seconds), one per PIF step.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    timers_of_interest = [
        'NativeNUFFT1', 'NativeNUFFT2',
        'scatterTimerNUFFT1', 'FFTNUFFT1', 'deconvolutionNUFFT1',
        'GatherNUFFT2', 'FFTNUFFT2', 'PrecorrectionNUFFT2',
    ]

    result = {}
    for timer in timers_of_interest:
        tdf = df[(df['timer_name'] == timer) & (df['rank'] == 0)]
        if tdf.empty:
            continue
        result[timer] = tdf['duration_seconds'].values

    return result


# ════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════

def _load_nufft_from_pif(grid_key, pruned=False, tol=None):
    """Load NUFFT-level scaling data extracted from Landau CSVs.

    Returns dict: cluster -> DataFrame with columns:
        gpus, t1_mean_ms, t1_min_ms, t1_max_ms,
              t2_mean_ms, t2_min_ms, t2_max_ms,
        and per-component means (scatter_ms, fft1_ms, deconv_ms,
                                  gather_ms, fft2_ms, precorr_ms)
    """
    data = {}
    warmup_steps = 3  # skip first 3 steps

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
                timers = load_nufft_timers_from_landau(path)
            except Exception as e:
                print(f'  [warn] {path}: {e}')
                continue

            if not timers:
                continue

            # Skip warmup steps
            for k in timers:
                if len(timers[k]) > warmup_steps + 2:
                    timers[k] = timers[k][warmup_steps:]

            row = {'gpus': gpus}

            # Total NUFFT times
            for label, timer in [('t1', 'NativeNUFFT1'), ('t2', 'NativeNUFFT2')]:
                vals = timers.get(timer)
                if vals is not None and len(vals) > 0:
                    row[f'{label}_mean_ms'] = np.mean(vals) * 1000
                    row[f'{label}_min_ms'] = np.min(vals) * 1000
                    row[f'{label}_max_ms'] = np.max(vals) * 1000
                    row[f'{label}_median_ms'] = np.median(vals) * 1000

            # Component means
            comp_map = {
                'scatter_ms':  'scatterTimerNUFFT1',
                'fft1_ms':     'FFTNUFFT1',
                'deconv_ms':   'deconvolutionNUFFT1',
                'gather_ms':   'GatherNUFFT2',
                'fft2_ms':     'FFTNUFFT2',
                'precorr_ms':  'PrecorrectionNUFFT2',
            }
            for col, timer in comp_map.items():
                vals = timers.get(timer)
                if vals is not None and len(vals) > 0:
                    row[col] = np.mean(vals) * 1000
                else:
                    row[col] = 0.0

            rows.append(row)

        if rows:
            data[cluster] = pd.DataFrame(rows).sort_values('gpus').reset_index(drop=True)

    return data


# ════════════════════════════════════════════════════════════════════════
# 1. NUFFT scaling (from PIF): time vs GPUs
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_pif_scaling(output_dir, grid_key='current', tol=None):
    """Strong scaling of the NUFFT part extracted from PIF runs."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    tl = _tol_label(tol)
    print(f'\n=== NUFFT-from-PIF scaling (grid {grid_label}, {tl}) ===')

    data = _load_nufft_from_pif(grid_key, pruned=False, tol=tol)
    if not data:
        print('  [skip] no data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=False)

    for ax, prefix, title in [
        (axes[0], 't1', 'Type\u20111 (spreading)'),
        (axes[1], 't2', 'Type\u20112 (interpolation)'),
    ]:
        all_gpus = set()
        for cluster, df in data.items():
            col_mean = f'{prefix}_mean_ms'
            col_min = f'{prefix}_min_ms'
            col_max = f'{prefix}_max_ms'
            if col_mean not in df.columns:
                continue
            cdf = df.dropna(subset=[col_mean])
            if cdf.empty:
                continue
            all_gpus.update(cdf['gpus'])

            ax.plot(cdf['gpus'], cdf[col_mean],
                    marker=CLUSTER_MARKERS[cluster], color=CLUSTER_COLORS[cluster],
                    label=CLUSTER_LABELS[cluster],
                    markeredgecolor='black', markeredgewidth=0.3)
            ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                            color=CLUSTER_COLORS[cluster], alpha=0.15)

        # Ideal scaling
        if all_gpus:
            ref_cluster = list(data.keys())[0]
            ref_df = data[ref_cluster]
            col_mean = f'{prefix}_mean_ms'
            if col_mean in ref_df.columns:
                ref_df_clean = ref_df.dropna(subset=[col_mean])
                if not ref_df_clean.empty:
                    ref_gpus = ref_df_clean['gpus'].iloc[0]
                    ref_time = ref_df_clean[col_mean].iloc[0]
                    sorted_gpus = np.array(sorted(all_gpus))
                    ideal = ref_time * (ref_gpus / sorted_gpus)
                    ax.plot(sorted_gpus, ideal, 'k--', linewidth=0.8, alpha=0.5,
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
    _save(fig, os.path.join(output_dir, f'nufft_pif_scaling_{grid_label}{_tol_suffix(tol)}'))


# ════════════════════════════════════════════════════════════════════════
# 2. Full vs Pruned NUFFT (from PIF)
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_pif_full_vs_pruned(output_dir, grid_key='current', tol=None):
    """Compare Full vs Pruned FFT at the NUFFT level, per cluster."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    tl = _tol_label(tol)
    print(f'\n=== NUFFT-from-PIF Full vs Pruned (grid {grid_label}, {tl}) ===')

    full_data = _load_nufft_from_pif(grid_key, pruned=False, tol=tol)
    pruned_data = _load_nufft_from_pif(grid_key, pruned=True, tol=tol)

    clusters_with_both = [c for c in CLUSTERS if c in full_data and c in pruned_data]
    if not clusters_with_both:
        print('  [skip] no clusters with both full and pruned')
        return

    n = len(clusters_with_both)

    for prefix, type_label, suffix in [
        ('t1', 'Type\u20111', 'type1'),
        ('t2', 'Type\u20112', 'type2'),
    ]:
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 2.8), sharey=True, squeeze=False)
        axes = axes[0]

        for ai, cluster in enumerate(clusters_with_both):
            ax = axes[ai]
            col_mean = f'{prefix}_mean_ms'
            col_min = f'{prefix}_min_ms'
            col_max = f'{prefix}_max_ms'

            fdf = full_data[cluster].dropna(subset=[col_mean])
            pdf = pruned_data[cluster].dropna(subset=[col_mean])

            if not fdf.empty:
                ax.plot(fdf['gpus'], fdf[col_mean],
                        marker='o', color=CLUSTER_COLORS[cluster], linestyle='-',
                        label='Full FFT', markeredgecolor='black', markeredgewidth=0.3)
                ax.fill_between(fdf['gpus'], fdf[col_min], fdf[col_max],
                                color=CLUSTER_COLORS[cluster], alpha=0.12)

            if not pdf.empty:
                ax.plot(pdf['gpus'], pdf[col_mean],
                        marker='s', color=CLUSTER_COLORS[cluster], linestyle='--',
                        label='Pruned FFT', markeredgecolor='black', markeredgewidth=0.3)
                ax.fill_between(pdf['gpus'], pdf[col_min], pdf[col_max],
                                color=CLUSTER_COLORS[cluster], alpha=0.12)

            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Number of GPUs')
            if ai == 0:
                ax.set_ylabel('NUFFT time (ms)')
            ax.set_title(CLUSTER_LABELS[cluster], fontsize=9)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
            ax.legend(loc='best', fontsize=6)

        plt.suptitle(
            f'NUFFT {type_label} — Full vs Pruned FFT, Grid ${grid_label}^3$',
            fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(
            output_dir, f'nufft_pif_full_vs_pruned_{grid_label}_{suffix}{_tol_suffix(tol)}'))


# ════════════════════════════════════════════════════════════════════════
# 3. NUFFT component breakdown (from PIF)
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_pif_components(output_dir, grid_key='current', pruned=False, tol=None):
    """Stacked bar: NUFFT component breakdown vs GPU count from PIF data."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    fft_label = 'pruned' if pruned else 'full'
    tl = _tol_label(tol)
    print(f'\n=== NUFFT-from-PIF components (grid {grid_label}, {fft_label} FFT, {tl}) ===')

    data = _load_nufft_from_pif(grid_key, pruned=pruned, tol=tol)
    if not data:
        print('  [skip] no data')
        return

    # Component columns -> label mapping
    t1_comps = [
        ('scatter_ms',  'Spreading'),
        ('fft1_ms',     'FFT'),
        ('deconv_ms',   'Deconvolution'),
    ]
    t2_comps = [
        ('gather_ms',   'Interpolation'),
        ('fft2_ms',     'FFT'),
        ('precorr_ms',  'Precorrection'),
    ]

    for cluster, df in data.items():
        gpu_counts = df['gpus'].values
        if len(gpu_counts) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
        x = np.arange(len(gpu_counts))
        bw = 0.7

        for ax, comps, comp_order, title in [
            (axes[0], t1_comps, COMP_ORDER_T1, 'Type\u20111 (spreading)'),
            (axes[1], t2_comps, COMP_ORDER_T2, 'Type\u20112 (interpolation)'),
        ]:
            for gi in range(len(gpu_counts)):
                row = df.iloc[gi]
                bottom = 0.0
                for col, label in comps:
                    v = row.get(col, 0.0)
                    if v == 0:
                        continue
                    lbl = label if gi == 0 else None
                    ax.bar(x[gi], v, bw, bottom=bottom,
                           color=COMP_COLORS.get(label, '#999'),
                           edgecolor='black', linewidth=0.3, label=lbl)
                    bottom += v

            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=6)
            ax.set_xlabel('Number of GPUs')
            ax.set_ylabel('Time (ms)')
            ax.set_title(title, fontsize=9)
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=6)

        plt.suptitle(
            f'{CLUSTER_LABELS[cluster]} — Grid ${grid_label}^3$ ({fft_label} FFT, from PIF)',
            fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(
            output_dir, f'nufft_pif_components_{cluster}_{grid_label}_{fft_label}{_tol_suffix(tol)}'))


# ════════════════════════════════════════════════════════════════════════
# 4. Pruned vs Full component comparison for a single cluster
#    Shows FFT (combined T1+T2 avg), Spreading, Interpolation as lines,
#    with pruned (solid) and full (dashed) on the same axes.
# ════════════════════════════════════════════════════════════════════════

def plot_nufft_pif_pruned_vs_full_components(output_dir, grid_key='current',
                                              cluster='lumi', tol=None):
    """Line plot: pruned vs full component times (FFT, Spreading, Interpolation)
    on one axis for a single cluster."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    tl = _tol_label(tol)
    print(f'\n=== NUFFT pruned-vs-full components {cluster} '
          f'(grid {grid_label}, {tl}) ===')

    full_data = _load_nufft_from_pif(grid_key, pruned=False, tol=tol)
    pruned_data = _load_nufft_from_pif(grid_key, pruned=True, tol=tol)

    if cluster not in full_data and cluster not in pruned_data:
        print(f'  [skip] no data for {cluster}')
        return

    # Components to plot: (col_name, label, color)
    components = [
        ('fft_avg_ms',    'FFT',            '#CC79A7'),
        ('scatter_ms',    'Spreading',      '#0072B2'),
        ('gather_ms',     'Interpolation',  '#56B4E9'),
    ]

    def add_fft_avg(df):
        """Add fft_avg_ms = mean of FFT Type-1 and Type-2."""
        df = df.copy()
        df['fft_avg_ms'] = (df['fft1_ms'] + df['fft2_ms']) / 2.0
        return df

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))

    for data, fft_ls, fft_fill, label_suffix in [
        (pruned_data, '-', 'full', 'pruned'),
        (full_data, '--', 'none', 'full'),
    ]:
        if cluster not in data:
            continue
        df = add_fft_avg(data[cluster])

        for col, comp_label, color in components:
            if col not in df.columns:
                continue
            cdf = df.dropna(subset=[col])
            if cdf.empty:
                continue
            marker = 'o' if label_suffix == 'pruned' else 's'
            ax.plot(cdf['gpus'], cdf[col],
                    marker=marker, color=color, linestyle=fft_ls,
                    label=f'{comp_label} ({label_suffix})',
                    markeredgecolor='black', markeredgewidth=0.3,
                    markersize=3.5,
                    fillstyle=fft_fill if fft_fill != 'full' else 'full')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=5.5)

    plt.tight_layout()
    _save(fig, os.path.join(
        output_dir,
        f'nufft_pif_pruned_vs_full_components_{cluster}_{grid_label}{_tol_suffix(tol)}'))


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate NUFFT-level plots from PIF Landau damping data')
    parser.add_argument('--output-dir', '-o', default='plots_paper',
                        help='Output directory (default: plots_paper)')
    args = parser.parse_args()

    output_dir = str(BASE / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    for tol in [None, '1e8']:
        ts = _tol_suffix(tol)
        tl = _tol_label(tol)
        print(f'\n{"═"*60}\n  Tolerance: {tl}\n{"═"*60}')

        # ── NUFFT scaling from PIF ────────────────────────────────────
        plot_nufft_pif_scaling(output_dir, 'current', tol=tol)
        plot_nufft_pif_scaling(output_dir, '1024', tol=tol)

        # ── Full vs Pruned at NUFFT level ─────────────────────────────
        plot_nufft_pif_full_vs_pruned(output_dir, 'current', tol=tol)
        plot_nufft_pif_full_vs_pruned(output_dir, '1024', tol=tol)

        # ── NUFFT component breakdown from PIF ────────────────────────
        plot_nufft_pif_components(output_dir, 'current', pruned=False, tol=tol)
        plot_nufft_pif_components(output_dir, 'current', pruned=True, tol=tol)
        plot_nufft_pif_components(output_dir, '1024', pruned=False, tol=tol)
        plot_nufft_pif_components(output_dir, '1024', pruned=True, tol=tol)

        # ── Pruned vs Full component lines (per cluster) ─────────────
        for cluster in CLUSTERS:
            plot_nufft_pif_pruned_vs_full_components(
                output_dir, 'current', cluster=cluster, tol=tol)
            plot_nufft_pif_pruned_vs_full_components(
                output_dir, '1024', cluster=cluster, tol=tol)

    print('\nDone.')


if __name__ == '__main__':
    main()
