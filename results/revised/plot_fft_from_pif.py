#!/usr/bin/env python3
"""
Plot FFT time (pruned vs full) extracted from PIF Landau damping CSVs.

Uses FFTNUFFT1 and FFTNUFFT2 timers from LandauDampingPIF*.csv (full)
and LandauDampingPIFPruned*.csv (pruned) to compare FFT performance
across Alps, JUWELS, and LUMI.

Generates:
  fft_pif_pruned_vs_full_{512,1024}_{type1,type2}.{pdf,png}

Usage:
  python plot_fft_from_pif.py [--output-dir plots_paper]
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

# ── Style ─────────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────

def _save(fig, path_stem):
    fig.savefig(f'{path_stem}.pdf', facecolor='white')
    fig.savefig(f'{path_stem}.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'  -> {path_stem}.pdf')


def _find_landau_csv(cluster, grid_key, nodes, pruned=False):
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    prefix = 'LandauDampingPIFPruned' if pruned else 'LandauDampingPIF'
    base_dir = BASE / cluster / f'strong_landau_{grid_key}' / f'grid_{grid_size}' / f'nodes_{nodes}'
    path = base_dir / f'{prefix}{gpus}.csv'
    return path if path.exists() else None


def _load_fft_times(grid_key, pruned=False):
    """Load FFT timer data from Landau CSVs.

    Returns dict: cluster -> DataFrame with columns:
        gpus, fft1_mean_ms, fft1_min_ms, fft1_max_ms,
              fft2_mean_ms, fft2_min_ms, fft2_max_ms
    """
    warmup = 3
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

            path = _find_landau_csv(cluster, grid_key, nodes, pruned=pruned)
            if path is None:
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
            except Exception as e:
                print(f'  [warn] {path}: {e}')
                continue

            row = {'gpus': gpus}
            for label, timer in [('fft1', 'FFTNUFFT1'), ('fft2', 'FFTNUFFT2')]:
                tdf = df[(df['timer_name'] == timer) & (df['rank'] == 0)]
                if tdf.empty:
                    continue
                vals = tdf['duration_seconds'].values
                if len(vals) > warmup + 2:
                    vals = vals[warmup:]
                vals_ms = vals * 1000
                row[f'{label}_mean_ms'] = np.mean(vals_ms)
                row[f'{label}_min_ms'] = np.min(vals_ms)
                row[f'{label}_max_ms'] = np.max(vals_ms)

            if len(row) > 1:
                rows.append(row)

        if rows:
            data[cluster] = pd.DataFrame(rows).sort_values('gpus').reset_index(drop=True)

    return data


# ── Plot ──────────────────────────────────────────────────────────────

def plot_fft_pruned_vs_full(output_dir, grid_key='current'):
    """FFT time: full vs pruned, one subplot per cluster, separate figures per type."""
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== FFT-from-PIF pruned vs full (grid {grid_label}) ===')

    full_data = _load_fft_times(grid_key, pruned=False)
    pruned_data = _load_fft_times(grid_key, pruned=True)

    # Clusters with at least full data (pruned may be absent, e.g. LUMI)
    clusters_any = [c for c in CLUSTERS if c in full_data]
    clusters_both = [c for c in CLUSTERS if c in full_data and c in pruned_data]

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

            # Full
            fdf = full_data[cluster]
            if col_mean in fdf.columns:
                cdf = fdf.dropna(subset=[col_mean])
                if not cdf.empty:
                    ax.plot(cdf['gpus'], cdf[col_mean],
                            marker='o', color=CLUSTER_COLORS[cluster], linestyle='-',
                            label='Full FFT', markeredgecolor='black', markeredgewidth=0.3)
                    ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                    color=CLUSTER_COLORS[cluster], alpha=0.12)

            # Pruned
            if cluster in pruned_data:
                pdf = pruned_data[cluster]
                if col_mean in pdf.columns:
                    cdf = pdf.dropna(subset=[col_mean])
                    if not cdf.empty:
                        ax.plot(cdf['gpus'], cdf[col_mean],
                                marker='s', color=CLUSTER_COLORS[cluster], linestyle='--',
                                label='Pruned FFT', markeredgecolor='black', markeredgewidth=0.3)
                        ax.fill_between(cdf['gpus'], cdf[col_min], cdf[col_max],
                                        color=CLUSTER_COLORS[cluster], alpha=0.12)

            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xlabel('Number of GPUs')
            if ai == 0:
                ax.set_ylabel('FFT time (ms)')
            ax.set_title(CLUSTER_LABELS[cluster], fontsize=9)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
            ax.legend(loc='best', fontsize=6)

        plt.suptitle(
            f'FFT {type_label} — Full vs Pruned, Grid ${grid_label}^3$ (from PIF)',
            fontsize=9, y=1.02)
        plt.tight_layout()
        _save(fig, os.path.join(output_dir, f'fft_pif_pruned_vs_full_{grid_label}_{suffix}'))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot FFT pruned vs full from PIF Landau damping data')
    parser.add_argument('--output-dir', '-o', default='plots_paper',
                        help='Output directory (default: plots_paper)')
    args = parser.parse_args()

    output_dir = str(BASE / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    plot_fft_pruned_vs_full(output_dir, 'current')
    plot_fft_pruned_vs_full(output_dir, '1024')

    print('\nDone.')


if __name__ == '__main__':
    main()
