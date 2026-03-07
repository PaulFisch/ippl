#!/usr/bin/env python3
"""
plot_paper.py
=============
Unified plotting script for all figures in the PASC '26 paper
"A Performance-Portable, Massively Parallel Distributed Nonuniform FFT".

Generates figures with reviewer-requested modifications:
  Fig 1/2:  Kernel throughput vs tolerance (Alps and LUMI), with cuFINUFFT overlay
  Fig 3:    Tile-size heatmap – 2×2 layout: (Tiled, GridParallel) × (rho=1, rho=10)
  Fig 4:    NUFFT component time breakdown (Alps), with distinct Halo colour
  Fig 5:    Throughput comparison with cuFINUFFT vs density
  Fig 6:    Distributed NUFFT strong scaling (Alps, JUWELS, LUMI if available)
  Fig 7:    Distributed NUFFT component breakdown (Alps), with distinct Halo colour
  Fig 8:    Pruned vs Full-FFT strong scaling: time vs GPU count (replaces bar chart)
  Fig 9:    PIF solver strong scaling (Alps, JUWELS Booster, LUMI)

Usage
-----
  # From the results/ directory:
  python plot_paper.py --output-dir paper_figures/

  # Generate only specific figures:
  python plot_paper.py --figs 1 2 3 4 --output-dir paper_figures/

  # Override a data path:
  python plot_paper.py --scaling-lumi path/to/benchmark_results_lumi
"""

import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

def setup_style():
    plt.rcParams.update({
        'font.family':          'serif',
        'font.serif':           ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size':            8,
        'mathtext.fontset':     'stix',
        'axes.titlesize':       9,
        'axes.labelsize':       8,
        'axes.linewidth':       0.5,
        'axes.spines.top':      False,
        'axes.spines.right':    False,
        'axes.axisbelow':       True,
        'xtick.labelsize':      7,
        'ytick.labelsize':      7,
        'xtick.major.width':    0.5,
        'ytick.major.width':    0.5,
        'xtick.major.size':     2.5,
        'ytick.major.size':     2.5,
        'xtick.direction':      'out',
        'ytick.direction':      'out',
        'legend.fontsize':      7,
        'legend.frameon':       False,
        'legend.borderpad':     0.3,
        'legend.handlelength':  1.5,
        'legend.handletextpad': 0.4,
        'lines.linewidth':      1.2,
        'lines.markersize':     4,
        'figure.facecolor':     'white',
        'figure.dpi':           150,
        'savefig.dpi':          300,
        'savefig.bbox':         'tight',
        'savefig.pad_inches':   0.02,
        'grid.linewidth':       0.3,
        'grid.alpha':           0.5,
    })


# ---------------------------------------------------------------------------
# Shared colour / marker / label dictionaries
# ---------------------------------------------------------------------------

# Kernel methods
KERNEL_COLORS = {
    'Atomic':       '#E69F00',   # orange
    'Tiled':        '#009E73',   # bluish-green
    'GridParallel': '#0072B2',   # blue
    'Direct':       '#E69F00',
    'Sorted':       '#56B4E9',
    'TeamParallel': '#0072B2',
    'Native':       '#CC79A7',
    'cuFINUFFT':    '#D55E00',   # vermillion  (reviewer: add to Fig 1)
}
KERNEL_MARKERS = {
    'Atomic': 's', 'Tiled': '^', 'GridParallel': 'o',
    'Direct': 's', 'Sorted': 'D', 'TeamParallel': 'o',
    'Native': 'v', 'cuFINUFFT': 'X',
}
KERNEL_LABELS = {
    'Atomic': 'Atomic', 'Tiled': 'Tiled', 'GridParallel': 'Grid-Parallel',
    'Direct': 'Direct', 'Sorted': 'Sorted', 'TeamParallel': 'Team-Parallel',
    'Native': 'Native', 'cuFINUFFT': 'cuFINUFFT',
}

# Clusters
CLUSTER_COLORS  = {'alps': '#2166AC', 'juwels': '#D6604D', 'lumi': '#4DAF4A',
                   'booster': '#D6604D'}
CLUSTER_MARKERS = {'alps': 'o', 'juwels': 's', 'lumi': '^', 'booster': 's'}
CLUSTER_LABELS  = {'alps': 'Alps (GH200)', 'juwels': 'JUWELS (A100)',
                   'lumi': 'LUMI (MI250X)', 'booster': 'JUWELS Booster (A100)'}

# NUFFT component colours – reviewer: Halo must be clearly different from Scatter/Gather
# Old: Scatter/Gather='#0072B2' (blue), Halo='#56B4E9' (sky-blue) – too similar.
# New: Halo='#E69F00' (orange/amber), Scatter/Gather='#0072B2' (blue).
COMPONENT_COLORS = {
    'Scatter/Gather': '#0072B2',   # blue
    'FFT':            '#CC79A7',   # pink
    'Halo':           '#E69F00',   # amber  ← reviewer fix
    'Deconv/Precorr': '#009E73',   # green
}
COMPONENT_LABELS_T1 = {
    'Scatter/Gather': 'Spreading',
    'FFT':            'FFT',
    'Halo':           'Halo Exchange',
    'Deconv/Precorr': 'Deconvolution',
}
COMPONENT_LABELS_T2 = {
    'Scatter/Gather': 'Interpolation',
    'FFT':            'FFT',
    'Halo':           'Halo Exchange',
    'Deconv/Precorr': 'Precorrection',
}
STACK_ORDER = ['Scatter/Gather', 'FFT', 'Deconv/Precorr', 'Halo']

# Distributed NUFFT timer → category mappings
TIMER_CATS_T1 = {
    'scatterTimerNUFFT1':   'Scatter/Gather',
    'accumulateHaloNUFFT1': 'Halo',
    'FFTNUFFT1':            'FFT',
    'deconvolutionNUFFT1':  'Deconv/Precorr',
}
TIMER_CATS_T2 = {
    'PrecorrectionNUFFT2': 'Deconv/Precorr',
    'FFTNUFFT2':           'FFT',
    'FillHaloNUFFT2':      'Halo',
    'GatherNUFFT2':        'Scatter/Gather',
}

# Breakdown (single-node) timer → category
BREAKDOWN_TIMER_CATS = {
    'NativeNUFFT1::scatter':     'Scatter/Gather',
    'NativeNUFFT1::halo':        'Halo',
    'NativeNUFFT1::fft':         'FFT',
    'NativeNUFFT1::deconv':      'Deconv/Precorr',
    'NativeNUFFT2::gather':      'Scatter/Gather',
    'NativeNUFFT2::halo':        'Halo',
    'NativeNUFFT2::fft':         'FFT',
    'NativeNUFFT2::precorr':     'Deconv/Precorr',
    # fallback patterns matched in load_breakdown()
}

# Landau / pruned-FFT
PRUNED_COLORS  = {'PIF': '#0072B2', 'PIFPruned': '#E69F00'}
PRUNED_MARKERS = {'PIF': 'o',       'PIFPruned': 's'}
PRUNED_LABELS  = {'PIF': 'Full FFT', 'PIFPruned': 'Pruned FFT'}

# Landau NUFFT sub-component colours (plot_landau.py)
NUFFT_COMPONENT_COLORS = {
    'Spreading':     '#4477AA',
    'Deconvolution': '#EE6677',
    'Interpolation': '#228833',
    'Precorrection': '#CCBB44',
    'FFT':           '#66CCEE',
    'Halo Exchange': '#AA3377',
}
N_OUTLIERS_DISCARD = 5

# PIF solver component colours (plot_pif.py)
PIF_COMPONENT_COLORS = {
    'ScatterPIFNUFFT': '#0072B2',
    'GatherPIFNUFFT':  '#E69F00',
    'updateParticle':  '#56B4E9',
    'dumpData':        '#009E73',
}
PIF_COMPONENT_LABELS = {
    'ScatterPIFNUFFT': 'Scatter (Type-1)',
    'GatherPIFNUFFT':  'Gather (Type-2)',
    'updateParticle':  'Particle Update',
    'dumpData':        'I/O',
}


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save(fig, out_dir: Path, stem: str):
    for ext in ('pdf', 'png'):
        p = out_dir / f'{stem}.{ext}'
        fig.savefig(p, facecolor='white', dpi=300)
        print(f'  saved {p}')
    plt.close(fig)


# ===========================================================================
# Figure 1 / 2 – Kernel throughput vs tolerance, Alps and LUMI
# ===========================================================================

def load_kernel_csv(path: Path) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def plot_kernel_throughput_vs_tolerance(
    df: pd.DataFrame,
    density_df: pd.DataFrame | None,
    rho: float,
    cluster_label: str,
    out_dir: Path,
    stem: str,
):
    """
    One figure per (cluster, rho): throughput vs tolerance for scatter and
    gather kernels side by side, with cuFINUFFT horizontal reference overlay.
    """
    setup_style()

    # Filter to this rho
    data = df[np.isclose(df['rho'], rho)].copy()
    if data.empty:
        print(f'  [skip] no data for rho={rho} in {stem}')
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=False)

    for ax, op in zip(axes, ['scatter', 'gather']):
        op_data = data[data['operation'] == op]

        # Group by kernel name
        for kernel in op_data['kernel'].unique():
            kdf = op_data[op_data['kernel'] == kernel].sort_values('tolerance')
            if kdf.empty:
                continue
            color  = KERNEL_COLORS.get(kernel, '#999999')
            marker = KERNEL_MARKERS.get(kernel, 'o')
            label  = KERNEL_LABELS.get(kernel, kernel)
            ax.plot(kdf['tolerance'], kdf['throughput_Mpts_per_s'],
                    color=color, marker=marker, label=label,
                    markersize=4, linewidth=1.2)

        # cuFINUFFT horizontal reference (reviewer request)
        if density_df is not None:
            cuf = density_df[
                (density_df['method'] == 'cuFINUFFT') &
                np.isclose(density_df['density'], rho)
            ]
            if cuf.empty:
                # Use closest density
                avail = density_df[density_df['method'] == 'cuFINUFFT']['density']
                if not avail.empty:
                    closest = avail.iloc[(avail - rho).abs().argsort()[:1]].values[0]
                    cuf = density_df[
                        (density_df['method'] == 'cuFINUFFT') &
                        np.isclose(density_df['density'], closest)
                    ]
            if not cuf.empty:
                cuf_tp = cuf['throughput_mpts'].mean()
                tols = sorted(op_data['tolerance'].unique())
                ax.axhline(cuf_tp, color=KERNEL_COLORS['cuFINUFFT'],
                           linestyle='--', linewidth=1.0, alpha=0.85,
                           label='cuFINUFFT (ref.)')

        op_label = 'Spreading' if op == 'scatter' else 'Interpolation'
        ax.set_xscale('log')
        ax.set_xlabel('Tolerance ε')
        ax.set_ylabel('Throughput (Mpts/s)')
        ax.set_title(op_label, fontsize=9)
        ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_kernels(paths: dict, out_dir: Path, figs_requested: set):
    """Figs 1 & 2."""
    if not (figs_requested & {1, 2}):
        return

    density_df = None
    if paths.get('densities') and Path(paths['densities']).exists():
        density_df = pd.read_csv(paths['densities'])

    configs = [
        # (path key,               rho, cluster label,      fig#, stem)
        ('kernels_alps_rho1',   1.0, 'Alps (GH200)',    1, 'fig1a_alps_rho1'),
        ('kernels_alps_rho10', 10.0, 'Alps (GH200)',    1, 'fig1b_alps_rho10'),
        ('kernels_lumi_rho1',   1.0, 'LUMI (MI250X)',   2, 'fig2a_lumi_rho1'),
        ('kernels_lumi_rho10', 10.0, 'LUMI (MI250X)',   2, 'fig2b_lumi_rho10'),
    ]
    for key, rho, label, figno, stem in configs:
        if figno not in figs_requested:
            continue
        p = paths.get(key)
        df = load_kernel_csv(Path(p) if p else None)
        if df is None:
            print(f'  [skip fig {figno}] missing: {p}')
            continue
        print(f'  fig {figno}: {stem}')
        plot_kernel_throughput_vs_tolerance(df, density_df, rho, label, out_dir, stem)


# ===========================================================================
# Figure 3 – Tile-size heatmap, 2×2 layout
# ===========================================================================

def load_heatmap(path: Path) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def plot_heatmap_2x2(panels: list, out_dir: Path, stem: str):
    """
    panels: list of (title, df_or_None)  – 4 entries, row-major order:
        [Tiled-rho1, Tiled-rho10, GP-rho1, GP-rho10]
    reviewer: show Tiled vs Tiled and GP vs GP side by side for ρ=1 vs ρ=10.
    Layout: 2 rows × 2 cols
      row 0: Tiled | Tiled          (col 0: rho=1, col 1: rho=10)
      row 1: Grid-Parallel | GP     (col 0: rho=1, col 1: rho=10)
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.0))

    for idx, (title, df) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        if df is None:
            ax.set_visible(False)
            continue

        # Columns: tile_size, width_2, width_3, ...
        width_cols = [c for c in df.columns if c.startswith('width_')]
        widths = [int(c.split('_')[1]) for c in width_cols]
        tile_sizes = df['tile_size'].values

        Z = df[width_cols].values   # shape (n_tiles, n_widths)

        im = ax.imshow(Z, aspect='auto', origin='lower',
                       extent=[widths[0]-0.5, widths[-1]+0.5,
                               tile_sizes[0]-0.5, tile_sizes[-1]+0.5],
                       cmap='viridis')
        plt.colorbar(im, ax=ax, shrink=0.85, label='Throughput (Mpts/s)')

        ax.set_xlabel('Kernel width w')
        ax.set_ylabel('Tile size')
        ax.set_xticks(widths)
        ax.set_yticks(tile_sizes)
        ax.set_title(title, fontsize=8)

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_heatmap(paths: dict, out_dir: Path, figs_requested: set):
    """Fig 3."""
    if 3 not in figs_requested:
        return

    def _load(key):
        p = paths.get(key)
        return load_heatmap(Path(p) if p else None)

    panels = [
        ('Tiled – ρ = 1',           _load('heatmap_tiled_rho1')),
        ('Tiled – ρ = 10',          _load('heatmap_tiled_rho10')),
        ('Grid-Parallel – ρ = 1',   _load('heatmap_gp_rho1')),
        ('Grid-Parallel – ρ = 10',  _load('heatmap_gp_rho10')),
    ]
    if all(p is None for _, p in panels):
        print('  [skip fig 3] no heatmap data found')
        return
    print('  fig 3: tile heatmap')
    plot_heatmap_2x2(panels, out_dir, 'fig3_heatmap')


# ===========================================================================
# Figure 4 – NUFFT component time breakdown (single node, Alps)
# ===========================================================================

def load_breakdown(path: Path) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    # Map timer names to categories
    timer_map = {}
    for t in df['timer'].unique():
        tl = t.lower()
        if 'scatter' in tl or 'spreading' in tl:
            timer_map[t] = ('Scatter/Gather', 'type1')
        elif 'gather' in tl or 'interpolat' in tl:
            timer_map[t] = ('Scatter/Gather', 'type2')
        elif 'halo' in tl or 'accumulate' in tl or 'fillhalo' in tl:
            timer_map[t] = ('Halo', None)
        elif 'fft' in tl:
            timer_map[t] = ('FFT', None)
        elif 'deconv' in tl:
            timer_map[t] = ('Deconv/Precorr', 'type1')
        elif 'precorr' in tl or 'precondi' in tl:
            timer_map[t] = ('Deconv/Precorr', 'type2')
        # skip unknown

    rows = []
    for t, (cat, ttype) in timer_map.items():
        sub = df[df['timer'] == t].copy()
        sub['category'] = cat
        # infer type from timer name if not set
        if ttype is None:
            sub['transform_type'] = sub['timer'].apply(
                lambda x: 'type1' if 'NUFFT1' in x or 'nufft1' in x.lower() else 'type2')
        else:
            sub['transform_type'] = ttype
        rows.append(sub)

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def plot_breakdown(df: pd.DataFrame, out_dir: Path, stem: str):
    """Stacked bar chart for breakdown_combined.csv data."""
    setup_style()

    grids = sorted(df['grid'].unique())
    n = len(grids)
    x = np.arange(n)
    bar_width = 0.65

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for ax, (ttype, labels) in zip(axes, [('type1', COMPONENT_LABELS_T1),
                                           ('type2', COMPONENT_LABELS_T2)]):
        sub = df[df['transform_type'] == ttype]
        # mean over runs, then sum
        agg = sub.groupby(['grid', 'category'])['time_s'].mean().reset_index()
        agg['time_ms'] = agg['time_s'] * 1000

        bottoms = np.zeros(n)
        for cat in STACK_ORDER:
            heights = []
            for g in grids:
                row = agg[(agg['grid'] == g) & (agg['category'] == cat)]
                heights.append(row['time_ms'].values[0] if not row.empty else 0.0)
            heights = np.array(heights)
            color = COMPONENT_COLORS[cat]
            label = labels.get(cat, cat)
            ax.bar(x, heights, bar_width, bottom=bottoms,
                   color=color, edgecolor='black', linewidth=0.3, label=label)
            bottoms += heights

        for i, total in enumerate(bottoms):
            ax.annotate(f'{total:.0f}',
                        xy=(x[i], total), xytext=(0, 2),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=5)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}³' for g in grids])
        ax.set_xlabel('Grid size')
        ax.set_ylabel('Time (ms)')
        title = 'Type-1 (Spreading)' if ttype == 'type1' else 'Type-2 (Interpolation)'
        ax.set_title(title, fontsize=9)
        ax.legend(loc='upper left', fontsize=6)

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_breakdown(paths: dict, out_dir: Path, figs_requested: set):
    """Fig 4."""
    if 4 not in figs_requested:
        return
    p = paths.get('breakdown')
    df = load_breakdown(Path(p) if p else None)
    if df is None:
        print(f'  [skip fig 4] missing: {p}')
        return
    print('  fig 4: breakdown')
    plot_breakdown(df, out_dir, 'fig4_breakdown')


# ===========================================================================
# Figure 5 – cuFINUFFT comparison: throughput vs density
# ===========================================================================

def plot_densities(df: pd.DataFrame, out_dir: Path, stem: str):
    setup_style()

    # Method display names
    method_colors = {
        'Native_OutputFocused': KERNEL_COLORS['GridParallel'],
        'Native_Tiled':         KERNEL_COLORS['Tiled'],
        'Native_Atomic':        KERNEL_COLORS['Atomic'],
        'cuFINUFFT':            KERNEL_COLORS['cuFINUFFT'],
    }
    method_labels = {
        'Native_OutputFocused': 'Grid-Parallel',
        'Native_Tiled':         'Tiled',
        'Native_Atomic':        'Atomic',
        'cuFINUFFT':            'cuFINUFFT',
    }
    method_markers = {
        'Native_OutputFocused': 'o',
        'Native_Tiled':         '^',
        'Native_Atomic':        's',
        'cuFINUFFT':            'X',
    }

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Left: throughput vs density, Type-1
    ax = axes[0]
    t1 = df[df['type'] == 1]
    for method in t1['method'].unique():
        mdf = t1[t1['method'] == method].sort_values('density')
        ax.plot(mdf['density'], mdf['throughput_mpts'],
                color=method_colors.get(method, '#999999'),
                marker=method_markers.get(method, 'o'),
                label=method_labels.get(method, method))
    ax.set_xscale('log')
    ax.set_xlabel('Density ρ (pts/cell)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-1 (Spreading)', fontsize=9)
    ax.legend(loc='best', fontsize=6)

    # Right: throughput vs density, Type-2
    ax = axes[1]
    t2 = df[df['type'] == 2]
    for method in t2['method'].unique():
        mdf = t2[t2['method'] == method].sort_values('density')
        ax.plot(mdf['density'], mdf['throughput_mpts'],
                color=method_colors.get(method, '#999999'),
                marker=method_markers.get(method, 'o'),
                label=method_labels.get(method, method))
    ax.set_xscale('log')
    ax.set_xlabel('Density ρ (pts/cell)')
    ax.set_ylabel('Throughput (Mpts/s)')
    ax.set_title('Type-2 (Interpolation)', fontsize=9)
    ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_densities(paths: dict, out_dir: Path, figs_requested: set):
    """Fig 5."""
    if 5 not in figs_requested:
        return
    p = paths.get('densities')
    if not p or not Path(p).exists():
        print(f'  [skip fig 5] missing: {p}')
        return
    df = pd.read_csv(p)
    print('  fig 5: densities / cuFINUFFT comparison')
    plot_densities(df, out_dir, 'fig5_densities')


# ===========================================================================
# Figure 6 – Distributed NUFFT strong scaling (time, all clusters)
# ===========================================================================

def load_scaling_dir(base_dir: Path) -> pd.DataFrame | None:
    """Load timings_gpus*.csv files from a benchmark_results_* directory."""
    if base_dir is None or not base_dir.exists():
        return None
    cluster = base_dir.name.replace('benchmark_results_', '').lower()
    files = list(base_dir.glob('timings_gpus*.csv'))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            tmp['cluster'] = cluster
            frames.append(tmp)
        except Exception as e:
            print(f'  warning: {f}: {e}')
    return pd.concat(frames, ignore_index=True) if frames else None


def compute_scaling_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(['cluster', 'num_ranks', 'grid_size', 'transform_type'])
    stats = g['time_ms'].agg(['mean', 'std', 'min', 'max']).reset_index()
    parts = g['num_particles'].first().reset_index()
    stats = stats.merge(parts, on=['cluster', 'num_ranks', 'grid_size', 'transform_type'])
    stats['throughput_mpts'] = stats['num_particles'] / stats['mean'] * 1e3 / 1e6
    stats['throughput_std']  = stats['throughput_mpts'] * (stats['std'] / stats['mean'])
    return stats


def compute_efficiency(stats: pd.DataFrame, group_cols: list, rank_col='num_ranks') -> pd.DataFrame:
    stats = stats.copy()
    stats['efficiency'] = np.nan
    for keys, sub in stats.groupby(group_cols):
        min_r = sub[rank_col].min()
        ref   = sub.loc[sub[rank_col] == min_r, 'mean'].values[0]
        for idx in sub.index:
            n = stats.loc[idx, rank_col]
            t = stats.loc[idx, 'mean']
            stats.loc[idx, 'efficiency'] = ref * min_r / t / n * 100
    return stats


def plot_strong_scaling(stats: pd.DataFrame, out_dir: Path, stem: str, title_suffix=''):
    """Line plot: time vs #GPUs per transform type, one line per cluster."""
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    type_labels = {'type1': 'Type-1 (Spreading)', 'type2': 'Type-2 (Interpolation)'}

    for ax, ttype in zip(axes, ['type1', 'type2']):
        sub = stats[stats['transform_type'] == ttype]

        for cluster in sorted(sub['cluster'].unique()):
            cdf = sub[sub['cluster'] == cluster].sort_values('num_ranks')
            ax.errorbar(cdf['num_ranks'], cdf['mean'], yerr=cdf['std'],
                        color=CLUSTER_COLORS.get(cluster, '#999'),
                        marker=CLUSTER_MARKERS.get(cluster, 'o'),
                        markersize=4, linewidth=1, capsize=2, capthick=0.5,
                        label=CLUSTER_LABELS.get(cluster, cluster))

        # Ideal scaling reference
        if not sub.empty:
            ranks = sorted(sub['num_ranks'].unique())
            ref_t = sub[sub['num_ranks'] == min(ranks)]['mean'].mean()
            ax.plot(ranks, [ref_t * min(ranks) / r for r in ranks],
                    'k:', linewidth=0.8, alpha=0.5, label='Ideal')

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(type_labels.get(ttype, ttype), fontsize=9)
        if not sub.empty:
            ranks = sorted(sub['num_ranks'].unique())
            ax.set_xticks(ranks)
            ax.set_xticklabels([str(r) for r in ranks])
        ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_scaling(paths: dict, out_dir: Path, figs_requested: set):
    """Fig 6."""
    if 6 not in figs_requested:
        return

    frames = []
    for key in ('scaling_alps', 'scaling_juwels', 'scaling_lumi'):
        p = paths.get(key)
        df = load_scaling_dir(Path(p) if p else None)
        if df is not None:
            frames.append(df)
            print(f'  loaded scaling: {key}')
        else:
            print(f'  [no data] {key}: {p}')

    if not frames:
        print('  [skip fig 6] no scaling data')
        return

    combined = pd.concat(frames, ignore_index=True)
    stats    = compute_scaling_stats(combined)
    print('  fig 6: strong scaling')
    plot_strong_scaling(stats, out_dir, 'fig6_scaling')


# ===========================================================================
# Figure 7 – Distributed component breakdown, distinct Halo colour (Alps)
# ===========================================================================

def load_component_dir(base_dir: Path) -> pd.DataFrame | None:
    if base_dir is None or not base_dir.exists():
        return None
    cluster = base_dir.name.replace('benchmark_results_', '').lower()
    files   = list(base_dir.glob('nufft_components_*.csv'))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            tmp['cluster'] = cluster
            frames.append(tmp)
        except Exception as e:
            print(f'  warning: {f}: {e}')
    return pd.concat(frames, ignore_index=True) if frames else None


def compute_component_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Map timer names to categories and compute mean time per (cluster,ranks,type,cat)."""
    df = df.copy()

    def _cat(timer):
        if timer in TIMER_CATS_T1:
            return TIMER_CATS_T1[timer], 'type1'
        if timer in TIMER_CATS_T2:
            return TIMER_CATS_T2[timer], 'type2'
        return None, None

    df['category'], df['transform_type'] = zip(*df['timer'].map(_cat))
    df = df.dropna(subset=['category'])
    df = df[df['run'] > 0]
    df['time_ms'] = df['time_s'] * 1000

    g = df.groupby(['cluster', 'num_ranks', 'transform_type', 'category'])
    return g['time_ms'].agg(['mean', 'std']).reset_index()


def plot_component_breakdown(comp_stats: pd.DataFrame, cluster: str, out_dir: Path, stem: str):
    """Stacked bar chart for one cluster, both transform types.
    reviewer: distinct colour for Halo vs Scatter/Gather."""
    setup_style()

    cdf = comp_stats[comp_stats['cluster'] == cluster]
    if cdf.empty:
        print(f'  [skip] no component data for cluster={cluster}')
        return

    gpu_counts = sorted(cdf['num_ranks'].unique())
    n   = len(gpu_counts)
    x   = np.arange(n)
    bw  = 0.65

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for ax, (ttype, labels) in zip(axes, [('type1', COMPONENT_LABELS_T1),
                                           ('type2', COMPONENT_LABELS_T2)]):
        sub = cdf[cdf['transform_type'] == ttype]
        bottoms = np.zeros(n)

        for cat in STACK_ORDER:
            heights = []
            for gi, g in enumerate(gpu_counts):
                row = sub[(sub['num_ranks'] == g) & (sub['category'] == cat)]
                heights.append(row['mean'].values[0] if not row.empty else 0.0)
            heights = np.array(heights)
            ax.bar(x, heights, bw, bottom=bottoms,
                   color=COMPONENT_COLORS[cat], edgecolor='black', linewidth=0.3,
                   label=labels.get(cat, cat))
            for i, h in enumerate(heights):
                if h > 0:
                    ax.annotate(f'{bottoms[i]+h:.0f}',
                                xy=(x[i], bottoms[i]+h), xytext=(0, 2),
                                textcoords='offset points',
                                ha='center', va='bottom', fontsize=5)
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gpu_counts])
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        title = 'Type-1 (Spreading)' if ttype == 'type1' else 'Type-2 (Interpolation)'
        ax.set_title(title, fontsize=9)
        ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_component_breakdown(paths: dict, out_dir: Path, figs_requested: set):
    """Fig 7."""
    if 7 not in figs_requested:
        return

    frames = []
    for key in ('scaling_alps', 'scaling_juwels', 'scaling_lumi'):
        p = paths.get(key)
        df = load_component_dir(Path(p) if p else None)
        if df is not None:
            frames.append(df)

    if not frames:
        print('  [skip fig 7] no component data')
        return

    combined   = pd.concat(frames, ignore_index=True)
    comp_stats = compute_component_stats(combined)
    print('  fig 7: component breakdown')

    for cluster in comp_stats['cluster'].unique():
        plot_component_breakdown(comp_stats, cluster,
                                 out_dir, f'fig7_components_{cluster}')


# ===========================================================================
# Figure 8 – Pruned vs Full FFT strong scaling (replaces bar chart)
# ===========================================================================

def _parse_landau_filename(fp: Path):
    """Return (version, num_gpus) from LandauDampingPIF{Pruned?}N_M.csv."""
    m = re.match(r'LandauDampingPIF(Pruned)?(\d+)_(\d+)', fp.stem)
    if m:
        return ('PIFPruned' if m.group(1) else 'PIF'), int(m.group(2))
    return None, None


def load_landau_data(base_dir: Path) -> pd.DataFrame | None:
    if base_dir is None or not base_dir.exists():
        return None
    files = list(base_dir.glob('LandauDampingPIF*_*.csv'))
    frames = []
    for fp in files:
        version, num_gpus = _parse_landau_filename(fp)
        if version is None:
            continue
        try:
            df = pd.read_csv(fp)
            df['version']  = version
            df['num_gpus'] = num_gpus
            frames.append(df)
        except Exception as e:
            print(f'  warning: {fp}: {e}')
    return pd.concat(frames, ignore_index=True) if frames else None


def _remove_outliers(group, n=N_OUTLIERS_DISCARD):
    if len(group) <= n:
        return group
    return group.sort_values('duration_seconds', ascending=False).iloc[n:]


# NUFFT sub-timer categories (same as plot_landau.py)
_LANDAU_GATHER_CATS = {
    'PrecorrectionNUFFT2': 'Precorrection',
    'FFTNUFFT2':           'FFT',
    'FillHaloNUFFT2':      'Halo Exchange',
    'GatherNUFFT2':        'Interpolation',
}
_LANDAU_SCATTER_CATS = {
    'scatterTimerNUFFT1':   'Spreading',
    'accumulateHaloNUFFT1': 'Halo Exchange',
    'FFTNUFFT1':            'FFT',
    'deconvolutionNUFFT1':  'Deconvolution',
}
_ALL_NUFFT_TIMERS = list(_LANDAU_GATHER_CATS) + list(_LANDAU_SCATTER_CATS)


def compute_landau_scaling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total NUFFT time per (version, num_gpus) from sub-component timers."""
    sub = df[df['timer_name'].isin(_ALL_NUFFT_TIMERS)].copy()

    # Remove outliers
    sub = sub.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name'],
        group_keys=False
    ).apply(_remove_outliers).reset_index(drop=True)

    # Sum iterations per rank per timer
    per_timer = sub.groupby(
        ['version', 'num_gpus', 'source_file', 'rank', 'timer_name']
    )['duration_seconds'].sum().reset_index()

    # Sum timers → total per rank
    per_rank = per_timer.groupby(
        ['version', 'num_gpus', 'source_file', 'rank']
    )['duration_seconds'].sum().reset_index()

    # Average ranks → per run
    per_run = per_rank.groupby(
        ['version', 'num_gpus', 'source_file']
    )['duration_seconds'].mean().reset_index()

    stats = per_run.groupby(['version', 'num_gpus'])['duration_seconds'].agg(
        mean='mean', std='std').reset_index()
    stats['mean_ms'] = stats['mean'] * 1000
    stats['std_ms']  = stats['std'].fillna(0) * 1000
    return stats


def plot_landau_strong_scaling(stats: pd.DataFrame, out_dir: Path, stem: str):
    """
    Reviewer: Replace bar chart with strong-scaling line comparison
    (Pruned FFT vs Full FFT), like Fig 6.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    # Left: execution time
    ax = axes[0]
    for version in sorted(stats['version'].unique()):
        vdf = stats[stats['version'] == version].sort_values('num_gpus')
        ax.errorbar(vdf['num_gpus'], vdf['mean_ms'], yerr=vdf['std_ms'],
                    color=PRUNED_COLORS.get(version, '#999'),
                    marker=PRUNED_MARKERS.get(version, 'o'),
                    markersize=4, linewidth=1.2, capsize=2, capthick=0.5,
                    label=PRUNED_LABELS.get(version, version))

    if not stats.empty:
        gpus  = sorted(stats['num_gpus'].unique())
        ref_t = stats[stats['num_gpus'] == min(gpus)]['mean_ms'].mean()
        ax.plot(gpus, [ref_t * min(gpus) / g for g in gpus],
                'k--', linewidth=1.0, alpha=0.6, label='Ideal')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('NUFFT Time (ms)')
    ax.set_title('(a) Execution Time', loc='left', fontweight='medium')
    ax.legend(loc='best', fontsize=7)
    if not stats.empty:
        gpus = sorted(stats['num_gpus'].unique())
        ax.set_xticks(gpus)
        ax.set_xticklabels([str(g) for g in gpus])

    # Right: parallel efficiency
    ax = axes[1]
    eff_stats = stats.copy()
    eff_stats['efficiency'] = np.nan
    for version in eff_stats['version'].unique():
        mask = eff_stats['version'] == version
        sub  = eff_stats[mask]
        min_g = sub['num_gpus'].min()
        ref   = sub.loc[sub['num_gpus'] == min_g, 'mean_ms'].values[0]
        for idx in sub.index:
            g = eff_stats.loc[idx, 'num_gpus']
            t = eff_stats.loc[idx, 'mean_ms']
            eff_stats.loc[idx, 'efficiency'] = ref * min_g / t / g * 100

    for version in sorted(eff_stats['version'].unique()):
        vdf = eff_stats[eff_stats['version'] == version].sort_values('num_gpus')
        ax.plot(vdf['num_gpus'], vdf['efficiency'],
                color=PRUNED_COLORS.get(version, '#999'),
                marker=PRUNED_MARKERS.get(version, 'o'),
                markersize=4, linewidth=1.2,
                label=PRUNED_LABELS.get(version, version))

    ax.axhline(100, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 110)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('(b) Efficiency', loc='left', fontweight='medium')
    if not eff_stats.empty:
        gpus = sorted(eff_stats['num_gpus'].unique())
        ax.set_xticks(gpus)
        ax.set_xticklabels([str(g) for g in gpus])

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_pruned(paths: dict, out_dir: Path, figs_requested: set):
    """Fig 8 – pruned vs full FFT strong scaling."""
    if 8 not in figs_requested:
        return
    p = paths.get('landau_alps')
    df = load_landau_data(Path(p) if p else None)
    if df is None:
        print(f'  [skip fig 8] no landau data: {p}')
        return
    # Attach dummy source_file col if absent
    if 'source_file' not in df.columns:
        df['source_file'] = df['version'].astype(str) + '_' + df['num_gpus'].astype(str)
    stats = compute_landau_scaling_stats(df)
    print('  fig 8: pruned vs full FFT strong scaling')
    plot_landau_strong_scaling(stats, out_dir, 'fig8_pruned_scaling')


# ===========================================================================
# Figure 9 – PIF solver strong scaling (Alps + JUWELS Booster + LUMI)
# ===========================================================================

def _parse_timing_dat(filepath: Path) -> dict | None:
    """Parse a timing.dat file (from the PIF solver benchmark)."""
    result = {'ranks': None, 'wall_tot': None, 'timers': {}}
    try:
        lines = filepath.read_text().splitlines()
    except Exception:
        return None

    in_detail = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('='):
            continue
        if 'Wall max' in line and 'Wall min' in line:
            in_detail = True
            continue

        m = re.match(r'^(\S+?)\.{2,}\s+(.+)$', line)
        if not m:
            m = re.match(r'^(\S+)\s+(\d.*)$', line)
        if not m:
            continue

        name  = m.group(1)
        parts = m.group(2).split()
        if len(parts) < 2:
            continue
        try:
            if not in_detail:
                if len(parts) >= 2:
                    r, wt = int(parts[0]), float(parts[1])
                    if name == 'mainTimer':
                        result['ranks']    = r
                        result['wall_tot'] = wt
            else:
                if len(parts) >= 4:
                    r = int(parts[0])
                    result['timers'][name] = {
                        'max': float(parts[1]),
                        'min': float(parts[2]),
                        'avg': float(parts[3]),
                    }
                    if name == 'mainTimer':
                        result['ranks'] = r
        except (ValueError, IndexError):
            continue

    return result if result['ranks'] is not None else None


def load_pif_data(base_dir: Path) -> dict:
    """
    Load solver_comparison_paper_scalings/ → nested dict
    {cluster: {config: {n_nodes: timing_data}}}
    """
    data = defaultdict(lambda: defaultdict(dict))
    if base_dir is None or not base_dir.exists():
        return data

    for cluster_dir in base_dir.iterdir():
        if not cluster_dir.is_dir():
            continue
        cluster = cluster_dir.name.lower()
        for config_dir in cluster_dir.iterdir():
            if not config_dir.is_dir():
                continue
            config = config_dir.name
            for nodes_dir in config_dir.iterdir():
                if not nodes_dir.is_dir():
                    continue
                m = re.search(r'nodes_(\d+)', nodes_dir.name)
                if not m:
                    continue
                n_nodes = int(m.group(1))
                tf = nodes_dir / 'timing.dat'
                if tf.exists():
                    td = _parse_timing_dat(tf)
                    if td:
                        data[cluster][config][n_nodes] = td
    return data


def _tol_from_config(cfg: str) -> str:
    if '1em4' in cfg:
        return '1e-4'
    if '1em7' in cfg:
        return '1e-7'
    return 'unknown'


def plot_pif_scaling(all_data: dict, out_dir: Path, stem: str, grid_filter='512'):
    """
    Two-panel strong-scaling comparison across clusters.
    reviewer: show LUMI alongside Alps and JUWELS Booster; add "Full FFT" to caption.
    """
    setup_style()

    tolerances = ['1e-4', '1e-7']
    tol_keys   = {'1e-4': '1em4', '1e-7': '1em7'}

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for ax, tol in zip(axes, tolerances):
        tk = tol_keys[tol]

        for cluster in sorted(all_data.keys()):
            cfg_match = next(
                (c for c in all_data[cluster]
                 if tk in c and c.startswith(grid_filter)),
                None
            )
            if not cfg_match:
                continue

            cfg_data = all_data[cluster][cfg_match]
            nodes    = sorted(cfg_data.keys())
            times    = [cfg_data[n]['wall_tot'] for n in nodes]

            ax.plot(nodes, times,
                    color=CLUSTER_COLORS.get(cluster, '#999'),
                    marker=CLUSTER_MARKERS.get(cluster, 'o'),
                    markersize=4, linewidth=1,
                    label=CLUSTER_LABELS.get(cluster, cluster))

        # Ideal scaling reference
        all_n, all_t = [], []
        for cluster in all_data:
            for c in all_data[cluster]:
                if tk in c and c.startswith(grid_filter):
                    for n, td in all_data[cluster][c].items():
                        all_n.append(n)
                        all_t.append(td['wall_tot'])

        if all_n:
            min_n = min(all_n)
            ref_times = [all_t[i] for i, n in enumerate(all_n) if n == min_n]
            ref_t = np.mean(ref_times)
            ideal_nodes = sorted(set(all_n))
            ax.plot(ideal_nodes, [ref_t * min_n / n for n in ideal_nodes],
                    'k:', linewidth=0.8, alpha=0.5, label='Ideal')
            ax.set_xticks(ideal_nodes)
            ax.set_xticklabels([str(n) for n in ideal_nodes], rotation=45)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Wall Time (s)')
        ax.set_title(f'ε = {tol}', fontsize=9)
        ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    save(fig, out_dir, stem)


def fig_pif(paths: dict, out_dir: Path, figs_requested: set):
    """Fig 9."""
    if 9 not in figs_requested:
        return
    p = paths.get('pif_data')
    all_data = load_pif_data(Path(p) if p else None)
    if not all_data:
        print(f'  [skip fig 9] no PIF solver data at: {p}')
        return
    print('  fig 9: PIF solver scaling')
    plot_pif_scaling(all_data, out_dir, 'fig9_pif_scaling')


# ===========================================================================
# Argument parsing and main
# ===========================================================================

def build_paths(args: argparse.Namespace, results_dir: Path) -> dict:
    def resolve(attr, default_rel):
        val = getattr(args, attr, None)
        if val:
            return str(val)
        return str(results_dir / default_rel)

    return {
        # Fig 1 / 2 – kernel throughput
        'kernels_alps_rho1':   resolve('kernels_alps_rho1',
            'single_node/kernels/kernels_128_alps_rho1/benchmark_throughput.csv'),
        'kernels_alps_rho10':  resolve('kernels_alps_rho10',
            'single_node/kernels/kernels_128_alps/benchmark_throughput.csv'),
        'kernels_lumi_rho1':   resolve('kernels_lumi_rho1',
            'single_node/kernels/kernels_lumi/benchmark_throughput.csv'),
        'kernels_lumi_rho10':  resolve('kernels_lumi_rho10',
            'single_node/kernels/kernels_lumi_10/benchmark_throughput.csv'),
        # Fig 3 – heatmap
        'heatmap_tiled_rho1':   resolve('heatmap_tiled_rho1',
            'single_node/heatmap/tile_sweep_rho1/tile_sweep_heatmap_Tiled.csv'),
        'heatmap_tiled_rho10':  resolve('heatmap_tiled_rho10',
            'single_node/heatmap/tile_sweep_rho10/tile_sweep_heatmap_Tiled.csv'),
        'heatmap_gp_rho1':      resolve('heatmap_gp_rho1',
            'single_node/heatmap/tile_sweep_rho1/tile_sweep_heatmap_OutputFocused.csv'),
        'heatmap_gp_rho10':     resolve('heatmap_gp_rho10',
            'single_node/heatmap/tile_sweep_rho10/tile_sweep_heatmap_OutputFocused.csv'),
        # Fig 4 – breakdown
        'breakdown': resolve('breakdown',
            'single_node/breakdown/breakdown_results/breakdown_combined.csv'),
        # Fig 5 – densities
        'densities': resolve('densities',
            'single_node/densities/density_sweep_results.csv'),
        # Fig 6 / 7 – distributed scaling
        'scaling_alps':   resolve('scaling_alps',
            'scaling/benchmark_results_alps'),
        'scaling_juwels': resolve('scaling_juwels',
            'scaling/benchmark_results_juwels'),
        'scaling_lumi':   resolve('scaling_lumi',
            'scaling/benchmark_results_lumi'),
        # Fig 8 – pruned vs full (Landau)
        'landau_alps': resolve('landau_alps',
            'landau/new/alps'),
        # Fig 9 – PIF solver
        'pif_data': resolve('pif_data',
            'PIF/solver_comparison_paper_scalings'),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--output-dir', '-o', type=Path, default=Path('paper_figures'),
                   help='Directory for output figures (default: ./paper_figures)')
    p.add_argument('--results-dir', type=Path,
                   default=Path(__file__).parent,
                   help='Root of results/ directory (default: same dir as this script)')
    p.add_argument('--figs', nargs='+', type=int, metavar='N',
                   help='Which figures to generate (1-9). Default: all.')

    # Per-figure data path overrides
    p.add_argument('--kernels-alps-rho1',  type=Path)
    p.add_argument('--kernels-alps-rho10', type=Path)
    p.add_argument('--kernels-lumi-rho1',  type=Path)
    p.add_argument('--kernels-lumi-rho10', type=Path)
    p.add_argument('--heatmap-tiled-rho1',  type=Path)
    p.add_argument('--heatmap-tiled-rho10', type=Path)
    p.add_argument('--heatmap-gp-rho1',     type=Path)
    p.add_argument('--heatmap-gp-rho10',    type=Path)
    p.add_argument('--breakdown', type=Path)
    p.add_argument('--densities', type=Path)
    p.add_argument('--scaling-alps',   type=Path)
    p.add_argument('--scaling-juwels', type=Path)
    p.add_argument('--scaling-lumi',   type=Path,
                   help='Path to benchmark_results_lumi/ (optional; adds LUMI to Fig 6/7)')
    p.add_argument('--landau-alps', type=Path)
    p.add_argument('--pif-data',    type=Path,
                   help='Path to solver_comparison_paper_scalings/ for Fig 9')
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    figs_requested = set(args.figs) if args.figs else set(range(1, 10))

    paths = build_paths(args, args.results_dir)

    print(f'Output dir: {out_dir}')
    print(f'Figures:    {sorted(figs_requested)}\n')

    setup_style()

    fig_kernels(paths, out_dir, figs_requested)
    fig_heatmap(paths, out_dir, figs_requested)
    fig_breakdown(paths, out_dir, figs_requested)
    fig_densities(paths, out_dir, figs_requested)
    fig_scaling(paths, out_dir, figs_requested)
    fig_component_breakdown(paths, out_dir, figs_requested)
    fig_pruned(paths, out_dir, figs_requested)
    fig_pif(paths, out_dir, figs_requested)

    print('\n[Done] All requested figures generated.')


if __name__ == '__main__':
    main()
