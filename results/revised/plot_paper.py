#!/usr/bin/env python3
"""
Generate the 12 figures used in the PASC '26 paper.

Output (in ./plots):
  Fig 1  kernel_throughput_alps_rho10
  Fig 2  kernel_throughput_lumi_rho10
  Fig 3  density_sweep_alps
  Fig 4  nufft_breakdown_alps
  Fig 5  pruned_fft_scaling_all_conc
  Fig 6  juwels_fft_scaling
  Fig 7  comparison_scaling_combined_512_256
  Fig 8  nufft_scaling_components_alps
  Fig 9  comparison_scaling_combined_1024
  Fig 10 pif_scaling_combined_1024
  Fig 11 pif_tol_overlay_512_full
  Fig 12 nufft_pif_components_lumi_512_full_tol1e8

ACM-compliant: pdf.fonttype=42 (TrueType, no Type 3); stacked bars use hatch
patterns and line plots use distinct markers / linestyles for grayscale
readability.

Usage:
  python plot_paper.py [--output-dir plots]
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
# Paths and constants
# ════════════════════════════════════════════════════════════════════════

BASE = Path(__file__).resolve().parent
CLUSTERS = ['alps', 'juwels', 'lumi']
GPUS_PER_NODE = {'alps': 4, 'juwels': 4, 'lumi': 8}

FFT_DIR = BASE.parent / 'fft'
FFT_GRID_DIRS = {
    1024: FFT_DIR / 'benchmark_results_20251209_1024',
    2048: FFT_DIR / 'benchmark_results_20251209_2048',
}

# Older single-shot Alps NUFFT components benchmark — used by Fig 8 because
# it includes the accumulateHaloNUFFT1 / FillHaloNUFFT2 halo-exchange timers
# that the newer upsampled benchmark does not export.
ALPS_COMPONENTS_DIR = BASE.parent / 'scaling' / 'benchmark_results_alps'

# ════════════════════════════════════════════════════════════════════════
# Style — ACM-compliant
# ════════════════════════════════════════════════════════════════════════

def setup_style():
    plt.rcParams.update({
        # ACM: TrueType (Type 42) so the embedded fonts show as TrueType in
        # pdffonts and we never produce Type 3 bitmap fonts.
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
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

# Colors are Okabe-Ito (colorblind-safe). Markers/linestyles/hatches make the
# plots readable when printed in grayscale.

COLORS_T1 = {'Atomic': '#E69F00', 'Tiled': '#009E73',
             'Grid-Parallel': '#0072B2', 'cuFINUFFT': '#D55E00'}
MARKERS_T1 = {'Atomic': 's', 'Tiled': '^',
              'Grid-Parallel': 'o', 'cuFINUFFT': 'X'}
T1_ORDER = ['Atomic', 'Tiled', 'Grid-Parallel', 'cuFINUFFT']

COLORS_T2 = {'Direct': '#E69F00', 'Sorted': '#56B4E9', 'cuFINUFFT': '#D55E00'}
MARKERS_T2 = {'Direct': 's', 'Sorted': 'D', 'cuFINUFFT': 'X'}
T2_ORDER = ['Direct', 'Sorted', 'cuFINUFFT']

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

COMP_HATCHES = {
    'Spreading':     '',
    'Interpolation': '',
    'FFT':           '///',
    'Halo':          'xxx',
    'Deconvolution': '...',
    'Precorrection': '...',
}

CLUSTER_COLORS = {'alps': '#2166AC', 'juwels': '#D6604D', 'lumi': '#4DAF4A'}
CLUSTER_MARKERS = {'alps': 'o', 'juwels': 's', 'lumi': '^'}
CLUSTER_LABELS = {
    'alps':   'Alps (GH200)',
    'juwels': 'JUWELS Booster (A100)',
    'lumi':   'LUMI (MI250X)',
}

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


def _tol_label(tol):
    if tol is None:
        return r'$\varepsilon=10^{-4}$'
    return r'$\varepsilon=10^{-' + tol.lstrip('1e') + r'}$'


def _tol_suffix(tol):
    return '' if tol is None else f'_tol{tol}'


# ── path lookups ────────────────────────────────────────────────────────

def _find_landau_csv(cluster, grid_key, nodes, pruned=False, tol=None):
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    prefix = 'LandauDampingPIFPruned' if pruned else 'LandauDampingPIF'
    folder = f'strong_landau_{grid_key}'
    if tol is not None:
        base_dir = BASE / f'tol_{tol}' / cluster / folder / f'grid_{grid_size}' / f'nodes_{nodes}'
    else:
        base_dir = BASE / cluster / folder / f'grid_{grid_size}' / f'nodes_{nodes}'
    path = base_dir / f'{prefix}{gpus}.csv'
    return path if path.exists() else None


def _find_upsampled_csv(cluster, grid_key, nodes, kind):
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
    grid_size = 512 if grid_key == 'current' else 1024
    gpus = nodes * GPUS_PER_NODE[cluster]
    base_dir = BASE / cluster / f'strong_nufft_{grid_key}' / f'grid_{grid_size}' / f'nodes_{nodes}'
    pattern = f'nufft_{kind}_{gpus}.csv'
    for p in [base_dir / pattern, base_dir / 'results' / pattern]:
        if p.exists():
            return p
    return None


# ── data loaders ────────────────────────────────────────────────────────

def _load_scaling_data(grid_key, upsampled=False):
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


def _extract_fft_times(grid_key, upsampled=False):
    """Per-cluster DataFrame with fft1 / fft2 mean/min/max in ms."""
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


def _load_landau_timings(path):
    """Per-step Scatter / Gather / updateParticle timings from a Landau CSV.

    The PIF benchmark emits two timer entries per step: a real measurement and
    a tiny one (<1ms) that doesn't bracket actual work. Filter on a 10% of
    median threshold to drop the tiny ones.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    timers = ['ScatterPIFNUFFT', 'GatherPIFNUFFT', 'updateParticle']
    result = {}
    for timer in timers:
        tdf = df[df['timer_name'] == timer]
        if tdf.empty:
            continue
        tdf = tdf[tdf['rank'] == 0].copy()
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
        rows.append({'step': i, 'scatter_s': s, 'gather_s': g,
                     'update_s': u, 'total_s': s + g + u})
    return pd.DataFrame(rows)


def _load_landau_scaling_robust(grid_key, pruned=False, tol=None):
    """Per-cluster median+trimmed min/max of total per-step time vs GPUs."""
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
                tdf = _load_landau_timings(path)
            except Exception as e:
                print(f'  [warn] {path}: {e}')
                continue
            if tdf is None or tdf.empty:
                continue
            if len(tdf) > 5:
                tdf = tdf.iloc[3:]
            totals = np.sort(tdf['total_s'].values)
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


def _load_nufft_from_pif(grid_key, pruned=False, tol=None):
    """Per-cluster DataFrame with NativeNUFFT1/2 totals + per-component means."""
    data = {}
    warmup_steps = 3
    timers_of_interest = [
        'NativeNUFFT1', 'NativeNUFFT2',
        'scatterTimerNUFFT1', 'FFTNUFFT1', 'deconvolutionNUFFT1',
        'GatherNUFFT2', 'FFTNUFFT2', 'PrecorrectionNUFFT2',
    ]
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
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()
                timers = {}
                for t in timers_of_interest:
                    tdf = df[(df['timer_name'] == t) & (df['rank'] == 0)]
                    if not tdf.empty:
                        timers[t] = tdf['duration_seconds'].values
            except Exception as e:
                print(f'  [warn] {path}: {e}')
                continue
            if not timers:
                continue
            for k in timers:
                if len(timers[k]) > warmup_steps + 2:
                    timers[k] = timers[k][warmup_steps:]
            row = {'gpus': gpus}
            for label, timer in [('t1', 'NativeNUFFT1'), ('t2', 'NativeNUFFT2')]:
                vals = timers.get(timer)
                if vals is not None and len(vals) > 0:
                    row[f'{label}_mean_ms'] = np.mean(vals) * 1000
                    row[f'{label}_min_ms'] = np.min(vals) * 1000
                    row[f'{label}_max_ms'] = np.max(vals) * 1000
                    row[f'{label}_median_ms'] = np.median(vals) * 1000
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
                row[col] = np.mean(vals) * 1000 if vals is not None and len(vals) > 0 else 0.0
            rows.append(row)
        if rows:
            data[cluster] = pd.DataFrame(rows).sort_values('gpus').reset_index(drop=True)
    return data


# ════════════════════════════════════════════════════════════════════════
# Figure 1, 2: kernel throughput vs tolerance
# ════════════════════════════════════════════════════════════════════════

def fig_kernel_throughput(output_dir, clusters):
    setup_style()
    print('\n=== Figs 1, 2: kernel throughput vs tolerance ===')
    for cluster in clusters:
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
        ax.set_title('Type‑1 (spreading)', fontsize=9, pad=6)
        ax.invert_xaxis()
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc='best')

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
        ax.set_title('Type‑2 (interpolation)', fontsize=9, pad=6)
        ax.invert_xaxis()
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(loc='best')

        plt.tight_layout(w_pad=3.0)
        _save(fig, os.path.join(output_dir, f'kernel_throughput_{cluster}_rho10'))


# ════════════════════════════════════════════════════════════════════════
# Figure 3: density sweep
# ════════════════════════════════════════════════════════════════════════

def fig_density_sweep(output_dir, clusters):
    setup_style()
    print('\n=== Fig 3: density sweep ===')
    for cluster in clusters:
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
            (axes[0], t1, T1_ORDER, COLORS_T1, MARKERS_T1, 'Type‑1 (spreading)'),
            (axes[1], t2, T2_ORDER, COLORS_T2, MARKERS_T2, 'Type‑2 (interpolation)'),
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
# Figure 4: NUFFT breakdown (single node)
# ════════════════════════════════════════════════════════════════════════

def fig_nufft_breakdown(output_dir, clusters):
    setup_style()
    print('\n=== Fig 4: NUFFT breakdown ===')
    for cluster in clusters:
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
            (axes[0], t1, COMP_ORDER_T1, 'Type‑1 (spreading)'),
            (axes[1], t2, COMP_ORDER_T2, 'Type‑2 (interpolation)'),
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
                           hatch=COMP_HATCHES.get(comp, ''),
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
# Figure 5: pruned vs full FFT scaling, all concurrencies (Alps)
# ════════════════════════════════════════════════════════════════════════

def fig_pruned_fft_all_conc(output_dir):
    setup_style()
    print('\n=== Fig 5: pruned vs full FFT (all concurrencies) ===')

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
        # Lighter shade for the smaller upsampled grid, darker for the bigger one
        _grid_method_colors = {
            (min(grid_sizes), 'full'): '#999999', (min(grid_sizes), 'pruned'): '#F0A050',
            (max(grid_sizes), 'full'): '#333333', (max(grid_sizes), 'pruned'): '#D55E00',
        }
        for gi, gs in enumerate(grid_sizes):
            orig_gs = gs // 2
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
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'pruned_fft_scaling_all_conc'))


# ════════════════════════════════════════════════════════════════════════
# Figure 6: JUWELS pruned vs full FFT scaling
# ════════════════════════════════════════════════════════════════════════

def fig_juwels_fft_scaling(output_dir):
    setup_style()
    print('\n=== Fig 6: JUWELS FFT pruned vs full scaling ===')

    method_labels = {'full': 'Full FFT', 'pruned': 'Pruned FFT'}
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

    grid_keys = [('current', 512), ('1024', 1024)]
    _grid_method_colors = {
        (512,  'full'): '#999999', (512,  'pruned'): '#F0A050',
        (1024, 'full'): '#333333', (1024, 'pruned'): '#D55E00',
    }
    has_data = False

    for grid_key, grid_size in grid_keys:
        pruned = _extract_fft_times(grid_key, upsampled=False)
        full = _extract_fft_times(grid_key, upsampled=True)
        ls = '-' if grid_size == 1024 else '--'

        if 'juwels' in pruned:
            pdf = pruned['juwels']
            if 'fft1_mean_ms' in pdf.columns:
                cdf = pdf.dropna(subset=['fft1_mean_ms'])
                if not cdf.empty:
                    has_data = True
                    color = _grid_method_colors[(grid_size, 'pruned')]
                    ax.plot(cdf['gpus'], cdf['fft1_mean_ms'],
                            marker='o', color=color, linestyle=ls,
                            label=f'{method_labels["pruned"]} ({grid_size}$^3$)',
                            markeredgecolor='black', markeredgewidth=0.3)
                    ax.fill_between(cdf['gpus'], cdf['fft1_min_ms'], cdf['fft1_max_ms'],
                                    color=color, alpha=0.1)

        if 'juwels' in full:
            fdf = full['juwels']
            if 'fft1_mean_ms' in fdf.columns:
                cdf = fdf.dropna(subset=['fft1_mean_ms'])
                if not cdf.empty:
                    has_data = True
                    color = _grid_method_colors[(grid_size, 'full')]
                    ax.plot(cdf['gpus'], cdf['fft1_mean_ms'],
                            marker='s', color=color, linestyle=ls,
                            label=f'{method_labels["full"]} ({grid_size}$^3$)',
                            markeredgecolor='black', markeredgewidth=0.3)
                    ax.fill_between(cdf['gpus'], cdf['fft1_min_ms'], cdf['fft1_max_ms'],
                                    color=color, alpha=0.1)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Time (ms)')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which='major', linewidth=0.3, alpha=0.5)
    ax.legend(loc='best', fontsize=6)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'juwels_fft_scaling'))


# ════════════════════════════════════════════════════════════════════════
# Figures 7, 9: combined pruned vs upsampled scaling, all clusters
# ════════════════════════════════════════════════════════════════════════

def fig_comparison_scaling_combined(output_dir, grid_key='current', max_gpus=None):
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    suffix = f'_{max_gpus}' if max_gpus else ''
    print(f'\n=== Fig {7 if max_gpus else 9}: combined scaling (grid {grid_label}{suffix}) ===')

    data_pruned = _load_scaling_data(grid_key, upsampled=False)
    data_upsampled = _load_scaling_data(grid_key, upsampled=True)
    if not data_pruned and not data_upsampled:
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    for ax, ttype, title in [
        (axes[0], 'type1', 'Type‑1 (spreading)'),
        (axes[1], 'type2', 'Type‑2 (interpolation)'),
    ]:
        for cluster in CLUSTERS:
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

    plt.tight_layout()
    _save(fig, os.path.join(output_dir,
                            f'comparison_scaling_combined_{grid_label}{suffix}'))


# ════════════════════════════════════════════════════════════════════════
# Figure 8: NUFFT components on Alps at 512^3 (full-FFT benchmark)
#
# Uses the older results/scaling/benchmark_results_alps/ run because it
# exports the halo-exchange timers (accumulateHaloNUFFT1, FillHaloNUFFT2)
# that the newer upsampled benchmark dropped.
# ════════════════════════════════════════════════════════════════════════

# Per-type timer name -> stack-order component label
_FIG8_TIMER_T1 = {
    'scatterTimerNUFFT1':   'Spreading',
    'accumulateHaloNUFFT1': 'Halo',
    'FFTNUFFT1':            'FFT',
    'deconvolutionNUFFT1':  'Deconvolution',
}
_FIG8_TIMER_T2 = {
    'GatherNUFFT2':         'Interpolation',
    'FillHaloNUFFT2':       'Halo',
    'FFTNUFFT2':            'FFT',
    'PrecorrectionNUFFT2':  'Precorrection',
}


def fig_nufft_scaling_components(output_dir, clusters):
    setup_style()
    print('\n=== Fig 8: full-FFT NUFFT components ===')

    if 'alps' not in clusters:
        return
    if not ALPS_COMPONENTS_DIR.exists():
        print(f'  [skip] {ALPS_COMPONENTS_DIR} missing')
        return

    frames = []
    for path in sorted(ALPS_COMPONENTS_DIR.glob('nufft_components_*.csv')):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    # Skip warm-up: the first run is the one with multi-second halo outliers.
    df = df[df['run'] > 0].copy()
    df['time_ms'] = df['time_s'] * 1000

    # Cap at 256 GPUs to match the original Fig 8 layout.
    df = df[df['num_ranks'] <= 256]
    gpu_counts = sorted(df['num_ranks'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    x = np.arange(len(gpu_counts))
    bw = 0.65

    for ax, timer_map, comp_order, title in [
        (axes[0], _FIG8_TIMER_T1, COMP_ORDER_T1, 'Type‑1 (spreading)'),
        (axes[1], _FIG8_TIMER_T2, COMP_ORDER_T2, 'Type‑2 (interpolation)'),
    ]:
        for gi, gpus in enumerate(gpu_counts):
            gdf = df[df['num_ranks'] == gpus]
            bottom = 0.0
            for comp in comp_order:
                timer_names = [k for k, v in timer_map.items() if v == comp]
                total = 0.0
                for tn in timer_names:
                    tdf = gdf[gdf['timer'] == tn]
                    if not tdf.empty:
                        total += tdf['time_ms'].mean()
                if total == 0:
                    continue
                lbl = comp if gi == 0 else None
                ax.bar(x[gi], total, bw, bottom=bottom,
                       color=COMP_COLORS.get(comp, '#999'),
                       hatch=COMP_HATCHES.get(comp, ''),
                       edgecolor='black', linewidth=0.3, label=lbl)
                bottom += total

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=7)
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(title, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, 'nufft_scaling_components_alps'))


# ════════════════════════════════════════════════════════════════════════
# Figure 10: PIF combined pruned vs full at 1024^3
# ════════════════════════════════════════════════════════════════════════

def fig_pif_scaling_combined(output_dir, grid_key='1024', tol=None):
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    print(f'\n=== Fig 10: PIF combined (grid {grid_label}, {_tol_label(tol)}) ===')

    data_pruned = _load_landau_scaling_robust(grid_key, pruned=True, tol=tol)
    data_full = _load_landau_scaling_robust(grid_key, pruned=False, tol=tol)
    if not data_pruned and not data_full:
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))

    all_gpus = set()
    for cluster in CLUSTERS:
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
    _save(fig, os.path.join(output_dir,
                            f'pif_scaling_combined_{grid_label}{_tol_suffix(tol)}'))


# ════════════════════════════════════════════════════════════════════════
# Figure 11: PIF tolerance overlay (1e-4 solid vs 1e-8 dashed) — full FFT
# ════════════════════════════════════════════════════════════════════════

def fig_pif_tol_overlay(output_dir, grid_key='current', pruned=False):
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    fft_label = 'pruned' if pruned else 'full'
    print(f'\n=== Fig 11: PIF tolerance overlay (grid {grid_label}, {fft_label}) ===')

    data_1e4 = _load_landau_scaling_robust(grid_key, pruned=pruned, tol=None)
    data_1e8 = _load_landau_scaling_robust(grid_key, pruned=pruned, tol='1e8')
    if not data_1e4 and not data_1e8:
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.8))
    all_gpus = set()
    for cluster in CLUSTERS:
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
    _save(fig, os.path.join(output_dir,
                            f'pif_tol_overlay_{grid_label}_{fft_label}'))


# ════════════════════════════════════════════════════════════════════════
# Figure 12: NUFFT-from-PIF component breakdown (LUMI 512 full tol=1e-8)
# ════════════════════════════════════════════════════════════════════════

def fig_nufft_pif_components(output_dir, grid_key='current', pruned=False,
                              tol='1e8', clusters=None):
    setup_style()
    grid_label = '512' if grid_key == 'current' else '1024'
    fft_label = 'pruned' if pruned else 'full'
    print(f'\n=== Fig 12: NUFFT-from-PIF components '
          f'(grid {grid_label}, {fft_label}, {_tol_label(tol)}) ===')

    data = _load_nufft_from_pif(grid_key, pruned=pruned, tol=tol)
    if not data:
        return
    if clusters is not None:
        data = {c: data[c] for c in clusters if c in data}

    t1_comps = [('scatter_ms', 'Spreading'), ('fft1_ms', 'FFT'),
                ('deconv_ms', 'Deconvolution')]
    t2_comps = [('gather_ms', 'Interpolation'), ('fft2_ms', 'FFT'),
                ('precorr_ms', 'Precorrection')]

    for cluster, df in data.items():
        gpu_counts = df['gpus'].values
        if len(gpu_counts) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
        x = np.arange(len(gpu_counts))
        bw = 0.7

        for ax, comps, title in [
            (axes[0], t1_comps, 'Type‑1 (spreading)'),
            (axes[1], t2_comps, 'Type‑2 (interpolation)'),
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
                           hatch=COMP_HATCHES.get(label, ''),
                           edgecolor='black', linewidth=0.3, label=lbl)
                    bottom += v

            ax.set_xticks(x)
            ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=6)
            ax.set_xlabel('Number of GPUs')
            ax.set_ylabel('Time (ms)')
            ax.set_title(title, fontsize=9)
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=6)

        plt.tight_layout()
        _save(fig, os.path.join(
            output_dir,
            f'nufft_pif_components_{cluster}_{grid_label}_{fft_label}{_tol_suffix(tol)}'))


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generate the 12 figures used in the PASC \'26 paper.')
    parser.add_argument('--output-dir', '-o', default='plots',
                        help='Output directory (default: plots)')
    args = parser.parse_args()

    output_dir = str(BASE / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output directory: {output_dir}')

    fig_kernel_throughput(output_dir, clusters=['alps', 'lumi'])     # Figs 1, 2
    fig_density_sweep(output_dir, clusters=['alps'])                 # Fig 3
    fig_nufft_breakdown(output_dir, clusters=['alps'])               # Fig 4
    fig_pruned_fft_all_conc(output_dir)                              # Fig 5
    fig_juwels_fft_scaling(output_dir)                               # Fig 6
    fig_comparison_scaling_combined(output_dir, 'current', max_gpus=256)  # Fig 7
    fig_nufft_scaling_components(output_dir, clusters=['alps'])      # Fig 8
    fig_comparison_scaling_combined(output_dir, '1024')              # Fig 9
    fig_pif_scaling_combined(output_dir, '1024')                     # Fig 10
    fig_pif_tol_overlay(output_dir, 'current', pruned=False)         # Fig 11
    fig_nufft_pif_components(output_dir, 'current', pruned=False,
                              tol='1e8', clusters=['lumi'])          # Fig 12

    print('\nDone.')


if __name__ == '__main__':
    main()
