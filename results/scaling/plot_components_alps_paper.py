#!/usr/bin/env python3
"""Regenerate Alps component breakdown plot without title and with distinct Halo color."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

BASE = Path(__file__).resolve().parent

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

COMPONENT_COLORS = {
    'Scatter/Gather': '#0072B2',   # Blue
    'FFT': '#CC79A7',              # Pink
    'Halo': '#009E73',             # Teal green (distinct from blue)
    'Deconv/Precorr': '#E69F00',   # Orange
}

TIMER_CATEGORIES_T1 = {
    'scatterTimerNUFFT1': 'Scatter/Gather',
    'accumulateHaloNUFFT1': 'Halo',
    'FFTNUFFT1': 'FFT',
    'deconvolutionNUFFT1': 'Deconv/Precorr',
}

TIMER_CATEGORIES_T2 = {
    'PrecorrectionNUFFT2': 'Deconv/Precorr',
    'FFTNUFFT2': 'FFT',
    'FillHaloNUFFT2': 'Halo',
    'GatherNUFFT2': 'Scatter/Gather',
}

STACK_ORDER = ['Scatter/Gather', 'FFT', 'Deconv/Precorr', 'Halo']

COMPONENT_LABELS_T1 = {
    'Scatter/Gather': 'Spreading',
    'FFT': 'FFT',
    'Halo': 'Halo',
    'Deconv/Precorr': 'Deconvolution',
}

COMPONENT_LABELS_T2 = {
    'Scatter/Gather': 'Interpolation',
    'FFT': 'FFT',
    'Halo': 'Halo',
    'Deconv/Precorr': 'Precorrection',
}

TYPE_LABELS = {
    'type1': 'Type\u20111',
    'type2': 'Type\u20112',
}


def load_component_data(base_dir):
    base_path = Path(base_dir)
    all_data = []
    for filepath in sorted(base_path.glob('nufft_components_*.csv')):
        df = pd.read_csv(filepath)
        df['cluster'] = 'alps'
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def compute_component_statistics(df):
    df = df.copy()

    def categorize_timer(timer_name):
        if timer_name in TIMER_CATEGORIES_T1:
            return TIMER_CATEGORIES_T1[timer_name], 'type1'
        elif timer_name in TIMER_CATEGORIES_T2:
            return TIMER_CATEGORIES_T2[timer_name], 'type2'
        return None, None

    df['category'], df['transform_type'] = zip(*df['timer'].map(categorize_timer))
    df = df.dropna(subset=['category'])
    df = df[df['run'] > 0]
    df['time_ms'] = df['time_s'] * 1000

    grouped = df.groupby(['cluster', 'num_ranks', 'transform_type', 'category'])
    stats = grouped['time_ms'].agg(['mean', 'std']).reset_index()
    return stats


def main():
    setup_style()

    data_dir = BASE / 'benchmark_results_alps'
    output_dir = BASE / 'plots_paper'
    output_dir.mkdir(exist_ok=True)

    raw = load_component_data(data_dir)
    component_stats = compute_component_statistics(raw)
    component_stats = component_stats[component_stats['num_ranks'] <= 256]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    gpu_counts = sorted(component_stats['num_ranks'].unique())
    n_gpus = len(gpu_counts)
    x = np.arange(n_gpus)
    bar_width = 0.65

    for ax, ttype in zip(axes, ['type1', 'type2']):
        subset = component_stats[component_stats['transform_type'] == ttype]
        labels = COMPONENT_LABELS_T1 if ttype == 'type1' else COMPONENT_LABELS_T2

        for gpu_idx, gpu_count in enumerate(gpu_counts):
            gpu_data = subset[subset['num_ranks'] == gpu_count]
            bottom = 0

            for category in STACK_ORDER:
                cat_data = gpu_data[gpu_data['category'] == category]
                if not cat_data.empty:
                    height = cat_data['mean'].values[0]
                    color = COMPONENT_COLORS[category]
                    label = labels.get(category) if gpu_idx == 0 else None

                    ax.bar(x[gpu_idx], height, bar_width, bottom=bottom,
                           color=color, edgecolor='black', linewidth=0.3,
                           label=label)
                    bottom += height

            pass  # no total annotation

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gpu_counts])
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (ms)')
        ax.set_title(TYPE_LABELS[ttype], fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=6)

    # No suptitle
    plt.tight_layout()

    stem = output_dir / 'nufft_scaling_components_alps'
    fig.savefig(f'{stem}.pdf', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'Saved: {stem}.pdf and {stem}.png')


if __name__ == '__main__':
    main()
