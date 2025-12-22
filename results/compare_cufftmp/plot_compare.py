import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Update these paths to your actual folder locations
cufftmp_folder = Path('cufft_results')
heffte_folder = Path('heffte_results')

def load_csvs_from_folder(folder):
    """Load all CSV files from a folder and concatenate them."""
    dfs = []
    for csv_file in folder.glob('*.csv'):
        df_temp = pd.read_csv(csv_file)
        dfs.append(df_temp)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"No CSV files found in {folder}")

df_cufftmp = load_csvs_from_folder(cufftmp_folder)
df_heffte = load_csvs_from_folder(heffte_folder)

# Filter to only forward direction and full method
df_cufftmp = df_cufftmp[(df_cufftmp['direction'] == 'forward') & (df_cufftmp['method'] == 'full')]
df_heffte = df_heffte[(df_heffte['direction'] == 'forward') & (df_heffte['method'] == 'full')]

# Get unique GPU counts from both datasets
all_gpu_counts = set(df_cufftmp['num_gpus'].unique()) | set(df_heffte['num_gpus'].unique())
gpu_counts = sorted(all_gpu_counts)

print(f"Found GPU counts: {gpu_counts}")
print(f"cuFFTMp records: {len(df_cufftmp)}, heFFTe records: {len(df_heffte)}")

# Set up professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'figure.dpi': 150,
})

# Colors for the two libraries
colors = {
    'cuFFTMp': '#1f77b4',   # Blue
    'heFFTe': '#ff7f0e',    # Orange
}

# Compute statistics for each GPU count
def compute_stats(df, gpu_counts):
    means = []
    stds = []
    mins = []
    maxs = []
    for num_gpu in gpu_counts:
        times = df[df['num_gpus'] == num_gpu]['time_ms'].values
        if len(times) > 0:
            means.append(np.mean(times))
            stds.append(np.std(times))
            mins.append(np.min(times))
            maxs.append(np.max(times))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    return np.array(means), np.array(stds), np.array(mins), np.array(maxs)

cufftmp_means, cufftmp_stds, cufftmp_mins, cufftmp_maxs = compute_stats(df_cufftmp, gpu_counts)
heffte_means, heffte_stds, heffte_mins, heffte_maxs = compute_stats(df_heffte, gpu_counts)

# Create figure
fig, ax = plt.subplots(figsize=(5, 3.5))

x = np.arange(len(gpu_counts))

# Plot cuFFTMp
ax.plot(x, cufftmp_means, 'o-', color=colors['cuFFTMp'], label='cuFFTMp',
        linewidth=1.5, markersize=6, markeredgecolor='white', markeredgewidth=0.8)
ax.fill_between(x, cufftmp_means - cufftmp_stds, cufftmp_means + cufftmp_stds,
                color=colors['cuFFTMp'], alpha=0.2)
# Also show min-max as thinner band
ax.fill_between(x, cufftmp_mins, cufftmp_maxs, color=colors['cuFFTMp'], alpha=0.1)

# Plot heFFTe
ax.plot(x, heffte_means, 's-', color=colors['heFFTe'], label='heFFTe',
        linewidth=1.5, markersize=6, markeredgecolor='white', markeredgewidth=0.8)
ax.fill_between(x, heffte_means - heffte_stds, heffte_means + heffte_stds,
                color=colors['heFFTe'], alpha=0.2)
# Also show min-max as thinner band
ax.fill_between(x, heffte_mins, heffte_maxs, color=colors['heFFTe'], alpha=0.1)

# Customize axes
ax.set_xticks(x)
ax.set_xticklabels(gpu_counts)
ax.set_xlabel('# GPUs')
ax.set_ylabel('Time (ms)')
ax.set_title('FFT $1024^3$ Alps', pad=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Auto log scale if needed
all_vals = np.concatenate([cufftmp_means[~np.isnan(cufftmp_means)],
                           heffte_means[~np.isnan(heffte_means)]])
# if len(all_vals) > 0 and all_vals.max() / all_vals.min() > 5:
#     ax.set_yscale('log')

ax.legend(loc='best')

plt.tight_layout()
plt.savefig('fft_comparison_lines.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fft_comparison_lines.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Plots saved to fft_comparison_lines.png and fft_comparison_lines.pdf")