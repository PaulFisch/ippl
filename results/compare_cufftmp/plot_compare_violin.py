import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

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

# Set up compact professional style
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

# Create figure
fig, ax = plt.subplots(figsize=(5.5, 4))

violin_data = []
positions = []
violin_colors = []
scatter_data = []  # Store (position, times, color) for scatter overlay

for i, num_gpu in enumerate(gpu_counts):
    # cuFFTMp data
    cufftmp_times = df_cufftmp[df_cufftmp['num_gpus'] == num_gpu]['time_ms'].values
    if len(cufftmp_times) > 0:
        pos = i - 0.2
        violin_data.append(cufftmp_times)
        positions.append(pos)
        violin_colors.append(colors['cuFFTMp'])
        scatter_data.append((pos, cufftmp_times, colors['cuFFTMp']))

    # heFFTe data
    heffte_times = df_heffte[df_heffte['num_gpus'] == num_gpu]['time_ms'].values
    if len(heffte_times) > 0:
        pos = i + 0.2
        violin_data.append(heffte_times)
        positions.append(pos)
        violin_colors.append(colors['heFFTe'])
        scatter_data.append((pos, heffte_times, colors['heFFTe']))

# Create violin plot
if violin_data:
    vp = ax.violinplot(violin_data, positions=positions, widths=0.35,
                       showmeans=False, showmedians=False, showextrema=False)

    for j, body in enumerate(vp['bodies']):
        body.set_facecolor(violin_colors[j])
        body.set_edgecolor('black')
        body.set_linewidth(0.8)
        body.set_alpha(0.4)

# Overlay individual data points with jitter
np.random.seed(42)
for pos, times, color in scatter_data:
    jitter = np.random.uniform(-0.08, 0.08, size=len(times))
    ax.scatter(pos + jitter, times, c=color, s=25, alpha=0.8,
               edgecolors='white', linewidths=0.5, zorder=3)
    # Add mean marker
    mean_val = np.mean(times)
    ax.scatter(pos, mean_val, c='black', s=60, marker='_', linewidths=2, zorder=4)

# Add boxplot-style quartile indicators
for pos, times, color in scatter_data:
    q1, median, q3 = np.percentile(times, [25, 50, 75])
    # Draw median line
    ax.hlines(median, pos - 0.1, pos + 0.1, colors='black', linewidths=1.5, zorder=4)
    # Draw IQR box
    ax.vlines(pos, q1, q3, colors='black', linewidths=1.5, zorder=4)

# Customize axes
ax.set_xticks(range(len(gpu_counts)))
ax.set_xticklabels(gpu_counts)
ax.set_xlabel('# GPUs')
ax.set_ylabel('Time (ms)')
ax.set_title('Forward FFT Performance: cuFFTMp vs heFFTe', fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)

# Use log scale if data spans multiple orders of magnitude
if len(violin_data) > 0:
    all_times = np.concatenate(violin_data)
    if all_times.max() / all_times.min() > 5:
        ax.set_yscale('log')

# Add legend with scatter points
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['cuFFTMp'],
           markersize=8, label='cuFFTMp', markeredgecolor='white', markeredgewidth=0.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['heFFTe'],
           markersize=8, label='heFFTe', markeredgecolor='white', markeredgewidth=0.5),
]
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig('fft_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fft_comparison.pdf', bbox_inches='tight', facecolor='white')
plt.show()

print("Plots saved to fft_comparison.png and fft_comparison.pdf")