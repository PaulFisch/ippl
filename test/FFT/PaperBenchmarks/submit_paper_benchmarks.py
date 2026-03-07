#!/usr/bin/env python3
"""
submit_paper_benchmarks.py
==========================
Generates and submits SLURM batch jobs for all paper benchmarks, then a
final job that runs plot_paper.py once all data is collected.

Detects the HPC cluster automatically, or accepts --cluster to override.

Supported clusters
------------------
  alps    – CSCS Alps (GH200, 4 GPUs/node)
             uenv prgenv-gnu, Cray MPICH, mpi_wrapper.sh
  lumi    – CSC LUMI-G (MI250X = 8 GCDs/node)
             MPICH GPU-aware MPI, select_gpu wrapper, cpu-bind map
  juwels  – JSC JUWELS Booster (A100, 4 GPUs/node)
             --gres=gpu:4, standard srun, no wrapper

Jobs submitted
--------------
  1. single_node  – BenchKernels, TileSweep, BreakdownNUFFT, BenchDensities
  2. nufft_scaling – BenchmarkNUFFT strong-scaling (type-1 + type-2 + components)
  3. landau        – LandauDampingPIF + LandauDampingPIFPruned strong-scaling
  4. plot          – python plot_paper.py (depends on all three data jobs)

Usage
-----
  # Minimal – auto-detects cluster, must supply build dir:
  python submit_paper_benchmarks.py --build-dir /path/to/build/test/FFT/PaperBenchmarks

  # Full example on LUMI:
  python submit_paper_benchmarks.py \\
      --cluster lumi \\
      --account project_462000001 \\
      --build-dir /scratch/project.../build/test/FFT/PaperBenchmarks \\
      --solver-bin-dir /scratch/project.../build/test/Solver \\
      --results-dir /scratch/project.../results \\
      --plot-script /scratch/project.../results/plot_paper.py \\
      --wrapper /scratch/project.../mpi_wrapper.sh \\
      --max-gpus 256 --grid 512

  # Dry-run (print generated scripts, do not submit):
  python submit_paper_benchmarks.py --build-dir /path/to/build --dry-run
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Cluster configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClusterConfig:
    name: str
    gpus_per_node: int
    cpus_per_task: int
    omp_threads: int

    # SLURM header extras (applied unconditionally)
    extra_sbatch: list = field(default_factory=list)   # e.g. ["--uenv=..."]
    partition: Optional[str] = None
    account: Optional[str] = None

    # GPU resource directive variant
    # 'per_task'  → --gpus-per-task=1                  (Alps)
    # 'per_node'  → --gpus-per-node=N                  (LUMI)
    # 'gres'      → --gres=gpu:N                       (JUWELS)
    gpu_mode: str = 'per_task'

    # Environment exported inside the job
    env: dict = field(default_factory=dict)

    # srun prefix per step: e.g. "CPU_BIND=... $WRAPPER" etc.
    # These are shell fragments, can reference $WRAPPER / $CPU_BIND variables
    srun_prefix: str = ''

    # Shell setup snippet placed once at the top of every sbatch script
    # (create wrapper files, load modules, export env, etc.)
    preamble: str = ''


# ---------------------------------------------------------------------------
# Per-cluster defaults
# ---------------------------------------------------------------------------

def _alps_config(account: str, uenv: str, wrapper_path: str) -> ClusterConfig:
    """
    CSCS Alps – GH200 nodes, 4 GPUs per node.
    Uses mpi_wrapper.sh to set MPICH, NIC, GPU affinity.
    """
    preamble = textwrap.dedent(f"""\
        export OMP_PROC_BIND=spread
        export OMP_PLACES=threads
        export OMP_NUM_THREADS=72
        WRAPPER={wrapper_path}
    """)
    return ClusterConfig(
        name='alps',
        gpus_per_node=4,
        cpus_per_task=72,
        omp_threads=72,
        extra_sbatch=[
            f'--uenv={uenv}',
            '--view=default',
            '--exclusive',
        ],
        account=account,
        gpu_mode='per_task',
        preamble=preamble,
        srun_prefix='$WRAPPER',
    )


def _lumi_config(account: str) -> ClusterConfig:
    """
    CSC LUMI-G – MI250X nodes, 8 GCDs per node.
    Uses inline select_gpu wrapper and CPU-bind map.
    """
    preamble = textwrap.dedent("""\
        # Create per-job GPU-selector wrapper (LUMI requirement)
        cat << 'SELECT_GPU_EOF' > select_gpu
        #!/bin/bash
        export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
        exec "$@"
        SELECT_GPU_EOF
        chmod +x ./select_gpu

        export CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
        export MPICH_GPU_SUPPORT_ENABLED=1
        export OMP_NUM_THREADS=7
    """)
    return ClusterConfig(
        name='lumi',
        gpus_per_node=8,
        cpus_per_task=7,
        omp_threads=7,
        extra_sbatch=['--exclusive'],
        partition='standard-g',
        account=account,
        gpu_mode='per_node',
        preamble=preamble,
        srun_prefix='--cpu-bind=${CPU_BIND} ./select_gpu',
        env={'MPICH_GPU_SUPPORT_ENABLED': '1'},
    )


def _juwels_config(account: str) -> ClusterConfig:
    """
    JSC JUWELS Booster – A100 nodes, 4 GPUs per node.
    No special wrapper; --gres requests GPUs.
    """
    preamble = textwrap.dedent("""\
        export OMP_NUM_THREADS=12
    """)
    return ClusterConfig(
        name='juwels',
        gpus_per_node=4,
        cpus_per_task=12,
        omp_threads=12,
        extra_sbatch=['--exclusive'],
        partition='booster',
        account=account,
        gpu_mode='gres',
        preamble=preamble,
        srun_prefix='',
    )


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def detect_cluster() -> Optional[str]:
    slurm = os.environ.get('SLURM_CLUSTER_NAME', '').lower()
    host  = socket.gethostname().lower()
    lumi_marker = os.environ.get('LUMI_CLUSTER_NAME', '')
    uenv = os.environ.get('UENV_LOADED', '') or os.environ.get('UENV_MOUNT_POINT', '')

    if lumi_marker or slurm == 'lumi' or host.startswith('uan') and 'lumi' in host:
        return 'lumi'
    if slurm in ('alps', 'daint', 'clariden') or uenv:
        return 'alps'
    if slurm == 'juwels' or host.startswith('jwb') or host.startswith('jw'):
        return 'juwels'
    return None


# ---------------------------------------------------------------------------
# sbatch header builder
# ---------------------------------------------------------------------------

def sbatch_header(cfg: ClusterConfig, job_name: str, nodes: int, time_limit: str,
                  log_dir: Path) -> str:
    lines = [
        '#!/bin/bash -l',
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --output={log_dir}/{job_name}_%j.out',
        f'#SBATCH --error={log_dir}/{job_name}_%j.err',
        f'#SBATCH --nodes={nodes}',
        f'#SBATCH --time={time_limit}',
    ]
    if cfg.account:
        lines.append(f'#SBATCH --account={cfg.account}')
    if cfg.partition:
        lines.append(f'#SBATCH --partition={cfg.partition}')

    # GPU resource directive
    if cfg.gpu_mode == 'per_task':
        lines.append(f'#SBATCH --ntasks-per-node={cfg.gpus_per_node}')
        lines.append(f'#SBATCH --gpus-per-task=1')
        lines.append(f'#SBATCH --cpus-per-task={cfg.cpus_per_task}')
    elif cfg.gpu_mode == 'per_node':
        lines.append(f'#SBATCH --ntasks-per-node={cfg.gpus_per_node}')
        lines.append(f'#SBATCH --gpus-per-node={cfg.gpus_per_node}')
        lines.append(f'#SBATCH --cpus-per-task={cfg.cpus_per_task}')
    elif cfg.gpu_mode == 'gres':
        lines.append(f'#SBATCH --ntasks-per-node={cfg.gpus_per_node}')
        lines.append(f'#SBATCH --gres=gpu:{cfg.gpus_per_node}')
        lines.append(f'#SBATCH --cpus-per-task={cfg.cpus_per_task}')

    for extra in cfg.extra_sbatch:
        lines.append(f'#SBATCH {extra}')

    lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# srun helper: one step for a given GPU count
# ---------------------------------------------------------------------------

def srun_cmd(cfg: ClusterConfig, n_gpus: int, exe: str, exe_args: str) -> str:
    """Build a single srun invocation for n_gpus GPUs using the cluster config."""
    gpn = cfg.gpus_per_node
    nodes   = (n_gpus + gpn - 1) // gpn
    ntasks  = n_gpus

    flags = [
        f'--nodes={nodes}',
        f'--ntasks={ntasks}',
        f'--ntasks-per-node={min(n_gpus, gpn)}',
    ]
    if cfg.gpu_mode == 'per_task':
        flags.append('--gpus-per-task=1')
    elif cfg.gpu_mode == 'per_node':
        flags.append(f'--gpus-per-node={gpn}')
    elif cfg.gpu_mode == 'gres':
        flags.append(f'--gres=gpu:{min(n_gpus, gpn)}')

    flags_str  = ' '.join(flags)
    prefix_str = cfg.srun_prefix + ' ' if cfg.srun_prefix else ''
    return f'srun {flags_str} {prefix_str}{exe} {exe_args}'


# ---------------------------------------------------------------------------
# Job 1 – Single-node benchmarks
# ---------------------------------------------------------------------------

def make_single_node_script(cfg: ClusterConfig, build_dir: Path, results_dir: Path,
                             warmup: int, runs: int, grid: int, rho: float, tol: float,
                             log_dir: Path) -> str:
    sn_dir  = results_dir / 'single_node'
    n_nodes = 1

    # We run with all GPUs on one node for single-node benchmarks
    n_gpus    = cfg.gpus_per_node
    bench_exe = str(build_dir / 'BenchKernels')
    tile_exe  = str(build_dir / 'TileSweep')
    break_exe = str(build_dir / 'BreakdownNUFFT')
    dens_exe  = str(build_dir / 'BenchDensities')

    header = sbatch_header(cfg, 'nufft_single_node', n_nodes, '06:00:00', log_dir)

    def step(exe, args, label):
        cmd = srun_cmd(cfg, n_gpus, exe, args)
        return f'\necho "=== {label} ==="\n{cmd} || echo "[WARN] {label} returned non-zero"\n'

    body = textwrap.dedent(f"""\
        set -euo pipefail
        set +e   # allow individual benchmark failures without aborting

        {cfg.preamble}

        mkdir -p "{sn_dir}/kernels" "{sn_dir}/tilesweep" "{sn_dir}/breakdown" "{sn_dir}/densities"

        # ── 1. TileSweep (ρ=1 and ρ=10) ────────────────────────────────────────
        # Bayesian-optimisation sweep: takes ~1-2 h per density.
        # Output: tile_sweep_sa_optimal.csv in --output directory.
    """)

    for rho_val in [1, 10]:
        ts_out = sn_dir / 'tilesweep' / f'tilesweep_rho{rho_val}'
        args = (f'--grid {grid} --rho {rho_val} --warmup {warmup} --runs {runs} '
                f'--output "{ts_out}" --optimize --bo-budget 80 --min-width 2 --max-width 8')
        body += step(tile_exe, args, f'TileSweep rho={rho_val}')

    body += textwrap.dedent(f"""\

        # Export optimal tile CSVs so BenchKernels can use them (best-effort).
        # Concatenate both density sweeps into one combined CSV if both exist.
        TILE_CSV_RHO1="{sn_dir}/tilesweep/tilesweep_rho1/tile_sweep_sa_optimal.csv"
        TILE_CSV_RHO10="{sn_dir}/tilesweep/tilesweep_rho10/tile_sweep_sa_optimal.csv"
        COMBINED_TILE_CSV="{sn_dir}/tilesweep/tile_sweep_sa_optimal_combined.csv"
        if [ -f "$TILE_CSV_RHO1" ] && [ -f "$TILE_CSV_RHO10" ]; then
            head -1 "$TILE_CSV_RHO1" > "$COMBINED_TILE_CSV"
            tail -n +2 "$TILE_CSV_RHO1" >> "$COMBINED_TILE_CSV"
            tail -n +2 "$TILE_CSV_RHO10" >> "$COMBINED_TILE_CSV"
            export IPPL_TILE_CSV="$COMBINED_TILE_CSV"
        elif [ -f "$TILE_CSV_RHO1" ]; then
            export IPPL_TILE_CSV="$TILE_CSV_RHO1"
        elif [ -f "$TILE_CSV_RHO10" ]; then
            export IPPL_TILE_CSV="$TILE_CSV_RHO10"
        fi

        # ── 2. BenchKernels (ρ=1 and ρ=10, complex and real) ────────────────────
    """)

    for rho_val in [1, 10]:
        for vtype in ['complex', 'real']:
            tag    = f'kernels_rho{rho_val}_{vtype}'
            kn_out = sn_dir / 'kernels' / tag
            real_flag = '--real' if vtype == 'real' else ''
            args = (f'--grid {grid} --rho {rho_val} --warmup {warmup} --runs {runs} '
                    f'--output "{kn_out}" {real_flag}')
            body += step(bench_exe, args, f'BenchKernels rho={rho_val} {vtype}')

    body += '\n        # ── 3. BreakdownNUFFT (grid sizes 64, 128, 192, 256) ─────────────────────\n'

    for gs in [64, 128, 192, 256]:
        bd_out = sn_dir / 'breakdown' / f'breakdown_{gs}'
        # BreakdownNUFFT writes IpplTimings to stdout – redirect to file.
        # Positional args: grid_size rho tol warmup_runs benchmark_runs output_csv
        args = f'{gs} {rho} {tol} {warmup} {runs} "{bd_out}.csv"'
        cmd  = srun_cmd(cfg, n_gpus, break_exe, args)
        body += (f'\necho "=== BreakdownNUFFT grid={gs} ==="\n'
                 f'mkdir -p "{sn_dir}/breakdown"\n'
                 f'{cmd} || echo "[WARN] BreakdownNUFFT grid={gs} returned non-zero"\n')

    body += textwrap.dedent(f"""\

        # ── 4. BenchDensities / cuFINUFFT comparison (optional) ─────────────────
        # Requires build with -DENABLE_FINUFFT=ON.  Non-fatal if executable absent.
    """)
    dens_out = sn_dir / 'densities' / 'comparison'
    args = (f'--grid {grid} --rho {rho} --tol {tol} --warmup {warmup} '
            f'--runs {runs} --output "{dens_out}"')
    cmd = srun_cmd(cfg, n_gpus, dens_exe, args)
    body += (f'if [ -x "{dens_exe}" ]; then\n'
             f'  echo "=== BenchDensities ==="\n'
             f'  {cmd} || echo "[WARN] BenchDensities returned non-zero"\n'
             f'else\n'
             f'  echo "[SKIP] BenchDensities not found (build without ENABLE_FINUFFT?)"\n'
             f'fi\n')

    body += '\necho "Single-node benchmarks done."\n'
    return header + body


# ---------------------------------------------------------------------------
# Job 2 – Distributed NUFFT strong-scaling
# ---------------------------------------------------------------------------

def make_nufft_scaling_script(cfg: ClusterConfig, build_dir: Path, results_dir: Path,
                               gpu_counts: list, grid: int, rho: float, tol: float,
                               warmup: int, runs: int, log_dir: Path) -> str:
    max_gpus = max(gpu_counts)
    max_nodes = (max_gpus + cfg.gpus_per_node - 1) // cfg.gpus_per_node
    out_dir = results_dir / 'scaling' / f'benchmark_results_{cfg.name}'
    bench_exe = str(build_dir / 'BenchmarkNUFFT')

    header = sbatch_header(cfg, 'nufft_scaling', max_nodes, '02:00:00', log_dir)

    body = textwrap.dedent(f"""\
        set +e

        {cfg.preamble}

        mkdir -p "{out_dir}"

    """)

    for ng in gpu_counts:
        csv_timing = out_dir / f'timings_gpus{ng}.csv'
        csv_comp   = out_dir / f'nufft_components_{ng}.csv'
        # BenchmarkNUFFT positional: grid rho tol warmup runs output_csv
        # A second optional positional for the components CSV may exist;
        # adjust if the executable signature differs.
        args = f'{grid} {rho} {tol} {warmup} {runs} "{csv_timing}" "{csv_comp}"'
        cmd  = srun_cmd(cfg, ng, bench_exe, args)
        body += (f'echo "=== BenchmarkNUFFT {ng} GPUs ==="\n'
                 f'{cmd} || echo "[WARN] BenchmarkNUFFT {ng} GPUs returned non-zero"\n\n')

    body += 'echo "NUFFT scaling benchmarks done."\n'
    return header + body


# ---------------------------------------------------------------------------
# Job 3 – Landau damping (PIF + PIFPruned strong-scaling)
# ---------------------------------------------------------------------------

def make_landau_script(cfg: ClusterConfig, solver_bin_dir: Path, results_dir: Path,
                        gpu_counts: list, grid: int, warmup: int, runs: int,
                        log_dir: Path) -> str:
    """
    LandauDampingPIF writes per-rank timing CSVs automatically (IPPL framework)
    into the current working directory, named LandauDampingPIF{N}_{run}.csv etc.
    We cd into the output directory so they land there.
    Positional args (from existing sbatch): gx gy gz num_particles nt dt shape order tol --info 5
    """
    max_gpus  = max(gpu_counts)
    max_nodes = (max_gpus + cfg.gpus_per_node - 1) // cfg.gpus_per_node
    out_dir   = results_dir / 'landau' / 'new' / cfg.name

    n_particles = grid ** 3 * 10   # rho = 10
    nt  = 20
    dt  = 0.05
    tol = '1e-4'

    pif_exe        = str(solver_bin_dir / 'LandauDampingPIF')
    pif_pruned_exe = str(solver_bin_dir / 'LandauDampingPIFPruned')

    header = sbatch_header(cfg, 'landau_scaling', max_nodes, '02:00:00', log_dir)

    body = textwrap.dedent(f"""\
        set +e

        {cfg.preamble}

        mkdir -p "{out_dir}"
        cd "{out_dir}"

    """)

    for ng in gpu_counts:
        # Standard (full FFT)
        args_std = (f'{grid} {grid} {grid} {n_particles} {nt} {dt} '
                    f'B-spline 1 {tol} --info 5')
        cmd_std = srun_cmd(cfg, ng, pif_exe, args_std)
        # Pruned FFT
        cmd_pruned = srun_cmd(cfg, ng, pif_pruned_exe, args_std)

        body += (f'echo "=== LandauDampingPIF {ng} GPUs (Full FFT) ==="\n'
                 f'if [ -x "{pif_exe}" ]; then\n'
                 f'  {cmd_std} || echo "[WARN] LandauDampingPIF {ng} GPUs failed"\n'
                 f'else\n'
                 f'  echo "[SKIP] LandauDampingPIF not found: {pif_exe}"\n'
                 f'fi\n\n'
                 f'echo "=== LandauDampingPIFPruned {ng} GPUs (Pruned FFT) ==="\n'
                 f'if [ -x "{pif_pruned_exe}" ]; then\n'
                 f'  {cmd_pruned} || echo "[WARN] LandauDampingPIFPruned {ng} GPUs failed"\n'
                 f'else\n'
                 f'  echo "[SKIP] LandauDampingPIFPruned not found: {pif_pruned_exe}"\n'
                 f'fi\n\n')

    body += f'echo "Landau benchmarks done. Results in {out_dir}"\n'
    return header + body


# ---------------------------------------------------------------------------
# Job 4 – Plotting (CPU-only, no GPU)
# ---------------------------------------------------------------------------

def make_plot_script(cfg: ClusterConfig, plot_script: Path, results_dir: Path,
                     log_dir: Path) -> str:
    """
    Runs plot_paper.py on a single CPU node.
    Uses the same account/partition as the data jobs so the job can be
    submitted to the same cluster. Requests 1 node (no GPUs needed).
    """
    # Build a minimal header without GPU resources
    lines = [
        '#!/bin/bash -l',
        '#SBATCH --job-name=nufft_plot',
        f'#SBATCH --output={log_dir}/nufft_plot_%j.out',
        f'#SBATCH --error={log_dir}/nufft_plot_%j.err',
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=8',
        '#SBATCH --time=00:30:00',
    ]
    if cfg.account:
        lines.append(f'#SBATCH --account={cfg.account}')
    # On LUMI, use standard (CPU) partition for the plot job
    if cfg.name == 'lumi':
        lines.append('#SBATCH --partition=small')
    elif cfg.partition and cfg.name == 'juwels':
        lines.append('#SBATCH --partition=batch')  # CPU nodes on JUWELS
    # Alps: no partition needed
    if cfg.name == 'alps':
        for extra in cfg.extra_sbatch:
            if '--uenv' in extra or '--view' in extra:
                lines.append(f'#SBATCH {extra}')
    lines.append('')

    header = '\n'.join(lines)
    fig_dir = results_dir / 'paper_figures'
    body = textwrap.dedent(f"""\

        mkdir -p "{fig_dir}"
        python3 "{plot_script}" \\
            --results-dir "{results_dir}" \\
            --output-dir  "{fig_dir}"
        echo "Plotting done. Figures in {fig_dir}"
    """)
    return header + body


# ---------------------------------------------------------------------------
# Submit helper
# ---------------------------------------------------------------------------

def submit(script_path: Path, dependency: Optional[str] = None,
           dry_run: bool = False) -> Optional[str]:
    """Submit script_path with sbatch, return job ID string."""
    cmd = ['sbatch']
    if dependency:
        cmd += [f'--dependency=afterok:{dependency}']
    cmd.append(str(script_path))

    print(f'  sbatch {"(DRY) " if dry_run else ""}{" ".join(cmd[1:])}')
    if dry_run:
        return '00000'

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  [ERROR] sbatch failed:\n{result.stderr}', file=sys.stderr)
        return None

    # "Submitted batch job 1234567"
    m = re.search(r'(\d+)', result.stdout)
    job_id = m.group(1) if m else None
    print(f'  → job {job_id}')
    return job_id


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument('--build-dir', type=Path, required=True,
                   help='CMake build directory containing PaperBenchmarks executables')

    # Optional locations
    p.add_argument('--solver-bin-dir', type=Path, default=None,
                   help='Directory containing LandauDampingPIF[Pruned] executables '
                        '(default: --build-dir/../../Solver)')
    p.add_argument('--results-dir', type=Path, default=None,
                   help='Root results directory (default: <this-script>/../../../results)')
    p.add_argument('--plot-script', type=Path, default=None,
                   help='Path to plot_paper.py (default: <results-dir>/plot_paper.py)')
    p.add_argument('--wrapper', type=Path, default=None,
                   help='Path to mpi_wrapper.sh for Alps '
                        '(default: <this-script>/../../../../mpi_wrapper.sh)')

    # Cluster
    p.add_argument('--cluster', choices=['alps', 'lumi', 'juwels'], default=None,
                   help='Target cluster (auto-detected if omitted)')
    p.add_argument('--account', default=None,
                   help='SLURM account/project (e.g. c41, project_462000001, jsc-priority)')
    p.add_argument('--uenv', default='prgenv-gnu/25.11:v1',
                   help='Alps uenv image:version (default: prgenv-gnu/25.11:v1)')

    # Benchmark parameters
    p.add_argument('--grid', type=int, default=512,
                   help='Grid size for distributed benchmarks (default: 512)')
    p.add_argument('--rho', type=float, default=10.0,
                   help='Particles per grid cell (default: 10)')
    p.add_argument('--tol', type=float, default=1e-4,
                   help='NUFFT tolerance (default: 1e-4)')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--max-gpus', type=int, default=None,
                   help='Largest GPU count for strong-scaling jobs '
                        '(default: 256 for Alps/JUWELS, 256 for LUMI)')
    p.add_argument('--gpu-counts', type=str, default=None,
                   help='Comma-separated GPU counts for scaling '
                        '(overrides --max-gpus, e.g. 8,16,32,64,128,256)')

    # Control
    p.add_argument('--skip-single-node', action='store_true',
                   help='Skip single-node benchmarks (jobs 1)')
    p.add_argument('--skip-scaling', action='store_true',
                   help='Skip distributed NUFFT scaling (job 2)')
    p.add_argument('--skip-landau', action='store_true',
                   help='Skip Landau damping runs (job 3)')
    p.add_argument('--skip-plot', action='store_true',
                   help='Skip plotting job (job 4)')
    p.add_argument('--dry-run', action='store_true',
                   help='Generate scripts and print sbatch commands without submitting')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    script_dir   = Path(__file__).resolve().parent
    repo_root    = script_dir.parent.parent.parent   # test/FFT/PaperBenchmarks → ippl/

    build_dir = args.build_dir.resolve()

    results_dir = (args.results_dir or repo_root / 'results').resolve()

    solver_bin_dir = (
        args.solver_bin_dir or build_dir.parent.parent / 'Solver'
    ).resolve()

    plot_script = (
        args.plot_script or results_dir / 'plot_paper.py'
    ).resolve()

    wrapper = (
        args.wrapper or repo_root / 'mpi_wrapper.sh'
    ).resolve()

    log_dir = results_dir / 'logs'
    job_dir = results_dir / 'jobs'
    log_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)

    # ── Detect / validate cluster ────────────────────────────────────────────
    cluster = args.cluster or detect_cluster()
    if cluster is None:
        print('[ERROR] Could not auto-detect cluster. Use --cluster alps|lumi|juwels.',
              file=sys.stderr)
        sys.exit(1)

    account = args.account or {
        'alps': 'c41', 'lumi': 'project_XXXXXX', 'juwels': 'budget-XXXX'
    }[cluster]

    print(f'Cluster  : {cluster}')
    print(f'Account  : {account}')
    print(f'Build    : {build_dir}')
    print(f'Solver   : {solver_bin_dir}')
    print(f'Results  : {results_dir}')
    print(f'Logs     : {log_dir}')
    print(f'Jobs dir : {job_dir}')
    print()

    # ── Build cluster config ─────────────────────────────────────────────────
    if cluster == 'alps':
        cfg = _alps_config(account, args.uenv, str(wrapper))
    elif cluster == 'lumi':
        cfg = _lumi_config(account)
    else:  # juwels
        cfg = _juwels_config(account)

    # ── GPU counts ───────────────────────────────────────────────────────────
    if args.gpu_counts:
        gpu_counts = [int(x) for x in args.gpu_counts.split(',')]
    else:
        max_g = args.max_gpus or 256
        gpu_counts = []
        g = cfg.gpus_per_node  # start from 1 full node
        while g <= max_g:
            gpu_counts.append(g)
            g *= 2

    print(f'GPU counts for scaling: {gpu_counts}')
    print()

    submitted_ids = []

    # ── Job 1: single-node ───────────────────────────────────────────────────
    sn_job_id = None
    if not args.skip_single_node:
        script = make_single_node_script(
            cfg, build_dir, results_dir,
            args.warmup, args.runs, args.grid, args.rho, args.tol, log_dir,
        )
        sn_path = job_dir / 'single_node.sbatch'
        sn_path.write_text(script)
        print(f'[Job 1] Single-node benchmarks → {sn_path}')
        sn_job_id = submit(sn_path, dry_run=args.dry_run)
        if sn_job_id:
            submitted_ids.append(sn_job_id)

    # ── Job 2: NUFFT distributed scaling ────────────────────────────────────
    sc_job_id = None
    if not args.skip_scaling:
        script = make_nufft_scaling_script(
            cfg, build_dir, results_dir,
            gpu_counts, args.grid, args.rho, args.tol,
            args.warmup, args.runs, log_dir,
        )
        sc_path = job_dir / 'nufft_scaling.sbatch'
        sc_path.write_text(script)
        print(f'[Job 2] NUFFT scaling → {sc_path}')
        sc_job_id = submit(sc_path, dry_run=args.dry_run)
        if sc_job_id:
            submitted_ids.append(sc_job_id)

    # ── Job 3: Landau ────────────────────────────────────────────────────────
    la_job_id = None
    if not args.skip_landau:
        script = make_landau_script(
            cfg, solver_bin_dir, results_dir,
            gpu_counts, args.grid, args.warmup, args.runs, log_dir,
        )
        la_path = job_dir / 'landau.sbatch'
        la_path.write_text(script)
        print(f'[Job 3] Landau → {la_path}')
        la_job_id = submit(la_path, dry_run=args.dry_run)
        if la_job_id:
            submitted_ids.append(la_job_id)

    # ── Job 4: Plotting ──────────────────────────────────────────────────────
    if not args.skip_plot:
        script = make_plot_script(cfg, plot_script, results_dir, log_dir)
        pl_path = job_dir / 'plot.sbatch'
        pl_path.write_text(script)
        print(f'[Job 4] Plotting → {pl_path}')
        dep = ':'.join(submitted_ids) if submitted_ids else None
        submit(pl_path, dependency=dep, dry_run=args.dry_run)

    print('\nAll scripts written to', job_dir)
    if args.dry_run:
        print('[dry-run] No jobs were actually submitted.')
    else:
        print(f'Monitor with:  squeue -u $USER')
        print(f'Logs in:       {log_dir}')
        print(f'Figures will land in: {results_dir}/paper_figures/')


if __name__ == '__main__':
    main()
