#!/usr/bin/env python3
"""
run_paper_benchmarks.py
=======================
Orchestrator for all single-node and distributed benchmarks used in the PASC '26
paper "A Performance-Portable, Massively Parallel Distributed Nonuniform FFT".

Each subcommand generates the data for one or more figures:

  kernels   – Fig 1/2   BenchKernels:            throughput vs tolerance (ρ=1, ρ=10)
  tilesweep – Fig 3     TileSweep (--optimize):  tile-size / width heatmap
  breakdown – Fig 4     BreakdownNUFFT:          component time breakdown
  comparison– Fig 5     BenchDensities:          cuFINUFFT comparison
  scaling   – Fig 6/7   BenchmarkNUFFT:          distributed strong-scaling
  pruned    – Fig 8     TestPrunedFFTPerformance: pruned vs full FFT
  all                   Run every benchmark in sequence

Usage
-----
  python run_paper_benchmarks.py <subcommand> [options]

Required
--------
  --build-dir DIR      Path to the CMake build directory containing the
                       compiled executables (e.g. build/test/FFT/PaperBenchmarks).

Common options
--------------
  --output-dir DIR     Directory for CSV output   (default: ./paper_results)
  --warmup N           Warmup iterations          (default: 5)
  --runs N             Timed iterations           (default: 20)
  --grid N             Grid size per dimension    (default: 128)
  --rho R              Particles per grid point   (default: 10)
  --tol T              NUFFT tolerance            (default: 1e-4)
  --mpirun CMD         MPI launcher (e.g. srun)   (default: mpirun)
  --ranks N            MPI ranks for scaling run  (default: 1)
  --tile-csv PATH      Path to tile_sweep_sa_optimal.csv for BenchKernels
  --dry-run            Print commands without executing them
  -v, --verbose        Echo every command before running it

Examples
--------
  # Single-node kernel benchmark (Figs 1, 2) at two densities:
  python run_paper_benchmarks.py kernels --build-dir build/test/FFT/PaperBenchmarks

  # Tile-size BO sweep (Fig 3) – expensive, use a good GPU node:
  python run_paper_benchmarks.py tilesweep --build-dir build/... --grid 128

  # Full suite (single node):
  python run_paper_benchmarks.py all --build-dir build/... --output-dir ./results

  # Distributed scaling on Alps (Figs 6, 7):
  python run_paper_benchmarks.py scaling --build-dir build/... \\
      --mpirun "srun -C gpu" --ranks 8,16,32,64,128,256,512,1024 --grid 512
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], *, dry_run: bool = False, verbose: bool = False,
        env: dict | None = None) -> int:
    """Run *cmd*, optionally echoing it first.  Returns the exit code."""
    cmd_str = " ".join(str(c) for c in cmd)
    if verbose or dry_run:
        print(f"  $ {cmd_str}", flush=True)
    if dry_run:
        return 0
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, env=merged_env)
    if result.returncode != 0:
        print(f"[WARNING] Command returned {result.returncode}: {cmd_str}", file=sys.stderr)
    return result.returncode


def exe(build_dir: Path, name: str) -> Path:
    """Return path to a compiled executable, checking it exists."""
    p = build_dir / name
    if not p.exists():
        sys.exit(f"[ERROR] Executable not found: {p}\n"
                 "        Did you build with CMake? Check --build-dir.")
    return p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def mpi_prefix(mpirun: str, ranks: int) -> list[str]:
    """Build MPI prefix for the given launcher string and rank count."""
    if ranks <= 1:
        return []
    return mpirun.split() + ["-n", str(ranks)]


# ---------------------------------------------------------------------------
# Individual benchmark runners
# ---------------------------------------------------------------------------

def run_kernels(args: argparse.Namespace, build: Path, out: Path) -> None:
    """Figure 1 & 2 – kernel throughput vs tolerance, two densities."""
    section("BenchKernels  (Figs 1, 2)  –  throughput vs tolerance")
    b = exe(build, "BenchKernels")
    env = {}
    if args.tile_csv:
        env["IPPL_TILE_CSV"] = str(args.tile_csv)

    for rho in [1, 10]:
        for value_type in ["complex", "real"]:
            tag = f"kernels_rho{rho}_{value_type}"
            prefix = mpi_prefix(args.mpirun, args.ranks)
            cmd = prefix + [
                str(b),
                "--grid",    str(args.grid),
                "--rho",     str(rho),
                "--warmup",  str(args.warmup),
                "--runs",    str(args.runs),
                "--output",  str(out / tag),
            ]
            if value_type == "real":
                cmd.append("--real")
            print(f"\n  → rho={rho}, {value_type}", flush=True)
            run(cmd, dry_run=args.dry_run, verbose=args.verbose, env=env)


def run_tilesweep(args: argparse.Namespace, build: Path, out: Path) -> None:
    """Figure 3 – spreading throughput heatmap (tile size × kernel width)."""
    section("TileSweep  (Fig 3)  –  tile-size / kernel-width heatmap")
    b = exe(build, "TileSweep")

    for rho in [1, 10]:
        tag = f"tilesweep_rho{rho}"
        prefix = mpi_prefix(args.mpirun, args.ranks)
        cmd = prefix + [
            str(b),
            "--grid",       str(args.grid),
            "--rho",        str(rho),
            "--warmup",     str(args.warmup),
            "--runs",       str(args.runs),
            "--output",     str(out / tag),
            "--optimize",                      # Bayesian optimisation
            "--bo-budget",  "80",
            "--min-width",  "2",
            "--max-width",  "8",
        ]
        print(f"\n  → rho={rho}", flush=True)
        run(cmd, dry_run=args.dry_run, verbose=args.verbose)


def run_breakdown(args: argparse.Namespace, build: Path, out: Path) -> None:
    """Figure 4 – NUFFT component time breakdown for various grid sizes."""
    section("BreakdownNUFFT  (Fig 4)  –  component time breakdown")
    b = exe(build, "BreakdownNUFFT")

    # Grid sizes shown in the paper: 64, 128, 192, 256
    grid_sizes = [64, 128, 192, 256]
    for n in grid_sizes:
        prefix = mpi_prefix(args.mpirun, args.ranks)
        cmd = prefix + [
            str(b),
            str(n),           # positional: grid_size
            str(args.rho),    # positional: rho
            str(args.tol),    # positional: tol
            str(args.warmup), # positional: warmup_runs
            str(args.runs),   # positional: benchmark_runs
        ]
        print(f"\n  → grid={n}", flush=True)
        run(cmd, dry_run=args.dry_run, verbose=args.verbose)

    # Redirect / capture output: BreakdownNUFFT writes IpplTimings to stdout,
    # so the caller should redirect to a file if needed; the script echoes
    # commands for shell redirection when --dry-run is active.
    print("\n  [Note] Redirect stdout per run to capture IpplTimings output.\n"
          "         E.g.: BreakdownNUFFT 128 10 1e-4 5 20 > breakdown_128.txt",
          flush=True)


def run_comparison(args: argparse.Namespace, build: Path, out: Path) -> None:
    """Figure 5 – throughput comparison with cuFINUFFT (needs ENABLE_FINUFFT)."""
    section("BenchDensities  (Fig 5)  –  cuFINUFFT comparison")
    b = exe(build, "BenchDensities")
    prefix = mpi_prefix(args.mpirun, args.ranks)
    cmd = prefix + [
        str(b),
        "--grid",    str(args.grid),
        "--rho",     str(args.rho),
        "--tol",     str(args.tol),
        "--warmup",  str(args.warmup),
        "--runs",    str(args.runs),
        "--output",  str(out / "comparison"),
    ]
    print("\n  [Note] Requires build with -DENABLE_FINUFFT=ON (cuFINUFFT present).",
          flush=True)
    run(cmd, dry_run=args.dry_run, verbose=args.verbose)


def run_scaling(args: argparse.Namespace, build: Path, out: Path) -> None:
    """Figures 6 & 7 – distributed strong-scaling of the full NUFFT."""
    section("BenchmarkNUFFT  (Figs 6, 7)  –  distributed strong-scaling")
    b = exe(build, "BenchmarkNUFFT")

    # Parse the ranks list (comma-separated or single value)
    try:
        rank_list = [int(r) for r in str(args.ranks).split(",")]
    except ValueError:
        sys.exit(f"[ERROR] --ranks must be a comma-separated list of integers, got: {args.ranks}")

    for n_ranks in rank_list:
        csv_name = out / f"scaling_{n_ranks}ranks_{args.grid}grid.csv"
        prefix = mpi_prefix(args.mpirun, n_ranks)
        cmd = prefix + [
            str(b),
            str(args.grid),    # grid_size
            str(args.rho),     # rho
            str(args.tol),     # tol
            str(args.warmup),
            str(args.runs),
            str(csv_name),     # output CSV
        ]
        print(f"\n  → ranks={n_ranks}", flush=True)
        run(cmd, dry_run=args.dry_run, verbose=args.verbose)


def run_pruned(args: argparse.Namespace, build: Path, out: Path) -> None:
    """Figure 8 – pruned vs full FFT performance at large grid sizes."""
    section("TestPrunedFFTPerformance  (Fig 8)  –  pruned vs full FFT")
    b = exe(build, "TestPrunedFFTPerformance")

    # Paper uses 1024³ and 2048³ for this comparison
    grid_sizes = [512, 1024, 2048]
    for n in grid_sizes:
        for n_concurrent in [1, 4]:  # concurrency levels shown in Fig 8
            prefix = mpi_prefix(args.mpirun, args.ranks)
            cmd = prefix + [
                str(b),
                str(n),             # grid_size
                str(n_concurrent),  # concurrency
                str(args.warmup),
                str(args.runs),
            ]
            print(f"\n  → grid={n}, concurrency={n_concurrent}", flush=True)
            run(cmd, dry_run=args.dry_run, verbose=args.verbose)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand list
    cmds = {
        "kernels":    "Fig 1/2: kernel throughput vs tolerance",
        "tilesweep":  "Fig 3:   tile-size / kernel-width heatmap",
        "breakdown":  "Fig 4:   NUFFT component time breakdown",
        "comparison": "Fig 5:   cuFINUFFT comparison",
        "scaling":    "Fig 6/7: distributed strong-scaling",
        "pruned":     "Fig 8:   pruned vs full FFT",
        "all":        "Run all benchmarks in sequence",
    }
    subparsers = {}
    for name, help_text in cmds.items():
        subparsers[name] = sub.add_parser(name, help=help_text)

    # Attach common arguments to every subparser and to the parent
    for sp in list(subparsers.values()) + [parser]:
        sp.add_argument("--build-dir", metavar="DIR", type=Path,
                        help="CMake build directory containing compiled executables")
        sp.add_argument("--output-dir", metavar="DIR", type=Path,
                        default=Path("paper_results"),
                        help="Directory for output CSV files (default: ./paper_results)")
        sp.add_argument("--warmup", metavar="N", type=int, default=5,
                        help="Warmup iterations (default: 5)")
        sp.add_argument("--runs", metavar="N", type=int, default=20,
                        help="Timed iterations (default: 20)")
        sp.add_argument("--grid", metavar="N", type=int, default=128,
                        help="Grid size per dimension (default: 128)")
        sp.add_argument("--rho", metavar="R", type=float, default=10.0,
                        help="Particles per grid point (default: 10)")
        sp.add_argument("--tol", metavar="T", type=float, default=1e-4,
                        help="NUFFT tolerance (default: 1e-4)")
        sp.add_argument("--mpirun", metavar="CMD", type=str, default="mpirun",
                        help="MPI launcher (default: mpirun).  Use e.g. 'srun -C gpu' on Alps.")
        sp.add_argument("--ranks", metavar="N", default="1",
                        help="MPI ranks.  Comma-separated list for 'scaling' subcommand "
                             "(default: 1).")
        sp.add_argument("--tile-csv", metavar="PATH", type=Path, default=None,
                        help="Path to tile_sweep_sa_optimal.csv; sets IPPL_TILE_CSV env var "
                             "for BenchKernels so it uses profiled optimal tile configs.")
        sp.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
        sp.add_argument("-v", "--verbose", action="store_true",
                        help="Echo every command before running it")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNNERS = {
    "kernels":    run_kernels,
    "tilesweep":  run_tilesweep,
    "breakdown":  run_breakdown,
    "comparison": run_comparison,
    "scaling":    run_scaling,
    "pruned":     run_pruned,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.build_dir is None:
        sys.exit("[ERROR] --build-dir is required.")

    build = args.build_dir.resolve()
    out   = ensure_dir(args.output_dir.resolve())

    print(f"Build dir  : {build}")
    print(f"Output dir : {out}")
    print(f"Timestamp  : {datetime.now().isoformat(timespec='seconds')}")

    if args.subcommand == "all":
        for name, runner in RUNNERS.items():
            runner(args, build, out)
    else:
        RUNNERS[args.subcommand](args, build, out)

    print("\n[Done]  All requested benchmarks finished.", flush=True)


if __name__ == "__main__":
    main()
