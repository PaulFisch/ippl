#!/bin/bash
#SBATCH --job-name=ns_cur_32n
#SBATCH --account=c41
#SBATCH --time=00:30:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=ns_cur_32n-%j.out
#SBATCH --error=ns_cur_32n-%j.err

set -euo pipefail

export WRAPPER="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/mpi_wrapper.sh"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

mkdir -p results

srun --nodes=32 --ntasks=128 --ntasks-per-node=4 --gpus-per-task=1     $WRAPPER "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/test/FFT/PaperBenchmarks/BenchmarkNUFFT"     --grid 512 --ppp 8 --tol 1e-4 --warmup 3 --runs 10     --spread output_focused --gather atomic_sort     --csv nufft_scaling_128.csv --component-csv nufft_components_128.csv     > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
