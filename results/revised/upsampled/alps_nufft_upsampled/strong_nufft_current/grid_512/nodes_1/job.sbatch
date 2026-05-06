#!/bin/bash
#SBATCH --job-name=ns_up_cur_1n
#SBATCH --account=c41
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=ns_up_cur_1n-%j.out
#SBATCH --error=ns_up_cur_1n-%j.err

set -euo pipefail

export WRAPPER="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/mpi_wrapper.sh"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

cd "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build"
cd "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/generated_benchmark_jobs/alps_nufft_upsampled/strong_nufft_current/grid_512/nodes_1"
mkdir -p results data

srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 \
    "$WRAPPER" "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/test/FFT/PaperBenchmarks/BenchmarkNUFFT" \
    --grid 512 --ppp 8 --tol 1e-4 --warmup 3 --runs 10 \
    --spread output_focused --gather atomic_sort --upsampled \
    --csv results/nufft_scaling_4_upsampled.csv --component-csv results/nufft_components_4_upsampled.csv \
    > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
