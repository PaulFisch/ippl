#!/bin/bash
#SBATCH --job-name=ns_up_1024_512n
#SBATCH --account=c41
#SBATCH --time=00:30:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=ns_up_1024_512n-%j.out
#SBATCH --error=ns_up_1024_512n-%j.err

set -euo pipefail

export WRAPPER="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/mpi_wrapper.sh"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

cd "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build"
cd "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/generated_benchmark_jobs/alps_nufft_upsampled/strong_nufft_1024/grid_1024/nodes_512"
mkdir -p results data

srun --nodes=512 --ntasks=2048 --ntasks-per-node=4 --gpus-per-task=1 \
    "$WRAPPER" "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/test/FFT/PaperBenchmarks/BenchmarkNUFFT" \
    --grid 1024 --ppp 8 --tol 1e-4 --warmup 3 --runs 10 \
    --spread output_focused --gather atomic_sort --upsampled \
    --csv results/nufft_scaling_2048_upsampled.csv --component-csv results/nufft_components_2048_upsampled.csv \
    > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
