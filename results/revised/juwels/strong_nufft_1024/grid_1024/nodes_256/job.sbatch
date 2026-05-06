#!/bin/bash
#SBATCH --job-name=juwels_pruned_1024_256n
#SBATCH --account=hai_1073
#SBATCH --partition=booster
#SBATCH --time=00:30:00
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=juwels_pruned_1024_256n-%j.out
#SBATCH --error=juwels_pruned_1024_256n-%j.err
#SBATCH --gres=gpu:4

set -euo pipefail

ml Stages/2026 GCC OpenMPI CMake

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export UCC_TL_NCCL_TUNE=allreduce:cuda:inf

cd "/p/home/jusers/fischill1/juwels/ippl/build_fin"
cd "/p/scratch/hai_1073/pfischill/generated_benchmark_jobs/juwels_nufft_pruned_selected_1024/grid_1024/nodes_256"
mkdir -p results data

srun --nodes=256 --ntasks=1024 --ntasks-per-node=4 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/test/FFT/PaperBenchmarks/BenchmarkNUFFT"     --grid 1024 --ppp 8 --tol 1e-4 --warmup 3 --runs 10     --spread output_focused --gather atomic_sort --csv results/nufft_scaling_1024_pruned.csv --component-csv results/nufft_components_1024_pruned.csv     > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
