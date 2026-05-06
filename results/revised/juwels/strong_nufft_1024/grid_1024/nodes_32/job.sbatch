#!/bin/bash
#SBATCH --job-name=ns_1024_32n
#SBATCH --account=hai_1073
#SBATCH --partition=booster
#SBATCH --time=00:30:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=ns_1024_32n-%j.out
#SBATCH --error=ns_1024_32n-%j.err
#SBATCH --gres=gpu:4

set -euo pipefail

cd "/p/scratch/hai_1073/pfischill/generated_benchmark_jobs/juwels_single_nufft/strong_nufft_1024/grid_1024/nodes_32"
mkdir -p results data

ml Stages/2026 GCC OpenMPI CMake

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export UCC_TL_NCCL_TUNE=allreduce:cuda:inf

srun --nodes=32 --ntasks=128 --ntasks-per-node=4 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/test/FFT/PaperBenchmarks/BenchmarkNUFFT"     --grid 1024 --ppp 8 --tol 1e-4 --warmup 3 --runs 10     --spread output_focused --gather atomic_sort     --csv results/nufft_scaling_128.csv --component-csv results/nufft_components_128.csv     > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
