#!/bin/bash
#SBATCH --job-name=ns_up_cur_128n
#SBATCH --account=hai_1073
#SBATCH --partition=booster
#SBATCH --time=00:30:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=ns_up_cur_128n-%j.out
#SBATCH --error=ns_up_cur_128n-%j.err
#SBATCH --gres=gpu:4

set -euo pipefail

cd "/p/scratch/hai_1073/pfischill/generated_benchmark_jobs/juwels_nufft_upsampled/strong_nufft_current/grid_512/nodes_128"
mkdir -p results data

ml Stages/2026 GCC OpenMPI CMake

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export UCC_TL_NCCL_TUNE=allreduce:cuda:inf

srun --nodes=128 --ntasks=512 --ntasks-per-node=4 --gpus-per-task=1 \
    "/p/home/jusers/fischill1/juwels/ippl/build_fin/test/FFT/PaperBenchmarks/BenchmarkNUFFT" \
    --grid 512 --ppp 8 --tol 1e-4 --warmup 3 --runs 10 \
    --spread output_focused --gather atomic_sort --upsampled \
    --csv results/nufft_scaling_512_upsampled.csv --component-csv results/nufft_components_512_upsampled.csv \
    > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
