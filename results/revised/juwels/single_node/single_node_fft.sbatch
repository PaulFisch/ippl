#!/bin/bash
#SBATCH --job-name=single_fft
#SBATCH --account=hai_1073
#SBATCH --partition=booster
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=single_fft-%j.out
#SBATCH --error=single_fft-%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

cd "/p/scratch/hai_1073/pfischill/generated_benchmark_jobs/juwels_single_node_rerun/single_node"
mkdir -p results data

ml Stages/2026 GCC OpenMPI CMake

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export UCC_TL_NCCL_TUNE=allreduce:cuda:inf
export IPPL_TILE_CSV="/p/home/jusers/fischill1/juwels/ippl/build_fin/tile_sweep_sa_optimal.csv"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

echo "Running SingleNode --mode all --grid 144 --grids 64,96,128,160 --breakdown-rho 5"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/test/FFT/PaperBenchmarks/SingleNode"     --mode all     --grid 144     --grids 64,96,128,160     --breakdown-rho 5     > results/mode_all_grid144.out     2> results/mode_all_grid144.err

echo "Running SingleNode --mode density --grid 112"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/test/FFT/PaperBenchmarks/SingleNode"     --mode density     --grid 112     > results/mode_density_grid112.out     2> results/mode_density_grid112.err
