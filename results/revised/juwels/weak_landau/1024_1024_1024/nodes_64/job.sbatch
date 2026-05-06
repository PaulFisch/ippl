#!/bin/bash
#SBATCH --job-name=lw_64n
#SBATCH --account=hai_1073
#SBATCH --partition=booster
#SBATCH --time=00:10:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=lw_64n-%j.out
#SBATCH --error=lw_64n-%j.err
#SBATCH --gres=gpu:4

set -euo pipefail

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export UCC_TL_NCCL_TUNE=allreduce:cuda:inf
export IPPL_TILE_CSV="/p/home/jusers/fischill1/juwels/ippl/build_fin/tile_sweep_sa_optimal.csv"

mkdir -p data

srun --nodes=64 --ntasks=256 --ntasks-per-node=4 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/alpine/ElectrostaticPIF/LandauDampingPIF"     1024 1024 1024 8589934592 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIF.out 2> data/LandauDampingPIF.err

srun --nodes=64 --ntasks=256 --ntasks-per-node=4 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/alpine/ElectrostaticPIF/LandauDampingPIFPruned"     1024 1024 1024 8589934592 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIFPruned.out 2> data/LandauDampingPIFPruned.err
