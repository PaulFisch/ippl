#!/bin/bash
#SBATCH --job-name=lw_32n
#SBATCH --account=hai_1073
#SBATCH --partition=booster
#SBATCH --time=00:10:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=lw_32n-%j.out
#SBATCH --error=lw_32n-%j.err
#SBATCH --gres=gpu:4

set -euo pipefail

export OMPI_MCA_coll_ucc_enable=1
export OMPI_MCA_coll_ucc_priority=100
export UCC_TL_NCCL_TUNE=allreduce:cuda:inf
export IPPL_TILE_CSV="/p/home/jusers/fischill1/juwels/ippl/build_fin/tile_sweep_sa_optimal.csv"

mkdir -p data

srun --nodes=32 --ntasks=128 --ntasks-per-node=4 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/alpine/ElectrostaticPIF/LandauDampingPIF"     512 1024 1024 4294967296 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIF.out 2> data/LandauDampingPIF.err

srun --nodes=32 --ntasks=128 --ntasks-per-node=4 --gpus-per-task=1     "/p/home/jusers/fischill1/juwels/ippl/build_fin/alpine/ElectrostaticPIF/LandauDampingPIFPruned"     512 1024 1024 4294967296 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIFPruned.out 2> data/LandauDampingPIFPruned.err
