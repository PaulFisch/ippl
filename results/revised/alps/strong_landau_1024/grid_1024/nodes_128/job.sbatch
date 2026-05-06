#!/bin/bash
#SBATCH --job-name=ls_1024_128n
#SBATCH --account=c41
#SBATCH --time=00:30:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=ls_1024_128n-%j.out
#SBATCH --error=ls_1024_128n-%j.err

set -euo pipefail

export WRAPPER="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/mpi_wrapper.sh"
export IPPL_TILE_CSV="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/tile_sweep_sa_optimal.csv"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

mkdir -p data

srun --nodes=128 --ntasks=512 --ntasks-per-node=4 --gpus-per-task=1     $WRAPPER "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/alpine/ElectrostaticPIF/LandauDampingPIF"     1024 1024 1024 8589934592 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIF.out 2> data/LandauDampingPIF.err

srun --nodes=128 --ntasks=512 --ntasks-per-node=4 --gpus-per-task=1     $WRAPPER "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/alpine/ElectrostaticPIF/LandauDampingPIFPruned"     1024 1024 1024 8589934592 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIFPruned.out 2> data/LandauDampingPIFPruned.err
