#!/bin/bash
#SBATCH --job-name=pif_1e8_1024_64n
#SBATCH --account=c41
#SBATCH --time=00:30:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=pif_1e8_1024_64n-%j.out
#SBATCH --error=pif_1e8_1024_64n-%j.err

set -euo pipefail

export WRAPPER="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/mpi_wrapper.sh"
export IPPL_TILE_CSV="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/tile_sweep_sa_optimal.csv"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

cd "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/generated_benchmark_jobs/alps_pif_tol_1e8/strong_landau_1024/grid_1024/nodes_64"
mkdir -p data

srun --nodes=64 --ntasks=256 --ntasks-per-node=4 --gpus-per-task=1 \
    "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/alpine/ElectrostaticPIF/LandauDampingPIF" \
    1024 1024 1024 8589934592 10 0.0122718463 "B-spline" 1 "1e-8" --info 5 --overallocate 1.2 \
    > data/LandauDampingPIF.out 2> data/LandauDampingPIF.err

srun --nodes=64 --ntasks=256 --ntasks-per-node=4 --gpus-per-task=1 \
    "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/alpine/ElectrostaticPIF/LandauDampingPIFPruned" \
    1024 1024 1024 8589934592 10 0.0122718463 "B-spline" 1 "1e-8" --info 5 --overallocate 1.2 \
    > data/LandauDampingPIFPruned.out 2> data/LandauDampingPIFPruned.err
