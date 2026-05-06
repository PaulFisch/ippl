#!/bin/bash
#SBATCH --job-name=single_fft
#SBATCH --account=c41
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --uenv=prgenv-gnu/25.11:v1
#SBATCH --view=default
#SBATCH --output=single_fft-%j.out
#SBATCH --error=single_fft-%j.err

set -euo pipefail

export WRAPPER="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/mpi_wrapper.sh"
export IPPL_TILE_CSV="/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/tile_sweep_sa_optimal.csv"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=72

mkdir -p results/singlenode

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1     $WRAPPER "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/test/FFT/PaperBenchmarks/SingleNode" --mode all --grid 200     > results/singlenode/mode_all_grid200.out     2> results/singlenode/mode_all_grid200.err

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1     $WRAPPER "/capstor/scratch/cscs/pfischil/ippl_singlenode/ippl/build/test/FFT/PaperBenchmarks/SingleNode" --mode density --grid 200     > results/singlenode/mode_density_grid200.out     2> results/singlenode/mode_density_grid200.err
