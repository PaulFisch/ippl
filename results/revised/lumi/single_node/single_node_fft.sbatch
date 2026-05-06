#!/bin/bash -l
#SBATCH --job-name=single_fft
#SBATCH --error=single_fft-%j.error
#SBATCH --output=single_fft-%j.out
#SBATCH --partition=standard-g
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=project_465002453
#SBATCH --hint=nomultithread

set -euo pipefail

module load LUMI/25.03 partition/G cpeAMD rocm buildtools/25.03

CPU_BIND="map_cpu:49"
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -s unlimited

cat << 'EOS' > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOS
chmod +x ./select_gpu

export IPPL_TILE_CSV="/users/fischill/ippl/build_fin/tile_sweep_sa_optimal.csv"
mkdir -p results/singlenode

srun --cpu-bind=${CPU_BIND} --nodes=1 --ntasks=1 --ntasks-per-node=1     ./select_gpu "/scratch/project_465002453/pfischill/ippl/build/test/FFT/PaperBenchmarks/SingleNode" --mode all --grid 200     > results/singlenode/mode_all_grid200.out     2> results/singlenode/mode_all_grid200.err

srun --cpu-bind=${CPU_BIND} --nodes=1 --ntasks=1 --ntasks-per-node=1     ./select_gpu "/scratch/project_465002453/pfischill/ippl/build/test/FFT/PaperBenchmarks/SingleNode" --mode density --grid 200     > results/singlenode/mode_density_grid200.out     2> results/singlenode/mode_density_grid200.err
