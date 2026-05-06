#!/bin/bash -l
#SBATCH --job-name=lw_8n
#SBATCH --error=lw_8n-%j.error
#SBATCH --output=lw_8n-%j.out
#SBATCH --partition=standard-g
#SBATCH --time=00:10:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465002453
#SBATCH --hint=nomultithread

set -euo pipefail

module load LUMI/25.03 partition/G cpeAMD rocm buildtools/25.03

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
export MPICH_GPU_SUPPORT_ENABLED=1
ulimit -s unlimited

cat << 'EOS' > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOS
chmod +x ./select_gpu

export IPPL_TILE_CSV="/users/fischill/ippl/build_fin/tile_sweep_sa_optimal.csv"
mkdir -p data

srun --cpu-bind=${CPU_BIND} --nodes=8 --ntasks=64 --ntasks-per-node=8     ./select_gpu "/scratch/project_465002453/pfischill/ippl/build/alpine/ElectrostaticPIF/LandauDampingPIF"     512 512 512 1073741824 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIF.out 2> data/LandauDampingPIF.err

srun --cpu-bind=${CPU_BIND} --nodes=8 --ntasks=64 --ntasks-per-node=8     ./select_gpu "/scratch/project_465002453/pfischill/ippl/build/alpine/ElectrostaticPIF/LandauDampingPIFPruned"     512 512 512 1073741824 10 0.0122718463 "B-spline" 1 "1e-4" --info 5 --overallocate 1.2     > data/LandauDampingPIFPruned.out 2> data/LandauDampingPIFPruned.err
