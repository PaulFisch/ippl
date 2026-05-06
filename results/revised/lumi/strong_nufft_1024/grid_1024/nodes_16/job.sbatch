#!/bin/bash -l
#SBATCH --job-name=ns_1024_16n
#SBATCH --error=ns_1024_16n-%j.error
#SBATCH --output=ns_1024_16n-%j.out
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=16
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

mkdir -p results

srun --cpu-bind=${CPU_BIND} --nodes=16 --ntasks=128 --ntasks-per-node=8     ./select_gpu "/scratch/project_465002453/pfischill/ippl/build/test/FFT/PaperBenchmarks/BenchmarkNUFFT"     --grid 1024 --ppp 8 --tol 1e-4 --warmup 3 --runs 10     --spread output_focused --gather atomic_sort     --csv nufft_scaling_128.csv --component-csv nufft_components_128.csv     > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
