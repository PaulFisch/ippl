#!/bin/bash -l
#SBATCH --job-name=ns_cur_64n
#SBATCH --error=ns_cur_64n-%j.error
#SBATCH --output=ns_cur_64n-%j.out
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=64
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

srun --cpu-bind=${CPU_BIND} --nodes=64 --ntasks=512 --ntasks-per-node=8     ./select_gpu "/scratch/project_465002453/pfischill/ippl/build/test/FFT/PaperBenchmarks/BenchmarkNUFFT"     --grid 512 --ppp 8 --tol 1e-4 --warmup 3 --runs 10     --spread output_focused --gather atomic_sort     --csv nufft_scaling_512.csv --component-csv nufft_components_512.csv     > results/BenchmarkNUFFT.out 2> results/BenchmarkNUFFT.err
