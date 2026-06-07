#!/bin/bash
#SBATCH --job-name=coflip_shear_layer
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --output=coflip_shear_layer_%A_%a.out
#SBATCH --error=coflip_shear_layer_%A_%a.err

# Double-shear-layer mesh / particle-density sweep.
# Job array index = sweep_idx → SHEAR_LAYER_SWEEP entry:
#   1: nel=40²   ppc=10      4: nel=40²   ppc=20
#   2: nel=96²   ppc=10      5: nel=96²   ppc=20
#   3: nel=128²  ppc=10      6: nel=128²  ppc=20
# Submit:    sbatch run_shear_layer.sh
# Override:  sbatch --array=1-3 run_shear_layer.sh   # ppc=10 meshes only

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# All array tasks for one sweep deposit their output into the same parent
# directory so the resolution/density grid can be browsed side by side.
export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/shear_layer_$SLURM_ARRAY_JOB_ID
mkdir -p $OUTPUT_DIR
mkdir -p /scratch/opelegriartola/slurm_logs

cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

export COFLIP_CASE=shear_layer
export COFLIP_SWEEP_IDX=$SLURM_ARRAY_TASK_ID

julia --project=. CO-FLIP/CO-FLIP_solver.jl
