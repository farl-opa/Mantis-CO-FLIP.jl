#!/bin/bash
#SBATCH --job-name=coflip_tg_conv_space
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=coflip_tg_conv_space_%A_%a.out
#SBATCH --error=coflip_tg_conv_space_%A_%a.err

# Decaying Taylor-Green spatial-convergence sweep.
# Job array index = sweep_idx → TG_NEL_SPACE_SWEEP entry, dt fixed.
# Submit:    sbatch run_tg_convergence_space.sh
# Override:  sbatch --array=1-5 run_tg_convergence_space.sh

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# All array tasks for one sweep deposit CSVs into the same directory so
# post-processing can stitch them into a convergence plot trivially.
export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/tg_conv_space_$SLURM_ARRAY_JOB_ID
mkdir -p $OUTPUT_DIR
mkdir -p /scratch/opelegriartola/slurm_logs

cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

export COFLIP_CASE=tg_convergence
export COFLIP_SWEEP_MODE=space
export COFLIP_SWEEP_IDX=$SLURM_ARRAY_TASK_ID

julia --project=. CO-FLIP/CO-FLIP_periodic.jl
