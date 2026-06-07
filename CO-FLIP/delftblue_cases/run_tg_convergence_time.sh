#!/bin/bash
#SBATCH --job-name=coflip_tg_conv_time
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=coflip_tg_conv_time_%A_%a.out
#SBATCH --error=coflip_tg_conv_time_%A_%a.err

# Decaying Taylor-Green temporal-convergence sweep.
# Job array index = sweep_idx → TG_DT_TIME_SWEEP entry, nel fixed at 128.
# Submit:    sbatch run_tg_convergence_time.sh

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/tg_conv_time_$SLURM_ARRAY_JOB_ID
mkdir -p $OUTPUT_DIR
mkdir -p /scratch/opelegriartola/slurm_logs

cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

export COFLIP_CASE=tg_convergence
export COFLIP_SWEEP_MODE=time
export COFLIP_SWEEP_IDX=$SLURM_ARRAY_TASK_ID

julia --project=. CO-FLIP/CO-FLIP_solver.jl
