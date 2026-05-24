#!/bin/bash
#SBATCH --job-name=coflip_von_karman
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --output=coflip_von_karman_%j.out
#SBATCH --error=coflip_von_karman_%j.err

# Von Karman vortex shedding at Re=100 on a 50x30 box with free-slip top/bottom
# walls. Writes a per-step probe CSV used for offline Strouhal post-processing.

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/von_karman_$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR
mkdir -p /scratch/opelegriartola/slurm_logs

cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

export COFLIP_CASE=von_karman

julia --project=. CO-FLIP/CO-FLIP_periodic.jl
