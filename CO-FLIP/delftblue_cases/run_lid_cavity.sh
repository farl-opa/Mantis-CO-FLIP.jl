#!/bin/bash
#SBATCH --job-name=coflip_lid_cavity
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --output=coflip_lid_cavity_%j.out
#SBATCH --error=coflip_lid_cavity_%j.err

# Lid-driven cavity at Re=100 with Brinkman-style tangential lid enforcement.
# Note: the boundary layer is resolved over the Brinkman band rather than at
# the wall exactly — flag this caveat when comparing against benchmark data.

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/lid_cavity_$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR
mkdir -p /scratch/opelegriartola/slurm_logs

cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

export COFLIP_CASE=lid_cavity

julia --project=. CO-FLIP/CO-FLIP_periodic.jl
