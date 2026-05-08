#!/bin/bash
#SBATCH --job-name=coflip_stuart
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --time=15:00:00
#SBATCH --output=coflip_stuart_%j.out 
#SBATCH --error=coflip_stuart_%j.err

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Main scratch output folder for this run
export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/stuart_$SLURM_JOB_ID

# Create folders if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p /scratch/opelegriartola/slurm_logs

# Go to repository root (where Project.toml is)
cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

# Instantiate environment if needed
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# If Mantis is a local package, register it in the environment
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

# Run the simulation
julia --project=. CO-FLIP/delftblue_cases/CO-FLIP_periodic_stuart.jl
