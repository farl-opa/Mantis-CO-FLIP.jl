#!/bin/bash
#SBATCH --job-name=coflip_vk_restart
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=24:00:00
#SBATCH --output=coflip_vk_restart_%j.out
#SBATCH --error=coflip_vk_restart_%j.err

# Restart the Von Karman vortex-shedding case from a saved particle .vtu and
# run the full probe / force / Strouhal / Cd-Cl analysis when it finishes.
#
# Continues from step 1398 up to step 4901 (3503 steps) at dt=0.01. The step
# count is fixed: adaptive CFL may shrink dt but will not change how many steps
# run, so the run stops exactly at step 4901.

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /scratch/opelegriartola/slurm_logs

# Directory of the PREVIOUS run that produced particles_1398.vtu. New VTK frames
# (particles_1399 .. particles_4901) and the analysis CSVs are written back into
# this same tree so the series continues. Edit to match your previous job.
export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/von_karman_PREVIOUS_JOBID

cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

export COFLIP_CASE=von_karman_restart
export COFLIP_RESTART_VTU=$OUTPUT_DIR/von_karman/vk_re150_vtk/particles_1398.vtu
export COFLIP_RESTART_STEP=1398
export COFLIP_FINAL_STEP=4901
export COFLIP_DT=0.01

julia --project=. CO-FLIP/CO-FLIP_periodic.jl
