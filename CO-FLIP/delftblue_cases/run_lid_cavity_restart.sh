#!/bin/bash
#SBATCH --job-name=coflip_lid_restart
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=10:00:00
#SBATCH --output=coflip_lid_restart_%j.out
#SBATCH --error=coflip_lid_restart_%j.err

# Restart the lid-driven cavity case from a saved particle .vtu and run the
# full steady-state + Ghia (1982) comparison when it finishes.
#
# Continues from step 1550 up to step 5760 (4210 steps). The step count is
# fixed: adaptive CFL may shrink dt but will not change how many steps run, so
# the run stops exactly at step 5760. The starting dt is auto-computed from the
# CFL condition of the loaded state (set COFLIP_DT to override).

module load julia

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Output tree for the restarted run (new VTK frames particles_1551 .. 5760 and
# the analysis / Ghia CSVs are written under $OUTPUT_DIR/lid_cavity/).
export OUTPUT_DIR=/scratch/opelegriartola/coflip_outputs/lid_cavity_restart_$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR
mkdir -p /scratch/opelegriartola/slurm_logs

cd /home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using Pkg; Pkg.develop(path="Mantis")'

# Path to the snapshot to restart from. "main directory" = the repo root here;
# edit if particles_1550.vtu lives elsewhere on the cluster.
export COFLIP_CASE=lid_cavity_restart
export COFLIP_RESTART_VTU=/home/opelegriartola/CO-FLIP/Mantis-CO-FLIP.jl/particles_5100.vtu
export COFLIP_RESTART_STEP=5100
export COFLIP_FINAL_STEP=5760
# export COFLIP_DT=0.01   # optional: omit to auto-set dt from CFL

julia --project=. CO-FLIP/CO-FLIP_periodic.jl
