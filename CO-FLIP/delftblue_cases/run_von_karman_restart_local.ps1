# Restart the Von Karman vortex-shedding case LOCALLY (no SLURM / DelftBlue).
#
# Continues from step 1398 up to step 4901 (3503 steps) at dt=0.01, then runs
# the full probe / force / Strouhal / Cd-Cl analysis. The step count is fixed:
# adaptive CFL may shrink dt but will not change how many steps run, so the run
# stops exactly at step 4901.
#
# Run from the project root:
#     pwsh CO-FLIP/delftblue_cases/run_von_karman_restart_local.ps1

$ErrorActionPreference = "Stop"

# Move to the project root (two levels up from this script's folder).
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root

# Parent of the von_karman/ folder that holds vk_re150_vtk/particles_1398.vtu.
# Defaults to the project root, where the file currently lives.
$env:OUTPUT_DIR = $root.Path

# Case is selected by running cases/von_karman.jl; the restart env vars below
# trigger its restart path (COFLIP_RESTART_VTU set => restart instead of fresh).
$env:COFLIP_RESTART_VTU   = Join-Path $root "von_karman\vk_re150_vtk\particles_1398.vtu"
$env:COFLIP_RESTART_STEP  = "1398"
$env:COFLIP_FINAL_STEP    = "4901"
$env:COFLIP_DT            = "0.01"

if (-not (Test-Path $env:COFLIP_RESTART_VTU)) {
    throw "Restart file not found: $($env:COFLIP_RESTART_VTU)"
}

julia --project=. CO-FLIP/cases/von_karman.jl
