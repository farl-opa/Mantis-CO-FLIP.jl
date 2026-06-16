# Double shear-layer benchmark (one sweep point) -- standalone CO-FLIP case.
#
# Self-contained: includes the CO-FLIP engine, defines this case's driver
# (which consumes the full SimulationConfig `CFG` defined at the top of this
# file), and runs one sweep point
# when executed directly. The (nel, particles/cell) point is selected from
# the environment (same var the SLURM job array sets):
#     COFLIP_SWEEP_IDX  1-based index into SHEAR_LAYER_SWEEP (default: 1)
#     julia --project=. CO-FLIP/cases/shear_layer.jl

include(joinpath(@__DIR__, "..", "CO-FLIP_solver.jl"))

# Double-shear-layer mesh/particle-density sweep. Each entry is one
# independent simulation point `(nel, particles_per_cell)`; a SLURM job
# array (1:6) fills out the whole grid in parallel. Three mesh sizes
# (40², 96², 128²) crossed with two particle densities (10, 20 ppc).
const SHEAR_LAYER_SWEEP = (
    (40,  20), (96,  20), (128, 20),
)

# ============================================================================
# Configuration -- edit here to change the double-shear-layer case.
# ============================================================================
# Complete base simulation configuration; every field is set explicitly. The
# swept quantities (`nel`, `particles_per_cell`) are applied per run by the
# driver via `reconfigure`; the values below are placeholders the sweep
# overrides.
const CFG = SimulationConfig(
    nel                          = (96, 96),   # overridden per sweep point
    p                            = (2, 2),
    k                            = (1, 1),
    box_size                     = (1.0, 1.0),
    starting_point               = (0.0, 0.0),
    boundary_condition           = :periodic,
    inlet_U_inf                  = 1.0,
    farfield_velocity            = (0.0, 0.0),
    farfield_corner_pin_cells    = 4,
    obstacle                     = nothing,
    particles_per_cell           = 10,         # overridden per sweep point
    stratified_seeding           = true,
    volume_convention            = :physical,
    rng_seed                     = nothing,
    flow_type                    = :shear,
    target_cfl                   = 0.5,
    T_final                      = 3.0,
    viscosity                    = 0.0,
    viscosity_splitting          = :strang,
    viscous_time_scheme          = :backward_euler,
    max_fp_iter                  = 4,
    fp_tol                       = 1e-7,
    enable_energy_correction     = true,
    enable_pressure_kick         = true,
    pic_blend_alpha              = 0.02,
    min_particles_per_element    = 4,
    max_particles_per_element    = 16,
    min_particles_per_quarter    = 1,
    ftle_threshold               = 1.0,
    max_longterm_delta_t         = 1.0,
    ftle_use_rate                = true,
    global_ftle_gate             = -Inf,
    delayed_reinit_frequency     = 0,
    output_every                 = 10,
    clear_memo_every             = 0,
    clear_memo_every_fp_iter     = 1,
    lsqr_atol                    = 1e-6,
    lsqr_btol                    = 1e-6,
    lsqr_maxiter                 = 2000,
    lsqr_error_on_nonconvergence = false,
    cfl_recheck_tolerance        = 0.3,
    cfl_adaptive                 = true,
    projection_mean_subtract     = true,
    advection_time_integrator    = :rk2,
    vorticity_method             = :strong,
)

"""
Run ONE point of the double-shear-layer benchmark (Bell-Colella-Glaz
roll-up) on the periodic unit box, for a given mesh resolution and
particle-per-cell density.

`sweep_idx` indexes into `SHEAR_LAYER_SWEEP`, picking the
`(nel, particles_per_cell)` pair for this run. A SLURM job array (1:6)
covers the three mesh sizes (40², 96², 128²) × two densities (10, 20).

The shear flow (`flow_shear`) is the classic two anti-parallel layers
(thickness set by ρ=30) plus a single-mode `v`-perturbation (δ=0.05) that
seeds the Kelvin-Helmholtz roll-up. Each run writes vorticity / velocity
VTK frames every `save_vtk_every` steps so the roll-up can be visualised,
plus a per-step energy/enstrophy/circulation history CSV. Comparing the
frames across the sweep shows how mesh resolution and particle density
affect the resolved vortex structure.

Defaults: ν=1e-4 (Re≈10⁴) on the 1×1 periodic box, p=(3,3) splines, run
to T_final=1.8 with adaptive CFL (target 0.5).
"""
function test_shear_layer(cfg::SimulationConfig = CFG;
        sweep_idx::Int = 1,
        output_dir::String = joinpath(get(ENV, "OUTPUT_DIR", pwd()), "shear_layer"),
        record_every::Int = 1,
        shear_sweep = SHEAR_LAYER_SWEEP,
    )
    1 <= sweep_idx <= length(shear_sweep) ||
        error("sweep_idx=$sweep_idx out of range for shear-layer sweep (1:$(length(shear_sweep)))")
    nel_val, ppc_val = shear_sweep[sweep_idx]

    # Case quantities the print needs, pulled from the config.
    viscosity      = cfg.viscosity
    T_final        = cfg.T_final
    box_size       = cfg.box_size
    save_vtk_every = cfg.output_every

    run_label = @sprintf("shear_nel%03d_ppc%02d", nel_val, ppc_val)
    run_dir   = joinpath(output_dir, run_label)
    mkpath(run_dir)

    println("\n========== DOUBLE SHEAR LAYER ==========")
    println(@sprintf("  sweep_idx=%d  →  nel=(%d,%d)  ppc=%d  ν=%.1e  T=%.2f  box=%s",
                     sweep_idx, nel_val, nel_val, ppc_val, viscosity, T_final, string(box_size)))
    println("  Output dir: $run_dir")

    # Apply the swept mesh / particle density to the base config.
    cfg = reconfigure(cfg; nel = (nel_val, nel_val), particles_per_cell = ppc_val)

    result = run_diagnostic_simulation(cfg;
        save_vtk        = true,
        save_vtk_every  = save_vtk_every,
        output_dir      = run_dir,
        record_every    = record_every,
        case_name       = run_label,
    )

    history_csv = joinpath(output_dir, run_label * "_history.csv")
    open(history_csv, "w") do io
        println(io, "nel,ppc,t,energy,enstrophy,circulation")
        for hi in result.history
            @printf(io, "%d,%d,%.10e,%.16e,%.16e,%.16e\n",
                    nel_val, ppc_val, hi.t, hi.energy, hi.enstrophy, hi.circulation)
        end
    end
    println("Saved per-step history: $history_csv")

    return (result=result, run_label=run_label,
            nel=nel_val, ppc=ppc_val, history_csv=history_csv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    with_run_logging() do
        idx = parse(Int, get(ENV, "COFLIP_SWEEP_IDX", "1"))
        test_shear_layer(; sweep_idx=idx)
    end
end
