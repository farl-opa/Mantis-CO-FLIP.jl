# Decaying Taylor-Green convergence study (one sweep point) -- standalone case.
#
# Self-contained: includes the CO-FLIP engine + shared analysis helpers,
# defines this case's driver (which consumes the full SimulationConfig `CFG`
# defined at the top of this file),
# and runs one sweep point when executed directly. The sweep point is
# selected from the environment (same vars the SLURM job array sets):
#     COFLIP_SWEEP_MODE in {space, time}   (default: space)
#     COFLIP_SWEEP_IDX  1-based index      (default: 1)
#     julia --project=. CO-FLIP/cases/taylor_green_convergence.jl

include(joinpath(@__DIR__, "..", "CO-FLIP_solver.jl"))
isdefined(Main, :compute_decaying_tg_l2_error) || include(joinpath(@__DIR__, "case_common.jl"))

# ============================================================================
# Configuration -- edit here to change the decaying-TG convergence study.
# ============================================================================
# Complete base simulation configuration; every field is set explicitly. The
# swept quantity (mesh `nel` in :space mode, time step `dt` in :time mode) is
# applied per run by the driver: `nel` via `reconfigure`, `dt` via the
# `fixed_dt` argument of `run_diagnostic_simulation`. `nel` below is a
# placeholder that the sweep overrides.
const CFG = SimulationConfig(
    nel                          = (64, 64),   # overridden per sweep point
    p                            = (3, 3),
    k                            = (1, 1),
    box_size                     = (2π, 2π),
    starting_point               = (0.0, 0.0),
    boundary_condition           = :periodic,
    inlet_U_inf                  = 1.0,
    farfield_velocity            = (0.0, 0.0),
    farfield_corner_pin_cells    = 4,
    obstacle                     = nothing,
    particles_per_cell           = 16,
    stratified_seeding           = true,
    volume_convention            = :physical,
    rng_seed                     = nothing,
    flow_type                    = :decaying_tg,
    target_cfl                   = 0.5,
    T_final                      = 2.0,
    viscosity                    = 0.05,
    # Strang split + Crank–Nicolson diffusion → 2nd-order time discretisation.
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
    output_every                 = 0,
    clear_memo_every             = 0,
    clear_memo_every_fp_iter     = 1,
    lsqr_atol                    = 1e-6,
    lsqr_btol                    = 1e-6,
    lsqr_maxiter                 = 2000,
    lsqr_error_on_nonconvergence = false,
    cfl_recheck_tolerance        = 0.3,
    cfl_adaptive                 = false,      # fixed dt across all sweep points
    projection_mean_subtract     = true,
    advection_time_integrator    = :rk2,
    vorticity_method             = :strong,
)

const TG_NEL_SPACE_SWEEP    = (32, 48, 64, 96, 128)
const TG_DT_TIME_SWEEP      = (0.04, 0.02, 0.01, 0.005, 0.0025)
const TG_FIXED_NEL_FOR_TIME = 128
const TG_FIXED_DT_FOR_SPACE = 0.005

"""
Run ONE point of the decaying Taylor-Green convergence study.

`mode` selects whether this run is part of the spatial sweep (fixed dt,
varying nel) or the temporal sweep (fixed nel, varying dt). `sweep_idx`
picks the point within the corresponding sweep array
(`TG_NEL_SPACE_SWEEP` / `TG_DT_TIME_SWEEP`).

Each run samples the L2 velocity error against the analytical viscous
Navier-Stokes solution at `t=2` and `t=5`, and writes a CSV with one
row per sample plus a per-step history CSV. Cluster post-processing
should gather every `*_errors.csv` from one sweep and plot the resulting
convergence curve.

Defaults: ν=0.05 on the 2π×2π periodic box, T_final=5.0, p=(3,3) splines,
CFL adaptation off (so dt is constant across all spatial-sweep points).
"""
function test_decaying_tg_convergence(cfg::SimulationConfig = CFG;
        mode::Symbol = :space,
        sweep_idx::Int = 1,
        output_dir::String = joinpath(get(ENV, "OUTPUT_DIR", pwd()), "tg_convergence"),
        sample_times::Vector{Float64} = [2.0],
        nel_space_sweep = TG_NEL_SPACE_SWEEP,
        dt_time_sweep   = TG_DT_TIME_SWEEP,
        fixed_nel_for_time::Int   = TG_FIXED_NEL_FOR_TIME,
        fixed_dt_for_space::Float64 = TG_FIXED_DT_FOR_SPACE,
        save_vtk::Bool = false,
        save_vtk_at_samples::Bool = true,
        nq_extra::Int = 2,
    )
    mkpath(output_dir)
    # Case quantities the analysis needs, pulled from the config.
    viscosity = cfg.viscosity
    T_final   = cfg.T_final

    if mode === :space
        1 <= sweep_idx <= length(nel_space_sweep) ||
            error("sweep_idx=$sweep_idx out of range for space sweep (1:$(length(nel_space_sweep)))")
        nel_val = nel_space_sweep[sweep_idx]
        dt_val  = fixed_dt_for_space
        run_label = @sprintf("space_nel%03d_dt%.5f", nel_val, dt_val)
    elseif mode === :time
        1 <= sweep_idx <= length(dt_time_sweep) ||
            error("sweep_idx=$sweep_idx out of range for time sweep (1:$(length(dt_time_sweep)))")
        nel_val = fixed_nel_for_time
        dt_val  = dt_time_sweep[sweep_idx]
        run_label = @sprintf("time_nel%03d_dt%.5f", nel_val, dt_val)
    else
        error("mode must be :space or :time, got $mode")
    end

    println("\n========== DECAYING-TG CONVERGENCE ==========")
    println(@sprintf("  mode=%s  sweep_idx=%d  →  nel=(%d,%d)  dt=%.5f  ν=%.4f  T=%.2f",
                     mode, sweep_idx, nel_val, nel_val, dt_val, viscosity, T_final))
    println("  Output dir: $output_dir")

    # Apply the swept mesh to the base config (dt is applied via `fixed_dt`).
    cfg = reconfigure(cfg; nel = (nel_val, nel_val))

    sample_records = NamedTuple[]
    sample_cb = function (t, u_coeffs, domain, particles, step)
        err  = compute_decaying_tg_l2_error(domain, u_coeffs, t, viscosity;
                                            nq_extra=nq_extra)
        diag = compute_conservation_diagnostics(u_coeffs, domain;
                                                 vorticity_method=cfg.vorticity_method)
        rec  = (t=t, step=step, nel=nel_val, dt=dt_val,
                l2_err=err.l2_err, rel_l2_err=err.rel_l2_err,
                l2_norm_exact=err.l2_norm_exact,
                energy=diag.energy, enstrophy=diag.enstrophy,
                circulation=diag.circulation)
        push!(sample_records, rec)
        @printf("  [SAMPLE] t=%.3f  L2_err=%.6e  rel_L2_err=%.6e  ||u_exact||=%.6e  E=%.6e  Z=%.6e\n",
                t, err.l2_err, err.rel_l2_err, err.l2_norm_exact,
                diag.energy, diag.enstrophy)

        if save_vtk_at_samples
            u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
            u_phys_form = Assemblers.solve_L2_projection(domain.R1, ★(u_form), domain.dΩ)
            ω_h         = compute_vorticity_form(u_coeffs, u_phys_form, domain, cfg.vorticity_method)
            t_tag = @sprintf("t%05.2f", t)
            Plot.export_form_fields_to_vtk((u_form,),
                run_label * "_u_" * t_tag; output_directory_tree=[output_dir])
            Plot.export_form_fields_to_vtk((ω_h,),
                run_label * "_w_" * t_tag; output_directory_tree=[output_dir])
        end
    end

    result = run_diagnostic_simulation(cfg;
        save_vtk        = save_vtk,
        save_vtk_every  = 0,
        output_dir      = joinpath(output_dir, run_label * "_history"),
        record_every    = 5,
        case_name       = run_label,
        fixed_dt        = dt_val,
        sample_times    = sample_times,
        sample_callback = sample_cb,
    )

    csv_path = joinpath(output_dir, run_label * "_errors.csv")
    open(csv_path, "w") do io
        println(io, "mode,sweep_idx,nel,dt,viscosity,t,step,l2_err,rel_l2_err,l2_norm_exact,energy,enstrophy,circulation")
        for r in sample_records
            @printf(io, "%s,%d,%d,%.10e,%.10e,%.6f,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    String(mode), sweep_idx, r.nel, r.dt, viscosity,
                    r.t, r.step, r.l2_err, r.rel_l2_err, r.l2_norm_exact,
                    r.energy, r.enstrophy, r.circulation)
        end
    end
    println("Saved per-sample errors: $csv_path")

    history_csv = joinpath(output_dir, run_label * "_history.csv")
    open(history_csv, "w") do io
        println(io, "t,energy,enstrophy,circulation")
        for hi in result.history
            @printf(io, "%.10e,%.16e,%.16e,%.16e\n",
                    hi.t, hi.energy, hi.enstrophy, hi.circulation)
        end
    end
    println("Saved per-step history: $history_csv")

    return (result=result, records=sample_records, run_label=run_label,
            csv_path=csv_path, history_csv=history_csv)
end

if abspath(PROGRAM_FILE) == @__FILE__
    with_run_logging() do
        mode = Symbol(get(ENV, "COFLIP_SWEEP_MODE", "space"))
        idx  = parse(Int, get(ENV, "COFLIP_SWEEP_IDX", "1"))
        test_decaying_tg_convergence(; mode=mode, sweep_idx=idx)
    end
end
