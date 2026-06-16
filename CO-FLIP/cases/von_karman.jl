# Von Karman vortex shedding past a cylinder -- standalone CO-FLIP case.
#
# Self-contained: includes the CO-FLIP engine, defines this case's driver
# (which consumes the full SimulationConfig `CFG` defined at the top of this
# file) plus its Strouhal helper, and
# runs it when executed directly. Supports restart from a saved particle .vtu
# via the environment (same vars the SLURM restart script sets):
#     COFLIP_RESTART_VTU   path to particle .vtu (unset => fresh run)
#     COFLIP_RESTART_STEP  step the .vtu was saved at (default: 0)
#     COFLIP_FINAL_STEP    last step to integrate to  (default: 0)
#     COFLIP_DT            restart dt                 (default: 0.01)
#     julia --project=. CO-FLIP/cases/von_karman.jl

include(joinpath(@__DIR__, "..", "CO-FLIP_solver.jl"))

# ============================================================================
# Configuration -- edit here to change the Von Kármán shedding case.
# ============================================================================
# Complete simulation configuration: every SimulationConfig field is set
# explicitly so this file fully determines the run on its own. The cylinder
# geometry and free-stream speed live in `obstacle` / `inlet_U_inf`, and the
# driver derives Re, D and the force normalisation from them.
const CFG = SimulationConfig(
    nel                          = (160, 80),
    p                            = (3, 3),
    k                            = (1, 1),
    box_size                     = (40.0, 20.0),
    starting_point               = (0.0, 0.0),
    boundary_condition           = (:inlet, :outlet, :periodic, :periodic),
    inlet_U_inf                  = 1.5,
    # Far-field policy (only active when a top/bottom side is `:farfield`):
    # replenish depleted edge cells with `(U_inf, 0)` and pin v=0 on the first
    # few cells at each inlet-touching corner to stabilise the projection.
    farfield_velocity            = (1.5, 0.0),
    farfield_corner_pin_cells    = 4,
    obstacle                     = (kind=:cylinder, center=(10.0, 10.0), radius=1.0),
    particles_per_cell           = 10,
    stratified_seeding           = true,
    volume_convention            = :physical,
    rng_seed                     = nothing,
    flow_type                    = :cylinder,
    target_cfl                   = 0.3,
    T_final                      = 100.0,
    viscosity                    = 0.02,
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
    output_every                 = 3,
    clear_memo_every             = 1,
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

# Case-specific settings that are NOT part of SimulationConfig.
const OUTPUT_DIR   = joinpath(get(ENV, "OUTPUT_DIR", pwd()), "von_karman")
const PROBE_POINTS = [(20.0, 15.0), (25.0, 15.0), (30.0, 15.0)]  # downstream v(t) probes
const CASE_LABEL   = "vk_re150"

"""
Estimate the dominant frequency of a probe v(t) signal via a brute-force
periodogram over candidate frequencies. Returns `(f, St, power)` where
`St = f·D/U_inf` is the Strouhal number. Uses the second half of the
record by default to skip the start-up transient.

This is intentionally simple (no FFTW dependency). For a thesis-quality
spectrum the user should re-process the saved probe CSV in
Python/MATLAB; this estimate is meant as an in-run sanity check.
"""
function estimate_strouhal_from_v_signal(
        ts::Vector{Float64}, vs::Vector{Float64};
        U_inf::Float64=1.0, D::Float64=2.0,
        skip_fraction::Float64=0.5, n_freq::Int=4096,
    )
    n  = length(ts)
    n  >= 16 || return (f=NaN, St=NaN, power=0.0)
    n1 = max(1, floor(Int, skip_fraction * n))
    t_seg = ts[n1:end]
    v_seg = vs[n1:end]
    Nu = length(t_seg)
    t0, tf = first(t_seg), last(t_seg)
    window = tf - t0
    window > 0 || return (f=NaN, St=NaN, power=0.0)
    dt_avg = window / max(Nu - 1, 1)

    v_mean = sum(v_seg) / Nu
    v_c    = v_seg .- v_mean

    f_min = 1.0 / window
    f_max = 0.5 / dt_avg
    f_max <= f_min && return (f=NaN, St=NaN, power=0.0)

    df    = (f_max - f_min) / (n_freq - 1)
    best_power = 0.0
    best_f     = f_min
    for k in 0:(n_freq - 1)
        f = f_min + k * df
        ω = 2π * f
        a = 0.0;  b = 0.0
        @inbounds for j in 1:Nu
            tj = t_seg[j]
            a += v_c[j] * cos(ω * tj)
            b += v_c[j] * sin(ω * tj)
        end
        power = a*a + b*b
        if power > best_power
            best_power = power
            best_f     = f
        end
    end
    return (f=best_f, St=best_f * D / U_inf, power=best_power)
end

"""
Run a Von Kármán vortex-shedding case past a cylinder and write probe
velocity data for offline Strouhal estimation.

Defaults set up Re=100 (D=2, U_inf=1, ν=0.02) on a 50×30 box with
nel=(200,120) → dx=dy=0.25. Top and bottom are :wall (free-slip) to
avoid the periodic feedthrough of the older default cfg. Three probe
points are sampled every step and dumped to CSV. A coarse periodogram
estimate of the dominant frequency is printed at the end as a sanity
check; for thesis-grade spectra, post-process the CSV with FFTW in
Python/MATLAB.
"""
function test_von_karman_strouhal(cfg::SimulationConfig = CFG;
        output_dir::String       = OUTPUT_DIR,
        probe_points::Vector{NTuple{2,Float64}} = PROBE_POINTS,
        case_label::String       = CASE_LABEL,
        # Restart support. When `restart_vtu` is set, particles are loaded from
        # that .vtu instead of generated fresh, the run continues from
        # `restart_step`+1 up to `final_step` (a fixed step count, so adaptive
        # CFL shrinks dt but never changes how many steps run), starting at
        # `restart_dt`. All probe / force / Strouhal / Cd-Cl analysis runs the
        # same as a fresh case.
        restart_vtu::Union{String,Nothing} = nothing,
        restart_step::Int                  = 0,
        final_step::Union{Int,Nothing}     = nothing,
        restart_dt::Float64                = 0.01,
    )
    mkpath(output_dir)
    # Case quantities the analysis needs, pulled from the config. The original
    # local names are kept so the diagnostics below read unchanged.
    nel             = cfg.nel
    box_size        = cfg.box_size
    U_inf           = cfg.inlet_U_inf
    viscosity       = cfg.viscosity
    T_final         = cfg.T_final
    cylinder_center = cfg.obstacle.center
    cylinder_radius = cfg.obstacle.radius
    bc_top_bottom   = cfg.boundary_condition[3]
    save_vtk_every  = cfg.output_every
    Re = U_inf * (2 * cylinder_radius) / viscosity

    is_restart = restart_vtu !== nothing
    if is_restart
        final_step !== nothing && final_step > restart_step ||
            error("test_von_karman_strouhal: restart requires final_step > restart_step " *
                  "(got restart_step=$(restart_step), final_step=$(final_step))")
        isfinite(restart_dt) && restart_dt > 0 ||
            error("test_von_karman_strouhal: restart_dt must be a finite positive Float64")
    end
    n_steps_override = is_restart ? (final_step - restart_step) : nothing

    println("\n========== VON KARMAN VORTEX SHEDDING ==========")
    println(@sprintf("  box=%s  nel=%s  cyl=%s  r=%.3f  U=%.3f  ν=%.4f  Re=%.1f  T=%.1f  bc_y=%s",
                     box_size, nel, cylinder_center, cylinder_radius,
                     U_inf, viscosity, Re, T_final, bc_top_bottom))
    if is_restart
        println(@sprintf("  RESTART from %s: step %d → %d (%d steps), dt0=%.4g",
                         restart_vtu, restart_step, final_step,
                         n_steps_override, restart_dt))
    end

    n_probes = length(probe_points)
    probe_records = NamedTuple[]

    probe_cb = function (t, u_coeffs, domain, step)
        cache = EvaluationCache(evaluation_cache_size(domain))
        us = Vector{Float64}(undef, n_probes)
        vs = Vector{Float64}(undef, n_probes)
        @inbounds for k in 1:n_probes
            xp, yp = probe_points[k]
            (u, v), _ = probe_field_at_point(xp, yp, u_coeffs, domain, cache)
            us[k] = Float64(u);  vs[k] = Float64(v)
        end
        push!(probe_records, (t=t, step=step, us=us, vs=vs))
    end

    # Per-step force history for drag/lift coefficient post-processing.
    # `force_callback` fires after every step_co_flip!. `fx`, `fy` are
    # already converted from impulse to force (divided by dt) by the
    # runner; here we normalise to (Cd, Cl) using ρ=1, D=2r, U=U_inf.
    D_cyl = 2 * cylinder_radius
    cd_norm = 2.0 / (U_inf^2 * D_cyl)  # ρ = 1
    force_records = NamedTuple[]
    force_cb = function (t, fx, fy, dt, step)
        push!(force_records,
              (t=t, step=step, dt=dt, fx=fx, fy=fy,
               Cd=cd_norm * fx, Cl=cd_norm * fy))
    end

    result = run_diagnostic_simulation(cfg;
        save_vtk         = true,
        save_vtk_every   = save_vtk_every,
        output_dir       = joinpath(output_dir, case_label * "_vtk"),
        record_every     = max(1, save_vtk_every),
        case_name        = case_label,
        probe_callback   = probe_cb,
        force_callback   = force_cb,
        save_particles   = true,
        restart_vtu      = restart_vtu,
        restart_step     = restart_step,
        n_steps_override = n_steps_override,
        initial_dt       = is_restart ? restart_dt : nothing,
    )

    probe_csv = joinpath(output_dir, case_label * "_probes.csv")
    open(probe_csv, "w") do io
        header = "t,step"
        for k in 1:n_probes
            header *= @sprintf(",u_p%d,v_p%d", k, k)
        end
        println(io, header)
        for r in probe_records
            line = @sprintf("%.10e,%d", r.t, r.step)
            for k in 1:n_probes
                line *= @sprintf(",%.16e,%.16e", r.us[k], r.vs[k])
            end
            println(io, line)
        end
    end
    println("Saved probe CSV: $probe_csv")

    history_csv = joinpath(output_dir, case_label * "_history.csv")
    open(history_csv, "w") do io
        println(io, "t,energy,enstrophy,circulation")
        for hi in result.history
            @printf(io, "%.10e,%.16e,%.16e,%.16e\n",
                    hi.t, hi.energy, hi.enstrophy, hi.circulation)
        end
    end
    println("Saved history CSV: $history_csv")

    # ----- Force history CSV -----
    force_csv = joinpath(output_dir, case_label * "_forces.csv")
    open(force_csv, "w") do io
        println(io, "t,step,dt,fx,fy,Cd,Cl")
        for r in force_records
            @printf(io, "%.10e,%d,%.10e,%.16e,%.16e,%.16e,%.16e\n",
                    r.t, r.step, r.dt, r.fx, r.fy, r.Cd, r.Cl)
        end
    end
    println("Saved force CSV:   $force_csv")

    # Drag/lift coefficient statistics over the second half of the record
    # (the developed-shedding regime; the first half captures the start-up
    # transient where Cd is artificially high and Cl is still ramping up).
    cd_mean  = NaN
    cl_rms   = NaN
    cl_mean  = NaN
    n_window = 0
    if length(force_records) >= 16
        start_idx = div(length(force_records), 2) + 1
        n_window  = length(force_records) - start_idx + 1
        cds = [r.Cd for r in @view force_records[start_idx:end]]
        cls = [r.Cl for r in @view force_records[start_idx:end]]
        cd_mean = sum(cds) / n_window
        cl_mean = sum(cls) / n_window
        # RMS-about-the-mean (i.e. standard deviation). For Kármán shedding
        # the mean Cl ≈ 0 by symmetry; using std rather than √<Cl²> keeps
        # the value invariant to a small mean drift from start-up bias.
        cl_var = 0.0
        @inbounds for cl in cls
            d = cl - cl_mean
            cl_var += d * d
        end
        cl_rms = sqrt(cl_var / n_window)
    end

    # ----- Strouhal + Cd/Cl summary -----
    st_f = NaN; st_T = NaN; st_St = NaN
    if length(probe_records) > 16
        ts  = [r.t for r in probe_records]
        vs1 = [r.vs[1] for r in probe_records]
        st  = estimate_strouhal_from_v_signal(ts, vs1;
                                              U_inf=U_inf, D=2*cylinder_radius)
        st_f  = st.f
        st_T  = 1.0 / max(st.f, eps())
        st_St = st.St
        println("\n--- In-run Strouhal estimate (probe 1, second half of record) ---")
        @printf("  Re=%.1f  U=%.3f  D=%.3f\n", Re, U_inf, 2*cylinder_radius)
        @printf("  f_dominant=%.6f  T_period=%.4f  St=f·D/U=%.4f\n",
                st_f, st_T, st_St)
    end

    if length(force_records) >= 16
        println("\n--- Drag / lift coefficient (second half of record, $n_window samples) ---")
        @printf("  Cd_mean  = %.4f   (Re=%.1f reference ≈ 1.32-1.37)\n", cd_mean, Re)
        @printf("  Cl_mean  = %.4f   (expected ≈ 0 by top/bottom symmetry)\n", cl_mean)
        @printf("  Cl_rms   = %.4f   (Re=%.1f reference ≈ 0.22-0.26)\n", cl_rms, Re)
    end

    open(joinpath(output_dir, case_label * "_summary.txt"), "w") do io
        println(io, @sprintf("Re=%.2f  U=%.4f  D=%.4f  ν=%.4f", Re, U_inf,
                             2*cylinder_radius, viscosity))
        if length(probe_records) > 16
            println(io, @sprintf("Probe 1 location = (%.3f, %.3f)",
                                 probe_points[1][1], probe_points[1][2]))
            println(io, @sprintf("Strouhal       St = %.6f", st_St))
            println(io, @sprintf("Dominant freq f = %.6f", st_f))
            println(io, @sprintf("Period         T = %.4f", st_T))
        end
        if length(force_records) >= 16
            println(io, @sprintf("Cd_mean = %.6f  (window: last %d samples)", cd_mean, n_window))
            println(io, @sprintf("Cl_mean = %.6f", cl_mean))
            println(io, @sprintf("Cl_rms  = %.6f", cl_rms))
        end
    end

    return (result=result, probes=probe_records, forces=force_records,
            probe_csv=probe_csv, history_csv=history_csv, force_csv=force_csv,
            Cd_mean=cd_mean, Cl_rms=cl_rms, Cl_mean=cl_mean,
            St=st_St, f_shed=st_f, T_shed=st_T)
end

if abspath(PROGRAM_FILE) == @__FILE__
    with_run_logging() do
        vtu = get(ENV, "COFLIP_RESTART_VTU", "")
        if vtu == ""
            test_von_karman_strouhal()
        else
            rstep = parse(Int, get(ENV, "COFLIP_RESTART_STEP", "0"))
            fstep = parse(Int, get(ENV, "COFLIP_FINAL_STEP", "0"))
            rdt   = parse(Float64, get(ENV, "COFLIP_DT", "0.01"))
            test_von_karman_strouhal(;
                restart_vtu  = vtu,
                restart_step = rstep,
                final_step   = fstep,
                restart_dt   = rdt,
            )
        end
    end
end
