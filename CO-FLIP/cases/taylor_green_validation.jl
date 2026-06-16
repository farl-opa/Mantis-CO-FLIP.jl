# Decaying Taylor-Green viscous validation -- standalone CO-FLIP case.
#
# Self-contained: includes the CO-FLIP engine, defines this case's driver
# (which consumes the full SimulationConfig `CFG` defined at the top of this
# file), and runs it when executed
# directly:
#     julia --project=. CO-FLIP/cases/taylor_green_validation.jl
#
# Edit the SimulationConfig / keyword defaults in `test_decaying_taylor_green`
# below to change this case independently of the engine and other cases.

include(joinpath(@__DIR__, "..", "CO-FLIP_solver.jl"))

# ============================================================================
# Configuration -- edit here to change the decaying Taylor-Green case.
# ============================================================================
# Complete simulation configuration for the VISCOUS run; every field is set
# explicitly. The inviscid control reuses this config with `viscosity = 0`
# (see `reconfigure` in the driver).
const CFG = SimulationConfig(
    nel                          = (96, 96),
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
    T_final                      = 10.0,
    viscosity                    = 0.2,
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
    cfl_adaptive                 = true,
    projection_mean_subtract     = true,
    advection_time_integrator    = :rk2,
    vorticity_method             = :strong,
)

# Case-specific settings that are NOT part of SimulationConfig.
const OUTPUT_DIR = "decaying_tg_test"

"""
Comprehensive validation of the FEEC viscous diffusion implementation against
the decaying Taylor–Green vortex.

For the 2π×2π periodic box with U₀=1, kx=ky=1 the exact Navier–Stokes solution is

    u(x,y,t) =  sin(x)·cos(y)·exp(-2νt)
    v(x,y,t) = -cos(x)·sin(y)·exp(-2νt)
    ω(x,y,t) =  2·sin(x)·sin(y)·exp(-2νt)

with conserved-but-decaying integrals

    E(t) = E₀ · exp(-2ν·(kx²+ky²)·t)         (E₀ = π² for these constants)
    Z(t) = Z₀ · exp(-2ν·(kx²+ky²)·t)         (Z₀ = 2π²)
    Γ    ≡ 0                                  (periodic ⇒ zero circulation)

Runs:
  1. Viscous case (cfg.viscosity = `viscosity`, decaying_tg IC, periodic BCs)
  2. Inviscid baseline (ν = 0) — should preserve energy within CO-FLIP's usual
     conservation tolerance, isolating numerical drift from viscous dissipation.

Reports:
  • Initial E and Z vs analytical π² and 2π².
  • Per-step relative error of E(t), Z(t) vs the analytical exponential.
  • Best-fit decay rate from log-linear regression of E and Z, compared to
    the analytical 2ν(kx²+ky²).
  • Circulation drift (should remain ≈ 0).
  • Inviscid control drift (sanity check that ν=0 → no extra decay).

Saves a CSV of (t, E_num, E_exact, Z_num, Z_exact, Γ) and a 3-panel plot of
E, Z, and log(E) vs time in `output_dir`.

Returns a NamedTuple with all numerics for downstream automated checks.
"""
function test_decaying_taylor_green(cfg::SimulationConfig = CFG;
        output_dir::String="decaying_tg_test",
        save_vtk::Bool=true,
        run_inviscid_baseline::Bool=true,
    )
    mkpath(output_dir)

    # Case quantities the analysis needs, pulled from the config (original local
    # names kept so the diagnostics below read unchanged).
    nel                = cfg.nel
    particles_per_cell = cfg.particles_per_cell
    viscosity          = cfg.viscosity
    T_final            = cfg.T_final
    target_cfl         = cfg.target_cfl

    println("\n========== DECAYING TAYLOR-GREEN VISCOUS VALIDATION ==========")
    @printf("  nel = %s, particles/cell = %d\n", string(nel), particles_per_cell)
    @printf("  ν   = %.6g, T_final = %.3f, target_cfl = %.3f\n",
            viscosity, T_final, target_cfl)

    kx, ky   = 1.0, 1.0
    decay_rate_exact = 2 * viscosity * (kx^2 + ky^2)      # exponent for E and Z
    E_exact_init     = π^2                                # ½∫|u₀|² over 2π×2π
    Z_exact_init     = 2 * π^2                            # ½∫ω₀² (ω₀ = 2 sin x sin y)

    println("\n--- Running viscous case (ν=$(viscosity)) ---")
    cfg_v = cfg
    result_v = run_diagnostic_simulation(cfg_v;
        save_vtk     = save_vtk,
        output_dir   = joinpath(output_dir, "viscous"),
        record_every = 1,
        case_name    = "decaying_tg_viscous",
    )

    result_inv = nothing
    if run_inviscid_baseline
        println("\n--- Running inviscid baseline (ν=0.0) ---")
        cfg_inv = reconfigure(cfg; viscosity = 0.0)
        result_inv = run_diagnostic_simulation(cfg_inv;
            save_vtk     = true,
            output_dir   = joinpath(output_dir, "inviscid"),
            record_every = 1,
            case_name    = "decaying_tg_inviscid",
        )
    end

    ts      = [hi.t           for hi in result_v.history]
    E_num   = [hi.energy      for hi in result_v.history]
    Z_num   = [hi.enstrophy   for hi in result_v.history]
    Γ_num   = [hi.circulation for hi in result_v.history]
    E_exact = [E_num[1] * exp(-decay_rate_exact * t) for t in ts]
    Z_exact = [Z_num[1] * exp(-decay_rate_exact * t) for t in ts]

    E0_err       = abs(E_num[1] - E_exact_init) / E_exact_init
    Z0_err       = abs(Z_num[1] - Z_exact_init) / Z_exact_init
    E_final_err  = abs(E_num[end] - E_exact[end]) / E_exact[end]
    Z_final_err  = abs(Z_num[end] - Z_exact[end]) / Z_exact[end]
    rel_err_E    = [abs(E_num[i] - E_exact[i]) / E_exact[i] for i in eachindex(ts)]
    rel_err_Z    = [abs(Z_num[i] - Z_exact[i]) / Z_exact[i] for i in eachindex(ts)]
    mean_err_E   = sum(rel_err_E) / length(rel_err_E)
    mean_err_Z   = sum(rel_err_Z) / length(rel_err_Z)
    max_err_E    = maximum(rel_err_E)
    max_err_Z    = maximum(rel_err_Z)
    Γ_max_abs    = maximum(abs.(Γ_num))

    # Best-fit decay rate from log-linear regression (slope of log(y) vs t).
    fit_log_slope = function (t, y)
        n     = length(t)
        ly    = log.(max.(y, 1e-300))
        tbar  = sum(t) / n
        ybar  = sum(ly) / n
        num   = sum((t .- tbar) .* (ly .- ybar))
        den   = sum((t .- tbar).^2)
        return num / den
    end
    slope_E_fit  = -fit_log_slope(ts, E_num)
    slope_Z_fit  = -fit_log_slope(ts, Z_num)
    slope_err_E  = abs(slope_E_fit - decay_rate_exact) / max(decay_rate_exact, eps())
    slope_err_Z  = abs(slope_Z_fit - decay_rate_exact) / max(decay_rate_exact, eps())

    inviscid_E_drift = nothing
    inviscid_Z_drift = nothing
    if !isnothing(result_inv)
        E_inv = [hi.energy    for hi in result_inv.history]
        Z_inv = [hi.enstrophy for hi in result_inv.history]
        inviscid_E_drift = abs(E_inv[end] - E_inv[1]) / max(abs(E_inv[1]), eps())
        inviscid_Z_drift = abs(Z_inv[end] - Z_inv[1]) / max(abs(Z_inv[1]), eps())
    end

    println("\n========== VALIDATION REPORT ==========")
    @printf("Analytical decay rate (energy & enstrophy): %.6f  (half-life %.3f)\n",
            decay_rate_exact, decay_rate_exact > 0 ? log(2)/decay_rate_exact : Inf)
    println()
    @printf("Initial energy:      num=%.6e | exact=%.6e | rel err=%.3e\n",
            E_num[1], E_exact_init, E0_err)
    @printf("Initial enstrophy:   num=%.6e | exact=%.6e | rel err=%.3e\n",
            Z_num[1], Z_exact_init, Z0_err)
    println()
    @printf("Final energy:        num=%.6e | exact=%.6e | rel err=%.3e\n",
            E_num[end], E_exact[end], E_final_err)
    @printf("Final enstrophy:     num=%.6e | exact=%.6e | rel err=%.3e\n",
            Z_num[end], Z_exact[end], Z_final_err)
    println()
    @printf("Fit decay rate E:    %.6f  (target %.6f, rel err %.3e)\n",
            slope_E_fit, decay_rate_exact, slope_err_E)
    @printf("Fit decay rate Z:    %.6f  (target %.6f, rel err %.3e)\n",
            slope_Z_fit, decay_rate_exact, slope_err_Z)
    println()
    @printf("Mean rel err E(t):   %.3e   (max %.3e)\n", mean_err_E, max_err_E)
    @printf("Mean rel err Z(t):   %.3e   (max %.3e)\n", mean_err_Z, max_err_Z)
    println()
    if !isnothing(inviscid_E_drift)
        @printf("Inviscid ΔE/E₀:      %.3e   (should be ≪ %.3e from viscosity)\n",
                inviscid_E_drift, 1 - exp(-decay_rate_exact * T_final))
        @printf("Inviscid ΔZ/Z₀:      %.3e\n", inviscid_Z_drift)
    end
    @printf("max |Γ| over run:    %.3e   (should be ≈ 0)\n", Γ_max_abs)

    csv_path = joinpath(output_dir, "decaying_tg_diagnostics.csv")
    open(csv_path, "w") do io
        println(io, "t,E_num,E_exact,Z_num,Z_exact,circulation")
        for i in eachindex(ts)
            @printf(io, "%.10e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                    ts[i], E_num[i], E_exact[i], Z_num[i], Z_exact[i], Γ_num[i])
        end
    end
    println("\nSaved per-step CSV: $csv_path")

    try
        p_E = plot(ts, E_num, label="numerical", lw=2,
                   xlabel="t", ylabel="energy",
                   title="Decaying TG (ν=$(viscosity)): energy")
        plot!(p_E, ts, E_exact, label="analytical exp(-$(round(decay_rate_exact,digits=4))·t)",
              ls=:dash, lw=2)
        if !isnothing(result_inv)
            ts_i = [hi.t for hi in result_inv.history]
            E_i  = [hi.energy for hi in result_inv.history]
            plot!(p_E, ts_i, E_i, label="inviscid (ν=0)", ls=:dot, lw=1.5)
        end

        p_Z = plot(ts, Z_num, label="numerical", lw=2,
                   xlabel="t", ylabel="enstrophy",
                   title="Decaying TG: enstrophy")
        plot!(p_Z, ts, Z_exact, label="analytical", ls=:dash, lw=2)

        p_logE = plot(ts, log.(max.(E_num, 1e-300)), label="log(E_num)", lw=2,
                      xlabel="t", ylabel="log(E)",
                      title="log-energy slope ≈ -$(round(slope_E_fit,digits=4)) (target -$(round(decay_rate_exact,digits=4)))")
        plot!(p_logE, ts, log.(E_exact), label="log(E_exact)", ls=:dash, lw=2)

        layout = plot(p_E, p_Z, p_logE, layout=(3, 1), size=(800, 1000))
        plot_path = joinpath(output_dir, "decaying_tg_decay.png")
        savefig(layout, plot_path)
        println("Saved plot:         $plot_path")
    catch err
        @warn "Plot save failed (continuing)" exception=err
    end

    return (
        history          = result_v.history,
        ts               = ts,
        E_num            = E_num,
        E_exact          = E_exact,
        Z_num            = Z_num,
        Z_exact          = Z_exact,
        circulation      = Γ_num,
        decay_rate_exact = decay_rate_exact,
        decay_rate_fit_E = slope_E_fit,
        decay_rate_fit_Z = slope_Z_fit,
        slope_err_E      = slope_err_E,
        slope_err_Z      = slope_err_Z,
        E0_err           = E0_err,
        Z0_err           = Z0_err,
        E_final_err      = E_final_err,
        Z_final_err      = Z_final_err,
        mean_err_E       = mean_err_E,
        max_err_E        = max_err_E,
        mean_err_Z       = mean_err_Z,
        max_err_Z        = max_err_Z,
        Γ_max_abs        = Γ_max_abs,
        inviscid_E_drift = inviscid_E_drift,
        inviscid_Z_drift = inviscid_Z_drift,
        result_viscous   = result_v,
        result_inviscid  = result_inv,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    with_run_logging() do
        test_decaying_taylor_green()
    end
end
