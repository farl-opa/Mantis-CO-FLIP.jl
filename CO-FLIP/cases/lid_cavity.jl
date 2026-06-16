# Lid-driven cavity (Ghia 1982 benchmark) -- standalone CO-FLIP case.
#
# Self-contained: includes the CO-FLIP engine, defines this case's driver
# (which consumes the full SimulationConfig `CFG` defined at the top of this
# file) plus the Ghia reference data
# and centerline-validation helpers, and runs it when executed directly.
# Supports restart from a saved particle .vtu via the environment (same vars
# the SLURM restart script sets):
#     COFLIP_RESTART_VTU   path to particle .vtu (unset => fresh run)
#     COFLIP_RESTART_STEP  step the .vtu was saved at (default: 0)
#     COFLIP_FINAL_STEP    last step to integrate to  (default: 0)
#     COFLIP_DT            restart dt (unset => auto-set from CFL)
#     julia --project=. CO-FLIP/cases/lid_cavity.jl

include(joinpath(@__DIR__, "..", "CO-FLIP_solver.jl"))

# ============================================================================
# Configuration -- edit here to change the lid-driven cavity case.
# ============================================================================
# Complete simulation configuration: every SimulationConfig field is set
# explicitly so this file fully determines the run on its own. `CFG` is passed
# straight to `run_diagnostic_simulation` by the driver below.
const CFG = SimulationConfig(
    nel                          = (96, 96),
    p                            = (3, 3),
    k                            = (1, 1),
    box_size                     = (1.0, 1.0),
    starting_point               = (0.0, 0.0),
    boundary_condition           = (:wall, :wall, :wall, :wall),
    inlet_U_inf                  = 1.0,
    farfield_velocity            = (0.0, 0.0),
    farfield_corner_pin_cells    = 4,
    obstacle                     = nothing,
    particles_per_cell           = 10,
    stratified_seeding           = true,
    volume_convention            = :physical,
    rng_seed                     = nothing,
    flow_type                    = :zero,
    target_cfl                   = 0.5,
    T_final                      = 30.0,
    viscosity                    = 0.01,
    viscosity_splitting          = :strang,
    viscous_time_scheme          = :backward_euler,
    max_fp_iter                  = 4,
    fp_tol                       = 1e-7,
    enable_energy_correction     = false,
    enable_pressure_kick         = true,
    pic_blend_alpha              = 0.05,
    min_particles_per_element    = 8,
    max_particles_per_element    = 32,
    min_particles_per_quarter    = 1,
    ftle_threshold               = Inf,
    max_longterm_delta_t         = 1.0e9,
    ftle_use_rate                = true,
    global_ftle_gate             = -Inf,
    delayed_reinit_frequency     = 0,
    output_every                 = 50,
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

# Case-specific settings that are NOT part of SimulationConfig (they drive the
# Brinkman lid/wall stamping, steady-state probe, Ghia validation and output
# naming rather than the core solver).
const OUTPUT_DIR      = joinpath(get(ENV, "OUTPUT_DIR", pwd()), "lid_cavity")
const U_LID           = 1.0    # lid tangential speed (Brinkman-stamped each step)
const LID_BAND_CELLS  = 2.0    # Brinkman band thickness at the moving lid (cells)
const WALL_BAND_CELLS = 1.0    # Brinkman band thickness at the static walls (cells)
const SS_RECORD_EVERY = 5      # steady-state probe sampling cadence (steps)
const VALIDATE_GHIA   = true   # run the Ghia (1982) centerline comparison
const CASE_LABEL      = "lid_re100"

"""
Run a lid-driven cavity case using a Brinkman-style tangential
enforcement to mimic no-slip side walls and a moving top lid.

The solver's `:wall` BC pins the wall-normal velocity trace to zero
strongly at the operator level. The tangential component is not
strongly constrained, so we re-stamp it onto the particles each step
via `apply_lid_cavity_brinkman!`. The result is a recognisable lid-
driven cavity flow but **not** strict no-slip — the boundary layer is
resolved over the Brinkman band rather than at the wall exactly. Note
this in the thesis when reporting results.

Default Re = U_lid·L/ν = 1·1/0.01 = 100.
"""

# ----------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982) benchmark — lid-driven cavity centerline velocities
# ----------------------------------------------------------------------------
# Tables I & II, normalised by U_lid. Top-wall lid is +1, all other walls 0.
# Sampled along the vertical centerline (x = L/2, u(y)) and the horizontal
# centerline (y = L/2, v(x)). Only Re=100, 400, 1000 included here — the
# higher-Re columns are available in the paper and can be added later.
const GHIA_Y_OVER_L = (
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
    0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000)
const GHIA_U_RE100 = (
     0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
    -0.20581, -0.13641,  0.00332,  0.23151,  0.68717,  0.73722,  0.78871,  0.84123,  1.00000)
const GHIA_U_RE400 = (
     0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119,
    -0.11477,  0.02135,  0.16256,  0.29093,  0.55892,  0.61756,  0.68439,  0.75837,  1.00000)
const GHIA_U_RE1000 = (
     0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648,
    -0.06080,  0.05702,  0.18719,  0.33304,  0.46604,  0.51117,  0.57492,  0.65928,  1.00000)
const GHIA_X_OVER_L = (
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
    0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000)
const GHIA_V_RE100 = (
    0.00000,  0.09233,  0.10091,  0.10890,  0.12317,  0.16077,  0.17507,  0.17527,
    0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906,  0.00000)
const GHIA_V_RE400 = (
    0.00000,  0.18360,  0.19713,  0.20920,  0.22965,  0.28124,  0.30203,  0.30174,
    0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146,  0.00000)
const GHIA_V_RE1000 = (
    0.00000,  0.27485,  0.29012,  0.30353,  0.32627,  0.37095,  0.33075,  0.32235,
    0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388,  0.00000)

"""Return `(ys, us, xs, vs)` Ghia (1982) benchmark vectors for the given
Reynolds number, or `nothing` when no benchmark is tabulated. `ys`/`xs`
are in units of `L` (cavity side length); `us`/`vs` are in units of
`U_lid`."""
function ghia_reference_data(Re::Real)
    Re_int = Int(round(Re))
    if Re_int == 100
        return (collect(GHIA_Y_OVER_L), collect(GHIA_U_RE100),
                collect(GHIA_X_OVER_L), collect(GHIA_V_RE100))
    elseif Re_int == 400
        return (collect(GHIA_Y_OVER_L), collect(GHIA_U_RE400),
                collect(GHIA_X_OVER_L), collect(GHIA_V_RE400))
    elseif Re_int == 1000
        return (collect(GHIA_Y_OVER_L), collect(GHIA_U_RE1000),
                collect(GHIA_X_OVER_L), collect(GHIA_V_RE1000))
    end
    return nothing
end

"""Sample the *physical* velocity `(u, v)` at point `(x, y)` from an
already-projected `u_phys_form`. Cheaper than `probe_field_at_point`
when the caller has the L2-projected form already (and unlike the raw
R1 probe, this returns the physical-velocity components rather than the
rotated R1-form components)."""
function sample_phys_velocity_at_point(
        u_phys_form, domain::Domain, x::Float64, y::Float64,
    )
    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    dx = Lx / nx;  dy = Ly / ny
    modes = bc_to_position_modes(domain.bc_sides)
    x_w = wrap_axis(Float64(x), Lx, modes[1], 1e-12)
    y_w = wrap_axis(Float64(y), Ly, modes[2], 1e-12)
    ei  = clamp(floor(Int, x_w / dx) + 1, 1, nx)
    ej  = clamp(floor(Int, y_w / dy) + 1, 1, ny)
    eid = (ej - 1) * nx + ei
    xi  = (x_w - (ei - 1) * dx) / dx
    eta = (y_w - (ej - 1) * dy) / dy
    sample_points = Mantis.Points.CartesianPoints(([xi], [eta]))
    pushfwd_eval, _ = Forms.evaluate_sharp_pushforward(u_phys_form, eid, sample_points)
    u = reduce(+, pushfwd_eval[1], dims=2)[1]
    v = reduce(+, pushfwd_eval[2], dims=2)[1]
    return u, v
end

"""Extract `u(y)` along the vertical centerline `x = L/2` and `v(x)` along
the horizontal centerline `y = L/2`, sampled uniformly with `n_samples`
points. Used both for full-resolution centerline CSVs and for Ghia-point
sampling. Returns `(ys, us, xs, vs)`."""
function extract_centerline_profiles(
        u_coeffs::AbstractVector{Float64}, domain::Domain;
        n_samples::Int = 401,
    )
    Lx, Ly = domain.box_size
    x_mid = 0.5 * Lx
    y_mid = 0.5 * Ly

    u_form      = Forms.build_form_field(domain.R1, u_coeffs)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, ★(u_form), domain.dΩ)

    ys = collect(range(0.0, Ly, length=n_samples))
    xs = collect(range(0.0, Lx, length=n_samples))
    us = Vector{Float64}(undef, n_samples)
    vs = Vector{Float64}(undef, n_samples)

    @inbounds for k in 1:n_samples
        u, _ = sample_phys_velocity_at_point(u_phys_form, domain, x_mid, ys[k])
        us[k] = u
    end
    @inbounds for k in 1:n_samples
        _, v = sample_phys_velocity_at_point(u_phys_form, domain, xs[k], y_mid)
        vs[k] = v
    end
    return ys, us, xs, vs
end

"""
Validate a cavity simulation against the Ghia, Ghia, Shin (1982)
benchmark for a matching Reynolds number.

Steps:
  1. Build the L2-projected physical-velocity form from `u_coeffs`.
  2. Sample `u/U_lid` at the 17 Ghia vertical-centerline `y/L` points
     and `v/U_lid` at the 17 horizontal-centerline `x/L` points.
  3. Compute pointwise abs/rel differences and trapezoidal L2 / max
     errors against the tabulated reference.
  4. Also extract dense centerline profiles (`n_dense` samples per axis)
     for full-curve plots alongside the 17 reference points.
  5. Write three CSVs (`*_ghia_u_centerline.csv`, `*_ghia_v_centerline.csv`,
     `*_centerline_{u,v}_dense.csv`) and a `*_ghia_summary.txt` file
     with the summary errors.

Returns `nothing` when no Ghia data exists for the rounded `Re`.
"""
function validate_lid_cavity_against_ghia(
        u_coeffs::AbstractVector{Float64}, domain::Domain;
        Re::Real, U_lid::Float64,
        output_dir::String, case_label::String,
        n_dense::Int = 401,
    )
    ref = ghia_reference_data(Re)
    if ref === nothing
        @printf("  [Ghia] no benchmark tabulated for Re=%.0f; skipping centerline validation.\n", Re)
        return nothing
    end
    ys_g, us_g, xs_g, vs_g = ref
    Lx, Ly = domain.box_size

    u_form      = Forms.build_form_field(domain.R1, u_coeffs)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, ★(u_form), domain.dΩ)

    # FEEC artefact: the dy-form derivative space `D_y` has reduced boundary
    # multiplicity, so physical `u` (which lives in `comp2 = P_x ⊗ D_y`)
    # vanishes structurally at `y = 0` and `y = L_y`. Same for physical `v`
    # at `x = 0` and `x = L_x`. Sampling exactly on the wall therefore
    # returns 0 by construction, which has nothing to do with the lid
    # enforcement. We probe one cell inward so the comparison reflects the
    # solver's behaviour rather than the discretisation's boundary trace.
    nx, ny = domain.nel
    dx_dom = Lx / nx;  dy_dom = Ly / ny
    nudge_into_domain_y(y) = clamp(y, 0.5 * dy_dom, Ly - 0.5 * dy_dom)
    nudge_into_domain_x(x) = clamp(x, 0.5 * dx_dom, Lx - 0.5 * dx_dom)

    us_num = Vector{Float64}(undef, length(ys_g))
    @inbounds for k in eachindex(ys_g)
        y_sample = nudge_into_domain_y(ys_g[k] * Ly)
        u, _ = sample_phys_velocity_at_point(u_phys_form, domain, 0.5*Lx, y_sample)
        us_num[k] = u / U_lid
    end
    vs_num = Vector{Float64}(undef, length(xs_g))
    @inbounds for k in eachindex(xs_g)
        x_sample = nudge_into_domain_x(xs_g[k] * Lx)
        _, v = sample_phys_velocity_at_point(u_phys_form, domain, x_sample, 0.5*Ly)
        vs_num[k] = v / U_lid
    end

    # Build masks that exclude the wall endpoints (y/L=0 and y/L=1 etc.)
    # when computing L2 / max errors — those samples report a structural
    # zero from the FEEC space, not a real solver disagreement.
    interior_y = (ys_g .> 1e-6) .& (ys_g .< 1.0 - 1e-6)
    interior_x = (xs_g .> 1e-6) .& (xs_g .< 1.0 - 1e-6)

    # Trapezoidal L2 norm over the (irregular) reference grid, restricted
    # to the interior indices.
    function trapz_norm_interior(gs, vals, mask)
        s = 0.0
        idxs = findall(mask)
        @inbounds for k in 1:length(idxs)-1
            i, j = idxs[k], idxs[k+1]
            s += 0.5 * (gs[j] - gs[i]) * (vals[i]^2 + vals[j]^2)
        end
        sqrt(s)
    end

    eu       = us_num .- us_g
    ev       = vs_num .- vs_g
    l2_u     = trapz_norm_interior(ys_g, eu, interior_y)
    l2_v     = trapz_norm_interior(xs_g, ev, interior_x)
    den_u    = trapz_norm_interior(ys_g, us_g, interior_y)
    den_v    = trapz_norm_interior(xs_g, vs_g, interior_x)
    rel_l2_u = l2_u / max(den_u, eps())
    rel_l2_v = l2_v / max(den_v, eps())
    max_u    = maximum(abs.(eu[interior_y]))
    max_v    = maximum(abs.(ev[interior_x]))

    ghia_csv_u = joinpath(output_dir, case_label * "_ghia_u_centerline.csv")
    open(ghia_csv_u, "w") do io
        println(io, "y_over_L,u_ghia,u_num,abs_err,rel_err,is_wall_endpoint")
        for k in eachindex(ys_g)
            rel = abs(us_num[k] - us_g[k]) / max(abs(us_g[k]), 1e-12)
            is_wall = !interior_y[k]
            @printf(io, "%.6f,%.6f,%.6f,%.6e,%.6e,%d\n",
                    ys_g[k], us_g[k], us_num[k],
                    abs(us_num[k] - us_g[k]), rel, is_wall ? 1 : 0)
        end
    end
    ghia_csv_v = joinpath(output_dir, case_label * "_ghia_v_centerline.csv")
    open(ghia_csv_v, "w") do io
        println(io, "x_over_L,v_ghia,v_num,abs_err,rel_err,is_wall_endpoint")
        for k in eachindex(xs_g)
            rel = abs(vs_num[k] - vs_g[k]) / max(abs(vs_g[k]), 1e-12)
            is_wall = !interior_x[k]
            @printf(io, "%.6f,%.6f,%.6f,%.6e,%.6e,%d\n",
                    xs_g[k], vs_g[k], vs_num[k],
                    abs(vs_num[k] - vs_g[k]), rel, is_wall ? 1 : 0)
        end
    end

    ys_d, us_d, xs_d, vs_d = extract_centerline_profiles(u_coeffs, domain; n_samples=n_dense)
    dense_u = joinpath(output_dir, case_label * "_centerline_u_dense.csv")
    open(dense_u, "w") do io
        println(io, "y,u_phys,u_over_U_lid")
        for k in eachindex(ys_d)
            @printf(io, "%.10e,%.10e,%.10e\n", ys_d[k], us_d[k], us_d[k] / U_lid)
        end
    end
    dense_v = joinpath(output_dir, case_label * "_centerline_v_dense.csv")
    open(dense_v, "w") do io
        println(io, "x,v_phys,v_over_U_lid")
        for k in eachindex(xs_d)
            @printf(io, "%.10e,%.10e,%.10e\n", xs_d[k], vs_d[k], vs_d[k] / U_lid)
        end
    end

    summary_path = joinpath(output_dir, case_label * "_ghia_summary.txt")
    open(summary_path, "w") do io
        println(io, @sprintf("Ghia (1982) benchmark validation for cavity Re=%.0f", Re))
        println(io, "=" ^ 60)
        println(io, "Errors are reported over INTERIOR Ghia points only.")
        println(io, "The wall endpoints (y/L=0, y/L=1, x/L=0, x/L=1) are excluded")
        println(io, "because the FEEC dy-form derivative space `D_y` has reduced")
        println(io, "boundary multiplicity → physical u/v vanishes at y/x walls")
        println(io, "by construction, regardless of the lid Brinkman stamping.")
        println(io, "Endpoints are still listed in the per-point CSV with the")
        println(io, "`is_wall_endpoint=1` flag for completeness.")
        println(io, "")
        println(io, "u along vertical centerline (x = L/2):")
        println(io, @sprintf("  L2 error (relative) : %.4e", rel_l2_u))
        println(io, @sprintf("  L2 error (absolute) : %.4e", l2_u))
        println(io, @sprintf("  max pointwise error : %.4e", max_u))
        println(io, "")
        println(io, "v along horizontal centerline (y = L/2):")
        println(io, @sprintf("  L2 error (relative) : %.4e", rel_l2_v))
        println(io, @sprintf("  L2 error (absolute) : %.4e", l2_v))
        println(io, @sprintf("  max pointwise error : %.4e", max_v))
    end

    println("")
    @printf("  [Ghia Re=%.0f]  u centerline: rel_L2=%.3e  max=%.3e\n", Re, rel_l2_u, max_u)
    @printf("  [Ghia Re=%.0f]  v centerline: rel_L2=%.3e  max=%.3e\n", Re, rel_l2_v, max_v)
    println("  Saved: $ghia_csv_u")
    println("  Saved: $ghia_csv_v")
    println("  Saved: $dense_u")
    println("  Saved: $dense_v")
    println("  Saved: $summary_path")

    return (rel_l2_u=rel_l2_u, rel_l2_v=rel_l2_v, max_u=max_u, max_v=max_v,
            ys_g=ys_g, us_g=us_g, us_num=us_num,
            xs_g=xs_g, vs_g=vs_g, vs_num=vs_num,
            ghia_csv_u=ghia_csv_u, ghia_csv_v=ghia_csv_v,
            dense_u=dense_u, dense_v=dense_v, summary_path=summary_path)
end

function test_lid_driven_cavity(cfg::SimulationConfig = CFG;
        output_dir::String       = OUTPUT_DIR,
        U_lid::Float64           = U_LID,
        lid_band_cells::Float64  = LID_BAND_CELLS,
        wall_band_cells::Float64 = WALL_BAND_CELLS,
        ss_record_every::Int     = SS_RECORD_EVERY,
        validate_ghia::Bool      = VALIDATE_GHIA,
        case_label::String       = CASE_LABEL,
        # Restart support. When `restart_vtu` is set, particles are loaded from
        # that .vtu instead of generated fresh, the run continues from
        # `restart_step`+1 up to `final_step` (a fixed step count, so adaptive
        # CFL shrinks dt but never changes how many steps run). `restart_dt`
        # is the starting dt; leave it `nothing` to auto-set it from the CFL
        # condition of the loaded state. The Ghia validation runs the same as
        # a fresh case.
        restart_vtu::Union{String,Nothing} = nothing,
        restart_step::Int                  = 0,
        final_step::Union{Int,Nothing}     = nothing,
        restart_dt::Union{Float64,Nothing} = nothing,
    )
    mkpath(output_dir)
    Re = U_lid * cfg.box_size[1] / cfg.viscosity

    is_restart = restart_vtu !== nothing
    if is_restart
        final_step !== nothing && final_step > restart_step ||
            error("test_lid_driven_cavity: restart requires final_step > restart_step " *
                  "(got restart_step=$(restart_step), final_step=$(final_step))")
        restart_dt === nothing || (isfinite(restart_dt) && restart_dt > 0) ||
            error("test_lid_driven_cavity: restart_dt must be a finite positive Float64 when provided")
    end
    n_steps_override = is_restart ? (final_step - restart_step) : nothing

    println("\n========== LID DRIVEN CAVITY ==========")
    println(@sprintf("  nel=%s  L=%.3f  U_lid=%.3f  ν=%.4f  Re=%.1f  T=%.1f",
                     cfg.nel, cfg.box_size[1], U_lid, cfg.viscosity, Re, cfg.T_final))
    if is_restart
        println(@sprintf("  RESTART from %s: step %d → %d (%d steps), dt0=%s",
                         restart_vtu, restart_step, final_step, n_steps_override,
                         restart_dt === nothing ? "auto-CFL" : @sprintf("%.4g", restart_dt)))
    end
    println(@sprintf("  Brinkman bands:  lid=%.2f cells  walls=%.2f cells",
                     lid_band_cells, wall_band_cells))
    println(@sprintf("  energy_correction=%s  ftle_threshold=%s  pic_blend=%.3f",
                     cfg.enable_energy_correction, cfg.ftle_threshold, cfg.pic_blend_alpha))

    step_hook = function (particles, domain, _cfg, step, t)
        apply_lid_cavity_brinkman!(particles, domain, U_lid;
                                   lid_band_cells=lid_band_cells,
                                   wall_band_cells=wall_band_cells)
    end

    # Steady-state convergence tracking. probe_callback receives u_coeffs every
    # step; we sample every `ss_record_every` steps and record the
    # mass-weighted L2 norm of (u_h - u_h_prev). When this plateaus near
    # round-off the cavity has reached steady state.
    ss_records  = NamedTuple[]
    u_prev_ref  = Ref{Vector{Float64}}(Float64[])
    ss_probe_cb = function (t, u_coeffs, domain, step)
        step == 0 && return  # first call is the initial state — no diff yet
        step % max(1, ss_record_every) == 0 || return
        u_prev = u_prev_ref[]
        if length(u_prev) == length(u_coeffs)
            diff       = u_coeffs .- u_prev
            Mdiff      = domain.Mass_matrix * diff
            Mu         = domain.Mass_matrix * u_coeffs
            norm_diff  = sqrt(max(dot(diff,    Mdiff), 0.0))
            norm_u     = sqrt(max(dot(u_coeffs, Mu),    0.0))
            rel_diff   = norm_diff / max(norm_u, eps())
            push!(ss_records, (t=t, step=step, abs_diff=norm_diff,
                               norm_u=norm_u, rel_diff=rel_diff))
            @printf("  [SS] t=%.3f  ||Δu||_M=%.3e  ||u||_M=%.3e  rel=%.3e\n",
                    t, norm_diff, norm_u, rel_diff)
        end
        u_prev_ref[] = copy(u_coeffs)
    end

    result = run_diagnostic_simulation(cfg;
        save_vtk         = true,
        save_vtk_every   = cfg.output_every,
        output_dir       = joinpath(output_dir, case_label * "_vtk"),
        record_every     = max(1, cfg.output_every),
        case_name        = case_label,
        step_hook        = step_hook,
        probe_callback   = ss_probe_cb,
        save_particles   = true,
        restart_vtu      = restart_vtu,
        restart_step     = restart_step,
        n_steps_override = n_steps_override,
        initial_dt       = is_restart ? restart_dt : nothing,
    )

    history_csv = joinpath(output_dir, case_label * "_history.csv")
    open(history_csv, "w") do io
        println(io, "t,energy,enstrophy,circulation")
        for hi in result.history
            @printf(io, "%.10e,%.16e,%.16e,%.16e\n",
                    hi.t, hi.energy, hi.enstrophy, hi.circulation)
        end
    end
    println("Saved history CSV: $history_csv")

    ss_csv = joinpath(output_dir, case_label * "_steady_state.csv")
    open(ss_csv, "w") do io
        println(io, "t,step,abs_diff_L2,norm_u_L2,rel_diff")
        for r in ss_records
            @printf(io, "%.10e,%d,%.16e,%.16e,%.16e\n",
                    r.t, r.step, r.abs_diff, r.norm_u, r.rel_diff)
        end
    end
    println("Saved steady-state CSV: $ss_csv")

    ghia = nothing
    if validate_ghia
        ghia = validate_lid_cavity_against_ghia(
            result.u_final, result.domain;
            Re=Re, U_lid=U_lid,
            output_dir=output_dir, case_label=case_label,
        )
    end

    return (result=result, history_csv=history_csv, ss_csv=ss_csv,
            ss_records=ss_records, ghia=ghia)
end

if abspath(PROGRAM_FILE) == @__FILE__
    with_run_logging() do
        vtu = get(ENV, "COFLIP_RESTART_VTU", "")
        if vtu == ""
            test_lid_driven_cavity()
        else
            rstep = parse(Int, get(ENV, "COFLIP_RESTART_STEP", "0"))
            fstep = parse(Int, get(ENV, "COFLIP_FINAL_STEP", "0"))
            rdt   = haskey(ENV, "COFLIP_DT") ? parse(Float64, ENV["COFLIP_DT"]) : nothing
            test_lid_driven_cavity(;
                restart_vtu  = vtu,
                restart_step = rstep,
                final_step   = fstep,
                restart_dt   = rdt,
            )
        end
    end
end
