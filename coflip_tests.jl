using Test
using Random
using LinearAlgebra
using Statistics
using DelimitedFiles

# Load CO-FLIP operators without executing the simulation main loop.
include(joinpath(@__DIR__, "CO-FLIP", "CO-FLIP_periodic.jl"))

function tg_proxy_analytical_form(dom::Domain)
    lx, ly = dom.box_size
    wx = 2.0 * pi / lx
    wy = 2.0 * pi / ly

    function proxy_solution(x::Matrix{Float64})
        xs = @view x[:, 1]
        ys = @view x[:, 2]

        u = @. sin(wx * xs) * cos(wy * ys)
        v = @. -cos(wx * xs) * sin(wy * ys)

        # Internal proxy ordering in this CO-FLIP script is [v, -u].
        return [v, -u]
    end

    return Forms.AnalyticalFormField(1, proxy_solution, dom.geo, "tg_proxy")
end

function physical_velocity_from_proxy(proxy_vel)
    # Convert proxy [v, -u] back to physical [u, v].
    return -proxy_vel[2], proxy_vel[1]
end

function make_uniform_probes(dom::Domain, n_per_dim::Int)
    lx, ly = dom.box_size
    # Avoid including periodic endpoint to prevent duplicate wrapped points.
    xs = range(0.0, lx, length=n_per_dim + 1)[1:end-1]
    ys = range(0.0, ly, length=n_per_dim + 1)[1:end-1]

    px = [x for y in ys, x in xs] |> vec
    py = [y for y in ys, x in xs] |> vec
    n = length(px)

    probes = Particles(
        px,
        py,
        zeros(n),
        zeros(n),
        ones(n),
        zeros(n),
        zeros(n),
        zeros(Int, prod(dom.nel)),
        zeros(Int, n),
        zeros(Int, n),
    )
    particle_sorter!(probes, dom)
    return probes
end

function l2_probe_error_against_tg(dom::Domain, coeffs::AbstractVector{T}; n_probe::Int=96) where {T<:Real}
    probes = make_uniform_probes(dom, n_probe)
    cache = EvaluationCache(evaluation_cache_size(dom))

    lx, ly = dom.box_size
    wx = 2.0 * pi / lx
    wy = 2.0 * pi / ly

    err2 = 0.0
    for i in eachindex(probes.x)
        proxy_vel, _ = probe_field_at_point(probes.x[i], probes.y[i], coeffs, dom, cache)
        u_num, v_num = physical_velocity_from_proxy(proxy_vel)

        u_ex = sin(wx * probes.x[i]) * cos(wy * probes.y[i])
        v_ex = -cos(wx * probes.x[i]) * sin(wy * probes.y[i])

        du = u_num - u_ex
        dv = v_num - v_ex
        err2 += du * du + dv * dv
    end

    return sqrt(err2 / length(probes.x))
end

function seed_particles_from_grid!(p::Particles, dom::Domain, coeffs::AbstractVector{T}) where {T<:Real}
    cache = EvaluationCache(evaluation_cache_size(dom))

    for pid in eachindex(p.x)
        proxy_vel, _ = probe_field_at_point(p.x[pid], p.y[pid], coeffs, dom, cache)
        u, v = physical_velocity_from_proxy(proxy_vel)
        p.mx[pid] = u
        p.my[pid] = v
    end

    return p
end

function p2g_rhs_from_particles(p::Particles)
    n = length(p.x)
    rhs = Vector{Float64}(undef, 2 * n)
    @inbounds for i in 1:n
        w = sqrt(p.volume[i])
        rhs[2 * i - 1] = p.my[i] * w
        rhs[2 * i] = -p.mx[i] * w
    end
    return rhs
end

function p2g_residual_diagnostics(B, rec_coeffs::AbstractVector{T}, rhs::AbstractVector{R}) where {T<:Real, R<:Real}
    rhs_norm = max(norm(rhs), eps(Float64))
    resid = B * rec_coeffs - rhs
    rel_resid = norm(resid) / rhs_norm

    normal_rhs = transpose(B) * rhs
    normal_rhs_norm = max(norm(normal_rhs), eps(Float64))
    normal_resid = transpose(B) * resid
    rel_normal_resid = norm(normal_resid) / normal_rhs_norm

    return rel_resid, rel_normal_resid
end

function run_advection_operator_test(; nel=(16, 16), p=(2, 2), k=(1, 1), n_particles=2000, dt=0.1)
    dom = GenerateDomain(nel, p, k)
    particles = generate_particles(n_particles, dom, :tg)
    particle_sorter!(particles, dom)

    x0 = copy(particles.x)
    y0 = copy(particles.y)
    mx0 = copy(particles.mx)
    my0 = copy(particles.my)

    x_out = copy(x0)
    y_out = copy(y0)
    mx_out = copy(mx0)
    my_out = copy(my0)

    n_dofs = FunctionSpaces.get_num_basis(dom.R1.fem_space)
    u_zero = zeros(Float64, n_dofs)
    thread_caches = [EvaluationCache(evaluation_cache_size(dom)) for _ in 1:Threads.maxthreadid()]

    adv_err = advect_particles_pullback_rk2!(
        particles,
        x0,
        y0,
        mx0,
        my0,
        u_zero,
        dom,
        dt,
        x_out,
        y_out,
        mx_out,
        my_out,
        thread_caches,
    )

    pos_err = maximum(abs.(x_out .- x0) .+ abs.(y_out .- y0))
    imp_err = maximum(abs.(mx_out .- mx0) .+ abs.(my_out .- my0))

    return [dt, adv_err, pos_err, imp_err]
end

function run_energy_correction_operator_test(; n=128)
    Random.seed!(20260325)
    f_original = randn(n)
    f_from_particles = randn(n)
    f_midpoint = randn(n)
    f_next = similar(f_original)

    star_apply = v -> copy(v)

    apply_energy_correction!(
        f_next,
        f_original,
        f_from_particles,
        f_midpoint;
        star_apply=star_apply,
        tol=1e-12,
        project_non_orthogonal=true,
    )

    corrected_diff = f_next .- f_original
    orthogonality = abs(dot(corrected_diff, star_apply(f_midpoint)))
    correction_norm = norm(corrected_diff)

    f_next_no_proj = similar(f_original)
    apply_energy_correction!(
        f_next_no_proj,
        f_original,
        f_from_particles,
        f_midpoint;
        star_apply=star_apply,
        tol=1e-12,
        project_non_orthogonal=false,
    )

    no_proj_err = norm(f_next_no_proj .- f_from_particles)

    return [orthogonality, correction_norm, no_proj_err]
end

function run_pressure_correction_operator_test(; nel=(16, 16), p=(2, 2), k=(1, 1), n_particles=2000)
    dom = GenerateDomain(nel, p, k)
    particles = generate_particles(n_particles, dom, :tg)
    particle_sorter!(particles, dom)

    tg_proxy = tg_proxy_analytical_form(dom)
    ref_form = Assemblers.solve_L2_projection(dom.R1, tg_proxy, dom.dΩ)

    n_dofs = length(ref_form.coefficients)
    f_pre = zeros(Float64, n_dofs)
    f_proj = copy(ref_form.coefficients)

    mx_before = copy(particles.mx)
    my_before = copy(particles.my)

    thread_caches = [EvaluationCache(evaluation_cache_size(dom)) for _ in 1:Threads.maxthreadid()]
    tau = apply_pressure_correction!(particles, f_proj, f_pre, dom, thread_caches)

    tau_err = norm(tau .- (f_proj .- f_pre))

    cache = EvaluationCache(evaluation_cache_size(dom))
    max_update_err = 0.0
    for i in eachindex(particles.x)
        val, _ = probe_field_at_point(particles.x[i], particles.y[i], tau, dom, cache)
        expected_mx = mx_before[i] - val[2]
        expected_my = my_before[i] + val[1]
        local_err = abs(particles.mx[i] - expected_mx) + abs(particles.my[i] - expected_my)
        if local_err > max_update_err
            max_update_err = local_err
        end
    end

    particles_unchanged = generate_particles(n_particles, dom, :tg)
    particle_sorter!(particles_unchanged, dom)
    mx_before_zero = copy(particles_unchanged.mx)
    my_before_zero = copy(particles_unchanged.my)
    tau_zero = apply_pressure_correction!(
        particles_unchanged,
        f_pre,
        f_pre,
        dom,
        thread_caches,
    )
    zero_tau_norm = norm(tau_zero)
    no_change_err = maximum(abs.(particles_unchanged.mx .- mx_before_zero) .+ abs.(particles_unchanged.my .- my_before_zero))

    return [tau_err, max_update_err, zero_tau_norm, no_change_err]
end

function convergence_rates(errors::AbstractVector{<:Real})
    rates = fill(NaN, max(length(errors) - 1, 0))
    for i in 1:length(rates)
        rates[i] = log(errors[i] / errors[i + 1]) / log(2.0)
    end
    return rates
end

function run_projection_convergence(; levels=((8, 8), (16, 16), (32, 32), (64, 64)), p=(2, 2), k=(1, 1))
    rows = Vector{Vector{Float64}}()

    for nel in levels
        dom = GenerateDomain(nel, p, k)

        tg_proxy = tg_proxy_analytical_form(dom)
        ref_form = Assemblers.solve_L2_projection(dom.R1, tg_proxy, dom.dΩ)

        proj_coeffs, _ = project_and_get_pressure(dom, ref_form)
        proj_form = Forms.build_form_field(dom.R1, proj_coeffs)

        # Divergence defect after projection should be close to solver tolerance.
        div_l2 = Analysis.L2_norm(Forms.d(proj_form), dom.dΩ)
        vel_l2 = l2_probe_error_against_tg(dom, proj_coeffs)

        push!(rows, [nel[1], nel[2], div_l2, vel_l2])
    end

    vel_errors = [r[4] for r in rows]
    rates = convergence_rates(vel_errors)

    return rows, rates
end

function run_p2g_convergence_mesh(; levels=((8, 8), (16, 16), (32, 32), (64, 64)), p=(2, 2), k=(1, 1), ppc=20)
    rows = Vector{Vector{Float64}}()

    for nel in levels
        dom = GenerateDomain(nel, p, k)

        tg_proxy = tg_proxy_analytical_form(dom)
        ref_form = Assemblers.solve_L2_projection(dom.R1, tg_proxy, dom.dΩ)
        ref_coeffs = ref_form.coefficients

        nparticles = nel[1] * nel[2] * ppc
        particles = generate_particles(nparticles, dom, :tg)
        particle_sorter!(particles, dom)
        seed_particles_from_grid!(particles, dom, ref_coeffs)

        b = build_B_matrix(particles, dom)
        rec_coeffs = solve_grid_velocity_lsqr(b, particles)
        rhs = p2g_rhs_from_particles(particles)

        coeff_err = norm(rec_coeffs - ref_coeffs) / max(norm(ref_coeffs), eps(Float64))
        vel_err = l2_probe_error_against_tg(dom, rec_coeffs)
        sampled_ref_err = norm(b * (rec_coeffs - ref_coeffs)) / max(norm(b * ref_coeffs), eps(Float64))
        rel_resid, rel_normal_resid = p2g_residual_diagnostics(b, rec_coeffs, rhs)

        push!(rows, [nel[1], nel[2], nparticles, coeff_err, vel_err, sampled_ref_err, rel_resid, rel_normal_resid])
    end

    vel_errors = [r[5] for r in rows]
    rates = convergence_rates(vel_errors)

    return rows, rates
end

function run_p2g_convergence_particles(; nel=(32, 32), p=(2, 2), k=(1, 1), ppc_levels=(5, 10, 20, 40, 80), n_trials=5)
    rows = Vector{Vector{Float64}}()

    dom = GenerateDomain(nel, p, k)
    tg_proxy = tg_proxy_analytical_form(dom)
    ref_form = Assemblers.solve_L2_projection(dom.R1, tg_proxy, dom.dΩ)
    ref_coeffs = ref_form.coefficients

    for ppc in ppc_levels
        nparticles = nel[1] * nel[2] * ppc
        coeff_errs = zeros(Float64, n_trials)
        vel_errs = zeros(Float64, n_trials)
        sampled_ref_errs = zeros(Float64, n_trials)
        rel_resids = zeros(Float64, n_trials)
        rel_normal_resids = zeros(Float64, n_trials)

        for trial in 1:n_trials
            Random.seed!(10_000 + 101 * ppc + trial)
            particles = generate_particles(nparticles, dom, :tg)
            particle_sorter!(particles, dom)
            seed_particles_from_grid!(particles, dom, ref_coeffs)

            b = build_B_matrix(particles, dom)
            rec_coeffs = solve_grid_velocity_lsqr(b, particles)
            rhs = p2g_rhs_from_particles(particles)

            coeff_errs[trial] = norm(rec_coeffs - ref_coeffs) / max(norm(ref_coeffs), eps(Float64))
            vel_errs[trial] = l2_probe_error_against_tg(dom, rec_coeffs)
            sampled_ref_errs[trial] = norm(b * (rec_coeffs - ref_coeffs)) / max(norm(b * ref_coeffs), eps(Float64))
            rel_resids[trial], rel_normal_resids[trial] = p2g_residual_diagnostics(b, rec_coeffs, rhs)
        end

        push!(
            rows,
            [
                ppc,
                nparticles,
                mean(coeff_errs),
                mean(vel_errs),
                mean(sampled_ref_errs),
                mean(rel_resids),
                mean(rel_normal_resids),
                std(coeff_errs),
                std(vel_errs),
                std(sampled_ref_errs),
                std(rel_resids),
                std(rel_normal_resids),
            ],
        )
    end

    vel_errors = [r[4] for r in rows]
    rates = convergence_rates(vel_errors)

    return rows, rates
end

function write_table(path::String, header::String, rows::Vector{Vector{Float64}}, rates::Vector{Float64}; rate_label::String)
    open(path, "w") do io
        println(io, header)
        for i in eachindex(rows)
            row = rows[i]
            if i <= length(rates)
                println(io, join(vcat(row, rates[i]), " "))
            else
                println(io, join(vcat(row, NaN), " "))
            end
        end
        println(io, "# last_column = $(rate_label)")
    end
end

function write_single_row(path::String, header::String, row::Vector{Float64})
    open(path, "w") do io
        println(io, header)
        println(io, join(row, " "))
    end
end

function write_rows(path::String, header::String, rows::Vector{Vector{Float64}})
    open(path, "w") do io
        println(io, header)
        for row in rows
            println(io, join(row, " "))
        end
    end
end

function max_advective_rate(p::Particles, d::Domain)
    nx, ny = d.nel
    lx, ly = d.box_size
    dx = lx / nx
    dy = ly / ny

    rate = 0.0
    @inbounds for i in eachindex(p.mx)
        local_rate = abs(p.mx[i]) / dx + abs(p.my[i]) / dy
        if local_rate > rate
            rate = local_rate
        end
    end
    return rate
end

function run_long_horizon_stability_case(
    label::String;
    nel=(32, 32),
    p=(2, 2),
    k=(1, 1),
    flow::Symbol=:convecting,
    ppc::Int=20,
    n_steps::Int=120,
    dt::Float64=0.003,
    adaptive_dt::Bool=true,
    target_cfl::Float64=0.5,
    min_particles_per_element::Int=10,
    max_particles_per_element::Int=25,
    min_particles_per_quarter::Int=3,
    speed_blowup_threshold::Float64=200.0,
)
    dom = GenerateDomain(nel, p, k)
    num_particles = nel[1] * nel[2] * ppc
    particles = generate_particles(num_particles, dom, flow)
    particle_sorter!(particles, dom)

    u_coeffs = coadjoint_step!(particles, dom)
    t = 0.0
    blew_up = 0.0

    rows = Vector{Vector{Float64}}()

    diag0 = compute_conservation_diagnostics(u_coeffs, dom)
    energy0 = abs(diag0.energy) > eps(Float64) ? diag0.energy : 1.0

    for step in 1:n_steps
        if adaptive_dt
            dt_cfl = compute_cfl_dt(particles, dom, target_cfl)
            dt_eff = isfinite(dt_cfl) ? min(dt, dt_cfl) : dt
        else
            dt_eff = dt
        end

        u_coeffs = step_co_flip!(
            particles,
            dom,
            dt_eff,
            u_coeffs;
            min_particles_per_element=min_particles_per_element,
            max_particles_per_element=max_particles_per_element,
            min_particles_per_quarter=min_particles_per_quarter,
        )
        t += dt_eff

        particle_sorter!(particles, dom)
        b = build_B_matrix(particles, dom)
        u_raw = solve_grid_velocity_lsqr(b, particles)
        rhs = p2g_rhs_from_particles(particles)
        rel_resid, rel_normal_resid = p2g_residual_diagnostics(b, u_raw, rhs)

        diag = compute_conservation_diagnostics(u_coeffs, dom)
        rel_energy_drift = abs(diag.energy - diag0.energy) / abs(energy0)

        max_speed = maximum(sqrt.(particles.mx .^ 2 .+ particles.my .^ 2))
        adv_rate = max_advective_rate(particles, dom)
        cfl = dt_eff * adv_rate

        push!(
            rows,
            [
                step,
                t,
                dt_eff,
                diag.energy,
                diag.circulation,
                diag.enstrophy,
                rel_energy_drift,
                rel_resid,
                rel_normal_resid,
                cfl,
                max_speed,
                length(particles.x),
            ],
        )

        if !all(isfinite, u_coeffs) || !isfinite(max_speed) || max_speed > speed_blowup_threshold
            blew_up = 1.0
            break
        end
    end

    out_path = joinpath(@__DIR__, "tests_output", "coflip_longrun_$(label).txt")
    write_rows(
        out_path,
        "# step time dt energy circulation enstrophy rel_energy_drift rel_resid rel_normal_resid cfl max_speed n_particles",
        rows,
    )

    last_row = rows[end]
    summary = [
        length(rows),
        blew_up,
        last_row[7],
        last_row[8],
        last_row[9],
        last_row[10],
        last_row[11],
        last_row[12],
    ]
    return summary
end

function run_long_horizon_stability_suite()
    baseline = run_long_horizon_stability_case(
        "baseline";
        min_particles_per_element=10,
        max_particles_per_element=25,
        min_particles_per_quarter=3,
    )

    no_reseed = run_long_horizon_stability_case(
        "no_reseed";
        min_particles_per_element=0,
        max_particles_per_element=0,
        min_particles_per_quarter=0,
    )

    return baseline, no_reseed
end

function main()
    Random.seed!(12345)
    mkpath(joinpath(@__DIR__, "tests_output"))

    baseline_summary, no_reseed_summary = run_long_horizon_stability_suite()

    write_rows(
        joinpath(@__DIR__, "tests_output", "coflip_longrun_summary.txt"),
        "# case steps_done blew_up rel_energy_drift rel_resid rel_normal_resid cfl max_speed n_particles",
        [
            vcat([1.0], baseline_summary),
            vcat([2.0], no_reseed_summary),
        ],
    )

    @test baseline_summary[2] == 0.0
    @test no_reseed_summary[2] == 0.0
    @test isfinite(baseline_summary[4])
    @test isfinite(no_reseed_summary[4])

    println("Long-horizon stability diagnostics completed.")
    println("Wrote: tests_output/coflip_longrun_baseline.txt")
    println("Wrote: tests_output/coflip_longrun_no_reseed.txt")
    println("Wrote: tests_output/coflip_longrun_summary.txt")
end

main()
