# P2G (particle-to-grid) Taylor-Green spatial convergence -- standalone case.
#
# Self-contained: includes the CO-FLIP engine + shared analysis helpers and
# defines this case's driver. This case builds its domain/particles directly
# (it tests the P2G least-squares reconstruction at t=0, not a full run), so
# it does not use SimulationConfig. Run directly:
#     julia --project=. CO-FLIP/cases/p2g_convergence.jl

include(joinpath(@__DIR__, "..", "CO-FLIP_solver.jl"))
isdefined(Main, :compute_decaying_tg_l2_error) || include(joinpath(@__DIR__, "case_common.jl"))

"""
Spatial convergence study of the P2G (particle-to-grid) least-squares
reconstruction at the **initial step**, on the periodic Taylor-Green
vortex.

For each `(p_order, nel)` pair this routine:
  1. Builds a periodic 2π×2π domain with `nel × nel` elements and B-spline
     order `p_order` (continuity k=1).
  2. Seeds particles via `seeding_mode`. Default is `:gauss_legendre`
     (tensor-product GL nodes per element with weights set as particle
     volumes), which turns the LSQR fit into an exact polynomial
     quadrature so the discretisation error is dominated by the spline
     approximation rather than particle-quadrature noise.
  3. Assembles the P2G matrix `B` (rows = 2·num_particles) and solves the
     least-squares system `min ‖B·u_grid − m_p‖₂` via LSQR.
  4. Measures the L2 error via `compute_decaying_tg_l2_error` (at t=0,
     ν=0), which uses Mantis's `Analysis.L2_norm` on the difference
     `u_phys_form − u_TG_analytical` over a high-order Gauss–Legendre
     quadrature.

The least-squares fit alone is tested — no Hodge projection / pressure
solve — so the resulting curve isolates the P2G reconstruction error.

For the rotated R1 storage on tensor-product B-splines of order `p`, the
mixed-degree components `(p-1, p)` and `(p, p-1)` make the L2-best-
approximation rate of a smooth physical-velocity field `O(h^p)`. The
reference slope drawn in the plot is `h^p`.

Returns the list of per-run `(p, nel, h, l2_err, rel_l2_err, lsqr_resid_rel)`
NamedTuples.
"""
function test_p2g_taylor_green_convergence(;
        nel_sweep::NTuple{N,Int}=(32, 48, 64, 96, 128),
        p_orders::NTuple{M,Int}=(2, 3),
        particles_per_cell::Int=16,
        seeding_mode::Symbol=:gauss_legendre,
        rng_seed::Union{Int,Nothing}=42,
        nq_extra::Int=2,
        lsqr_atol::Float64=1e-12,
        lsqr_btol::Float64=1e-12,
        lsqr_maxiter::Int=5000,
        output_dir::String=joinpath(get(ENV, "OUTPUT_DIR", pwd()), "p2g_tg_convergence"),
        save_plot::Bool=true,
    ) where {N,M}
    mkpath(output_dir)

    box_size = (2π, 2π)
    results  = NamedTuple[]

    println("\n========== P2G TAYLOR-GREEN CONVERGENCE ==========")
    @printf("  box=%s  ppc=%d  seeding_mode=%s  rng_seed=%s  nq_extra=%d\n",
            string(box_size), particles_per_cell, seeding_mode,
            string(rng_seed), nq_extra)

    for p_order in p_orders, n in nel_sweep
        nel = (n, n)
        p   = (p_order, p_order)
        k   = (1, 1)

        println(@sprintf("\n--- p=%d  nel=%d ---", p_order, n))

        domain = GenerateDomain(nel, p, k;
                                box_size=box_size,
                                starting_point=(0.0, 0.0),
                                boundary_condition=:periodic,
                                obstacle=nothing)

        num_particles = nel[1] * nel[2] * particles_per_cell
        particles = generate_particles(num_particles, domain, :tg;
                                       stratified_seeding=true,
                                       seeding_mode=seeding_mode,
                                       rng_seed=rng_seed,
                                       volume_convention=:physical,
                                       boundary_condition=:periodic,
                                       U_inf=0.0)
        # GL seeding may round the actual count down; use the resulting array.
        num_particles = length(particles.x)
        particle_sorter!(particles, domain)

        ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
        num_elements = prod(domain.nel)
        buf          = SimulationBuffers(num_particles, ndofs, num_elements, domain)

        t0 = time()
        B  = build_B_matrix(particles, domain, buf)
        t_build_B = time() - t0

        t0 = time()
        u_grid, _ = solve_grid_velocity_lsqr(B, particles, buf.V_p, buf.lsqr_warm,
                                          buf.ne_rhs_buf, buf.col_norms_buf,
                                          domain.Mass1_fact;
                                          atol=lsqr_atol, btol=lsqr_btol,
                                          maxiter=lsqr_maxiter,
                                          error_on_nonconvergence=true)
        t_solve = time() - t0

        # LSQR residual on the actual fit problem — quantifies whether the
        # least-squares optimum was reached. Build RHS afresh because the
        # LSQR call above scaled B in place via column-Jacobi preconditioner,
        # which leaves B with unit-norm columns rather than the original B.
        m_rhs = similar(buf.V_p, 2 * length(particles.x))
        build_lsqr_rhs!(m_rhs, particles)
        B_fresh = build_B_matrix(particles, domain, buf)
        residual = B_fresh * u_grid .- m_rhs
        lsqr_resid_rel = norm(residual) / max(norm(m_rhs), eps())

        err = compute_decaying_tg_l2_error(domain, u_grid, 0.0, 0.0;
                                           nq_extra=nq_extra)

        h   = box_size[1] / n
        rec = (p=p_order, nel=n, h=h,
               l2_err=err.l2_err,
               rel_l2_err=err.rel_l2_err,
               l2_norm_exact=err.l2_norm_exact,
               lsqr_resid_rel=lsqr_resid_rel,
               num_particles=num_particles,
               t_build_B=t_build_B, t_solve=t_solve)
        push!(results, rec)
        @printf("  nel=%d h=%.5f  L2=%.4e  rel_L2=%.4e  LSQR_resid_rel=%.2e  (build_B=%.2fs solve=%.2fs)\n",
                n, h, err.l2_err, err.rel_l2_err,
                lsqr_resid_rel, t_build_B, t_solve)
    end

    println("\n========== OBSERVED CONVERGENCE ORDERS ==========")
    for p_order in p_orders
        subset = filter(r -> r.p == p_order, results)
        length(subset) >= 2 || continue
        println(@sprintf("\n  p=%d  (expected order %d)", p_order, p_order))
        @printf("  %-6s %-10s  %-12s %-8s %-8s  %-10s\n",
                "nel", "h", "L2_err", "ratio", "order", "lsqr_res")
        prev = nothing
        for r in subset
            if prev === nothing
                @printf("  %-6d %-10.5e  %-12.4e %-8s %-8s  %-10.2e\n",
                        r.nel, r.h, r.l2_err, "—", "—", r.lsqr_resid_rel)
            else
                h_ratio = prev.h / r.h
                ratio   = prev.l2_err / max(r.l2_err, eps())
                order   = log(ratio) / log(h_ratio)
                @printf("  %-6d %-10.5e  %-12.4e %-8.3f %-8.3f  %-10.2e\n",
                        r.nel, r.h, r.l2_err, ratio, order, r.lsqr_resid_rel)
            end
            prev = r
        end
    end

    csv_path = joinpath(output_dir, "p2g_tg_convergence.csv")
    open(csv_path, "w") do io
        println(io, "p,nel,h,num_particles,l2_err,rel_l2_err,l2_norm_exact,lsqr_resid_rel,t_build_B,t_solve")
        for r in results
            @printf(io, "%d,%d,%.10e,%d,%.16e,%.16e,%.16e,%.16e,%.6f,%.6f\n",
                    r.p, r.nel, r.h, r.num_particles,
                    r.l2_err, r.rel_l2_err, r.l2_norm_exact,
                    r.lsqr_resid_rel,
                    r.t_build_B, r.t_solve)
        end
    end
    println("\nSaved per-run CSV: $csv_path")

    if save_plot
        try
            plt = plot(xscale=:log10, yscale=:log10,
                       xlabel="h", ylabel="L2 error",
                       title="P2G LSQR reconstruction error vs h\n(Taylor-Green, t=0)",
                       legend=:bottomright, minorgrid=true)
            for p_order in p_orders
                subset = filter(r -> r.p == p_order, results)
                isempty(subset) && continue
                hs   = [r.h for r in subset]
                errs = [r.l2_err for r in subset]
                plot!(plt, hs, errs, marker=:circle, lw=2,
                      label="p=$(p_order)  measured")
                refs = errs[end] .* (hs ./ hs[end]) .^ p_order
                plot!(plt, hs, refs, ls=:dash, lw=1,
                      label="∝ h^$(p_order)")
            end
            plot_path = joinpath(output_dir, "p2g_tg_convergence.png")
            savefig(plt, plot_path)
            println("Saved log-log plot: $plot_path")
        catch err
            @warn "Plot save failed (continuing)" exception=err
        end
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    with_run_logging() do
        test_p2g_taylor_green_convergence()
    end
end
