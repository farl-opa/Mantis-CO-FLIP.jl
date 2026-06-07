# =============================================================================
# DISPOSABLE one-off: recompute the vorticity snapshots for the Von Kármán
# Re=150 run WITHOUT re-running the simulation.
#
# The velocity `u_h_*.vtu` files are only grid samples (lossy), but each
# `particles_NNNN.vtu` stores the particle momentum field — exactly what the
# restart path reconstructs `u_coeffs` from (coadjoint P2G + H(div) projection).
# So for every particle snapshot we rebuild `u_coeffs`, then evaluate the
# vorticity with `compute_vorticity_form` and export a fresh `w_*` VTK.
#
# Usage (from the CO-FLIP/ directory):
#     julia --project=.. recompute_vorticity.jl            # default: :weak
#     COFLIP_VORTICITY_METHOD=strong julia --project=.. recompute_vorticity.jl
#
# Output files are named  `w_<method>_NNNN-<label>.vtu`  in the same VTK dir,
# so the original `w_h_*-★(d(fₕ)).vtu` (strong, from the live run) are kept.
# Delete this file when done.
# =============================================================================

include("CO-FLIP_solver.jl")
using Printf

function recompute_von_karman_vorticity(;
        vtk_dir::String = abspath(joinpath(@__DIR__, "..", "von_karman", "vk_re150_vtk")),
        method::Symbol  = Symbol(get(ENV, "COFLIP_VORTICITY_METHOD", "weak")),
        skip_existing::Bool = true,
        max_count::Int  = typemax(Int),   # cap snapshots processed (smoke-testing)
    )
    method in (:weak, :strong) ||
        error("method must be :weak or :strong, got $method")
    isdir(vtk_dir) || error("VTK directory not found: $vtk_dir")
    LinearAlgebra.BLAS.set_num_threads(1)

    # --- Domain: must match the original `test_von_karman_strouhal()` run -----
    # (defaults confirmed against von_karman/vk_re150_summary.txt: Re=150,
    #  U=1.5, ν=0.02, D=2, cylinder at (10,10), bc inlet/outlet/periodic).
    cfg = SimulationConfig(;
        nel                       = (160, 80),
        box_size                  = (40.0, 20.0),
        boundary_condition        = (:inlet, :outlet, :periodic, :periodic),
        inlet_U_inf               = 1.5,
        farfield_velocity         = (1.5, 0.0),
        farfield_corner_pin_cells = 4,
        obstacle                  = (kind=:cylinder, center=(10.0, 10.0), radius=1.0),
        flow_type                 = :cylinder,
        viscosity                 = 0.02,
        particles_per_cell        = 10,
        vorticity_method          = method,
    )

    domain = GenerateDomain(cfg.nel, cfg.p, cfg.k;
        box_size                  = cfg.box_size,
        starting_point            = cfg.starting_point,
        boundary_condition        = cfg.boundary_condition,
        obstacle                  = cfg.obstacle,
        farfield_corner_pin_cells = cfg.farfield_corner_pin_cells)

    num_elements = prod(domain.nel)
    ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
    warmup_evaluation_memo!(domain)

    # --- Discover the particle snapshots and their step indices ---------------
    existing = Set(readdir(vtk_dir))   # snapshot once for the skip-existing check
    pfiles = filter(f -> occursin(r"^particles_\d+\.vtu$", f), collect(existing))
    steps  = sort(parse.(Int, getindex.(match.(r"particles_(\d+)\.vtu", pfiles), 1)))
    isempty(steps) && error("No particles_*.vtu found in $vtk_dir")
    length(steps) > max_count && (steps = steps[1:max_count])
    println(@sprintf("Recomputing vorticity (method=:%s) for %d snapshots (steps %d…%d)",
                     method, length(steps), first(steps), last(steps)))
    println("  output dir: ", vtk_dir)

    for (i, step) in enumerate(steps)
        out_prefix = @sprintf("w_%s_%04d", method, step)
        # Skip if already produced (lets a crashed run resume). The exact
        # field-file name carries the form label, which differs per method;
        # prefix-match against the start-of-run listing instead of guessing it.
        if skip_existing && any(startswith(f, out_prefix * "-") for f in existing)
            continue
        end

        pvtu = joinpath(vtk_dir, @sprintf("particles_%04d.vtu", step))
        x, y, mx, my, vol = read_particles_from_vtu(pvtu)
        particles = particles_from_vtu_state(x, y, mx, my, vol, num_elements)

        apply_obstacle_brinkman!(particles, domain)
        particle_sorter!(particles, domain)

        sim_buf  = SimulationBuffers(length(particles.x), ndofs, num_elements, domain)
        u_coeffs = coadjoint_step!(particles, domain, sim_buf;
            lsqr_atol                    = cfg.lsqr_atol,
            lsqr_btol                    = cfg.lsqr_btol,
            lsqr_maxiter                 = cfg.lsqr_maxiter,
            lsqr_error_on_nonconvergence = cfg.lsqr_error_on_nonconvergence,
            projection_mean_subtract     = cfg.projection_mean_subtract)

        # :strong needs the L2-projected physical velocity; :weak rebuilds the
        # velocity from u_coeffs internally, so skip the extra projection there.
        ω_h = if method === :strong
            u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
            u_phys_form = Assemblers.solve_L2_projection(domain.R1, ★(u_form), domain.dΩ)
            compute_vorticity_form(u_coeffs, u_phys_form, domain, :strong)
        else
            compute_vorticity_form(u_coeffs, nothing, domain, :weak)
        end

        Plot.export_form_fields_to_vtk((ω_h,), out_prefix; output_directory_tree=[vtk_dir])

        # The live run cleared memo tables every step (clear_memo_every=1); do
        # the same so 1600+ reconstructions don't accumulate cached evaluations.
        maybe_clear_memo_tables!(step, 1, domain)
        println(@sprintf("  [%4d/%4d] step %04d done", i, length(steps), step))
    end

    println("Done. Wrote w_$(method)_NNNN-*.vtu next to the originals in $vtk_dir")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    recompute_von_karman_vorticity()
end
