using Mantis
using Random
using LinearAlgebra
using Plots
using SparseArrays
using IterativeSolvers
using LinearMaps
using StaticArrays
using CUDA
using DelimitedFiles
using Printf
using Memoization
using WriteVTK

gr()

const DEFAULT_OUTPUT_DIR = get(ENV, "OUTPUT_DIR", pwd())

struct Domain{F0, F1, F2, G, Q, MH, NH, MM, HF, KR0, KR0F, GG, M1F}
    R0::F0
    R1::F1
    R2::F2
    geo::G
    dΩ::Q
    nel::NTuple{2,Int}
    p::NTuple{2,Int}
    k::NTuple{2,Int}
    LHS_Hodge::MH
    LHS_Hodge_fact::HF
    N_Hodge::NH
    Mass_matrix::MM
    K_R0::KR0
    K_R0_fact::KR0F
    G_R0_R1::GG
    Mass1_fact::M1F
    eval_cache_size::Int
    box_size::NTuple{2,Float64}
    R1_basis_indices::Vector{Vector{Int}}
end

mutable struct Particles
    x::Vector{Float64}
    y::Vector{Float64}
    mx::Vector{Float64}
    my::Vector{Float64}
    volume::Vector{Float64}
    can_x::Vector{Float64}
    can_y::Vector{Float64}
    head::Vector{Int}
    next::Vector{Int}
    elem_ids::Vector{Int}
    P11::Vector{Float64}
    P12::Vector{Float64}
    P21::Vector{Float64}
    P22::Vector{Float64}
    delta_t::Vector{Float64}
end

struct EvaluationCache
    results::Vector{Vector{Vector{Matrix{Float64}}}}
    # Scratch buffers for the allocation-free 2D fast-path (per dimension):
    # bern_buf[d]  is (p+1, nder+1)  — Bernstein values
    # bsp_vals[d]  is (nder+1, n_bsp) — after BSpline extraction
    # gtb_vals[d]  is (nder+1, n_gtb) — after GTBSpline extraction
    bern_buf::NTuple{2, Matrix{Float64}}
    bsp_vals::NTuple{2, Matrix{Float64}}
    gtb_vals::NTuple{2, Matrix{Float64}}

    """Allocates basis-evaluation storage for fast local element evaluations."""
    function EvaluationCache(max_basis_size::Int)
        v0  = [[zeros(1, max_basis_size) for _ in 1:2]]
        v1  = [[zeros(1, max_basis_size) for _ in 1:2] for _ in 1:2]
        res = Vector{Vector{Vector{Matrix{Float64}}}}(undef, 2)
        res[1] = v0; res[2] = v1

        # Oversized to handle p ≤ 11 and nderivatives ≤ 3 (current use is p=3, nder=1)
        MAX_P1  = 12
        MAX_ND1 = 4
        bern = (zeros(MAX_P1, MAX_ND1), zeros(MAX_P1, MAX_ND1))
        bsp  = (zeros(MAX_ND1, MAX_P1), zeros(MAX_ND1, MAX_P1))
        gtb  = (zeros(MAX_ND1, MAX_P1), zeros(MAX_ND1, MAX_P1))

        return new(res, bern, bsp, gtb)
    end
end

mutable struct SimulationBuffers
    x_n::Vector{Float64}
    y_n::Vector{Float64}
    mx_n::Vector{Float64}
    my_n::Vector{Float64}
    x_np1::Vector{Float64}
    y_np1::Vector{Float64}
    mx_np1::Vector{Float64}
    my_np1::Vector{Float64}

    short_P11::Vector{Float64}
    short_P12::Vector{Float64}
    short_P21::Vector{Float64}
    short_P22::Vector{Float64}

    f_n_saved::Vector{Float64}
    f_star::Vector{Float64}
    f_np1::Vector{Float64}
    f_np1_raw::Vector{Float64}
    f_np1_proj::Vector{Float64}
    f_np1_proj_prev::Vector{Float64}

    lsqr_warm::Vector{Float64}

    thread_caches::Vector{EvaluationCache}

    quarter_pids::Vector{Vector{Int}}
    add_quarter::Matrix{Int}
    counts_elem::Vector{Int}
    counts_quarter::Matrix{Int}
    keep_mask::Vector{Bool}

    V_p::Vector{Float64}

    B_I::Vector{Int}
    B_J::Vector{Int}
    B_V::Vector{Float64}
    elem_basis_indices::Vector{Vector{Int}}

    max_err_per_thread::Vector{Float64}

    v_coeffs_buf::Vector{Float64}
    b_buf::Vector{Float64}
    sol_buf::Vector{Float64}
end

"""Allocates scratch buffers for particle-to-grid transfer and time stepping."""
function SimulationBuffers(num_particles_initial::Int, ndofs::Int, num_elements::Int, d::Domain)
    n_threads  = Threads.maxthreadid()
    cache_size = evaluation_cache_size(d)
    cap        = num_particles_initial * 2
    max_nnz    = 2 * cap * d.eval_cache_size

    return SimulationBuffers(
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        Vector{Float64}(undef, ndofs),
        zeros(Float64, ndofs),
        zeros(Float64, ndofs),
        [EvaluationCache(cache_size) for _ in 1:n_threads],
        [Int[] for _ in 1:num_elements * 4],
        zeros(Int, num_elements, 4),
        zeros(Int, num_elements),
        zeros(Int, num_elements, 4),
        trues(cap),
        Vector{Float64}(undef, 2 * cap),
        Vector{Int}(undef, max_nnz),
        Vector{Int}(undef, max_nnz),
        Vector{Float64}(undef, max_nnz),
        [Int[] for _ in 1:num_elements],
        zeros(Float64, n_threads),
        zeros(Float64, size(d.N_Hodge, 1)),
        zeros(Float64, size(d.N_Hodge, 1)),
        zeros(Float64, size(d.N_Hodge, 1)),
    )
end

Base.@kwdef struct SimulationConfig
    nel::NTuple{2,Int}                = (96, 96)
    p::NTuple{2,Int}                  = (3, 3)
    k::NTuple{2,Int}                  = (1, 1)
    box_size::NTuple{2,Float64}       = (1.0, 1.0)
    starting_point::NTuple{2,Float64} = (0.0, 0.0)
    boundary_condition::Symbol        = :periodic
    particles_per_cell::Int           = 20
    stratified_seeding::Bool          = true
    volume_convention::Symbol         = :physical
    rng_seed::Union{Int,Nothing}      = nothing
    flow_type::Symbol                 = :leapfrog
    target_cfl::Float64               = 0.5
    T_final::Float64                  = 1.0
    max_fp_iter::Int                  = 6
    fp_tol::Float64                   = 5e-9
    enable_energy_correction::Bool    = true
    enable_pressure_kick::Bool        = true
    pressure_kick_method::Symbol      = :curl_consistent
    pic_blend_alpha::Float64          = 0.0
    min_particles_per_element::Int    = 10
    max_particles_per_element::Int    = 25
    min_particles_per_quarter::Int    = 3
    ftle_threshold::Float64           = 0.2
    max_longterm_delta_t::Float64     = 1.0
    ftle_use_rate::Bool               = true
    global_ftle_gate::Float64         = -Inf
    delayed_reinit_frequency::Int     = 0
    output_every::Int                 = 1
    clear_memo_every::Int             = 0
    clear_memo_every_fp_iter::Int     = 1
    lsqr_atol::Float64                = 1e-9
    lsqr_btol::Float64                = 1e-9
    lsqr_maxiter::Int                 = 2000
    lsqr_error_on_nonconvergence::Bool = true
    cfl_recheck_tolerance::Float64    = 0.5
    cfl_adaptive::Bool                = false
    projection_mean_subtract::Bool    = true
    advection_time_integrator::Symbol = :rk2
end

"""Promote coefficient type for field probe output."""
probe_output_eltype(::Type{T}) where {T<:Real} = promote_type(Float64, T)

"""Get maximum local basis size for evaluation cache."""
evaluation_cache_size(d::Domain) = d.eval_cache_size

"""Build De Rham complex with quadrature and assembled Hodge-Laplace/mass operators."""
function GenerateDomain(
        nel::NTuple{2,Int},
        p::NTuple{2,Int},
        k::NTuple{2,Int};
        box_size::NTuple{2,Float64}=(1.0, 1.0),
        starting_point::NTuple{2,Float64}=(0.0, 0.0),
        boundary_condition::Symbol=:periodic,
    )
    if boundary_condition === :neumann
        @warn "GenerateDomain: :neumann boundary path is a stub; falling back to periodic."
    elseif boundary_condition !== :periodic
        throw(ArgumentError(
            "boundary_condition must be :periodic or :neumann, got $(boundary_condition)"
        ))
    end

    nq_assembly    = p .+ 1
    nq_error       = nq_assembly .* 2
    ∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, nq_assembly, nq_error)
    dΩ = Quadrature.StandardQuadrature(∫ₐ, prod(nel))

    periodic_flags = (boundary_condition === :periodic, boundary_condition === :periodic)
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_point, box_size, nel, p, k; periodic=periodic_flags,
    )

    A, N = assemble_hodge_laplacian_matrices(R[2], R[3], dΩ)
    A_fact = lu(A)
    M = assemble_1form_mass_matrix(R[2], dΩ)
    M_fact = lu(M)

    K_R0 = assemble_R0_stiffness_matrix(R[1], dΩ)
    G_R0_R1 = assemble_R0_R1_weak_grad_matrix(R[1], R[2], dΩ)

    n0 = size(K_R0, 1)
    ridge_R0 = 1e-10 * (sum(abs, diag(K_R0)) / max(n0, 1) + 1.0)
    K_R0_reg = K_R0 + ridge_R0 * sparse(I, n0, n0)
    K_R0_fact = lu(K_R0_reg)

    num_elements_R1 = prod(nel)
    R1_basis_indices = Vector{Vector{Int}}(undef, num_elements_R1)
    for eid in 1:num_elements_R1
        R1_basis_indices[eid] = collect(FunctionSpaces.get_basis_indices(R[2].fem_space, eid))
    end
    eval_cache_size = maximum(length(bi) for bi in R1_basis_indices)

    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k,
                  A, A_fact, N, M, K_R0, K_R0_fact, G_R0_R1, M_fact,
                  eval_cache_size, box_size, R1_basis_indices)
end

"""Evaluate analytic velocity field at specified point."""
function initial_velocity(flow_type::Symbol, px::Float64, py::Float64, Lx::Float64, Ly::Float64)
    if flow_type == :tg;             return flow_taylor_green(px, py, Lx, Ly)
    elseif flow_type == :vortex;     return flow_lamb_oseen(px, py, Lx, Ly)
    elseif flow_type == :gyre;       return flow_double_gyre(px, py, Lx, Ly)
    elseif flow_type == :decay;      return flow_decay(px, py, Lx, Ly)
    elseif flow_type == :convecting; return flow_convecting_vortex(px, py, Lx, Ly)
    elseif flow_type == :merging;    return flow_merging_vortices(px, py, Lx, Ly)
    elseif flow_type == :uniform;     return flow_uniform(px, py, Lx, Ly)
    elseif flow_type == :shear;       return flow_shear(px, py, Lx, Ly)
    elseif flow_type == :kh;          return flow_kelvin_helmholtz(px, py, Lx, Ly)
    elseif flow_type == :dipole;      return flow_dipole(px, py, Lx, Ly)
    elseif flow_type == :leapfrog;    return flow_leapfrog(px, py, Lx, Ly)
    elseif flow_type == :four_vortex; return flow_four_vortex(px, py, Lx, Ly)
    elseif flow_type == :stuart;      return flow_stuart(px, py, Lx, Ly)
    else; error("Unknown flow type: $flow_type")
    end
end

"""Generate initial particle distribution with stratified or random seeding."""
function generate_particles(
        num_particles::Int,
        domain::Domain,
        flow_type::Symbol=:vortex;
        stratified_seeding::Bool=true,
        rng_seed::Union{Int,Nothing}=nothing,
        volume_convention::Symbol=:physical,
        boundary_condition::Symbol=:periodic,
    )
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end

    if !(volume_convention in (:physical, :cell_fraction))
        throw(ArgumentError(
            "volume_convention must be :physical or :cell_fraction, got $(volume_convention)"
        ))
    end

    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    dx     = Lx / nx
    dy     = Ly / ny

    num_elements = nx * ny
    base_ppc     = num_particles ÷ num_elements
    extra        = num_particles - base_ppc * num_elements

    x   = Vector{Float64}(undef, num_particles)
    y   = Vector{Float64}(undef, num_particles)
    mx  = Vector{Float64}(undef, num_particles)
    my  = Vector{Float64}(undef, num_particles)
    vol = ones(Float64, num_particles)

    can_x    = zeros(Float64, num_particles)
    can_y    = zeros(Float64, num_particles)
    head     = zeros(Int, num_elements)
    next     = zeros(Int, num_particles)
    elem_ids = zeros(Int, num_particles)

    P11 = ones(Float64, num_particles)
    P12 = zeros(Float64, num_particles)
    P21 = zeros(Float64, num_particles)
    P22 = ones(Float64, num_particles)
    delta_t = zeros(Float64, num_particles)

    pid = 1
    if stratified_seeding && base_ppc > 0
        used_N = max(1, floor(Int, sqrt(base_ppc) + 1e-12))
        for ej in 1:ny, ei in 1:nx
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy
            for jj in 0:(used_N - 1), ii in 0:(used_N - 1)
                px = x0 + ((ii + rand()) / used_N) * dx
                py = y0 + ((jj + rand()) / used_N) * dy
                if boundary_condition === :periodic
                    px = mod(px, Lx)
                    py = mod(py, Ly)
                end
                x[pid]  = px
                y[pid]  = py
                u, v    = initial_velocity(flow_type, px, py, Lx, Ly)
                mx[pid] = u
                my[pid] = v
                pid += 1
            end
            remainder = base_ppc - used_N * used_N
            for _ in 1:remainder
                px = x0 + rand() * dx
                py = y0 + rand() * dy
                if boundary_condition === :periodic
                    px = mod(px, Lx)
                    py = mod(py, Ly)
                end
                x[pid]  = px
                y[pid]  = py
                u, v    = initial_velocity(flow_type, px, py, Lx, Ly)
                mx[pid] = u
                my[pid] = v
                pid += 1
            end
        end
        for _ in 1:extra
            px = rand() * Lx
            py = rand() * Ly
            x[pid]  = px
            y[pid]  = py
            u, v    = initial_velocity(flow_type, px, py, Lx, Ly)
            mx[pid] = u
            my[pid] = v
            pid += 1
        end
    else
        for i in 1:num_particles
            px = rand() * Lx
            py = rand() * Ly
            x[i]  = px
            y[i]  = py
            u, v  = initial_velocity(flow_type, px, py, Lx, Ly)
            mx[i] = u
            my[i] = v
        end
    end

    if volume_convention === :physical
        fill!(vol, (Lx * Ly) / num_particles)
    else
        ppc_for_volume = base_ppc > 0 ? base_ppc : num_particles
        fill!(vol, 1.0 / ppc_for_volume)
    end

    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids,
                     P11, P12, P21, P22, delta_t)
end

"""Build per-element linked lists and canonical reference coordinates for particles."""
function particle_sorter!(p::Particles, d::Domain)
    fill!(p.head, 0)

    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy

    @inbounds for i in 1:length(p.x)
        p.x[i] = mod(p.x[i], Lx)
        p.y[i] = mod(p.y[i], Ly)

        ei = clamp(floor(Int, p.x[i] * inv_dx) + 1, 1, nx)
        ej = clamp(floor(Int, p.y[i] * inv_dy) + 1, 1, ny)
        eid = (ej - 1) * nx + ei
        p.elem_ids[i] = eid

        p.can_x[i] = (p.x[i] - (ei - 1) * dx) * inv_dx
        p.can_y[i] = (p.y[i] - (ej - 1) * dy) * inv_dy

        p.next[i]   = p.head[eid]
        p.head[eid] = i
    end
end

"""Set particle position and velocity by sampling grid 1-form."""
function set_g2p_velocity(
        p::Particles, pid::Int,
        px::Float64, py::Float64, eid::Int,
        x0::Float64, y0::Float64, dx::Float64, dy::Float64,
        ref_phys_form, Lx::Float64, Ly::Float64,
    )
    pxw = mod(px, Lx)
    pyw = mod(py, Ly)

    xi  = (pxw - x0) / dx
    eta = (pyw - y0) / dy
    sample_points = Mantis.Points.CartesianPoints(([xi], [eta]))
    pushfwd_eval, _ = Forms.evaluate_sharp_pushforward(ref_phys_form, eid, sample_points)

    p.x[pid]  = pxw
    p.y[pid]  = pyw
    p.mx[pid] = reduce(+, pushfwd_eval[1], dims=2)[1]
    p.my[pid] = reduce(+, pushfwd_eval[2], dims=2)[1]

    return nothing
end

"""Spawn and cull particles to maintain per-element and per-quarter count bounds."""
function enforce_min_particles_per_element!(
        p::Particles, d::Domain,
        min_particles_per_element::Int,
        max_particles_per_element::Int,
        min_particles_per_quarter::Int,
        ref_grid_coeffs::AbstractVector{T},
        buf::SimulationBuffers,
    ) where {T<:Real}

    if min_particles_per_element <= 0 && max_particles_per_element <= 0 && min_particles_per_quarter <= 0
        return 0, 0
    end
    if max_particles_per_element > 0 && min_particles_per_element > max_particles_per_element
        throw(ArgumentError("min_particles_per_element must be <= max_particles_per_element"))
    end
    if max_particles_per_element > 0 && min_particles_per_quarter > 0 &&
            max_particles_per_element < 4 * min_particles_per_quarter
        throw(ArgumentError("max_particles_per_element must be >= 4 * min_particles_per_quarter"))
    end

    particle_sorter!(p, d)

    num_elements       = prod(d.nel)
    min_elem_effective = max(min_particles_per_element, 4 * max(min_particles_per_quarter, 0))

    counts_elem    = buf.counts_elem
    counts_quarter = buf.counts_quarter
    add_quarter    = buf.add_quarter

    fill!(counts_elem,    0)
    fill!(counts_quarter, 0)
    fill!(add_quarter,    0)

    @inbounds for pid in eachindex(p.x)
        eid = p.elem_ids[pid]
        qx  = p.can_x[pid] >= 0.5 ? 2 : 1
        qy  = p.can_y[pid] >= 0.5 ? 1 : 0
        qid = qx + 2 * qy
        counts_elem[eid]         += 1
        counts_quarter[eid, qid] += 1
    end

    if min_particles_per_quarter > 0
        @inbounds for eid in 1:num_elements
            for qid in 1:4
                deficit = min_particles_per_quarter - counts_quarter[eid, qid]
                if deficit > 0
                    add_quarter[eid, qid] += deficit
                end
            end
        end
    end

    if min_elem_effective > 0
        @inbounds for eid in 1:num_elements
            base_after_quarter = counts_elem[eid] + sum(@view add_quarter[eid, :])
            extra_needed = min_elem_effective - base_after_quarter
            while extra_needed > 0
                qbest = 1
                cbest = counts_quarter[eid, 1] + add_quarter[eid, 1]
                for qid in 2:4
                    c = counts_quarter[eid, qid] + add_quarter[eid, qid]
                    if c < cbest; cbest = c; qbest = qid; end
                end
                add_quarter[eid, qbest] += 1
                extra_needed -= 1
            end
        end
    end

    total_to_add = sum(add_quarter)

    if total_to_add == 0 && max_particles_per_element <= 0
        rebind_volumes_per_element!(p, d, counts_elem)
        return 0, 0
    end

    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny
    rng = Random.default_rng()

    if total_to_add > 0
        ref_coeffs_f64 = collect(Float64, ref_grid_coeffs)
        ref_form       = Forms.build_form_field(d.R1, ref_coeffs_f64)
        ref_phys_expr  = ★(ref_form)
        ref_phys_form  = Assemblers.solve_L2_projection(d.R1, ref_phys_expr, d.dΩ)

        old_count = length(p.x)
        new_count = old_count + total_to_add

        resize!(p.x,       new_count)
        resize!(p.y,       new_count)
        resize!(p.mx,      new_count)
        resize!(p.my,      new_count)
        resize!(p.volume,  new_count)
        resize!(p.can_x,   new_count)
        resize!(p.can_y,   new_count)
        resize!(p.next,    new_count)
        resize!(p.elem_ids, new_count)
        resize!(p.P11,     new_count)
        resize!(p.P12,     new_count)
        resize!(p.P21,     new_count)
        resize!(p.P22,     new_count)
        resize!(p.delta_t, new_count)

        ensure_particle_capacity!(buf, new_count)

        pid = old_count + 1
        @inbounds for eid in 1:num_elements
            ej = ((eid - 1) ÷ nx) + 1
            ei = eid - (ej - 1) * nx
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy

            for qid in 1:4
                n_add_q = add_quarter[eid, qid]
                n_add_q <= 0 && continue

                qx  = ((qid - 1) % 2)
                qy  = ((qid - 1) ÷ 2)
                qx0 = x0 + qx * 0.5 * dx
                qy0 = y0 + qy * 0.5 * dy
                qdx = 0.5 * dx
                qdy = 0.5 * dy

                for _ in 1:n_add_q
                    px = qx0 + rand(rng) * qdx
                    py = qy0 + rand(rng) * qdy
                    set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy, ref_phys_form, Lx, Ly)
                    p.P11[pid]     = 1.0
                    p.P12[pid]     = 0.0
                    p.P21[pid]     = 0.0
                    p.P22[pid]     = 1.0
                    p.delta_t[pid] = 0.0
                    pid += 1
                end
            end
        end

        particle_sorter!(p, d)
    end

    total_to_remove = 0
    if max_particles_per_element > 0 && length(p.x) > 0
        n_particles = length(p.x)

        ensure_particle_capacity!(buf, n_particles)
        keep_mask    = buf.keep_mask
        quarter_pids = buf.quarter_pids

        fill!(view(keep_mask, 1:n_particles), true)
        @inbounds for s in eachindex(quarter_pids); empty!(quarter_pids[s]); end

        fill!(counts_elem,    0)
        fill!(counts_quarter, 0)

        @inbounds for pid in 1:n_particles
            eid = p.elem_ids[pid]
            qx  = p.can_x[pid] >= 0.5 ? 2 : 1
            qy  = p.can_y[pid] >= 0.5 ? 1 : 0
            qid = qx + 2 * qy
            counts_elem[eid]         += 1
            counts_quarter[eid, qid] += 1
            push!(quarter_pids[(eid - 1) * 4 + qid], pid)
        end

        @inbounds for eid in 1:num_elements
            local_excess = counts_elem[eid] - max_particles_per_element
            while local_excess > 0
                qbest       = 0
                excess_best = 0
                for qid in 1:4
                    removable = counts_quarter[eid, qid] - max(min_particles_per_quarter, 0)
                    if removable > excess_best
                        excess_best = removable
                        qbest = qid
                    end
                end
                qbest == 0 && break

                qlist = quarter_pids[(eid - 1) * 4 + qbest]
                while !isempty(qlist)
                    victim = pop!(qlist)
                    if keep_mask[victim]
                        keep_mask[victim]            = false
                        counts_quarter[eid, qbest]  -= 1
                        counts_elem[eid]            -= 1
                        total_to_remove             += 1
                        local_excess                -= 1
                        break
                    end
                end
            end
        end

        if total_to_remove > 0
            compact_particles_inplace!(p, keep_mask, n_particles)
            particle_sorter!(p, d)
        end
    end

    fill!(counts_elem, 0)
    @inbounds for pid in 1:length(p.x)
        counts_elem[p.elem_ids[pid]] += 1
    end
    rebind_volumes_per_element!(p, d, counts_elem)

    return total_to_add, total_to_remove
end

"""Compact particle arrays in-place using keep mask."""
function compact_particles_inplace!(p::Particles, keep_mask::AbstractVector{Bool}, n::Int)
    dst = 0
    @inbounds for src in 1:n
        keep_mask[src] || continue
        dst += 1
        if dst != src
            p.x[dst]        = p.x[src]
            p.y[dst]        = p.y[src]
            p.mx[dst]       = p.mx[src]
            p.my[dst]       = p.my[src]
            p.volume[dst]   = p.volume[src]
            p.can_x[dst]    = p.can_x[src]
            p.can_y[dst]    = p.can_y[src]
            p.next[dst]     = p.next[src]
            p.elem_ids[dst] = p.elem_ids[src]
            p.P11[dst]      = p.P11[src]
            p.P12[dst]      = p.P12[src]
            p.P21[dst]      = p.P21[src]
            p.P22[dst]      = p.P22[src]
            p.delta_t[dst]  = p.delta_t[src]
        end
    end
    resize!(p.x,        dst)
    resize!(p.y,        dst)
    resize!(p.mx,       dst)
    resize!(p.my,       dst)
    resize!(p.volume,   dst)
    resize!(p.can_x,    dst)
    resize!(p.can_y,    dst)
    resize!(p.next,     dst)
    resize!(p.elem_ids, dst)
    resize!(p.P11,      dst)
    resize!(p.P12,      dst)
    resize!(p.P21,      dst)
    resize!(p.P22,      dst)
    resize!(p.delta_t,  dst)
    return nothing
end

"""Distribute cell volume equally among particles in each element."""
function rebind_volumes_per_element!(p::Particles, d::Domain, counts_elem::AbstractVector{Int})
    nx, ny = d.nel
    Lx, Ly = d.box_size
    cell_area = (Lx / nx) * (Ly / ny)
    @inbounds for pid in 1:length(p.x)
        c = counts_elem[p.elem_ids[pid]]
        p.volume[pid] = c > 0 ? cell_area / c : cell_area
    end
    return nothing
end

"""Expand scratch buffers to accommodate at least n particles."""
function ensure_particle_capacity!(buf::SimulationBuffers, n::Int)
    if length(buf.x_n) >= n
        return nothing
    end
    new_cap = max(n, length(buf.x_n) * 2)
    for arr in (buf.x_n, buf.y_n, buf.mx_n, buf.my_n,
                buf.x_np1, buf.y_np1, buf.mx_np1, buf.my_np1,
                buf.short_P11, buf.short_P12, buf.short_P21, buf.short_P22,
                buf.keep_mask)
        resize!(arr, new_cap)
    end
    if length(buf.V_p) < 2 * new_cap
        resize!(buf.V_p, 2 * new_cap)
    end
    return nothing
end

"""Taylor-Green vortex on periodic domain."""
function flow_taylor_green(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    kx = 2*π * x / Lx
    ky = 2*π * y / Ly
    u =  U0 * sin(kx) * cos(ky)
    v = -U0 * cos(kx) * sin(ky)
    return u, v
end

"""Lamb-Oseen vortex centered in domain."""
function flow_lamb_oseen(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Gamma = 2.0
    r_core = min(Lx, Ly) / 8.0

    dx = x - cx;  dy = y - cy
    r2 = dx^2 + dy^2
    r  = sqrt(r2)

    if r < 1e-8; return 0.0, 0.0; end

    v_theta = (Gamma / (2 * π * r)) * (1.0 - exp(-r2 / (r_core^2)))
    u = -v_theta * (dy / r)
    v =  v_theta * (dx / r)
    return u, v
end

"""Double-gyre flow field."""
function flow_double_gyre(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    X = x / Lx;  Y = y / Ly
    u =  U0 * sin(π * X) * cos(π * Y)
    v = -U0 * cos(π * X) * sin(π * Y)
    return u, v
end

"""Compactly-supported decaying vortex."""
function flow_decay(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Gamma = 10.0
    r_core = min(Lx, Ly) / 6.0

    dx = x - cx;  dy = y - cy
    r2 = dx^2 + dy^2
    r  = sqrt(r2)

    if r < 1e-8; return 0.0, 0.0; end

    sin_theta = dy/r
    cos_theta = dx/r
    u = -sin_theta*exp(-r2/(0.2*r_core^2))
    v =  cos_theta*exp(-r2/(0.2*r_core^2))
    return u, v
end

"""Gaussian vortex with uniform advection."""
function flow_convecting_vortex(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    Γ  = 5.0
    σ  = 0.1 * min(Lx, Ly)
    x0 = 0.5 * Lx;  y0 = 0.5 * Ly

    dx = x - x0;  dy = y - y0
    r2 = dx^2 + dy^2

    factor = (Γ / (2π)) * exp(-r2 / σ^2)
    u_vortex = -factor * dy / σ^2
    v_vortex =  factor * dx / σ^2

    u = u_vortex + U0
    v = v_vortex
    return u, v
end

"""Two co-rotating vortices configured to merge."""
function flow_merging_vortices(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Gamma = 3.0
    r_core = min(Lx, Ly) / 15.0
    separation = r_core * 3.5

    cx1 = cx - separation / 2.0;  cy1 = cy
    cx2 = cx + separation / 2.0;  cy2 = cy

    dx1 = x - cx1;  dy1 = y - cy1
    r1_2 = dx1^2 + dy1^2;  r1 = sqrt(r1_2)
    dx2 = x - cx2;  dy2 = y - cy2
    r2_2 = dx2^2 + dy2^2;  r2 = sqrt(r2_2)

    u = 0.0;  v = 0.0
    if r1 > 1e-8
        v_theta1 = (Gamma / (2 * π * r1)) * (1.0 - exp(-r1_2 / (r_core^2)))
        u += -v_theta1 * (dy1 / r1)
        v +=  v_theta1 * (dx1 / r1)
    end
    if r2 > 1e-8
        v_theta2 = (Gamma / (2 * π * r2)) * (1.0 - exp(-r2_2 / (r_core^2)))
        u += -v_theta2 * (dy2 / r2)
        v +=  v_theta2 * (dx2 / r2)
    end
    return u, v
end

"""Uniform translating flow — sanity check for advection and projection bugs."""
function flow_uniform(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    _ = (x, y, Lx, Ly)
    return 1.0, 0.0
end

"""Bell-Colella-Glaz double shear layer — vortex roll-up benchmark on periodic box."""
function flow_shear(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    ρ = 30.0
    δ = 0.05
    yh = y / Ly
    u = (yh ≤ 0.5) ? tanh(ρ * (yh - 0.25)) : tanh(ρ * (0.75 - yh))
    v = δ * sin(2π * x / Lx)
    return u, v
end

"""Kelvin-Helmholtz: two smoothed anti-parallel shear layers + single-mode perturbation."""
function flow_kelvin_helmholtz(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    δ  = 0.025 * Ly
    A  = 0.01 * U0
    base_u = (y ≤ 0.5*Ly) ?  U0 * tanh((y - 0.25*Ly) / δ) :
                             U0 * tanh((0.75*Ly - y) / δ)
    u = base_u
    v = A * sin(2π * x / Lx)
    return u, v
end

"""Counter-rotating Lamb-Oseen pair forming a self-propagating dipole (translates in +x)."""
function flow_dipole(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 4.0;  cy = Ly / 2.0
    Γ = 3.0
    r_core = min(Lx, Ly) / 25.0
    sep = min(Lx, Ly) / 10.0

    centers = ((cx, cy + sep/2, +Γ),
               (cx, cy - sep/2, -Γ))

    u = 0.0;  v = 0.0
    for (xc, yc, Γi) in centers
        dx = x - xc;  dy = y - yc
        r2 = dx^2 + dy^2;  r = sqrt(r2)
        if r > 1e-8
            v_theta = (Γi / (2π * r)) * (1.0 - exp(-r2 / r_core^2))
            u += -v_theta * dy / r
            v +=  v_theta * dx / r
        end
    end
    return u, v
end

"""Leapfrog: two coaxial dipoles arranged horizontally with different separations; both propagate in +y and leapfrog through each other (mirrors C++ COFLIPSolver2D::sampleLeapfrog)."""
function flow_leapfrog(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 4.0
    Γ = 2.0
    r_core = min(Lx, Ly) / 35.0
    dist_a = min(Lx, Ly) / 4.0
    dist_b = min(Lx, Ly) / 2.0

    centers = ((cx - 0.5*dist_a, cy, +Γ),
               (cx + 0.5*dist_a, cy, -Γ),
               (cx - 0.5*dist_b, cy, +Γ),
               (cx + 0.5*dist_b, cy, -Γ))

    u = 0.0;  v = 0.0
    for (xc, yc, Γi) in centers
        dx = x - xc;  dy = y - yc
        r2 = dx^2 + dy^2;  r = sqrt(r2)
        if r > 1e-8
            v_theta = (Γi / (2π * r)) * (1.0 - exp(-r2 / r_core^2))
            u += -v_theta * dy / r
            v +=  v_theta * dx / r
        end
    end
    return u, v
end

"""Four alternating-sign Lamb-Oseen vortices in checkerboard square — symmetry-preservation test (net Γ = 0)."""
function flow_four_vortex(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    cx = Lx / 2.0;  cy = Ly / 2.0
    Γ = 2.0
    r_core = min(Lx, Ly) / 20.0
    s = min(Lx, Ly) / 6.0

    centers = ((cx - s, cy - s, +Γ),
               (cx + s, cy + s, +Γ),
               (cx + s, cy - s, -Γ),
               (cx - s, cy + s, -Γ))

    u = 0.0;  v = 0.0
    for (xc, yc, Γi) in centers
        dx = x - xc;  dy = y - yc
        r2 = dx^2 + dy^2;  r = sqrt(r2)
        if r > 1e-8
            v_theta = (Γi / (2π * r)) * (1.0 - exp(-r2 / r_core^2))
            u += -v_theta * dy / r
            v +=  v_theta * dx / r
        end
    end
    return u, v
end

"""Stuart vortex — exact traveling-wave solution of 2D Euler. Periodic in x; in y, decays to a uniform shear (use a tall enough box)."""
function flow_stuart(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    ρ  = 0.5
    cy = Ly / 2.0
    kx = 2π / Lx
    ky = kx
    tx = kx * x
    ty = ky * (y - cy)
    D  = cosh(ty) - ρ * cos(tx)
    u  = -sinh(ty) * ky / D
    v  =  ρ * sin(tx) * kx / D
    return u, v
end

"""Evaluate basis and derivatives at sample point with caching."""
function evaluate_fast!(cache::EvaluationCache, space::S, element_id::Int, xi::P, nderivatives::Int) where {S, P}
    @inbounds for ord in 1:(nderivatives+1)
        for der in 1:length(cache.results[ord])
            for comp in 1:2
                fill!(cache.results[ord][der][comp], 0.0)
            end
        end
    end

    num_components = 2

    @inbounds for component_idx in 1:num_components
        extraction_coefficients, J = FunctionSpaces.get_extraction(space, element_id, component_idx)
        component_basis = FunctionSpaces.get_local_basis(space, element_id, xi, nderivatives, component_idx)

        for der_order in 1:(nderivatives+1)
             for der_idx in 1:length(cache.results[der_order])
                basis_val = component_basis[der_order][der_idx][1]
                out_mat   = cache.results[der_order][der_idx][component_idx]

                LinearAlgebra.mul!(view(out_mat, :, J), basis_val, extraction_coefficients)
            end
        end
    end
    return cache.results
end

"""
Allocation-free, thread-safe 2D fast-path evaluator that bypasses Mantis's allocating
`evaluate` chain for the CO-FLIP setup. Expects `space` to be a DirectSumSpace with 2
TensorProductSpace components, each having 2 1D GTBSplineSpace constituents (1 patch
each, single BSplineSpace inside). Writes into `cache.results` in the same layout as
`evaluate_fast!`.

Caller passes (xi, eta) ∈ [0,1] reference coords for the element directly — no
CartesianPoints wrapping. nderivatives ≤ 1 is supported in this implementation.
"""
@inline function evaluate_fast_2d!(
    cache::EvaluationCache,
    space,
    element_id::Int,
    xi_d1::Float64,
    xi_d2::Float64,
    nderivatives::Int,
)
    @inbounds for ord in 1:(nderivatives + 1)
        for der in 1:length(cache.results[ord])
            for comp in 1:2
                fill!(cache.results[ord][der][comp], 0.0)
            end
        end
    end

    J_offset = 0
    @inbounds for c in 1:2
        tp_space    = space.component_spaces[c]
        gtb_space_1 = tp_space.constituent_spaces[1]
        gtb_space_2 = tp_space.constituent_spaces[2]

        cart_idx = tp_space.cart_num_elements[element_id]
        e1 = cart_idx[1]
        e2 = cart_idx[2]

        # Single-patch GTBSpline assumed (CO-FLIP setup): patch_id = 1
        bsp_space_1 = gtb_space_1.patch_spaces[1]
        bsp_space_2 = gtb_space_2.patch_spaces[1]

        bsp_ext_1 = bsp_space_1.extraction_op.extraction_coefficients[e1][1]
        bsp_ext_2 = bsp_space_2.extraction_op.extraction_coefficients[e2][1]
        gtb_ext_1 = gtb_space_1.extraction_op.extraction_coefficients[e1][1]
        gtb_ext_2 = gtb_space_2.extraction_op.extraction_coefficients[e2][1]

        polynomial_1 = bsp_space_1.polynomials
        polynomial_2 = bsp_space_2.polynomials
        p1 = polynomial_1.p
        p2 = polynomial_2.p
        nd1 = nderivatives + 1
        n_bsp_1 = size(bsp_ext_1, 2)
        n_bsp_2 = size(bsp_ext_2, 2)
        n_gtb_1 = size(gtb_ext_1, 2)
        n_gtb_2 = size(gtb_ext_2, 2)

        # Leaf Bernstein evals — allocation-free
        bern_1 = @view cache.bern_buf[1][1:(p1 + 1), 1:nd1]
        bern_2 = @view cache.bern_buf[2][1:(p2 + 1), 1:nd1]
        Mantis.FunctionSpaces.evaluate_at!(bern_1, polynomial_1, xi_d1, nderivatives)
        Mantis.FunctionSpaces.evaluate_at!(bern_2, polynomial_2, xi_d2, nderivatives)

        # Apply BSpline extraction: (nd1, p+1) × (p+1, n_bsp) → (nd1, n_bsp)
        bsp_vals_1 = @view cache.bsp_vals[1][1:nd1, 1:n_bsp_1]
        bsp_vals_2 = @view cache.bsp_vals[2][1:nd1, 1:n_bsp_2]
        mul!(bsp_vals_1, transpose(bern_1), bsp_ext_1)
        mul!(bsp_vals_2, transpose(bern_2), bsp_ext_2)

        # Apply GTBSpline extraction: (nd1, n_bsp) × (n_bsp, n_gtb) → (nd1, n_gtb)
        gtb_vals_1 = @view cache.gtb_vals[1][1:nd1, 1:n_gtb_1]
        gtb_vals_2 = @view cache.gtb_vals[2][1:nd1, 1:n_gtb_2]
        mul!(gtb_vals_1, bsp_vals_1, gtb_ext_1)
        mul!(gtb_vals_2, bsp_vals_2, gtb_ext_2)

        # 2D kron pattern (matches Mantis): result_2d[(i-1)*n_gtb_1 + j] = vals_2[i] * vals_1[j]
        # Value key (0,0):
        out_val = cache.results[1][1][c]
        for i in 1:n_gtb_2
            v2 = gtb_vals_2[1, i]
            base = J_offset + (i - 1) * n_gtb_1
            for j in 1:n_gtb_1
                out_val[1, base + j] = v2 * gtb_vals_1[1, j]
            end
        end

        if nderivatives >= 1
            # d/dxi key (1,0): k_1=1, k_2=0
            out_dxi = cache.results[2][1][c]
            for i in 1:n_gtb_2
                v2 = gtb_vals_2[1, i]
                base = J_offset + (i - 1) * n_gtb_1
                for j in 1:n_gtb_1
                    out_dxi[1, base + j] = v2 * gtb_vals_1[2, j]
                end
            end
            # d/deta key (0,1): k_1=0, k_2=1
            out_deta = cache.results[2][2][c]
            for i in 1:n_gtb_2
                v2 = gtb_vals_2[2, i]
                base = J_offset + (i - 1) * n_gtb_1
                for j in 1:n_gtb_1
                    out_deta[1, base + j] = v2 * gtb_vals_1[1, j]
                end
            end
        end

        J_offset += n_gtb_1 * n_gtb_2
    end

    return cache.results
end

"""Evaluate velocity and Jacobian at point from grid coefficients."""
function probe_field_at_point(
        x::Real, y::Real,
        u_coeffs::AbstractVector{T},
        d::Domain,
        cache::EvaluationCache,
    ) where {T<:Real}
    Tout   = probe_output_eltype(T)
    Lx, Ly = d.box_size
    nx, ny = d.nel

    x_wrapped = mod(x, Lx)
    y_wrapped = mod(y, Ly)

    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy

    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    elem_idx = (ej - 1) * nx + ei

    xi  = (x_wrapped - (ei - 1) * dx) * inv_dx
    eta = (y_wrapped - (ej - 1) * dy) * inv_dy

    eval_out     = evaluate_fast_2d!(cache, d.R1.fem_space, elem_idx, xi, eta, 1)
    dof_indices  = d.R1_basis_indices[elem_idx]
    local_coeffs = @view u_coeffs[dof_indices]
    n_loc        = length(local_coeffs)

    val_x  = @view eval_out[1][1][1][1:n_loc]
    val_y  = @view eval_out[1][1][2][1:n_loc]
    u      = dot(val_x, local_coeffs)
    v      = dot(val_y, local_coeffs)

    dxi_u  = @view eval_out[2][1][1][1:n_loc]
    deta_u = @view eval_out[2][2][1][1:n_loc]
    dxi_v  = @view eval_out[2][1][2][1:n_loc]
    deta_v = @view eval_out[2][2][2][1:n_loc]

    du_dx = dot(dxi_u,  local_coeffs) * inv_dx
    du_dy = dot(deta_u, local_coeffs) * inv_dy
    dv_dx = dot(dxi_v,  local_coeffs) * inv_dx
    dv_dy = dot(deta_v, local_coeffs) * inv_dy

    return SVector{2, Tout}(u, v), SMatrix{2,2,Tout,4}(du_dx, dv_dx, du_dy, dv_dy)
end

"""Warm up basis-evaluation paths (compilation, memoization)."""
function warmup_evaluation_memo!(d::Domain)
    cache  = EvaluationCache(evaluation_cache_size(d))
    points = Mantis.Points.CartesianPoints(([0.5], [0.5]))
    evaluate_fast!(cache, d.R1.fem_space, 1, points, 1)
    evaluate_fast_2d!(cache, d.R1.fem_space, 1, 0.5, 0.5, 1)
    evaluate_fast_2d!(cache, d.R1.fem_space, 1, 0.5, 0.5, 0)
    return nothing
end

"""Export particle data to VTK format."""
function export_particles_to_vtk(particles::Particles, output_path::String)
    n_particles = length(particles.x)

    points = zeros(3, n_particles)
    points[1, :] .= particles.x
    points[2, :] .= particles.y

    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, vec([i])) for i in 1:n_particles]

    vtkfile = vtk_grid(output_path, points, cells)

    u_part     = copy(particles.mx)
    v_part     = copy(particles.my)
    speed_part = sqrt.(u_part.^2 .+ v_part.^2)

    vtkfile["mx",     VTKPointData()] = u_part
    vtkfile["my",     VTKPointData()] = v_part
    vtkfile["speed",  VTKPointData()] = speed_part
    vtkfile["volume", VTKPointData()] = particles.volume

    outfiles = vtk_save(vtkfile)
    return outfiles
end

"""Ensure triplet buffers can hold B-matrix nonzeros."""
function ensure_B_triplet_capacity!(buf::SimulationBuffers, num_p::Int, d::Domain)
    needed_nnz = 2 * num_p * d.eval_cache_size
    if length(buf.B_I) < needed_nnz
        resize!(buf.B_I, needed_nnz)
        resize!(buf.B_J, needed_nnz)
        resize!(buf.B_V, needed_nnz)
    end
    return nothing
end

function count_particles_per_element!(counts_elem::Vector{Int}, p::Particles, d::Domain)
    fill!(counts_elem, 0)
    @inbounds for eid in 1:length(counts_elem)
        pid = p.head[eid]
        while pid != 0
            counts_elem[eid] += 1
            pid = p.next[pid]
        end
    end
    return counts_elem
end

function physical_spatial_sort!(p::Particles, d::Domain)
    num_p    = length(p.x)
    num_elem = prod(d.nel)
    
    # Create a permutation array based on element IDs
    perm = sortperm(p.elem_ids[1:num_p])
    
    # Create temporary storage for reordered data
    x_new     = similar(p.x, num_p)
    y_new     = similar(p.y, num_p)
    mx_new    = similar(p.mx, num_p)
    my_new    = similar(p.my, num_p)
    vol_new   = similar(p.volume, num_p)
    can_x_new = similar(p.can_x, num_p)
    can_y_new = similar(p.can_y, num_p)
    P11_new   = similar(p.P11, num_p)
    P12_new   = similar(p.P12, num_p)
    P21_new   = similar(p.P21, num_p)
    P22_new   = similar(p.P22, num_p)
    elem_ids_new = similar(p.elem_ids, num_p)
    
    # Reorder all particle data according to element locality
    @inbounds for i in 1:num_p
        old_idx = perm[i]
        x_new[i]     = p.x[old_idx]
        y_new[i]     = p.y[old_idx]
        mx_new[i]    = p.mx[old_idx]
        my_new[i]    = p.my[old_idx]
        vol_new[i]   = p.volume[old_idx]
        can_x_new[i] = p.can_x[old_idx]
        can_y_new[i] = p.can_y[old_idx]
        P11_new[i]   = p.P11[old_idx]
        P12_new[i]   = p.P12[old_idx]
        P21_new[i]   = p.P21[old_idx]
        P22_new[i]   = p.P22[old_idx]
        elem_ids_new[i] = p.elem_ids[old_idx]
    end
    
    # Copy reordered data back into particle arrays
    copyto!(p.x, 1, x_new, 1, num_p)
    copyto!(p.y, 1, y_new, 1, num_p)
    copyto!(p.mx, 1, mx_new, 1, num_p)
    copyto!(p.my, 1, my_new, 1, num_p)
    copyto!(p.volume, 1, vol_new, 1, num_p)
    copyto!(p.can_x, 1, can_x_new, 1, num_p)
    copyto!(p.can_y, 1, can_y_new, 1, num_p)
    copyto!(p.P11, 1, P11_new, 1, num_p)
    copyto!(p.P12, 1, P12_new, 1, num_p)
    copyto!(p.P21, 1, P21_new, 1, num_p)
    copyto!(p.P22, 1, P22_new, 1, num_p)
    copyto!(p.elem_ids, 1, elem_ids_new, 1, num_p)
    
    # After physical reordering, rebuild the head/next linked list structure
    particle_sorter!(p, d)
    
    return nothing
end

@inline function probe_field_at_point_with_hint(
        x::Real, y::Real,
        u_coeffs::AbstractVector{T},
        d::Domain,
        cache::EvaluationCache,
        _hint_elem::Int,  # reserved for future hint optimization
    ) where {T<:Real}
    Tout   = probe_output_eltype(T)
    Lx, Ly = d.box_size
    nx, ny = d.nel

    x_wrapped = mod(x, Lx)
    y_wrapped = mod(y, Ly)

    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy

    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    elem_idx = (ej - 1) * nx + ei

    xi  = (x_wrapped - (ei - 1) * dx) * inv_dx
    eta = (y_wrapped - (ej - 1) * dy) * inv_dy

    eval_out     = evaluate_fast_2d!(cache, d.R1.fem_space, elem_idx, xi, eta, 1)
    dof_indices  = d.R1_basis_indices[elem_idx]
    local_coeffs = @view u_coeffs[dof_indices]
    n_loc        = length(local_coeffs)

    val_x  = @view eval_out[1][1][1][1:n_loc]
    val_y  = @view eval_out[1][1][2][1:n_loc]
    u      = dot(val_x, local_coeffs)
    v      = dot(val_y, local_coeffs)

    dxi_u  = @view eval_out[2][1][1][1:n_loc]
    deta_u = @view eval_out[2][2][1][1:n_loc]
    dxi_v  = @view eval_out[2][1][2][1:n_loc]
    deta_v = @view eval_out[2][2][2][1:n_loc]

    du_dx = dot(dxi_u,  local_coeffs) * inv_dx
    du_dy = dot(deta_u, local_coeffs) * inv_dy
    dv_dx = dot(dxi_v,  local_coeffs) * inv_dx
    dv_dy = dot(deta_v, local_coeffs) * inv_dy

    return SVector{2, Tout}(u, v), SMatrix{2,2,Tout,4}(du_dx, dv_dx, du_dy, dv_dy)
end

@inline function physical_velocity_and_jacobian_hint(
        x::T, y::T,
        u_coeffs::AbstractVector{U},
        d::Domain, cache::EvaluationCache,
        hint_elem::Int,
    ) where {T<:AbstractFloat, U<:Real}
    raw_vel, raw_grad = probe_field_at_point_with_hint(x, y, u_coeffs, d, cache, hint_elem)

    u = -raw_vel[2]
    v =  raw_vel[1]

    ux = -raw_grad[2, 1]
    uy = -raw_grad[2, 2]
    vx =  raw_grad[1, 1]
    vy =  raw_grad[1, 2]

    vel = SVector{2, T}(u, v)
    jac = SMatrix{2, 2, T, 4}(ux, vx, uy, vy)
    return vel, jac
end

"""Build particle-to-grid least-squares matrix with parallelization."""
function build_B_matrix(p::Particles, d::Domain, buf::SimulationBuffers)
    fes      = d.R1.fem_space
    num_p    = length(p.x)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    num_elem = prod(d.nel)

    ensure_B_triplet_capacity!(buf, num_p, d)
    I_idx  = buf.B_I
    J_idx  = buf.B_J
    V_val  = buf.B_V

    if length(buf.elem_basis_indices) != num_elem
        resize!(buf.elem_basis_indices, num_elem)
        for eid in 1:num_elem
            buf.elem_basis_indices[eid] = Int[]
        end
    end

    # Pre-compute basis indices for all elements
    @inbounds for eid in 1:num_elem
        dof_indices = buf.elem_basis_indices[eid]
        if isempty(dof_indices)
            dof_indices = collect(FunctionSpaces.get_basis_indices(fes, eid))
            buf.elem_basis_indices[eid] = dof_indices
        end
    end

    # STEP 1: Count particles per element and compute offsets
    counts_elem = buf.counts_elem
    count_particles_per_element!(counts_elem, p, d)
    
    offsets = Vector{Int}(undef, num_elem + 1)
    offsets[1] = 1
    
    @inbounds for eid in 1:num_elem
        n_loc = length(buf.elem_basis_indices[eid])
        num_particles_in_elem = counts_elem[eid]
        nnz_this_elem = num_particles_in_elem * n_loc * 2
        offsets[eid + 1] = offsets[eid] + nnz_this_elem
    end
    
    upper_bound_nnz = offsets[end] - 1
    ensure_B_triplet_capacity!(buf, upper_bound_nnz, d)

    # STEP 2: Parallel element loop with thread-local writes
    # Track actual cursor position for each element
    final_cursors = Vector{Int}(undef, num_elem)
    
    Threads.@threads :static for eid in 1:num_elem
        tid   = Threads.threadid()
        cache = buf.thread_caches[tid]
        
        pid = p.head[eid]
        if pid == 0
            final_cursors[eid] = offsets[eid]
            continue
        end

        dof_indices = buf.elem_basis_indices[eid]
        n_loc = length(dof_indices)
        
        cursor = offsets[eid]

        while pid != 0
            eval_out = evaluate_fast_2d!(cache, fes, eid, p.can_x[pid], p.can_y[pid], 0)

            vals_x = @view eval_out[1][1][1][1, 1:n_loc]
            vals_y = @view eval_out[1][1][2][1, 1:n_loc]
            w      = sqrt(p.volume[pid])
            row_x  = 2pid - 1
            row_y  = 2pid

            for k in 1:n_loc
                vx = vals_x[k] * w
                vy = vals_y[k] * w

                if abs(vx) > 1e-15
                    I_idx[cursor] = row_x
                    J_idx[cursor] = dof_indices[k]
                    V_val[cursor] = vx
                    cursor += 1
                end

                if abs(vy) > 1e-15
                    I_idx[cursor] = row_y
                    J_idx[cursor] = dof_indices[k]
                    V_val[cursor] = vy
                    cursor += 1
                end
            end

            pid = p.next[pid]
        end
        
        final_cursors[eid] = cursor
    end

    # STEP 3: Compact triplet arrays to remove gaps created by filtering
    write_pos = 1
    @inbounds for eid in 1:num_elem
        read_start = offsets[eid]
        read_end   = final_cursors[eid] - 1
        num_to_copy = max(0, read_end - read_start + 1)
        
        if num_to_copy > 0
            copyto!(I_idx, write_pos, I_idx, read_start, num_to_copy)
            copyto!(J_idx, write_pos, J_idx, read_start, num_to_copy)
            copyto!(V_val, write_pos, V_val, read_start, num_to_copy)
            write_pos += num_to_copy
        end
    end
    
    actual_nnz = write_pos - 1

    return sparse(
        @view(I_idx[1:actual_nnz]),
        @view(J_idx[1:actual_nnz]),
        @view(V_val[1:actual_nnz]),
        2num_p, num_dofs,
    )
end

"""Build LSQR right-hand side from particle impulses."""
function build_lsqr_rhs!(V_p::AbstractVector{Float64}, p::Particles)
    @inbounds for i in 1:length(p.x)
        w = sqrt(p.volume[i])
        V_p[2i-1] =  p.my[i] * w
        V_p[2i]   = -p.mx[i] * w
    end
    return V_p
end

"""Solve P2G least-squares system with warm-start for grid velocity."""
function solve_grid_velocity_lsqr(
        B::SparseMatrixCSC{Float64,Int},
        p::Particles,
        V_p_buf::AbstractVector{Float64},
        warm::AbstractVector{Float64};
        atol::Float64=1e-8,
        btol::Float64=1e-8,
        maxiter::Int=2000,
        error_on_nonconvergence::Bool=true,
    )
    n = 2 * length(p.x)
    V_p = @view V_p_buf[1:n]
    build_lsqr_rhs!(V_p, p)

    x0 = copy(warm)
    if length(x0) != size(B, 2)
        x0 = zeros(size(B, 2))
    end
    _, ch = IterativeSolvers.lsqr!(x0, B, V_p; atol=atol, btol=btol,
                                   maxiter=maxiter, log=true)
    if !ch.isconverged && error_on_nonconvergence
        @error "P2G LSQR did not converge" iters=ch.iters atol=atol btol=btol maxiter=maxiter
    end
    return x0
end

"""Project to divergence-free space via Hodge-Laplace and return pressure."""
function project_and_get_pressure(dom::Domain, v_h::V, buf::SimulationBuffers; subtract_mean::Bool=true) where {V}
    fill!(buf.v_coeffs_buf, 0.0)
    copyto!(buf.v_coeffs_buf, 1, v_h.coefficients, 1, length(v_h.coefficients))

    mul!(buf.b_buf, dom.N_Hodge, buf.v_coeffs_buf)
    copyto!(buf.sol_buf, buf.b_buf)
    ldiv!(dom.LHS_Hodge_fact, buf.sol_buf)
    u_corr, ϕ_corr = Forms.build_form_fields((dom.R1, dom.R2), buf.sol_buf; labels=("u¹ₕ", "ϕ²ₕ"))

    ϕ_coeffs = copy(ϕ_corr.coefficients)
    if subtract_mean && !isempty(ϕ_coeffs)
        ϕ_coeffs .-= sum(ϕ_coeffs) / length(ϕ_coeffs)
    end

    div_free_coeffs = v_h.coefficients - u_corr.coefficients
    return div_free_coeffs, ϕ_coeffs
end

"""Assemble Hodge-Laplace system matrices."""
function assemble_hodge_laplacian_matrices(R1, R2, dΩ)
    weak_form_inputs = Assemblers.WeakFormInputs((R1, R2))
    lhs_expressions, rhs_expressions = get_lhs_rhs_hodge_laplacian(weak_form_inputs, dΩ)
    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    A, N = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})

    return A, N

end

"""Build weak-form expressions for Hodge-Laplace system."""
function get_lhs_rhs_hodge_laplacian(
    inputs::Assemblers.WeakFormInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ϵ¹, ε² = Assemblers.get_test_forms(inputs)
    u¹, ϕ² = Assemblers.get_trial_forms(inputs)
    deps1 = d(ϵ¹)
    A_11 = ∫(ϵ¹ ∧ ★(u¹), dΩ)
    A_12 = -∫(d(ϵ¹) ∧ ★(ϕ²), dΩ)
    A_21 = ∫(ε² ∧ ★(d(u¹)), dΩ)
    lhs_expressions = ((A_11, A_12), (A_21, 0))
    b_21 = ∫(ε² ∧ ★(deps1), dΩ)
    rhs_expressions = ((0,0), (b_21,0))
    return lhs_expressions, rhs_expressions
end

"""Assemble Galerkin 1-form mass matrix."""
function assemble_1form_mass_matrix(R1_space::F, dΩ::Q) where {F, Q}
    weak_form_inputs = Assemblers.WeakFormInputs(R1_space)
    vᵏ = Assemblers.get_test_form(weak_form_inputs)
    uᵏ = Assemblers.get_trial_form(weak_form_inputs)

    A = ∫(vᵏ ∧ ★(uᵏ), dΩ)

    lhs_expressions = ((A,),)
    rhs_expressions = ((0,),)

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    M, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})
    return M
end

"""Assemble scalar Laplacian (R0 stiffness): K[i,j] = ∫(d(ψᵢ) ∧ ★ d(ψⱼ))."""
function assemble_R0_stiffness_matrix(R0_space::F, dΩ::Q) where {F, Q}
    weak_form_inputs = Assemblers.WeakFormInputs(R0_space)
    ε⁰ = Assemblers.get_test_form(weak_form_inputs)
    u⁰ = Assemblers.get_trial_form(weak_form_inputs)

    A = ∫(d(ε⁰) ∧ ★(d(u⁰)), dΩ)

    lhs_expressions = ((A,),)
    rhs_expressions = ((0,),)

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    K, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})
    return K
end

"""
Assemble rectangular weak-gradient block G[i,j] = ∫(d(ψᵢ⁰) ∧ ★ φⱼ¹) coupling R0 (rows)
with R1 (cols). Provides the weak-divergence operator τ̃ ↦ G·τ̃ ∈ R0 and, via its
transpose, satisfies M₁·d₀ = Gᵀ for the FEEC complex.
"""
function assemble_R0_R1_weak_grad_matrix(R0_space::F0, R1_space::F1, dΩ::Q) where {F0, F1, Q}
    weak_form_inputs = Assemblers.WeakFormInputs((R0_space, R1_space))
    ε⁰, _ε¹ = Assemblers.get_test_forms(weak_form_inputs)
    _u⁰, u¹ = Assemblers.get_trial_forms(weak_form_inputs)

    A_12 = ∫(d(ε⁰) ∧ ★(u¹), dΩ)
    lhs_expressions = ((0, A_12), (0, 0))
    rhs_expressions = ((0, 0), (0, 0))

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    sys, _ = Assemblers.assemble(weak_form; rhs_type=SparseArrays.SparseMatrixCSC{Float64, Int})

    n0 = FunctionSpaces.get_num_basis(R0_space.fem_space)
    n1 = FunctionSpaces.get_num_basis(R1_space.fem_space)
    return sys[1:n0, (n0 + 1):(n0 + n1)]
end

"""Apply energy-conserving correction orthogonal to midpoint velocity."""
function apply_energy_correction!(
        f_next::AbstractVector{Tn},
        f_original::AbstractVector{To},
        f_from_particles::AbstractVector{Tp},
        f_midpoint::AbstractVector{Tm};
        star_apply::Function,
        tol::Float64=1e-9,
        project_non_orthogonal::Bool=true,
    ) where {Tn<:Real, To<:Real, Tp<:Real, Tm<:Real}

    if project_non_orthogonal
        circulations_original = star_apply(f_midpoint)
        original_energy = dot(f_midpoint, circulations_original)

        if sqrt(abs(original_energy)) > tol
            raw_diff_dot = dot(f_from_particles .- f_original, circulations_original)
            scale = raw_diff_dot / original_energy

            @inbounds for i in eachindex(f_next)
                fd = (f_from_particles[i] - f_original[i]) - scale * f_midpoint[i]
                f_next[i] = f_original[i] + fd
            end
            return nothing
        end
    end

    @inbounds for i in eachindex(f_next)
        f_next[i] = f_from_particles[i]
    end
    return nothing
end

"""Compute pressure-gradient correction."""
@inline function compute_pressure_delta(
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
    ) where {Tp<:Real, Tq<:Real}
    return f_projected .- f_pre_projection
end

"""Apply FLIP-style pressure gradient correction to particles."""
function apply_pressure_correction!(
        p::Particles,
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
        d::Domain,
        thread_caches::Vector{EvaluationCache};
        delta_scale::Float64=1.0,
    ) where {Tp<:Real, Tq<:Real}
    tau_coeffs = compute_pressure_delta(f_projected, f_pre_projection)

    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end
    n_thread_slots = Threads.maxthreadid()
    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end

    num_p = length(p.x)
    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
            p.my[i] += delta_scale * val[1]
            p.mx[i] -= delta_scale * val[2]
        end
    end
end

"""
Curl-consistent pressure-force kick (Section 6.2.2 of the CO-FLIP paper).

Given τ = f_projected − f_pre_projection on the grid (stored in R1 with flux-form
convention), this routine:
  1. interpolates τ to the particles as a flux form (same read-out as the
     div-consistent path), producing per-particle physical force vectors;
  2. inverse-interpolates those particle force vectors back to a discrete 1-form τ̃
     ∈ R1 (1-form storage) via the LSQR P2G operator B;
  3. solves the weak Poisson reconstruction
         K_R0 · p̃ = G_R0_R1 · τ̃
     (regularised with a tiny diagonal ridge and mean-pinned) to find the closest
     exact 1-form;
  4. computes d₀p̃ in R1 via the FEEC identity M₁ · d₀ = (G_R0_R1)ᵀ and probes the
     result at every particle, adding the curl-free physical gradient to (mx, my).
"""
function apply_pressure_correction_curl!(
        p::Particles,
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
        d::Domain,
        buf::SimulationBuffers,
        thread_caches::Vector{EvaluationCache};
        delta_scale::Float64=1.0,
        atol::Float64=1e-9,
        btol::Float64=1e-9,
        maxiter::Int=2000,
        error_on_nonconvergence::Bool=true,
    ) where {Tp<:Real, Tq<:Real}
    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end
    n_thread_slots = Threads.maxthreadid()
    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end

    tau_coeffs = compute_pressure_delta(f_projected, f_pre_projection)
    num_p = length(p.x)

    Fx = Vector{Float64}(undef, num_p)
    Fy = Vector{Float64}(undef, num_p)

    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
            Fx[i] = -val[2]
            Fy[i] =  val[1]
        end
    end

    B = build_B_matrix(p, d, buf)

    V_p = @view buf.V_p[1:(2 * num_p)]
    @inbounds for i in 1:num_p
        w = sqrt(p.volume[i])
        V_p[2i - 1] = Fx[i] * w
        V_p[2i]     = Fy[i] * w
    end

    tilde_tau = zeros(Float64, size(B, 2))
    _, ch = IterativeSolvers.lsqr!(tilde_tau, B, V_p; atol=atol, btol=btol,
                                   maxiter=maxiter, log=true)
    if !ch.isconverged && error_on_nonconvergence
        @error "Curl-consistent P2G LSQR did not converge" iters=ch.iters atol=atol btol=btol maxiter=maxiter
    end

    rhs_R0 = d.G_R0_R1 * tilde_tau
    rhs_mean = sum(rhs_R0) / length(rhs_R0)
    @. rhs_R0 -= rhs_mean

    p_tilde = d.K_R0_fact \ rhs_R0
    p_mean = sum(p_tilde) / length(p_tilde)
    @. p_tilde -= p_mean

    rhs1 = transpose(d.G_R0_R1) * p_tilde
    g_R1 = d.Mass1_fact \ rhs1

    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            val, _ = probe_field_at_point(p.x[i], p.y[i], g_R1, d, cache)
            p.mx[i] += delta_scale * val[1]
            p.my[i] += delta_scale * val[2]
        end
    end
    return nothing
end

"""Blend particle velocity toward grid velocity."""
function apply_pic_blend!(
        p::Particles,
        f_coeffs::AbstractVector{Float64},
        d::Domain,
        thread_caches::Vector{EvaluationCache},
        alpha::Float64,
    )
    alpha <= 0.0 && return nothing

    if isempty(thread_caches)
        throw(ArgumentError("thread_caches must not be empty"))
    end
    n_thread_slots = Threads.maxthreadid()
    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end

    num_p = length(p.x)
    Threads.@threads :static for i in 1:num_p
        cache = thread_caches[Threads.threadid()]
        @inbounds begin
            raw_vel, _ = probe_field_at_point(p.x[i], p.y[i], f_coeffs, d, cache)

            u_g = -raw_vel[2]
            v_g =  raw_vel[1]

            p.mx[i] = (1.0 - alpha) * p.mx[i] + alpha * u_g
            p.my[i] = (1.0 - alpha) * p.my[i] + alpha * v_g
        end
    end
    return nothing
end

"""Integrate form expression over all elements."""
function integrate_form_expression(expr, dom::Domain)
    total = 0.0
    for element_id in 1:Quadrature.get_num_base_elements(dom.dΩ)
        total += sum(Forms.evaluate(expr, element_id)[1])
    end
    return total
end

"""Compute energy, circulation, and enstrophy diagnostics."""
function compute_conservation_diagnostics(u_coeffs::AbstractVector{T}, dom::Domain) where {T<:Real}
    u_h          = Forms.build_form_field(dom.R1, u_coeffs; label="u_h")
    u_phys_expr  = ★(u_h)
    u_phys_form  = Assemblers.solve_L2_projection(dom.R1, u_phys_expr, dom.dΩ)
    d_u_phys     = Forms.d(u_phys_form)
    ω_h          = ★(d_u_phys)

    energy      = 0.5 * integrate_form_expression(∫(u_h ∧ ★(u_h), dom.dΩ), dom)
    circulation =       integrate_form_expression(∫(d_u_phys,       dom.dΩ), dom)
    enstrophy   = 0.5 * integrate_form_expression(∫(ω_h ∧ ★(ω_h), dom.dΩ), dom)

    return (; energy, circulation, enstrophy)
end

"""Print conservation diagnostics."""
function print_conservation_diagnostics(step::Int, time_value::Real, diagnostics)
    @printf(
        "  Diagnostics [step=%d, t=%.6f]: energy=%.12e, circulation=%.12e, enstrophy=%.12e\n",
        step, time_value,
        diagnostics.energy, diagnostics.circulation, diagnostics.enstrophy,
    )
end

"""Apply periodic wrapping or clamping to position."""
@inline function apply_position_policy(x::T, y::T, Lx::T, Ly::T, mode::Symbol, tol::T) where {T<:AbstractFloat}
    if mode === :periodic
        return mod(x, Lx), mod(y, Ly)
    elseif mode === :clamp
        return clamp(x, tol, Lx - tol), clamp(y, tol, Ly - tol)
    else
        throw(ArgumentError("Unknown position mode: $mode. Use :periodic or :clamp"))
    end
end

"""Evaluate physical velocity and Jacobian at point."""
@inline function physical_velocity_and_jacobian(
        x::T, y::T,
        u_coeffs::AbstractVector{U},
        d::Domain, cache::EvaluationCache,
    ) where {T<:AbstractFloat, U<:Real}
    raw_vel, raw_grad = probe_field_at_point(x, y, u_coeffs, d, cache)

    u = -raw_vel[2]
    v =  raw_vel[1]

    ux = -raw_grad[2, 1]
    uy = -raw_grad[2, 2]
    vx =  raw_grad[1, 1]
    vy =  raw_grad[1, 2]

    vel = SVector{2, T}(u, v)
    jac = SMatrix{2, 2, T, 4}(ux, vx, uy, vy)
    return vel, jac
end

"""Iteratively rescale pullback to unit determinant."""
@inline function clamp_pullback(
        pullback::SMatrix{2, 2, T, 4};
        tol::T,
        max_iter::Int,
    ) where {T<:AbstractFloat}

    d_pb = det(pullback)
    if !isfinite(d_pb) || abs(d_pb) < eps(T)
        return pullback
    end

    Pt    = transpose(pullback)
    PtP   = Pt * pullback
    d_PtP = det(PtP)
    if !isfinite(d_PtP) || d_PtP < eps(T)
        return pullback
    end

    inv_Pt   = inv(Pt)
    tr_mixed = tr(inv(PtP))
    if !isfinite(tr_mixed) || abs(tr_mixed) < eps(T)
        return pullback
    end

    prev_mu         = zero(T)
    used_mu         = zero(T)
    output_pullback = pullback
    mixed_det       = d_pb

    iter = 0
    while abs(mixed_det - one(T)) > tol && iter < max_iter
        used_mu = prev_mu + ((one(T) / mixed_det) - one(T)) / tr_mixed
        !isfinite(used_mu) && break
        output_pullback = pullback + used_mu * inv_Pt
        prev_mu   = used_mu
        mixed_det = det(output_pullback)
        iter += 1
    end

    mixed_det > zero(T) && (output_pullback /= sqrt(mixed_det))



    return output_pullback
end

"""Advance position and pullback by one RK4 step."""
@inline function pullback_rk4_step(
        pos::SVector{2, T},
        input_pullback::SMatrix{2, 2, T, 4},
        dt::T,
        u_coeffs::AbstractVector{U},
        d::Domain,
        cache::EvaluationCache;
        mode::Symbol,
        pos_tol::T,
        pullback_tol::T,
        pullback_max_iter::Int,
    ) where {T<:AbstractFloat, U<:Real}
    Lx, Ly = d.box_size
    nx, ny = d.nel
    c1 = dt / 6
    c2 = dt / 3
    c3 = dt / 3
    c4 = dt / 6

    # Calculate initial element as hint for RK stages
    x_wrapped = mod(pos[1], Lx)
    y_wrapped = mod(pos[2], Ly)
    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy
    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    hint_elem = (ej - 1) * nx + ei

    v1, J1 = physical_velocity_and_jacobian_hint(pos[1], pos[2], u_coeffs, d, cache, hint_elem)
    Pdot1 = -(transpose(J1) * input_pullback)

    midp1_raw = pos + (dt * 0.5) * v1
    mx, my = apply_position_policy(midp1_raw[1], midp1_raw[2], Lx, Ly, mode, pos_tol)
    midp1 = SVector{2, T}(mx, my)
    midP1 = input_pullback + (dt * 0.5) * Pdot1

    v2, J2 = physical_velocity_and_jacobian_hint(midp1[1], midp1[2], u_coeffs, d, cache, hint_elem)
    Pdot2 = -(transpose(J2) * midP1)

    midp2_raw = pos + (dt * 0.5) * v2
    mx, my = apply_position_policy(midp2_raw[1], midp2_raw[2], Lx, Ly, mode, pos_tol)
    midp2 = SVector{2, T}(mx, my)
    midP2 = input_pullback + (dt * 0.5) * Pdot2

    v3, J3 = physical_velocity_and_jacobian_hint(midp2[1], midp2[2], u_coeffs, d, cache, hint_elem)
    Pdot3 = -(transpose(J3) * midP2)

    midp3_raw = pos + dt * v3
    mx, my = apply_position_policy(midp3_raw[1], midp3_raw[2], Lx, Ly, mode, pos_tol)
    midp3 = SVector{2, T}(mx, my)
    midP3 = input_pullback + dt * Pdot3

    v4, J4 = physical_velocity_and_jacobian_hint(midp3[1], midp3[2], u_coeffs, d, cache, hint_elem)
    Pdot4 = -(transpose(J4) * midP3)

    pullback = input_pullback + c1 * Pdot1 + c2 * Pdot2 + c3 * Pdot3 + c4 * Pdot4
    pullback = clamp_pullback(pullback; tol=pullback_tol, max_iter=pullback_max_iter)

    pos_raw = pos + c1 * v1 + c2 * v2 + c3 * v3 + c4 * v4
    px, py = apply_position_policy(pos_raw[1], pos_raw[2], Lx, Ly, mode, pos_tol)

    return SVector{2, T}(px, py), pullback
end

"""Advance position and pullback by one RK2 step with element hint optimization."""
@inline function pullback_rk2_step(
        pos::SVector{2, T},
        input_pullback::SMatrix{2, 2, T, 4},
        dt::T,
        u_coeffs::AbstractVector{U},
        d::Domain,
        cache::EvaluationCache;
        mode::Symbol,
        pos_tol::T,
        pullback_tol::T,
        pullback_max_iter::Int,
    ) where {T<:AbstractFloat, U<:Real}
    Lx, Ly = d.box_size
    nx, ny = d.nel

    # Calculate initial element as hint for RK stages
    x_wrapped = mod(pos[1], Lx)
    y_wrapped = mod(pos[2], Ly)
    dx = Lx / nx;  dy = Ly / ny
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy
    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    hint_elem = (ej - 1) * nx + ei

    v1, J1 = physical_velocity_and_jacobian_hint(pos[1], pos[2], u_coeffs, d, cache, hint_elem)
    Pdot1 = -(transpose(J1) * input_pullback)

    midp_raw = pos + (dt * 0.5) * v1
    mx, my = apply_position_policy(midp_raw[1], midp_raw[2], Lx, Ly, mode, pos_tol)
    midp = SVector{2, T}(mx, my)
    midP = input_pullback + (dt * 0.5) * Pdot1

    v2, J2 = physical_velocity_and_jacobian_hint(midp[1], midp[2], u_coeffs, d, cache, hint_elem)
    Pdot2 = -(transpose(J2) * midP)

    pullback = input_pullback + dt * Pdot2
    pullback = clamp_pullback(pullback; tol=pullback_tol, max_iter=pullback_max_iter)

    pos_raw = pos + dt * v2
    px, py = apply_position_policy(pos_raw[1], pos_raw[2], Lx, Ly, mode, pos_tol)

    return SVector{2, T}(px, py), pullback
end

"""Advect all particles using RK4 integration (threaded)."""
function advect_particles_pullback_rk4!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T},
        short_P11::AbstractVector{T}, short_P12::AbstractVector{T},
        short_P21::AbstractVector{T}, short_P22::AbstractVector{T},
        thread_caches::Vector{EvaluationCache},
        max_err_per_thread::Vector{T};
        position_mode::Symbol=:periodic,
        position_tol::T=T(1e-12),
        pullback_tol::T=T(1e-9),
        pullback_max_iter::Int=200,
    ) where {T<:AbstractFloat, U<:Real}

    num_p          = length(x0)
    Lx, Ly         = d.box_size
    n_thread_slots = Threads.maxthreadid()

    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end
    if length(max_err_per_thread) < n_thread_slots
        throw(ArgumentError("max_err_per_thread length must be at least Threads.maxthreadid()"))
    end
    @inbounds for i in 1:n_thread_slots
        max_err_per_thread[i] = zero(T)
    end

    Threads.@threads :static for i in 1:num_p
        tid   = Threads.threadid()
        cache = thread_caches[tid]

        sx, sy = apply_position_policy(x0[i], y0[i], Lx, Ly, position_mode, position_tol)
        pos    = SVector{2, T}(sx, sy)
        input_pullback = SMatrix{2, 2, T, 4}(one(T), zero(T), zero(T), one(T))

        pos_new, pullback = pullback_rk4_step(
            pos, input_pullback, dt, u_coeffs, d, cache;
            mode=position_mode, pos_tol=position_tol,
            pullback_tol=pullback_tol, pullback_max_iter=pullback_max_iter,
        )

        m_old = SVector{2, T}(mx0[i], my0[i])
        m_new = pullback * m_old

        ox, oy = x_out[i], y_out[i]
        ex = abs(pos_new[1] - ox)
        ey = abs(pos_new[2] - oy)
        if position_mode === :periodic
            ex > 0.5 * Lx && (ex = Lx - ex)
            ey > 0.5 * Ly && (ey = Ly - ey)
        end
        local_err = max(ex, ey)
        local_err > max_err_per_thread[tid] && (max_err_per_thread[tid] = local_err)

        x_out[i]  = pos_new[1]
        y_out[i]  = pos_new[2]
        mx_out[i] = m_new[1]
        my_out[i] = m_new[2]

        short_P11[i] = pullback[1, 1]
        short_P12[i] = pullback[1, 2]
        short_P21[i] = pullback[2, 1]
        short_P22[i] = pullback[2, 2]
    end

    return maximum(@view max_err_per_thread[1:n_thread_slots])
end

"""Advect all particles using RK2 integration (threaded)."""
function advect_particles_pullback_rk2!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T},
        short_P11::AbstractVector{T}, short_P12::AbstractVector{T},
        short_P21::AbstractVector{T}, short_P22::AbstractVector{T},
        thread_caches::Vector{EvaluationCache},
        max_err_per_thread::Vector{T};
        position_mode::Symbol=:periodic,
        position_tol::T=T(1e-12),
        pullback_tol::T=T(1e-9),
        pullback_max_iter::Int=200,
    ) where {T<:AbstractFloat, U<:Real}

    num_p          = length(x0)
    Lx, Ly         = d.box_size
    n_thread_slots = Threads.maxthreadid()

    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end
    if length(max_err_per_thread) < n_thread_slots
        throw(ArgumentError("max_err_per_thread length must be at least Threads.maxthreadid()"))
    end
    @inbounds for i in 1:n_thread_slots
        max_err_per_thread[i] = zero(T)
    end

    Threads.@threads :static for i in 1:num_p
        tid   = Threads.threadid()
        cache = thread_caches[tid]

        sx, sy = apply_position_policy(x0[i], y0[i], Lx, Ly, position_mode, position_tol)
        pos    = SVector{2, T}(sx, sy)
        input_pullback = SMatrix{2, 2, T, 4}(one(T), zero(T), zero(T), one(T))

        pos_new, pullback = pullback_rk2_step(
            pos, input_pullback, dt, u_coeffs, d, cache;
            mode=position_mode, pos_tol=position_tol,
            pullback_tol=pullback_tol, pullback_max_iter=pullback_max_iter,
        )

        m_old = SVector{2, T}(mx0[i], my0[i])
        m_new = pullback * m_old

        ox, oy = x_out[i], y_out[i]
        ex = abs(pos_new[1] - ox)
        ey = abs(pos_new[2] - oy)
        if position_mode === :periodic
            ex > 0.5 * Lx && (ex = Lx - ex)
            ey > 0.5 * Ly && (ey = Ly - ey)
        end
        local_err = max(ex, ey)
        local_err > max_err_per_thread[tid] && (max_err_per_thread[tid] = local_err)

        x_out[i]  = pos_new[1]
        y_out[i]  = pos_new[2]
        mx_out[i] = m_new[1]
        my_out[i] = m_new[2]

        short_P11[i] = pullback[1, 1]
        short_P12[i] = pullback[1, 2]
        short_P21[i] = pullback[2, 1]
        short_P22[i] = pullback[2, 2]
    end

    return maximum(@view max_err_per_thread[1:n_thread_slots])
end

"""Perform single P2G and projection pass."""
function coadjoint_step!(p::Particles, d::Domain, buf::SimulationBuffers;
        lsqr_atol::Float64=1e-9, lsqr_btol::Float64=1e-9, lsqr_maxiter::Int=2000,
        lsqr_error_on_nonconvergence::Bool=true,
        projection_mean_subtract::Bool=true,
    )
    step_t0 = time()

    ensure_particle_capacity!(buf, length(p.x))

    t0 = time()
    B = build_B_matrix(p, d, buf)
    t_build_B_ms = time() - t0

    if length(buf.lsqr_warm) != size(B, 2)
        buf.lsqr_warm = zeros(size(B, 2))
    end

    t0 = time()
    u_grid_n = solve_grid_velocity_lsqr(B, p, buf.V_p, buf.lsqr_warm;
                                        atol=lsqr_atol, btol=lsqr_btol,
                                        maxiter=lsqr_maxiter,
                                        error_on_nonconvergence=lsqr_error_on_nonconvergence)
    copy!(buf.lsqr_warm, u_grid_n)
    t_solve_raw_ms = time() - t0

    t0 = time()
    u_grid_n_h             = Forms.build_form_field(d.R1, u_grid_n)
    u_div_free_coeffs_n, _ = project_and_get_pressure(d, u_grid_n_h, buf;
                                                      subtract_mean=projection_mean_subtract)
    t_project_ms = time() - t0

    t_total_ms = time() - step_t0

    println(
        "  Coadjoint Timings [s]: " *
        "build_B=$(round(t_build_B_ms, digits=2)), " *
        "solve_raw=$(round(t_solve_raw_ms, digits=2)), " *
        "project=$(round(t_project_ms, digits=2)), " *
        "total=$(round(t_total_ms, digits=2))",
    )

    return u_div_free_coeffs_n
end

"""Advance one CO-FLIP timestep with fixed-point iteration."""
function step_co_flip!(
        p::Particles, d::Domain, dt::Float64,
        f_n::AbstractVector{Tf},
        buf::SimulationBuffers,
        cfg::SimulationConfig,
    ) where {Tf<:Real}

    @assert all(isfinite, f_n) "step_co_flip!: f_n contains non-finite values"

    step_t0       = time()
    thread_caches = buf.thread_caches

    mass_matrix_1form = d.Mass_matrix

    copy!(buf.f_n_saved, f_n)
    f_n_saved = buf.f_n_saved

    eff_min_pe = cfg.min_particles_per_element
    eff_max_pe = cfg.max_particles_per_element
    eff_min_pq = cfg.min_particles_per_quarter

    t0 = time()
    added_start, removed_start = enforce_min_particles_per_element!(
        p, d,
        eff_min_pe, eff_max_pe, eff_min_pq,
        f_n_saved, buf,
    )
    t_seed_start = time() - t0
    if added_start > 0 || removed_start > 0
        println("  Particle rebalance before solve: +$added_start / -$removed_start")
    end
    println("  Initial seeding time: $(round(t_seed_start, digits=2)) s")

    num_p = length(p.x)
    ensure_particle_capacity!(buf, num_p)

    x_n  = buf.x_n;   y_n  = buf.y_n
    mx_n = buf.mx_n;  my_n = buf.my_n
    @inbounds for i in 1:num_p
        x_n[i]  = p.x[i];   y_n[i]  = p.y[i]
        mx_n[i] = p.mx[i];  my_n[i] = p.my[i]
    end

    x_np1  = buf.x_np1;   y_np1  = buf.y_np1
    mx_np1 = buf.mx_np1;  my_np1 = buf.my_np1

    f_star          = buf.f_star
    f_np1           = buf.f_np1
    f_np1_raw       = buf.f_np1_raw
    f_np1_proj      = buf.f_np1_proj
    f_np1_proj_prev = buf.f_np1_proj_prev

    copy!(f_star, f_n_saved)
    copy!(f_np1_proj_prev, f_star)

    f0_norm = norm(f_n_saved)

    prev_err = Inf

    for iter in 1:cfg.max_fp_iter
        iter_t0 = time()

        maybe_clear_memo_tables_fp_iter!(iter, cfg.clear_memo_every_fp_iter, d)

        t0 = time()
        if cfg.advection_time_integrator === :rk2
            advect_particles_pullback_rk2!(
                p,
                @view(x_n[1:num_p]),  @view(y_n[1:num_p]),
                @view(mx_n[1:num_p]), @view(my_n[1:num_p]),
                f_star, d, dt,
                @view(x_np1[1:num_p]),  @view(y_np1[1:num_p]),
                @view(mx_np1[1:num_p]), @view(my_np1[1:num_p]),
                @view(buf.short_P11[1:num_p]), @view(buf.short_P12[1:num_p]),
                @view(buf.short_P21[1:num_p]), @view(buf.short_P22[1:num_p]),
                thread_caches, buf.max_err_per_thread;
                position_mode=:periodic,
            )
        elseif cfg.advection_time_integrator === :rk4
            advect_particles_pullback_rk4!(
                p,
                @view(x_n[1:num_p]),  @view(y_n[1:num_p]),
                @view(mx_n[1:num_p]), @view(my_n[1:num_p]),
                f_star, d, dt,
                @view(x_np1[1:num_p]),  @view(y_np1[1:num_p]),
                @view(mx_np1[1:num_p]), @view(my_np1[1:num_p]),
                @view(buf.short_P11[1:num_p]), @view(buf.short_P12[1:num_p]),
                @view(buf.short_P21[1:num_p]), @view(buf.short_P22[1:num_p]),
                thread_caches, buf.max_err_per_thread;
                position_mode=:periodic,
            )
        else
            throw(ArgumentError("Unknown advection_time_integrator=$(cfg.advection_time_integrator). Use :rk2 or :rk4"))
        end
        t_advect = time() - t0

        t0 = time()
        @inbounds for i in 1:num_p
            p.x[i]  = x_np1[i];   p.y[i]  = y_np1[i]
            p.mx[i] = mx_np1[i];  p.my[i] = my_np1[i]
        end
        particle_sorter!(p, d)
        t_sort = time() - t0

        t0    = time()
        B_np1 = build_B_matrix(p, d, buf)
        if length(buf.lsqr_warm) != size(B_np1, 2)
            buf.lsqr_warm = zeros(size(B_np1, 2))
        end
        t_build_B = time() - t0

        t0      = time()
        raw_sol = solve_grid_velocity_lsqr(B_np1, p, buf.V_p, buf.lsqr_warm;
                                           atol=cfg.lsqr_atol, btol=cfg.lsqr_btol,
                                           maxiter=cfg.lsqr_maxiter,
                                           error_on_nonconvergence=cfg.lsqr_error_on_nonconvergence)
        copy!(f_np1_raw, raw_sol)
        copy!(buf.lsqr_warm, raw_sol)
        t_solve = time() - t0

        t0 = time()
        if cfg.enable_energy_correction
            star_apply = v -> mass_matrix_1form * v
            apply_energy_correction!(
                f_np1, f_n_saved, f_np1_raw, f_star;
                star_apply=star_apply,
                tol=1e-14,
                project_non_orthogonal=true,
            )
        else
            copy!(f_np1, f_np1_raw)
        end
        t_energy = time() - t0

        t0             = time()
        f_np1_h        = Forms.build_form_field(d.R1, f_np1)
        proj_result, _ = project_and_get_pressure(d, f_np1_h, buf;
                                                   subtract_mean=cfg.projection_mean_subtract)
        copy!(f_np1_proj, proj_result)
        t_project = time() - t0

        err = norm(@view(f_np1_proj[1:end]) .- @view(f_np1_proj_prev[1:end]))
        err /= (f0_norm + eps(Float64))

        t_iter = time() - iter_t0
        println("  Iter $iter: Grid-vel error = $(Printf.@sprintf("%.6e", err))")
        println(
            "    Timings [s]: advect=$(round(t_advect, digits=2)), " *
            "sort=$(round(t_sort, digits=2)), " *
            "build_B=$(round(t_build_B, digits=2)), " *
            "solve_raw=$(round(t_solve, digits=2)), " *
            "energy=$(round(t_energy, digits=2)), " *
            "project=$(round(t_project, digits=2)), " *
            "iter_total=$(round(t_iter, digits=2))",
        )

        copy!(f_np1_proj_prev, f_np1_proj)
        @. f_star = 0.5 * (f_n_saved + f_np1_proj)

        if err < cfg.fp_tol
            break
        end
        if iter > 1 && err >= 0.99 * prev_err
            println("  Fixed-point stalled (err=$err >= 0.99 * prev_err=$prev_err) — break.")
            break
        end
        prev_err = err
    end

    compose_longterm_pullback!(p, buf, dt, num_p)

    if cfg.enable_pressure_kick
        t0 = time()
        if cfg.pressure_kick_method === :div_consistent
            apply_pressure_correction!(
                p, f_np1_proj, f_np1, d, thread_caches; delta_scale=1.0,
            )
            println("  [pressure kick enabled — div-consistent (FLIP/hybrid)] t=$(round(time()-t0,digits=2))s")
        elseif cfg.pressure_kick_method === :curl_consistent
            apply_pressure_correction_curl!(
                p, f_np1_proj, f_np1, d, buf, thread_caches;
                delta_scale=1.0,
                atol=cfg.lsqr_atol, btol=cfg.lsqr_btol,
                maxiter=cfg.lsqr_maxiter,
                error_on_nonconvergence=cfg.lsqr_error_on_nonconvergence,
            )
            println("  [pressure kick enabled — curl-consistent (FEEC Poisson)] t=$(round(time()-t0,digits=2))s")
        else
            throw(ArgumentError(
                "pressure_kick_method must be :div_consistent or :curl_consistent, got $(cfg.pressure_kick_method)"
            ))
        end
    end

    apply_pic_blend!(p, f_np1_proj, d, thread_caches, cfg.pic_blend_alpha)

    t0 = time()
    particle_sorter!(p, d)
    t_final_sort = time() - t0

    t0 = time()
    added_end, removed_end = enforce_min_particles_per_element!(
        p, d,
        eff_min_pe, eff_max_pe, eff_min_pq,
        f_np1_proj, buf,
    )
    t_seed_end = time() - t0
    if added_end > 0 || removed_end > 0
        println("  Particle rebalance after solve: +$added_end / -$removed_end")
    end

    t_total = time() - step_t0
    println(
        "  Post Timings [s]: " *
        "final_sort=$(round(t_final_sort, digits=2)), " *
        "final_rebalance=$(round(t_seed_end, digits=2)), " *
        "step_total=$(round(t_total, digits=2))",
    )

    return f_np1_proj
end

"""Compose long-term pullback from short-term step and accumulate time."""
function compose_longterm_pullback!(p::Particles, buf::SimulationBuffers, dt::Float64, num_p::Int)
    @inbounds for i in 1:num_p
        sp11 = buf.short_P11[i]; sp12 = buf.short_P12[i]
        sp21 = buf.short_P21[i]; sp22 = buf.short_P22[i]
        p11 = p.P11[i]; p12 = p.P12[i]
        p21 = p.P21[i]; p22 = p.P22[i]
        p.P11[i]     = sp11 * p11 + sp12 * p21
        p.P12[i]     = sp11 * p12 + sp12 * p22
        p.P21[i]     = sp21 * p11 + sp22 * p21
        p.P22[i]     = sp21 * p12 + sp22 * p22
        p.delta_t[i] += dt
    end
    return nothing
end

"""Reset particles exceeding FTLE or time-accumulated thresholds."""
function apply_ftle_reset!(
        p::Particles, d::Domain, u_phys_form,
        cfg::SimulationConfig,
    )
    Lx, Ly = d.box_size
    nx, ny = d.nel
    dx = Lx / nx;  dy = Ly / ny

    if isfinite(cfg.global_ftle_gate)
        max_ftle_global = -Inf
        @inbounds for pid in eachindex(p.x)
            dt_acc = p.delta_t[pid]
            dt_acc <= 0.0 && continue
            P11 = p.P11[pid]; P12 = p.P12[pid]
            P21 = p.P21[pid]; P22 = p.P22[pid]
            a11 = P11*P11 + P21*P21
            a22 = P12*P12 + P22*P22
            a12 = P11*P12 + P21*P22
            tra = a11 + a22
            det_a = a11*a22 - a12*a12
            disc  = max(tra*tra - 4.0 * det_a, 0.0)
            σ2    = 0.5 * (tra + sqrt(disc))
            σ_max = sqrt(max(σ2, 0.0))
            ftle  = σ_max > 0 ?
                (cfg.ftle_use_rate ? log(σ_max) / dt_acc : log(σ_max)) :
                0.0
            if ftle > max_ftle_global
                max_ftle_global = ftle
            end
        end
        if max_ftle_global < cfg.global_ftle_gate
            return 0
        end
    end

    reset_count = 0
    @inbounds for pid in eachindex(p.x)
        dt_acc = p.delta_t[pid]
        dt_acc <= 0.0 && continue

        P11 = p.P11[pid]; P12 = p.P12[pid]
        P21 = p.P21[pid]; P22 = p.P22[pid]

        a11 = P11*P11 + P21*P21
        a22 = P12*P12 + P22*P22
        a12 = P11*P12 + P21*P22
        tra = a11 + a22
        det_a = a11*a22 - a12*a12
        disc  = max(tra*tra - 4.0 * det_a, 0.0)
        σ2    = 0.5 * (tra + sqrt(disc))
        σ_max = sqrt(max(σ2, 0.0))
        ftle  = σ_max > 0 ?
            (cfg.ftle_use_rate ? log(σ_max) / dt_acc : log(σ_max)) :
            0.0

        if ftle > cfg.ftle_threshold || dt_acc > cfg.max_longterm_delta_t
            eid = p.elem_ids[pid]
            ej = ((eid - 1) ÷ nx) + 1
            ei = eid - (ej - 1) * nx
            set_g2p_velocity(
                p, pid, p.x[pid], p.y[pid], eid,
                (ei - 1) * dx, (ej - 1) * dy, dx, dy,
                u_phys_form, Lx, Ly,
            )
            p.P11[pid] = 1.0;  p.P12[pid] = 0.0
            p.P21[pid] = 0.0;  p.P22[pid] = 1.0
            p.delta_t[pid] = 0.0
            reset_count += 1
        end
    end
    return reset_count
end

"""Discard and re-seed all particles from current grid state."""
function global_reseed_from_grid!(
        p::Particles, dom::Domain, cfg::SimulationConfig,
        u_phys_form;
        rng_seed::Union{Int,Nothing}=nothing,
    )
    Lx, Ly = dom.box_size
    nx, ny = dom.nel
    dx     = Lx / nx
    dy     = Ly / ny

    target_total = nx * ny * cfg.particles_per_cell
    base_ppc     = target_total ÷ (nx * ny)
    extra        = target_total - base_ppc * (nx * ny)
    new_count    = target_total

    resize!(p.x,        new_count)
    resize!(p.y,        new_count)
    resize!(p.mx,       new_count)
    resize!(p.my,       new_count)
    resize!(p.volume,   new_count)
    resize!(p.can_x,    new_count)
    resize!(p.can_y,    new_count)
    resize!(p.next,     new_count)
    resize!(p.elem_ids, new_count)
    resize!(p.P11,      new_count)
    resize!(p.P12,      new_count)
    resize!(p.P21,      new_count)
    resize!(p.P22,      new_count)
    resize!(p.delta_t,  new_count)

    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end

    pid = 1
    if cfg.particles_per_cell > 0
        used_N = max(1, floor(Int, sqrt(base_ppc) + 1e-12))
        for ej in 1:ny, ei in 1:nx
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy
            eid = (ej - 1) * nx + ei
            for jj in 0:(used_N - 1), ii in 0:(used_N - 1)
                px = x0 + ((ii + rand()) / used_N) * dx
                py = y0 + ((jj + rand()) / used_N) * dy
                if cfg.boundary_condition === :periodic
                    px = mod(px, Lx)
                    py = mod(py, Ly)
                end
                p.x[pid] = px
                p.y[pid] = py
                set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy,
                                 u_phys_form, Lx, Ly)
                p.P11[pid] = 1.0; p.P12[pid] = 0.0
                p.P21[pid] = 0.0; p.P22[pid] = 1.0
                p.delta_t[pid] = 0.0
                p.volume[pid] = (cfg.volume_convention === :physical) ?
                    (Lx * Ly) / new_count : 1.0 / max(base_ppc, 1)
                pid += 1
            end
            remainder = base_ppc - used_N * used_N
            for _ in 1:remainder
                px = x0 + rand() * dx
                py = y0 + rand() * dy
                if cfg.boundary_condition === :periodic
                    px = mod(px, Lx)
                    py = mod(py, Ly)
                end
                p.x[pid] = px
                p.y[pid] = py
                set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy,
                                 u_phys_form, Lx, Ly)
                p.P11[pid] = 1.0; p.P12[pid] = 0.0
                p.P21[pid] = 0.0; p.P22[pid] = 1.0
                p.delta_t[pid] = 0.0
                p.volume[pid] = (cfg.volume_convention === :physical) ?
                    (Lx * Ly) / new_count : 1.0 / max(base_ppc, 1)
                pid += 1
            end
        end
        for _ in 1:extra
            px = rand() * Lx
            py = rand() * Ly
            ei = clamp(floor(Int, px * (nx / Lx)) + 1, 1, nx)
            ej = clamp(floor(Int, py * (ny / Ly)) + 1, 1, ny)
            eid = (ej - 1) * nx + ei
            x0 = (ei - 1) * dx
            y0 = (ej - 1) * dy
            p.x[pid] = px
            p.y[pid] = py
            set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy,
                             u_phys_form, Lx, Ly)
            p.P11[pid] = 1.0; p.P12[pid] = 0.0
            p.P21[pid] = 0.0; p.P22[pid] = 1.0
            p.delta_t[pid] = 0.0
            p.volume[pid] = (cfg.volume_convention === :physical) ?
                (Lx * Ly) / new_count : 1.0 / max(base_ppc, 1)
            pid += 1
        end
    end

    particle_sorter!(p, dom)

    return new_count
end

"""Clear and rewarm memoization cache periodically."""
function maybe_clear_memo_tables!(step::Int, clear_every::Int, d::Domain)
    if clear_every > 0 && (step % clear_every == 0)
        Memoization.empty_all_caches!()
        warmup_evaluation_memo!(d)
    end
end

"""Clear and rewarm memoization cache periodically during fixed-point iterations."""
function maybe_clear_memo_tables_fp_iter!(iter::Int, clear_every::Int, d::Domain)
    if clear_every > 0 && (iter % clear_every == 0)
        Memoization.empty_all_caches!()
        warmup_evaluation_memo!(d)
    end
end

"""Compute maximum dt satisfying CFL criterion from particles."""
function compute_cfl_dt_from_particles(p::Particles, d::Domain, target_cfl::T) where {T<:AbstractFloat}
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny

    max_rate = zero(T)
    @inbounds for i in 1:length(p.mx)
        u = p.mx[i];  v = p.my[i]
        rate = abs(u) / dx + abs(v) / dy
        rate > max_rate && (max_rate = rate)
    end
    max_rate <= eps(Float64) && return Inf
    return target_cfl / max_rate
end

"""Recheck CFL condition from grid velocity and update dt if needed."""
function recheck_cfl_dt(u_coeffs::AbstractVector{Float64}, d::Domain, target_cfl::Float64, cur_dt::Float64)
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny

    max_abs = 0.0
    @inbounds for c in u_coeffs
        a = abs(c)
        a > max_abs && (max_abs = a)
    end
    max_abs <= eps(Float64) && return cur_dt
    candidate = target_cfl * min(dx, dy) / max_abs
    return min(cur_dt, candidate)
end

"""Enforce CFL recheck and adapt dt if necessary."""
function enforce_cfl_recheck!(cur_dt::Float64, dt_check::Float64, cfg::SimulationConfig)
    threshold = cur_dt * (1.0 - cfg.cfl_recheck_tolerance)
    if dt_check >= threshold
        return cur_dt
    end
    if cfg.cfl_adaptive
        @info "CFL recheck shrinking dt: $(cur_dt) → $(dt_check) (tolerance=$(cfg.cfl_recheck_tolerance))"
        return dt_check
    else
        @error "CFL recheck failed: grid velocity suggests dt=$(dt_check) " *
               "but cur_dt=$(cur_dt) (threshold=$(threshold), " *
               "tolerance=$(cfg.cfl_recheck_tolerance)). Set " *
               "cfg.cfl_adaptive=true to allow the simulation to shrink dt automatically."
        return cur_dt
    end
end

"""Run startup sanity checks on De Rham complex."""
function run_startup_smoke_tests(d::Domain)
    ndof_R0 = FunctionSpaces.get_num_basis(d.R0.fem_space)
    ndof_R1 = FunctionSpaces.get_num_basis(d.R1.fem_space)
    ndof_R2 = FunctionSpaces.get_num_basis(d.R2.fem_space)
    println("  Smoke test DOFs: R0=$ndof_R0, R1=$ndof_R1, R2=$ndof_R2")
    @assert ndof_R0 > 0 && ndof_R1 > 0 && ndof_R2 > 0 "Smoke test: empty DOF space"

    c0  = fill(1.0, ndof_R0)
    f0  = Forms.build_form_field(d.R0, c0)
    df0 = Forms.d(f0)
    qrule = Quadrature.get_canonical_quadrature_rule(d.dΩ)
    df0_norm = 0.0
    for elem_id in 1:Geometry.get_num_elements(d.geo)
        df0_eval, _ = Forms.evaluate(df0, elem_id, Quadrature.get_nodes(qrule))
        for component in df0_eval
            df0_norm = max(df0_norm, maximum(abs.(component)))
        end
    end
    println("  Smoke test ||d(1_R0)||_max = $(Printf.@sprintf("%.3e", df0_norm))")
    @assert df0_norm < 1e-8 "Smoke test: d(constant) not zero (got $df0_norm)"

    c2       = fill(1.0, ndof_R2)
    f2       = Forms.build_form_field(d.R2, c2)
    integral = integrate_form_expression(∫(f2, d.dΩ), d)
    expected = d.box_size[1] * d.box_size[2]
    err = abs(integral - expected)
    println("  Smoke test ∫1 dΩ = $integral (expected $expected, err=$(Printf.@sprintf("%.3e", err)))")
    @assert err < 1e-6 "Smoke test: integral consistency failed"
    return nothing
end

"""L2 norm of a 1-form coefficient vector via the assembled mass matrix."""
function l2_velocity_norm(u_coeffs::AbstractVector{<:Real}, dom::Domain)
    return sqrt(max(0.0, dot(u_coeffs, dom.Mass_matrix * u_coeffs)))
end

"""Write per-step conservation diagnostics to a CSV file."""
function save_diagnostics_csv(history, path::String)
    open(path, "w") do io
        println(io, "t,energy,enstrophy,circulation")
        for h in history
            @printf(io, "%.10e,%.16e,%.16e,%.16e\n",
                    h.t, h.energy, h.enstrophy, h.circulation)
        end
    end
    return path
end

"""
Run one CO-FLIP simulation and record (t, energy, enstrophy, circulation) every `record_every` steps.
Slimmed-down counterpart to `main()` — skips VTK/particle dumps unless `save_vtk=true`.
Returns initial and final velocity coefficients so callers can compute custom error norms.
"""
function run_diagnostic_simulation(cfg::SimulationConfig;
        save_vtk::Bool=false,
        output_dir::String=joinpath(DEFAULT_OUTPUT_DIR, "diag_output"),
        record_every::Int=1,
        case_name::String="",
    )
    LinearAlgebra.BLAS.set_num_threads(1)
    save_vtk && mkpath(output_dir)

    num_particles = cfg.nel[1] * cfg.nel[2] * cfg.particles_per_cell
    domain    = GenerateDomain(cfg.nel, cfg.p, cfg.k;
                               box_size=cfg.box_size,
                               starting_point=cfg.starting_point,
                               boundary_condition=cfg.boundary_condition)
    particles = generate_particles(num_particles, domain, cfg.flow_type;
                                   stratified_seeding=cfg.stratified_seeding,
                                   rng_seed=cfg.rng_seed,
                                   volume_convention=cfg.volume_convention,
                                   boundary_condition=cfg.boundary_condition)
    particle_sorter!(particles, domain)

    dt_cfl = compute_cfl_dt_from_particles(particles, domain, cfg.target_cfl)
    if !isfinite(dt_cfl)
        n_steps = 1;  dt = cfg.T_final
    else
        n_steps = max(1, ceil(Int, cfg.T_final / dt_cfl))
        dt      = cfg.T_final / n_steps
    end

    ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
    num_elements = prod(domain.nel)
    sim_buf      = SimulationBuffers(num_particles, ndofs, num_elements, domain)

    u_coeffs = coadjoint_step!(particles, domain, sim_buf;
        lsqr_atol=cfg.lsqr_atol, lsqr_btol=cfg.lsqr_btol, lsqr_maxiter=cfg.lsqr_maxiter,
        lsqr_error_on_nonconvergence=cfg.lsqr_error_on_nonconvergence,
        projection_mean_subtract=cfg.projection_mean_subtract)
    u_initial = copy(u_coeffs)

    u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
    u_phys_expr = ★(u_form)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)
    particle_sorter!(particles, domain)

    nx, ny = domain.nel
    Lx, Ly = domain.box_size
    dx = Lx / nx;  dy = Ly / ny
    @inbounds for pid in eachindex(particles.x)
        eid = particles.elem_ids[pid]
        ej  = ((eid - 1) ÷ nx) + 1
        ei  = eid - (ej - 1) * nx
        set_g2p_velocity(particles, pid, particles.x[pid], particles.y[pid],
                         eid, (ei - 1) * dx, (ej - 1) * dy, dx, dy,
                         u_phys_form, Lx, Ly)
    end

    if save_vtk
        d_u_phys_0 = Forms.d(u_phys_form)
        ω_h_0      = ★(d_u_phys_0)
        Plot.export_form_fields_to_vtk((u_form,), @sprintf("u_h_0000"); output_directory_tree=[output_dir])
        Plot.export_form_fields_to_vtk((ω_h_0,),  @sprintf("w_h_0000"); output_directory_tree=[output_dir])
    end

    warmup_evaluation_memo!(domain)

    d0 = compute_conservation_diagnostics(u_coeffs, domain)
    history = [(t=0.0, energy=d0.energy, enstrophy=d0.enstrophy, circulation=d0.circulation)]

    for step in 1:n_steps
        println(@sprintf("Step %d/%d, case: %s", step, n_steps, case_name))
        u_coeffs = step_co_flip!(particles, domain, dt, u_coeffs, sim_buf, cfg)

        u_form_loc      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
        u_phys_expr_loc = ★(u_form_loc)
        u_phys_form_loc = Assemblers.solve_L2_projection(domain.R1, u_phys_expr_loc, domain.dΩ)

        if cfg.delayed_reinit_frequency > 0 && step % cfg.delayed_reinit_frequency == 0
            global_reseed_from_grid!(particles, domain, cfg, u_phys_form_loc; rng_seed=cfg.rng_seed)
        end
        apply_ftle_reset!(particles, domain, u_phys_form_loc, cfg)

        dt_check = recheck_cfl_dt(u_coeffs, domain, cfg.target_cfl, dt)
        new_dt   = enforce_cfl_recheck!(dt, dt_check, cfg)
        new_dt != dt && (dt = new_dt)

        if step % record_every == 0
            d = compute_conservation_diagnostics(u_coeffs, domain)
            push!(history, (t=step*dt, energy=d.energy, enstrophy=d.enstrophy, circulation=d.circulation))

            if save_vtk
                d_u_phys = Forms.d(u_phys_form_loc)
                ω_h      = ★(d_u_phys)
                Plot.export_form_fields_to_vtk((u_form_loc,),
                    @sprintf("u_h_%04d", step); output_directory_tree=[output_dir])
                Plot.export_form_fields_to_vtk((ω_h,),
                    @sprintf("w_h_%04d", step); output_directory_tree=[output_dir])
            end
        end

        maybe_clear_memo_tables!(step, cfg.clear_memo_every, domain)
    end

    return (history=history, u_initial=u_initial, u_final=u_coeffs,
            domain=domain, dt=dt, n_steps=n_steps)
end

"""
Run the canonical 2D periodic inviscid Euler test suite for CO-FLIP.

Each case records (energy, enstrophy, circulation) over time → CSV, and prints conservation
drifts. The Taylor-Green case additionally reports ‖u(T)-u(0)‖₂/‖u(0)‖₂ — that ratio should
be tiny because the IC is an exact stationary solution of 2D Euler.

Set `save_vtk=true` to dump VTK snapshots per case (useful for visually inspecting the
double-shear roll-up, leapfrog dynamics, four-vortex symmetry, etc.). Use `T_factor` to
shorten/lengthen all runs uniformly.
"""
function run_test_suite(;
        output_dir::String=joinpath(DEFAULT_OUTPUT_DIR, "diag_output"),
        save_vtk::Bool=true,
        T_factor::Float64=1.0,
        nel::NTuple{2,Int}=(96, 96),
        particles_per_cell::Int=20,
    )
    mkpath(output_dir)

    cases = (
        (name="leapfrog",     flow=:leapfrog,    T=1.0, note="Two coaxial dipoles"),
    )

    println("\n========== CO-FLIP TEST SUITE (nel=$(nel)) ==========")
    summary_rows = String[]

    for case in cases
        println("\n--- $(case.name): $(case.note) ---")
        cfg = SimulationConfig(;
            flow_type=case.flow,
            T_final=case.T * T_factor,
            nel=nel,
            particles_per_cell=particles_per_cell,
            output_every=0,
        )

        result = nothing
        try
            result = run_diagnostic_simulation(cfg;
                save_vtk=save_vtk,
                output_dir=joinpath(output_dir, case.name),
                record_every=1,
                case_name=case.name)
        catch err
            @warn "$(case.name) FAILED" exception=err
            push!(summary_rows, @sprintf("%-13s | FAILED: %s", case.name, err))
            continue
        end

        save_diagnostics_csv(result.history, joinpath(output_dir, case.name * ".csv"))

        h0, hT = result.history[1], result.history[end]
        ΔE = abs(hT.energy    - h0.energy)    / max(abs(h0.energy),    1e-30)
        ΔZ = abs(hT.enstrophy - h0.enstrophy) / max(abs(h0.enstrophy), 1e-30)
        ΔΓ = abs(hT.circulation - h0.circulation)

        extra = ""
        if case.flow === :tg
            diff_norm = l2_velocity_norm(result.u_final .- result.u_initial, result.domain)
            init_norm = l2_velocity_norm(result.u_initial, result.domain)
            tg_relerr = diff_norm / max(init_norm, 1e-30)
            extra = @sprintf(" | stationary L2 relerr=%.3e", tg_relerr)
        end

        @printf("  ΔE/E0 = %.3e | ΔZ/Z0 = %.3e | |ΔΓ| = %.3e%s\n", ΔE, ΔZ, ΔΓ, extra)
        push!(summary_rows,
              @sprintf("%-13s | ΔE/E0=%.2e | ΔZ/Z0=%.2e | |ΔΓ|=%.2e%s",
                       case.name, ΔE, ΔZ, ΔΓ, extra))
    end

    println("\n========== SUMMARY ==========")
    foreach(println, summary_rows)

    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "CO-FLIP test suite — $(length(cases)) cases, nel=$(nel)")
        for row in summary_rows; println(io, row); end
    end

    return summary_rows
end

"""Run complete CO-FLIP simulation."""
function main(cfg::SimulationConfig=SimulationConfig())
    println("Initializing Domain and Particles...")
    LinearAlgebra.BLAS.set_num_threads(1)

    num_particles = cfg.nel[1] * cfg.nel[2] * cfg.particles_per_cell

    domain    = GenerateDomain(cfg.nel, cfg.p, cfg.k;
                               box_size=cfg.box_size,
                               starting_point=cfg.starting_point,
                               boundary_condition=cfg.boundary_condition)
    particles = generate_particles(num_particles, domain, cfg.flow_type;
                                   stratified_seeding=cfg.stratified_seeding,
                                   rng_seed=cfg.rng_seed,
                                   volume_convention=cfg.volume_convention,
                                   boundary_condition=cfg.boundary_condition)
    particle_sorter!(particles, domain)

    run_startup_smoke_tests(domain)

    dt_cfl = compute_cfl_dt_from_particles(particles, domain, cfg.target_cfl)
    if !isfinite(dt_cfl)
        n_steps = 1;  dt = cfg.T_final
    else
        n_steps = max(1, ceil(Int, cfg.T_final / dt_cfl))
        dt      = cfg.T_final / n_steps
    end

    println("Starting Time Integration (T_final=$(cfg.T_final), dt=$dt, steps=$n_steps)...")

    output_dir          = "coflip_output";  mkpath(output_dir)
    particle_output_dir = "coflip_output";  mkpath(particle_output_dir)

    ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
    num_elements = prod(domain.nel)
    sim_buf      = SimulationBuffers(num_particles, ndofs, num_elements, domain)

    println("Saving true t=0 snapshot...")
    u_coeffs = coadjoint_step!(particles, domain, sim_buf;
        lsqr_atol=cfg.lsqr_atol, lsqr_btol=cfg.lsqr_btol, lsqr_maxiter=cfg.lsqr_maxiter,
        lsqr_error_on_nonconvergence=cfg.lsqr_error_on_nonconvergence,
        projection_mean_subtract=cfg.projection_mean_subtract)

    u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
    u_phys_expr = ★(u_form)
    u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

    particle_sorter!(particles, domain)
    nx, ny = domain.nel
    Lx, Ly = domain.box_size
    dx = Lx / nx;  dy = Ly / ny

    @inbounds for pid in eachindex(particles.x)
        eid = particles.elem_ids[pid]
        ej  = ((eid - 1) ÷ nx) + 1
        ei  = eid - (ej - 1) * nx
        set_g2p_velocity(
            particles, pid,
            particles.x[pid], particles.y[pid],
            eid, (ei - 1) * dx, (ej - 1) * dy, dx, dy,
            u_phys_form, Lx, Ly,
        )
    end

    d_u_phys = Forms.d(u_phys_form)
    ω_h      = ★(d_u_phys)
    Plot.export_form_fields_to_vtk((u_form,), "u_h_0000"; output_directory_tree=["coflip_output"])
    Plot.export_form_fields_to_vtk((ω_h,),    "w_h_0000"; output_directory_tree=["coflip_output"])
    export_particles_to_vtk(particles, joinpath(particle_output_dir, "particles_0000"))
    println("  Saved step 0 particles to VTK")

    warmup_evaluation_memo!(domain)

    for step in 1:n_steps
        println("Step $step / $n_steps (t = $(round(step * dt, digits=3)))...")

        u_coeffs = step_co_flip!(particles, domain, dt, u_coeffs, sim_buf, cfg)

        u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
        u_phys_expr = ★(u_form)
        u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

        if cfg.delayed_reinit_frequency > 0 &&
           step % cfg.delayed_reinit_frequency == 0
            new_count = global_reseed_from_grid!(particles, domain, cfg, u_phys_form;
                                                 rng_seed=cfg.rng_seed)
            println("  Delayed re-seed at step $step: $(new_count) particles re-generated from grid.")
        end

        reset_count = apply_ftle_reset!(particles, domain, u_phys_form, cfg)
        reset_count > 0 && println("  FTLE reset: $reset_count particles reseeded.")

        dt_check = recheck_cfl_dt(u_coeffs, domain, cfg.target_cfl, dt)
        new_dt   = enforce_cfl_recheck!(dt, dt_check, cfg)
        if new_dt != dt
            remaining_steps = max(1, ceil(Int, (cfg.T_final - step * dt) / new_dt))
            println("  Adaptive CFL: dt=$(new_dt) (was $(dt)); remaining_steps≈$(remaining_steps)")
            dt = new_dt
        end

        if cfg.output_every > 0 && step % cfg.output_every == 0
            d_u_phys = Forms.d(u_phys_form)
            ω_h      = ★(d_u_phys)
            Plot.export_form_fields_to_vtk((u_form,), @sprintf("u_h_%04d", step); output_directory_tree=["coflip_output"])
            Plot.export_form_fields_to_vtk((ω_h,),    @sprintf("w_h_%04d", step); output_directory_tree=["coflip_output"])

            diagnostics = compute_conservation_diagnostics(u_coeffs, domain)
            print_conservation_diagnostics(step, step * dt, diagnostics)

            export_particles_to_vtk(
                particles,
                joinpath(particle_output_dir, @sprintf("particles_%04d", step)),
            )
            println("  Saved step $step visualization and particle data.")
        end

        maybe_clear_memo_tables!(step, cfg.clear_memo_every, domain)
    end

    println("\nSimulation complete! Step files written to '$(output_dir)'.")
end

#main()

run_test_suite()
