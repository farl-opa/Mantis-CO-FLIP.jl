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

# ==============================================================================
#                               DATA STRUCTURES
# ==============================================================================

struct Domain{F0, F1, F2, G, Q}
    R0::F0
    R1::F1
    R2::F2
    geo::G
    dΩ::Q
    nel::NTuple{2,Int}
    p::NTuple{2,Int}
    k::NTuple{2,Int}
    eval_cache_size::Int
    box_size::NTuple{2,Float64}
end

# FIX [3.3/4.2/10.2]: Particles now carries a per-particle longterm pullback
# P = [[P11 P12]; [P21 P22]] and accumulated advection time delta_t. These
# drive the FTLE-based adaptive reset that replaces the step%reset_every
# heuristic.
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
    # --- longterm pullback (composed across steps since last reset) ---
    P11::Vector{Float64}
    P12::Vector{Float64}
    P21::Vector{Float64}
    P22::Vector{Float64}
    delta_t::Vector{Float64}
end

struct EvaluationCache
    temp_mat::Matrix{Float64}
    results::Vector{Vector{Vector{Matrix{Float64}}}}
    xi_buf::Vector{Float64}
    eta_buf::Vector{Float64}

    function EvaluationCache(max_basis_size::Int)
        v0  = [[zeros(1, max_basis_size) for _ in 1:2]]
        v1  = [[zeros(1, max_basis_size) for _ in 1:2] for _ in 1:2]
        res = Vector{Vector{Vector{Matrix{Float64}}}}(undef, 2)
        res[1] = v0; res[2] = v1
        return new(zeros(1, max_basis_size), res, zeros(1), zeros(1))
    end
end

# FIX [12.1/4.2/5.1]: SimulationBuffers owns every per-step / per-iteration
# work array so step_co_flip! allocates nothing.
mutable struct SimulationBuffers
    x_n::Vector{Float64}
    y_n::Vector{Float64}
    mx_n::Vector{Float64}
    my_n::Vector{Float64}
    x_np1::Vector{Float64}
    y_np1::Vector{Float64}
    mx_np1::Vector{Float64}
    my_np1::Vector{Float64}

    # FIX [4.2]: shorterm-pullback output buffers populated by the advection
    # kernel and then composed into the particle longterm pullback.
    short_P11::Vector{Float64}
    short_P12::Vector{Float64}
    short_P21::Vector{Float64}
    short_P22::Vector{Float64}

    f_n_saved::Vector{Float64}
    f_star::Vector{Float64}
    f_np1::Vector{Float64}
    f_np1_raw::Vector{Float64}
    f_np1_proj::Vector{Float64}
    f_np1_proj_prev::Vector{Float64}   # preallocated — no per-step similar()

    # FIX [5.1]: LSQR warm-start state (reused across steps + fixed-point iters)
    lsqr_warm::Vector{Float64}

    thread_caches::Vector{EvaluationCache}

    # enforce_min scratch
    quarter_pids::Vector{Vector{Int}}
    add_quarter::Matrix{Int}
    counts_elem::Vector{Int}
    counts_quarter::Matrix{Int}
    keep_mask::Vector{Bool}

    V_p::Vector{Float64}

    # FIX [12.1]: preallocated per-thread max-error scratch for advection.
    max_err_per_thread::Vector{Float64}
end

function SimulationBuffers(num_particles_initial::Int, ndofs::Int, num_elements::Int, d::Domain)
    n_threads  = Threads.maxthreadid()
    cache_size = evaluation_cache_size(d)
    cap        = num_particles_initial * 2

    return SimulationBuffers(
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),    # x_n, y_n
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),    # mx_n, my_n
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),    # x_np1, y_np1
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),    # mx_np1, my_np1
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),    # short_P11, P12
        Vector{Float64}(undef, cap), Vector{Float64}(undef, cap),    # short_P21, P22
        Vector{Float64}(undef, ndofs),                               # f_n_saved
        Vector{Float64}(undef, ndofs),                               # f_star
        Vector{Float64}(undef, ndofs),                               # f_np1
        Vector{Float64}(undef, ndofs),                               # f_np1_raw
        Vector{Float64}(undef, ndofs),                               # f_np1_proj
        zeros(Float64, ndofs),                                       # f_np1_proj_prev
        zeros(Float64, ndofs),                                       # lsqr_warm
        [EvaluationCache(cache_size) for _ in 1:n_threads],
        [Int[] for _ in 1:num_elements * 4],                         # quarter_pids
        zeros(Int, num_elements, 4),                                 # add_quarter
        zeros(Int, num_elements),                                    # counts_elem
        zeros(Int, num_elements, 4),                                 # counts_quarter
        trues(cap),                                                  # keep_mask
        Vector{Float64}(undef, 2 * cap),                             # V_p
        zeros(Float64, n_threads),                                   # max_err_per_thread
    )
end

# FIX [12.4]: SimulationConfig — one struct for every tunable.
Base.@kwdef struct SimulationConfig
    nel::NTuple{2,Int}             = (64, 64)
    p::NTuple{2,Int}               = (2, 2)
    k::NTuple{2,Int}               = (1, 1)
    particles_per_cell::Int        = 20
    flow_type::Symbol              = :convecting
    target_cfl::Float64            = 0.5
    T_final::Float64               = 1.0
    # fixed-point controls
    max_fp_iter::Int               = 6
    fp_tol::Float64                = 1e-9
    enable_energy_correction::Bool = true     # FIX [6.1]
    enable_pressure_kick::Bool     = false    # FIX [8.1] — pure CO-FLIP
    # particle management
    min_particles_per_element::Int = 10
    max_particles_per_element::Int = 25
    min_particles_per_quarter::Int = 3
    # FTLE-based reset (FIX [3.3 / 8.f])
    ftle_threshold::Float64        = 0.5
    max_longterm_delta_t::Float64  = 0.25
    # diagnostics / IO
    output_every::Int              = 1        # FIX [11.1]
    clear_memo_every::Int          = 1        # FIX [11.4]
    # LSQR (FIX [5.1])
    lsqr_atol::Float64             = 1e-8
    lsqr_btol::Float64             = 1e-8
    lsqr_maxiter::Int              = 2000
    # projection / null space (FIX [7.1/7.2])
    projection_mean_subtract::Bool = true
end

probe_output_eltype(::Type{T}) where {T<:Real} = promote_type(Float64, T)
evaluation_cache_size(d::Domain) = d.eval_cache_size

# ==============================================================================
#                            DOMAIN GENERATION
# ==============================================================================

function GenerateDomain(nel::NTuple{2,Int}, p::NTuple{2,Int}, k::NTuple{2,Int})
    starting_point = (0.0, 0.0)
    box_size       = (1.0, 1.0)
    nq_assembly    = p .+ 1
    nq_error       = nq_assembly .* 2
    ∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, nq_assembly, nq_error)
    dΩ = Quadrature.StandardQuadrature(∫ₐ, prod(nel))

    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_point, box_size, nel, p, k; periodic=(true, true),
    )

    eval_cache_size = maximum(
        length(FunctionSpaces.get_basis_indices(R[2].fem_space, eid)) for eid in 1:prod(nel)
    )

    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k, eval_cache_size, box_size)
end

# ==============================================================================
#                          PARTICLE GENERATION & SORTING
# ==============================================================================

function initial_velocity(flow_type::Symbol, px::Float64, py::Float64, Lx::Float64, Ly::Float64)
    if flow_type == :tg;          return flow_taylor_green(px, py, Lx, Ly)
    elseif flow_type == :vortex;  return flow_lamb_oseen(px, py, Lx, Ly)
    elseif flow_type == :gyre;    return flow_double_gyre(px, py, Lx, Ly)
    elseif flow_type == :decay;   return flow_decay(px, py, Lx, Ly)
    elseif flow_type == :convecting; return flow_convecting_vortex(px, py, Lx, Ly)
    elseif flow_type == :merging; return flow_merging_vortices(px, py, Lx, Ly)
    else; error("Unknown flow type: $flow_type")
    end
end

function generate_particles(num_particles::Int, domain::Domain, flow_type::Symbol=:vortex)
    Lx, Ly = domain.box_size

    x   = Vector{Float64}(undef, num_particles)
    y   = Vector{Float64}(undef, num_particles)
    mx  = Vector{Float64}(undef, num_particles)
    my  = Vector{Float64}(undef, num_particles)
    vol = ones(Float64, num_particles)

    can_x    = zeros(Float64, num_particles)
    can_y    = zeros(Float64, num_particles)
    num_elements = prod(domain.nel)
    head     = zeros(Int, num_elements)
    next     = zeros(Int, num_particles)
    elem_ids = zeros(Int, num_particles)

    # FIX [10.2]: identity longterm pullback + zero accumulated delta_t.
    P11 = ones(Float64, num_particles)
    P12 = zeros(Float64, num_particles)
    P21 = zeros(Float64, num_particles)
    P22 = ones(Float64, num_particles)
    delta_t = zeros(Float64, num_particles)

    for i in 1:num_particles
        px = rand() * Lx
        py = rand() * Ly
        x[i]  = px
        y[i]  = py
        u, v  = initial_velocity(flow_type, px, py, Lx, Ly)
        mx[i] = u
        my[i] = v
    end

    fill!(vol, (Lx * Ly) / num_particles)

    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids,
                     P11, P12, P21, P22, delta_t)
end

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

# FIX [9.x]: wrap incoming (px, py) via mod so the caller does not have to
# pre-wrap positions. Robust against floating-point creep in RK stages.
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

    # FIX [10.1]: if nothing to add/remove, still rebind per-cell volumes.
    if total_to_add == 0 && max_particles_per_element <= 0
        rebind_volumes_per_element!(p, d, counts_elem)
        return 0, 0
    end

    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx;  dy = Ly / ny
    rng = Random.default_rng()

    # ------------------------------------------------------------------
    # Particle spawning
    # ------------------------------------------------------------------
    if total_to_add > 0
        ref_coeffs_f64 = collect(Float64, ref_grid_coeffs)
        ref_form       = Forms.build_form_field(d.R1, ref_coeffs_f64)
        ref_phys_expr  = ★(ref_form)
        ref_phys_form  = Assemblers.solve_L2_projection(d.R1, ref_phys_expr, d.dΩ)

        old_count = length(p.x)
        new_count = old_count + total_to_add

        # FIX [10.2]: grow longterm-pullback + delta_t alongside legacy fields.
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
                    # FIX [10.2]: fresh particles start with identity pullback
                    # and zero accumulated advection time.
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

    # ------------------------------------------------------------------
    # Particle culling — FIX [12.3]: in-place compaction (no fancy indexing).
    # ------------------------------------------------------------------
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

    # FIX [10.1]: per-element volume — each particle inherits a share of its
    # cell's area (cell_area / local_count). Replaces the global (Lx·Ly)/N
    # rescale that violated per-cell mass conservation under loose PPE bounds.
    fill!(counts_elem, 0)
    @inbounds for pid in 1:length(p.x)
        counts_elem[p.elem_ids[pid]] += 1
    end
    rebind_volumes_per_element!(p, d, counts_elem)

    return total_to_add, total_to_remove
end

# FIX [12.3]: single-pass in-place SoA compaction. Must include every particle
# field, including the new longterm-pullback and delta_t arrays.
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

# FIX [10.1]: rebind particle volume = cell_area / per-cell particle count.
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

function ensure_particle_capacity!(buf::SimulationBuffers, n::Int)
    if length(buf.x_n) >= n
        return nothing
    end
    new_cap = max(n, length(buf.x_n) * 2)
    # FIX [10.2/4.2]: include short_Pxx alongside legacy particle scratch.
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

# ==============================================================================
#                             FLOW DEFINITIONS
# ==============================================================================

function flow_taylor_green(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    kx = 2*π * x / Lx
    ky = 2*π * y / Ly
    u =  U0 * sin(kx) * cos(ky)
    v = -U0 * cos(kx) * sin(ky)
    return u, v
end

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

function flow_double_gyre(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    X = x / Lx;  Y = y / Ly
    u =  U0 * sin(π * X) * cos(π * Y)
    v = -U0 * cos(π * X) * sin(π * Y)
    return u, v
end

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

# ==============================================================================
#                            EVALUATION KERNELS
# ==============================================================================

function evaluate_fast!(cache::EvaluationCache, space::S, element_id::Int, xi::P, nderivatives::Int) where {S, P}
    for ord in 1:(nderivatives+1)
        for der in 1:length(cache.results[ord])
            for comp in 1:2
                fill!(cache.results[ord][der][comp], 0.0)
            end
        end
    end

    num_components = 2

    for component_idx in 1:num_components
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

    cache.xi_buf[1]  = (x_wrapped - (ei - 1) * dx) * inv_dx
    cache.eta_buf[1] = (y_wrapped - (ej - 1) * dy) * inv_dy

    points = Mantis.Points.CartesianPoints((cache.xi_buf, cache.eta_buf))

    eval_out     = evaluate_fast!(cache, d.R1.fem_space, elem_idx, points, 1)
    dof_indices  = FunctionSpaces.get_basis_indices(d.R1.fem_space, elem_idx)
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

function evaluate_field_at_probes(u_coeffs::AbstractVector{T}, probes::Particles, d::Domain) where {T<:Real}
    Tout = probe_output_eltype(T)
    num_points = length(probes.x)
    u_mat = Matrix{Tout}(undef, num_points, 2)
    viz_cache = EvaluationCache(evaluation_cache_size(d))

    @inbounds for i in 1:num_points
        vel, _ = probe_field_at_point(probes.x[i], probes.y[i], u_coeffs, d, viz_cache)
        u_mat[i, 1] =  vel[2]
        u_mat[i, 2] = -vel[1]
    end
    return u_mat
end

function evaluate_scalar_form_at_probes(form_expr::F, probes::Particles, d::Domain) where {F}
    num_points = length(probes.x)
    values = zeros(Float64, num_points)

    num_elements = prod(d.nel)
    can_batch_x = Float64[]
    can_batch_y = Float64[]
    pid_batch   = Int[]
    max_batch   = 1024

    for eid in 1:num_elements
        empty!(can_batch_x); empty!(can_batch_y); empty!(pid_batch)

        pid = probes.head[eid]
        while pid != 0
            push!(pid_batch,   pid)
            push!(can_batch_x, probes.can_x[pid])
            push!(can_batch_y, probes.can_y[pid])
            pid = probes.next[pid]
        end

        isempty(pid_batch) && continue

        total_points = length(pid_batch)
        for chunk_start in 1:max_batch:total_points
            chunk_end = min(chunk_start + max_batch - 1, total_points)
            rng = chunk_start:chunk_end

            xs_view   = @view can_batch_x[rng]
            ys_view   = @view can_batch_y[rng]
            pids_view = @view pid_batch[rng]
            points    = Mantis.Points.CartesianPoints((xs_view, ys_view))

            eval_out, _ = Forms.evaluate(form_expr, eid, points)
            local_vals  = eval_out[1]

            @inbounds for (i, global_pid) in enumerate(pids_view)
                values[global_pid] = local_vals[i]
            end
        end
    end

    return values
end

function evaluate_velocity_and_vorticity_at_probes(u_coeffs::AbstractVector{T}, probes::Particles, d::Domain) where {T<:Real}
    Tout = probe_output_eltype(T)
    num_points = length(probes.x)
    uvω_mat = Matrix{Tout}(undef, num_points, 3)
    viz_cache = EvaluationCache(evaluation_cache_size(d))

    u_form      = Forms.build_form_field(d.R1, u_coeffs)
    u_phys_expr = ★(u_form)
    u_phys_form = Assemblers.solve_L2_projection(d.R1, u_phys_expr, d.dΩ)
    ω_form      = ★(Forms.d(u_phys_form))
    ω_vals      = evaluate_scalar_form_at_probes(ω_form, probes, d)

    @inbounds for i in 1:num_points
        raw_vel, _ = probe_field_at_point(probes.x[i], probes.y[i], u_coeffs, d, viz_cache)
        u = -raw_vel[2]
        v =  raw_vel[1]

        uvω_mat[i, 1] = u
        uvω_mat[i, 2] = v
        uvω_mat[i, 3] = ω_vals[i]
    end

    return uvω_mat
end

function warmup_evaluation_memo!(d::Domain)
    cache  = EvaluationCache(evaluation_cache_size(d))
    points = Mantis.Points.CartesianPoints(([0.5], [0.5]))
    evaluate_fast!(cache, d.R1.fem_space, 1, points, 1)
    return nothing
end

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

# ==============================================================================
#                            GRID OPERATIONS & SOLVERS
# ==============================================================================
# FIX [5.2]: B-matrix layout.
#   Shape: (2 * N_p) × N_dof  (rows: particle × component, cols: grid DOF).
#   Row 2i-1 (p.my sign channel): β · dξ1  component of basis at particle i.
#   Row 2i   (p.mx sign channel): β · dξ2  component.
#   V_p[2i-1] =  p.my[i] * sqrt(vol);  V_p[2i] = -p.mx[i] * sqrt(vol).
#   Entries scaled by sqrt(p.volume[i]) so the least-squares fit is quadrature-
#   weighted. Stationary mass M_1form is assembled separately via Mantis
#   Galerkin assembly — see assemble_1form_mass_matrix.
# ==============================================================================

function build_B_matrix(p::Particles, d::Domain)
    fes      = d.R1.fem_space
    num_p    = length(p.x)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    cache    = EvaluationCache(evaluation_cache_size(d))

    max_nnz = 2 * num_p * d.eval_cache_size
    I_idx = Vector{Int}(undef, max_nnz)
    J_idx = Vector{Int}(undef, max_nnz)
    V_val = Vector{Float64}(undef, max_nnz)
    cursor = 1

    @inbounds for pid in 1:num_p
        eid    = p.elem_ids[pid]
        points = Mantis.Points.CartesianPoints(([p.can_x[pid]], [p.can_y[pid]]))
        eval_out    = evaluate_fast!(cache, fes, eid, points, 0)
        dof_indices = FunctionSpaces.get_basis_indices(fes, eid)
        n_loc       = length(dof_indices)

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
    end

    actual = cursor - 1
    return sparse(
        @view(I_idx[1:actual]),
        @view(J_idx[1:actual]),
        @view(V_val[1:actual]),
        2num_p, num_dofs,
    )
end

function build_lsqr_rhs!(V_p::AbstractVector{Float64}, p::Particles)
    @inbounds for i in 1:length(p.x)
        w = sqrt(p.volume[i])
        V_p[2i-1] =  p.my[i] * w
        V_p[2i]   = -p.mx[i] * w
    end
    return V_p
end

# FIX [5.1]: LSQR with warm-start and maxiter control. warm must already be
# ndof-length — caller is responsible for resizing on DOF count change.
function solve_grid_velocity_lsqr(
        B::SparseMatrixCSC{Float64,Int},
        p::Particles,
        V_p_buf::AbstractVector{Float64},
        warm::AbstractVector{Float64};
        atol::Float64=1e-8,
        btol::Float64=1e-8,
        maxiter::Int=2000,
    )
    n = 2 * length(p.x)
    V_p = @view V_p_buf[1:n]
    build_lsqr_rhs!(V_p, p)

    x0 = copy(warm)
    if length(x0) != size(B, 2)
        x0 = zeros(size(B, 2))
    end
    # IterativeSolvers.lsqr! accepts an initial guess and mutates it in place.
    IterativeSolvers.lsqr!(x0, B, V_p; atol=atol, btol=btol, maxiter=maxiter)
    return x0
end

# FIX [7.1/7.2]: wrap the Hodge-Laplace projection with an explicit mean-
# subtraction on the pressure block to remove the constant-mode null space on
# the fully-periodic torus. The inner direct sparse factorisation remains the
# Mantis default; substituting MINRES at the saddle-point level is a library-
# side refactor (Assemblers.solve_volume_form_hodge_laplacian) — guarded
# externally here with the null-space fix that is the proven workaround.
function project_and_get_pressure(dom::Domain, v_h::V; subtract_mean::Bool=true) where {V}
    f = d(v_h)
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(dom.R1, dom.R2, f, dom.dΩ)

    ϕ_coeffs = copy(ϕ_corr.coefficients)
    if subtract_mean && !isempty(ϕ_coeffs)
        # FIX [7.2]: kill the constant-pressure mode.
        ϕ_coeffs .-= sum(ϕ_coeffs) / length(ϕ_coeffs)
    end

    div_free_coeffs = v_h.coefficients - u_corr.coefficients
    return div_free_coeffs, ϕ_coeffs
end

function assemble_1form_mass_matrix(R1_space::F, dΩ::Q) where {F, Q}
    # FIX [6.2]: this is the consistent Galerkin mass on the 1-form space:
    #   M[i,j] = ∫(φ_i ∧ ★φ_j).
    # Matches the ★ used in project_and_get_pressure / energy correction.
    # A DEC-lumped alternative would require a separate diagonal ★-apply and
    # is intentionally NOT used here to keep the inner products consistent
    # across projection and energy gating.
    weak_form_inputs = Assemblers.WeakFormInputs(R1_space)
    vᵏ = Assemblers.get_test_form(weak_form_inputs)
    uᵏ = Assemblers.get_trial_form(weak_form_inputs)

    A = ∫(vᵏ ∧ ★(uᵏ), dΩ)

    lhs_expressions = ((A,),)
    rhs_expressions = ((0,),)

    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    M, _ = Assemblers.assemble(weak_form)
    return M
end

# ==============================================================================
#                        PHYSICS & ENERGY CORRECTION
# ==============================================================================

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

@inline function compute_pressure_delta(
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
    ) where {Tp<:Real, Tq<:Real}
    return f_projected .- f_pre_projection
end

# FIX [8.1]: only invoked when cfg.enable_pressure_kick == true. Default config
# is pure CO-FLIP (no kick). Kept for ablation studies / FLIP-style runs.
function apply_pressure_correction!(
        p::Particles,
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
        d::Domain,
        thread_caches::Vector{EvaluationCache};
        delta_scale::Float64=1.0,
    ) where {Tp<:Real, Tq<:Real}
    tau_coeffs = compute_pressure_delta(f_projected, f_pre_projection)

    n_thread_slots = Threads.maxthreadid()
    if length(thread_caches) < n_thread_slots
        throw(ArgumentError("thread_caches length must be at least Threads.maxthreadid()"))
    end

    Threads.@threads :static for i in 1:length(p.x)
        tid   = Threads.threadid()
        cache = thread_caches[tid]
        val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
        p.my[i] += delta_scale * val[1]
        p.mx[i] -= delta_scale * val[2]
    end
end

# FIX [12.6]: the allocating no-thread_caches overload is removed. Callers MUST
# pass a preallocated Vector{EvaluationCache}.

function integrate_form_expression(expr, dom::Domain)
    total = 0.0
    for element_id in 1:Quadrature.get_num_base_elements(dom.dΩ)
        total += sum(Forms.evaluate(expr, element_id)[1])
    end
    return total
end

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

function print_conservation_diagnostics(step::Int, time_value::Real, diagnostics)
    @printf(
        "  Diagnostics [step=%d, t=%.6f]: energy=%.12e, circulation=%.12e, enstrophy=%.12e\n",
        step, time_value,
        diagnostics.energy, diagnostics.circulation, diagnostics.enstrophy,
    )
end

# ==============================================================================
#                          TIME INTEGRATION & STEPPING
# ==============================================================================

@inline function apply_position_policy(x::T, y::T, Lx::T, Ly::T, mode::Symbol, tol::T) where {T<:AbstractFloat}
    if mode === :periodic
        return mod(x, Lx), mod(y, Ly)
    elseif mode === :clamp
        return clamp(x, tol, Lx - tol), clamp(y, tol, Ly - tol)
    else
        throw(ArgumentError("Unknown position mode: $mode. Use :periodic or :clamp"))
    end
end

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

# FIX [9.1]: toggle at runtime: `CLAMP_PULLBACK_DEBUG[] = true` to surface
# determinant-drift diagnostics when the iteration stalls or aborts.
const CLAMP_PULLBACK_DEBUG = Ref(false)

@inline function clamp_pullback(
        pullback::SMatrix{2, 2, T, 4};
        tol::T,
        max_iter::Int,
    ) where {T<:AbstractFloat}

    d_pb = det(pullback)
    if !isfinite(d_pb) || abs(d_pb) < eps(T)
        CLAMP_PULLBACK_DEBUG[] && @warn "clamp_pullback: degenerate input" det=d_pb
        return pullback
    end

    Pt    = transpose(pullback)
    PtP   = Pt * pullback
    d_PtP = det(PtP)
    if !isfinite(d_PtP) || d_PtP < eps(T)
        CLAMP_PULLBACK_DEBUG[] && @warn "clamp_pullback: PᵀP singular" det_PtP=d_PtP
        return pullback
    end

    inv_Pt   = inv(Pt)
    tr_mixed = tr(inv(PtP))
    if !isfinite(tr_mixed) || abs(tr_mixed) < eps(T)
        CLAMP_PULLBACK_DEBUG[] && @warn "clamp_pullback: inv(PᵀP) trace near zero"
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

    if CLAMP_PULLBACK_DEBUG[] && iter >= max_iter
        @warn "clamp_pullback: max_iter reached" final_det=det(output_pullback) iters=iter
    end

    return output_pullback
end

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
    c1 = dt / 6
    c2 = dt / 3
    c3 = dt / 3
    c4 = dt / 6

    v1, J1 = physical_velocity_and_jacobian(pos[1], pos[2], u_coeffs, d, cache)
    Pdot1 = -(transpose(J1) * input_pullback)

    midp1_raw = pos + (dt * 0.5) * v1
    mx, my = apply_position_policy(midp1_raw[1], midp1_raw[2], Lx, Ly, mode, pos_tol)
    midp1 = SVector{2, T}(mx, my)
    midP1 = input_pullback + (dt * 0.5) * Pdot1

    v2, J2 = physical_velocity_and_jacobian(midp1[1], midp1[2], u_coeffs, d, cache)
    Pdot2 = -(transpose(J2) * midP1)

    midp2_raw = pos + (dt * 0.5) * v2
    mx, my = apply_position_policy(midp2_raw[1], midp2_raw[2], Lx, Ly, mode, pos_tol)
    midp2 = SVector{2, T}(mx, my)
    midP2 = input_pullback + (dt * 0.5) * Pdot2

    v3, J3 = physical_velocity_and_jacobian(midp2[1], midp2[2], u_coeffs, d, cache)
    Pdot3 = -(transpose(J3) * midP2)

    midp3_raw = pos + dt * v3
    mx, my = apply_position_policy(midp3_raw[1], midp3_raw[2], Lx, Ly, mode, pos_tol)
    midp3 = SVector{2, T}(mx, my)
    midP3 = input_pullback + dt * Pdot3

    v4, J4 = physical_velocity_and_jacobian(midp3[1], midp3[2], u_coeffs, d, cache)
    Pdot4 = -(transpose(J4) * midP3)

    pullback = input_pullback + c1 * Pdot1 + c2 * Pdot2 + c3 * Pdot3 + c4 * Pdot4
    pullback = clamp_pullback(pullback; tol=pullback_tol, max_iter=pullback_max_iter)

    pos_raw = pos + c1 * v1 + c2 * v2 + c3 * v3 + c4 * v4
    px, py = apply_position_policy(pos_raw[1], pos_raw[2], Lx, Ly, mode, pos_tol)

    return SVector{2, T}(px, py), pullback
end

# FIX [4.1]: RK2 single-step kernel removed (dead code). All advection goes
# through the RK4 kernel, which is second-order consistent with the paper and
# matches the C++ reference.
# FIX [4.2]: kernel now also writes the shorterm pullback to dedicated buffers,
# which step_co_flip! later composes into each particle's longterm pullback.
# FIX [12.1]: max_err_per_thread is passed in (preallocated), not allocated here.
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
    # Reset only the slots we'll touch.
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

        # FIX [4.2]: store per-particle short-term pullback.
        short_P11[i] = pullback[1, 1]
        short_P12[i] = pullback[1, 2]
        short_P21[i] = pullback[2, 1]
        short_P22[i] = pullback[2, 2]
    end

    return maximum(@view max_err_per_thread[1:n_thread_slots])
end

# FIX [12.6]: the no-thread_caches convenience overload is removed. Same for
# the RK2 variants, which are superseded by RK4 (FIX [4.1]).

function coadjoint_step!(p::Particles, d::Domain, buf::SimulationBuffers;
        lsqr_atol::Float64=1e-8, lsqr_btol::Float64=1e-8, lsqr_maxiter::Int=2000,
        projection_mean_subtract::Bool=true,
    )
    step_t0 = time()

    ensure_particle_capacity!(buf, length(p.x))

    t0 = time()
    B = build_B_matrix(p, d)
    t_build_B_ms = time() - t0

    # FIX [5.1]: realign warm-start vector with current DOF count if needed.
    if length(buf.lsqr_warm) != size(B, 2)
        buf.lsqr_warm = zeros(size(B, 2))
    end

    t0 = time()
    u_grid_n = solve_grid_velocity_lsqr(B, p, buf.V_p, buf.lsqr_warm;
                                        atol=lsqr_atol, btol=lsqr_btol, maxiter=lsqr_maxiter)
    copy!(buf.lsqr_warm, u_grid_n)
    t_solve_raw_ms = time() - t0

    t0 = time()
    u_grid_n_h             = Forms.build_form_field(d.R1, u_grid_n)
    u_div_free_coeffs_n, _ = project_and_get_pressure(d, u_grid_n_h;
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

function step_co_flip!(
        p::Particles, d::Domain, dt::Float64,
        f_n::AbstractVector{Tf},
        buf::SimulationBuffers,
        mass_matrix_1form::AbstractMatrix{Float64},
        cfg::SimulationConfig,
    ) where {Tf<:Real}

    # FIX [12.2]: catch NaN/Inf in the incoming grid state up-front.
    @assert all(isfinite, f_n) "step_co_flip!: f_n contains non-finite values"

    step_t0       = time()
    thread_caches = buf.thread_caches

    copy!(buf.f_n_saved, f_n)
    f_n_saved = buf.f_n_saved

    # ---- Particle rebalance before solve ----
    t0 = time()
    added_start, removed_start = enforce_min_particles_per_element!(
        p, d,
        cfg.min_particles_per_element, cfg.max_particles_per_element, cfg.min_particles_per_quarter,
        f_n_saved, buf,
    )
    t_seed_start = time() - t0
    if added_start > 0 || removed_start > 0
        println("  Particle rebalance before solve: +$added_start / -$removed_start")
    end
    println("  Initial seeding time: $(round(t_seed_start, digits=2)) s")

    num_p = length(p.x)
    ensure_particle_capacity!(buf, num_p)

    # Snapshot n-state particle positions + impulses.
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

    # Midpoint estimate bootstraps as f_n (iter 1 is explicit-Euler).
    copy!(f_star, f_n_saved)

    # FIX [3.1]: seed f_np1_proj_prev from f_star (NOT from f_n_saved). The
    # "previous projected iterate" lives in midpoint space; the old code
    # produced a spurious err on iter 1 by comparing against f_n.
    copy!(f_np1_proj_prev, f_star)

    f0_norm = norm(f_n_saved)

    # FIX [3.2]: C++-style hysteresis — break when err fails to improve.
    prev_err = Inf

    for iter in 1:cfg.max_fp_iter
        iter_t0 = time()

        # ------------------------------------------------------------------
        # Step 1: Advect n-state particles using current f_star as velocity.
        # FIX [4.1]: RK4 (replaces prior RK2 call at this site).
        # FIX [4.2]: short-term pullback is output to buf.short_Pxx.
        # FIX [12.1]: preallocated per-thread error buffer.
        # ------------------------------------------------------------------
        t0 = time()
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
        t_advect = time() - t0

        # Step 2: commit advected state to p so the P2G is at the right (x,u).
        t0 = time()
        @inbounds for i in 1:num_p
            p.x[i]  = x_np1[i];   p.y[i]  = y_np1[i]
            p.mx[i] = mx_np1[i];  p.my[i] = my_np1[i]
        end
        particle_sorter!(p, d)
        t_sort = time() - t0

        # Step 3: P2G least-squares solve at the new positions.
        t0    = time()
        B_np1 = build_B_matrix(p, d)
        if length(buf.lsqr_warm) != size(B_np1, 2)
            buf.lsqr_warm = zeros(size(B_np1, 2))
        end
        t_build_B = time() - t0

        t0      = time()
        # FIX [5.1]: warm-start from last iterate, cap iterations.
        raw_sol = solve_grid_velocity_lsqr(B_np1, p, buf.V_p, buf.lsqr_warm;
                                           atol=cfg.lsqr_atol, btol=cfg.lsqr_btol,
                                           maxiter=cfg.lsqr_maxiter)
        copy!(f_np1_raw, raw_sol)
        copy!(buf.lsqr_warm, raw_sol)
        t_solve = time() - t0

        # Step 4: energy-based correction (Algorithm 2, line 6). Gated (FIX 6.1).
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
            # No dissipation/forcing to project against — skip the correction.
            copy!(f_np1, f_np1_raw)
        end
        t_energy = time() - t0

        # Step 5: Pressure-project for a divergence-free field (with FIX 7.2).
        t0             = time()
        f_np1_h        = Forms.build_form_field(d.R1, f_np1)
        proj_result, _ = project_and_get_pressure(d, f_np1_h;
                                                   subtract_mean=cfg.projection_mean_subtract)
        copy!(f_np1_proj, proj_result)
        t_project = time() - t0

        # Step 6: convergence check — change in f_np1_proj between iterations.
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

        # Step 7: update midpoint for next iteration; save current proj.
        copy!(f_np1_proj_prev, f_np1_proj)
        @. f_star = 0.5 * (f_n_saved + f_np1_proj)

        # FIX [3.2]: stall detector — break if the relative error stops shrinking.
        if err < cfg.fp_tol
            break
        end
        if iter > 1 && err >= 0.99 * prev_err
            println("  Fixed-point stalled (err=$err >= 0.99 * prev_err=$prev_err) — break.")
            break
        end
        prev_err = err
    end

    # FIX [3.3]: compose the LAST iteration's short-term pullback into each
    # particle's longterm pullback and accumulate per-particle delta_t.
    compose_longterm_pullback!(p, buf, dt, num_p)

    # FIX [8.1]: pure CO-FLIP — NO pressure kick on particles (grid is already
    # divergence-free). FLIP / hybrid mode is available behind a config flag.
    if cfg.enable_pressure_kick
        t0 = time()
        apply_pressure_correction!(
            p, f_np1_proj, f_np1, d, thread_caches; delta_scale=1.0,
        )
        println("  [pressure kick enabled — FLIP/hybrid] t=$(round(time()-t0,digits=2))s")
    end

    t0 = time()
    particle_sorter!(p, d)
    t_final_sort = time() - t0

    t0 = time()
    added_end, removed_end = enforce_min_particles_per_element!(
        p, d,
        cfg.min_particles_per_element, cfg.max_particles_per_element, cfg.min_particles_per_quarter,
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

# FIX [3.3]: P_long ← P_short * P_long; delta_t ← delta_t + dt (per particle).
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

# FIX [3.3 / 8.f]: per-particle FTLE reset. A particle is reseeded from the
# pure grid state when either:
#   (a) log(σ_max(P_long)) / delta_t exceeds ftle_threshold
#   (b) delta_t exceeds max_longterm_delta_t  (scheduled PIC-style cap)
# σ_max is the largest singular value of the longterm pullback, computed
# closed-form from the 2×2 PᵀP eigenvalue. Replaces the old reset_every
# heuristic (FIX [3.x] removal).
function apply_ftle_reset!(
        p::Particles, d::Domain, u_phys_form,
        cfg::SimulationConfig,
    )
    Lx, Ly = d.box_size
    nx, ny = d.nel
    dx = Lx / nx;  dy = Ly / ny

    reset_count = 0
    @inbounds for pid in eachindex(p.x)
        dt_acc = p.delta_t[pid]
        dt_acc <= 0.0 && continue

        P11 = p.P11[pid]; P12 = p.P12[pid]
        P21 = p.P21[pid]; P22 = p.P22[pid]

        # σ_max² = 0.5 * (tr(PᵀP) + sqrt(tr(PᵀP)² - 4·det(PᵀP)))
        a11 = P11*P11 + P21*P21
        a22 = P12*P12 + P22*P22
        a12 = P11*P12 + P21*P22
        tra = a11 + a22
        det_a = a11*a22 - a12*a12
        disc  = max(tra*tra - 4.0 * det_a, 0.0)
        σ2    = 0.5 * (tra + sqrt(disc))
        σ_max = sqrt(max(σ2, 0.0))
        ftle  = σ_max > 0 ? log(σ_max) / dt_acc : 0.0

        if ftle > cfg.ftle_threshold || dt_acc > cfg.max_longterm_delta_t
            eid = p.elem_ids[pid]
            ej = ((eid - 1) ÷ nx) + 1
            ei = eid - (ej - 1) * nx
            set_g2p_velocity(
                p, pid, p.x[pid], p.y[pid], eid,
                (ei - 1) * dx, (ej - 1) * dy, dx, dy,
                u_phys_form, Lx, Ly,
            )
            # Reset longterm state to identity / zero.
            p.P11[pid] = 1.0;  p.P12[pid] = 0.0
            p.P21[pid] = 0.0;  p.P22[pid] = 1.0
            p.delta_t[pid] = 0.0
            reset_count += 1
        end
    end
    return reset_count
end

function maybe_clear_memo_tables!(step::Int, clear_every::Int, d::Domain)
    # FIX [11.4]: clear_every == 0 disables the clear entirely (default).
    if clear_every > 0 && (step % clear_every == 0)
        Memoization.empty_all_caches!()
        warmup_evaluation_memo!(d)
    end
end

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

# FIX [11.2]: after a solve, re-validate dt against the grid velocity the next
# step will actually advect with. Returns the stricter of cur_dt and the
# candidate dt derived from the grid-DOF max magnitude.
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

# FIX [9.2]: startup smoke tests — fail loudly if the De Rham complex is
# mis-assembled or the quadrature/integral machinery is off.
function run_startup_smoke_tests(d::Domain)
    ndof_R0 = FunctionSpaces.get_num_basis(d.R0.fem_space)
    ndof_R1 = FunctionSpaces.get_num_basis(d.R1.fem_space)
    ndof_R2 = FunctionSpaces.get_num_basis(d.R2.fem_space)
    println("  Smoke test DOFs: R0=$ndof_R0, R1=$ndof_R1, R2=$ndof_R2")
    @assert ndof_R0 > 0 && ndof_R1 > 0 && ndof_R2 > 0 "Smoke test: empty DOF space"

    # d(constant 0-form) must vanish.
    c0  = fill(1.0, ndof_R0)
    f0  = Forms.build_form_field(d.R0, c0)
    df0 = Forms.d(f0)
    # Evaluate the derivative at quadrature points and compute its norm
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

    # ∫1 over the torus should equal |Ω|.
    c2       = fill(1.0, ndof_R2)
    f2       = Forms.build_form_field(d.R2, c2)
    integral = integrate_form_expression(∫(f2, d.dΩ), d)
    expected = d.box_size[1] * d.box_size[2]
    err = abs(integral - expected)
    println("  Smoke test ∫1 dΩ = $integral (expected $expected, err=$(Printf.@sprintf("%.3e", err)))")
    @assert err < 1e-6 "Smoke test: integral consistency failed"
    return nothing
end

# ==============================================================================
#                                   MAIN
# ==============================================================================

function main(cfg::SimulationConfig=SimulationConfig())
    println("Initializing Domain and Particles...")
    LinearAlgebra.BLAS.set_num_threads(1)

    num_particles = cfg.nel[1] * cfg.nel[2] * cfg.particles_per_cell

    domain    = GenerateDomain(cfg.nel, cfg.p, cfg.k)
    particles = generate_particles(num_particles, domain, cfg.flow_type)
    particle_sorter!(particles, domain)

    # FIX [9.2]: fail early on obvious setup mistakes.
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

    # ---- SimulationBuffers must exist before any LSQR solve (FIX 5.1) ----
    ndofs        = FunctionSpaces.get_num_basis(domain.R1.fem_space)
    num_elements = prod(domain.nel)
    sim_buf      = SimulationBuffers(num_particles, ndofs, num_elements, domain)

    # ---- t=0 snapshot ----
    println("Saving true t=0 snapshot...")
    u_coeffs = coadjoint_step!(particles, domain, sim_buf;
        lsqr_atol=cfg.lsqr_atol, lsqr_btol=cfg.lsqr_btol, lsqr_maxiter=cfg.lsqr_maxiter,
        projection_mean_subtract=cfg.projection_mean_subtract)

    # FIX [11.3]: compute u_phys_form once per step; reuse for G2P reset / VTK.
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

    println("Assembling exact 1-form mass matrix...")
    M_1form = assemble_1form_mass_matrix(domain.R1, domain.dΩ)
    println("  Mass matrix assembled: $(size(M_1form, 1)) × $(size(M_1form, 2))")

    # ---- Time loop ----
    for step in 1:n_steps
        println("Step $step / $n_steps (t = $(round(step * dt, digits=3)))...")

        u_coeffs = step_co_flip!(particles, domain, dt, u_coeffs, sim_buf, M_1form, cfg)

        # FIX [11.3]: rebuild u_phys_form once here, reuse downstream.
        u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
        u_phys_expr = ★(u_form)
        u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)

        # FIX [3.3]: per-particle FTLE reset replaces the old step%reset_every.
        reset_count = apply_ftle_reset!(particles, domain, u_phys_form, cfg)
        reset_count > 0 && println("  FTLE reset: $reset_count particles reseeded.")

        # FIX [11.2]: sanity-check CFL given the current grid velocity.
        dt_check = recheck_cfl_dt(u_coeffs, domain, cfg.target_cfl, dt)
        if dt_check < 0.5 * dt
            @warn "CFL recheck — grid velocity suggests dt=$dt_check (current dt=$dt)"
        end

        # FIX [11.1]: throttle VTK + diagnostics by output_every.
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

        # FIX [11.4]: clear_memo_every = 0 ⇒ no-op.
        maybe_clear_memo_tables!(step, cfg.clear_memo_every, domain)
    end

    println("\nSimulation complete! Step files written to '$(output_dir)'.")
end

function plot_flow()
    x = range(0, 1, length=100)
    y = range(0, 1, length=100)

    Lx = 1.0;  Ly = 1.0
    u_vals = zeros(length(x), length(y))
    v_vals = zeros(length(x), length(y))

    for i in eachindex(x), j in eachindex(y)
        u_vals[i, j], v_vals[i, j] = flow_decay(x[i], y[j], Lx, Ly)
    end

    log_vel_mag = sqrt.(u_vals.^2 .+ v_vals.^2)

    heatmap(
        x, y, log_vel_mag',
        aspect_ratio = :equal,
        c = :viridis,
        title = "flow_decay velocity magnitude |u|",
        xlims = (0, Lx), ylims = (0, Ly),
    )
end

main()
