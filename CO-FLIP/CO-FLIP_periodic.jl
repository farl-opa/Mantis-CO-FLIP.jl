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

mutable struct Particles
    x::Vector{Float64}
    y::Vector{Float64}
    mx::Vector{Float64}      # Impulse (Covector)
    my::Vector{Float64}
    volume::Vector{Float64}
    can_x::Vector{Float64}   # Pre-computed Canonical Coordinates
    can_y::Vector{Float64}
    head::Vector{Int}        # Linked List: head[element_index]
    next::Vector{Int}        # Linked List: next[particle_index]
    elem_ids::Vector{Int}
end

struct EvaluationCache
    temp_mat::Matrix{Float64}
    results::Vector{Vector{Vector{Matrix{Float64}}}}
    xi_buf::Vector{Float64}    # single-element buffer for canonical x
    eta_buf::Vector{Float64}   # single-element buffer for canonical y

    function EvaluationCache(max_basis_size::Int)
        v0  = [[zeros(1, max_basis_size) for _ in 1:2]]
        v1  = [[zeros(1, max_basis_size) for _ in 1:2] for _ in 1:2]
        res = Vector{Vector{Vector{Matrix{Float64}}}}(undef, 2)
        res[1] = v0; res[2] = v1
        return new(zeros(1, max_basis_size), res, zeros(1), zeros(1))
    end
end

mutable struct SimulationBuffers
    # ------------------------------------------------------------------
    # Particle-sized working arrays — hold n-state and (n+1)-state copies.
    # Capacity may exceed the live particle count after a resize.
    # ------------------------------------------------------------------
    x_n::Vector{Float64}
    y_n::Vector{Float64}
    mx_n::Vector{Float64}
    my_n::Vector{Float64}
    x_np1::Vector{Float64}
    y_np1::Vector{Float64}
    mx_np1::Vector{Float64}
    my_np1::Vector{Float64}

    # ------------------------------------------------------------------
    # DOF-sized (grid) working arrays
    # ------------------------------------------------------------------
    f_n_saved::Vector{Float64}    # snapshot of f_n at step entry (avoids aliasing
                                  # when the caller passes buf.f_np1_proj as f_n)
    f_star::Vector{Float64}       # midpoint estimate / fixed-point iterate
    f_np1::Vector{Float64}        # energy-corrected field (pre-projection)
    f_np1_raw::Vector{Float64}    # raw P2G least-squares result
    f_np1_proj::Vector{Float64}   # divergence-free projected field (returned to caller)

    # ------------------------------------------------------------------
    # Per-thread evaluation caches (one per Julia thread slot)
    # ------------------------------------------------------------------
    thread_caches::Vector{EvaluationCache}

    # ------------------------------------------------------------------
    # enforce_min_particles_per_element! scratch buffers
    # ------------------------------------------------------------------
    quarter_pids::Vector{Vector{Int}}   # length = num_elements × 4; emptied and reused
    add_quarter::Matrix{Int}            # num_elements × 4
    counts_elem::Vector{Int}            # num_elements
    counts_quarter::Matrix{Int}         # num_elements × 4
    keep_mask::Vector{Bool}             # particle-capacity sized

    # ------------------------------------------------------------------
    # LSQR right-hand-side buffer (2 × num_particles sized)
    # ------------------------------------------------------------------
    V_p::Vector{Float64}
end

function SimulationBuffers(
        num_particles_initial::Int,
        ndofs::Int,
        num_elements::Int,
        d::Domain,
    )
    n_threads  = Threads.maxthreadid()
    cache_size = evaluation_cache_size(d)
    cap        = num_particles_initial * 2   # headroom for particle spawning

    return SimulationBuffers(
        # particle arrays (8 of them)
        Vector{Float64}(undef, cap),  # x_n
        Vector{Float64}(undef, cap),  # y_n
        Vector{Float64}(undef, cap),  # mx_n
        Vector{Float64}(undef, cap),  # my_n
        Vector{Float64}(undef, cap),  # x_np1
        Vector{Float64}(undef, cap),  # y_np1
        Vector{Float64}(undef, cap),  # mx_np1
        Vector{Float64}(undef, cap),  # my_np1
        # grid arrays (5 of them)
        Vector{Float64}(undef, ndofs),  # f_n_saved
        Vector{Float64}(undef, ndofs),  # f_star
        Vector{Float64}(undef, ndofs),  # f_np1
        Vector{Float64}(undef, ndofs),  # f_np1_raw
        Vector{Float64}(undef, ndofs),  # f_np1_proj
        # thread caches
        [EvaluationCache(cache_size) for _ in 1:n_threads],
        # enforce_min scratch
        [Int[] for _ in 1:num_elements * 4],     # quarter_pids
        zeros(Int, num_elements, 4),              # add_quarter
        zeros(Int, num_elements),                 # counts_elem
        zeros(Int, num_elements, 4),              # counts_quarter
        trues(cap),                               # keep_mask
        # LSQR rhs
        Vector{Float64}(undef, 2 * cap),          # V_p
    )
end

probe_output_eltype(::Type{T}) where {T<:Real} = promote_type(Float64, T)

evaluation_cache_size(d::Domain) = d.eval_cache_size

# ==============================================================================
#                            DOMAIN GENERATION
# ==============================================================================

function GenerateDomain(nel::NTuple{2,Int}, p::NTuple{2,Int}, k::NTuple{2,Int})
    starting_point = (0.0, 0.0)
    box_size       = (1.0, 1.0)
    nq_assembly = p .+ 1
    nq_error    = nq_assembly .* 2
    ∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, nq_assembly, nq_error)
    dΩ = Quadrature.StandardQuadrature(∫ₐ, prod(nel))
    
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_point,
        box_size,
        nel,
        p,
        k;
        periodic=(true, true),
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
    if flow_type == :tg
        return flow_taylor_green(px, py, Lx, Ly)
    elseif flow_type == :vortex
        return flow_lamb_oseen(px, py, Lx, Ly)
    elseif flow_type == :gyre
        return flow_double_gyre(px, py, Lx, Ly)
    elseif flow_type == :decay
        return flow_decay(px, py, Lx, Ly)
    elseif flow_type == :convecting
        return flow_convecting_vortex(px, py, Lx, Ly)
    elseif flow_type == :merging
        return flow_merging_vortices(px, py, Lx, Ly)
    else
        error("Unknown flow type: $flow_type")
    end
end

function generate_particles(num_particles::Int, domain::Domain, flow_type::Symbol=:vortex)
    Lx, Ly = domain.box_size
    
    x  = Vector{Float64}(undef, num_particles)
    y  = Vector{Float64}(undef, num_particles)
    mx = Vector{Float64}(undef, num_particles)
    my = Vector{Float64}(undef, num_particles)
    vol = ones(Float64, num_particles) # Volume is 1.0/density roughly
    
    # Pre-allocate arrays for sorting structures
    can_x = zeros(Float64, num_particles)
    can_y = zeros(Float64, num_particles)
    num_elements = prod(domain.nel)
    head = zeros(Int, num_elements)
    next = zeros(Int, num_particles)
    elem_ids = zeros(Int, num_particles)

    # Initialize Particle States
    for i in 1:num_particles
        # 1. Random Position (Stratified sampling is better, but random is fine for now)
        px = rand() * Lx
        py = rand() * Ly
        x[i] = px
        y[i] = py
        
        # 2. Select Flow Type
        u, v = initial_velocity(flow_type, px, py, Lx, Ly)
        
        # 3. Assign Impulse
        # In this metric (Cartesian), impulse (covector) components = velocity components
        mx[i] = u
        my[i] = v
    end
    
    # Estimate particle volume (Area / N) for integration weights
    # This helps the least-squares P2G solver converge better
    particle_vol = (Lx * Ly) / num_particles
    fill!(vol, particle_vol)
    
    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids)
end

function particle_sorter!(p::Particles, d::Domain)
    fill!(p.head, 0)
    
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx
    dy = Ly / ny
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    
    @inbounds for i in 1:length(p.x)
        p.x[i] = mod(p.x[i], Lx)
        p.y[i] = mod(p.y[i], Ly)
        
        ei = clamp(floor(Int, p.x[i] * inv_dx) + 1, 1, nx)
        ej = clamp(floor(Int, p.y[i] * inv_dy) + 1, 1, ny)
        
        eid = (ej - 1) * nx + ei
        p.elem_ids[i] = eid
        
        p.can_x[i] = (p.x[i] - (ei - 1) * dx) * inv_dx
        p.can_y[i] = (p.y[i] - (ej - 1) * dy) * inv_dy
        
        p.next[i] = p.head[eid]
        p.head[eid] = i
    end
end

function set_g2p_velocity(
        p::Particles,
        pid::Int,
        px::Float64,
        py::Float64,
        eid::Int,
        x0::Float64,
        y0::Float64,
        dx::Float64,
        dy::Float64,
        ref_phys_form,
    )
    xi = (px - x0) / dx
    eta = (py - y0) / dy
    sample_points = Mantis.Points.CartesianPoints(([xi], [eta]))
    pushfwd_eval, _ = Forms.evaluate_sharp_pushforward(ref_phys_form, eid, sample_points)

    p.x[pid] = px
    p.y[pid] = py
    p.mx[pid] = reduce(+, pushfwd_eval[1], dims=2)[1]
    p.my[pid] = reduce(+, pushfwd_eval[2], dims=2)[1]

    return nothing
end

function enforce_min_particles_per_element!(
        p::Particles,
        d::Domain,
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

    # ---- Reuse scratch arrays from buf (no allocation) ----
    counts_elem    = buf.counts_elem     # Vector{Int}, length num_elements
    counts_quarter = buf.counts_quarter  # Matrix{Int}, num_elements × 4
    add_quarter    = buf.add_quarter     # Matrix{Int}, num_elements × 4

    fill!(counts_elem,    0)
    fill!(counts_quarter, 0)
    fill!(add_quarter,    0)

    @inbounds for pid in eachindex(p.x)
        eid = p.elem_ids[pid]
        qx  = p.can_x[pid] >= 0.5 ? 2 : 1
        qy  = p.can_y[pid] >= 0.5 ? 1 : 0
        qid = qx + 2 * qy
        counts_elem[eid]       += 1
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
        return 0, 0
    end

    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx
    dy = Ly / ny
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

        resize!(p.x,       new_count)
        resize!(p.y,       new_count)
        resize!(p.mx,      new_count)
        resize!(p.my,      new_count)
        resize!(p.volume,  new_count)
        resize!(p.can_x,   new_count)
        resize!(p.can_y,   new_count)
        resize!(p.next,    new_count)
        resize!(p.elem_ids, new_count)

        # Grow SimulationBuffers particle capacity to match.
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
                    set_g2p_velocity(p, pid, px, py, eid, x0, y0, dx, dy, ref_phys_form)
                    pid += 1
                end
            end
        end

        particle_sorter!(p, d)
    end

    # ------------------------------------------------------------------
    # Particle culling
    # ------------------------------------------------------------------
    total_to_remove = 0
    if max_particles_per_element > 0 && length(p.x) > 0
        n_particles = length(p.x)

        ensure_particle_capacity!(buf, n_particles)
        keep_mask    = buf.keep_mask
        quarter_pids = buf.quarter_pids   # Vector{Vector{Int}}, pre-allocated

        fill!(view(keep_mask, 1:n_particles), true)

        # Empty the reusable per-quarter lists (no allocation; just sets length=0).
        @inbounds for s in eachindex(quarter_pids)
            empty!(quarter_pids[s])
        end

        # Rebuild counts and per-quarter particle lists.
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
                qbest      = 0
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
                        keep_mask[victim]          = false
                        counts_quarter[eid, qbest] -= 1
                        counts_elem[eid]           -= 1
                        total_to_remove            += 1
                        local_excess               -= 1
                        break
                    end
                end
            end
        end

        if total_to_remove > 0
            keep_idx   = findall(@view keep_mask[1:n_particles])
            p.x        = p.x[keep_idx]
            p.y        = p.y[keep_idx]
            p.mx       = p.mx[keep_idx]
            p.my       = p.my[keep_idx]
            p.volume   = p.volume[keep_idx]
            p.can_x    = p.can_x[keep_idx]
            p.can_y    = p.can_y[keep_idx]
            p.next     = p.next[keep_idx]
            p.elem_ids = p.elem_ids[keep_idx]
            particle_sorter!(p, d)
        end
    end

    fill!(p.volume, (Lx * Ly) / length(p.x))
    return total_to_add, total_to_remove
end

function ensure_particle_capacity!(buf::SimulationBuffers, n::Int)
    if length(buf.x_n) >= n
        return nothing
    end
    new_cap = max(n, length(buf.x_n) * 2)
    for arr in (buf.x_n, buf.y_n, buf.mx_n, buf.my_n,
                buf.x_np1, buf.y_np1, buf.mx_np1, buf.my_np1,
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
# ------------------------------------------------------------------------------
# 1. Taylor-Green Vortex (Periodic)
# ------------------------------------------------------------------------------
function flow_taylor_green(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    # Wavenumbers optimized for [0, Lx] x [0, Ly]
    kx = 2*π * x / Lx
    ky = 2*π * y / Ly
    
    u =  U0 * sin(kx) * cos(ky)
    v = -U0 * cos(kx) * sin(ky)
    return u, v
end

# ------------------------------------------------------------------------------
# 2. Lamb-Oseen Vortex (Decaying)
# ------------------------------------------------------------------------------
function flow_lamb_oseen(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    # Vortex Center
    cx = Lx / 2.0
    cy = Ly / 2.0
    
    # Parameters
    Gamma = 2.0   # Circulation strength
    r_core = min(Lx, Ly) / 8.0  # Radius where max velocity occurs
    
    dx = x - cx
    dy = y - cy
    r2 = dx^2 + dy^2
    r  = sqrt(r2)
    
    # Avoid singularity at center
    if r < 1e-8
        return 0.0, 0.0
    end

    # Tangential velocity: v_theta = (Gamma / 2πr) * (1 - exp(-r²/rc²))
    v_theta = (Gamma / (2 * π * r)) * (1.0 - exp(-r2 / (r_core^2)))
    
    # Convert polar to Cartesian (u = -y*omega, v = x*omega)
    u = -v_theta * (dy / r)
    v =  v_theta * (dx / r)
    
    return u, v
end

# ------------------------------------------------------------------------------
# 3. Double Gyre (Stationary)
# ------------------------------------------------------------------------------
function flow_double_gyre(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    
    # Normalize coordinates to [0, 1]
    X = x / Lx
    Y = y / Ly
    
    # Streamfunction-derived velocity
    # u =  sin(πx)cos(πy) -> 0 at x=0, x=1 (Tangential flow only)
    # v = -cos(πx)sin(πy) -> 0 at y=0, y=1
    u =  U0 * sin(π * X) * cos(π * Y)
    v = -U0 * cos(π * X) * sin(π * Y)
    
    return u, v
end

# ------------------------------------------------------------------------------
# 4. Exponential decaying flow
# ------------------------------------------------------------------------------
function flow_decay(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    # Vortex Center
    cx = Lx / 2.0
    cy = Ly / 2.0
    
    # Parameters
    Gamma = 10.0   # Circulation strength
    r_core = min(Lx, Ly) / 6.0  # Radius where max velocity occurs
    
    dx = x - cx
    dy = y - cy
    r2 = dx^2 + dy^2
    r  = sqrt(r2)
    
    # Avoid singularity at center
    if r < 1e-8
        return 0.0, 0.0
    end

    sin_theta = dy/r
    cos_theta = dx/r

    u = -sin_theta*exp(-r2/(0.2*r_core^2))
    v = cos_theta*exp(-r2/(0.2*r_core^2))
    
    return u, v
end

# ------------------------------------------------------------------------------
# 5. Convecting Vortex (Periodic Advection Test)
# ------------------------------------------------------------------------------
function flow_convecting_vortex(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    # --- Parameters ---
    U0 = 1.0              # background convection velocity
    Γ  = 5.0              # vortex strength
    σ  = 0.1 * min(Lx, Ly)  # vortex core size

    # Vortex center (middle of domain)
    x0 = 0.5 * Lx
    y0 = 0.5 * Ly

    # --- Relative coordinates ---
    dx = x - x0
    dy = y - y0
    r2 = dx^2 + dy^2

    # --- Gaussian vortex (smooth, no singularity) ---
    factor = (Γ / (2π)) * exp(-r2 / σ^2)

    u_vortex = -factor * dy / σ^2
    v_vortex =  factor * dx / σ^2

    # Periodic advection benchmark: do not damp near boundaries.
    # A boundary envelope artificially weakens the vortex as it approaches x=0/Lx or y=0/Ly.
    u = u_vortex + U0
    v = v_vortex

    return u, v
end

# ------------------------------------------------------------------------------
# 6. Merging Vortices (Co-rotating Pair)
# ------------------------------------------------------------------------------
function flow_merging_vortices(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    # 1. Domain Center
    cx = Lx / 2.0
    cy = Ly / 2.0
    
    # 2. Vortex Parameters
    Gamma = 3.0                # Both have the same positive circulation
    r_core = min(Lx, Ly) / 15.0 # Core radius
    separation = r_core * 3.5   # Distance between vortex centers
    
    # Centers of Vortex 1 and Vortex 2
    cx1 = cx - separation / 2.0
    cy1 = cy
    
    cx2 = cx + separation / 2.0
    cy2 = cy
    
    # 3. Distance calculations
    dx1 = x - cx1
    dy1 = y - cy1
    r1_2 = dx1^2 + dy1^2
    r1   = sqrt(r1_2)
    
    dx2 = x - cx2
    dy2 = y - cy2
    r2_2 = dx2^2 + dy2^2
    r2   = sqrt(r2_2)
    
    # 4. Initialize velocity components
    u = 0.0
    v = 0.0
    
    # 5. Add contribution from Vortex 1
    if r1 > 1e-8
        v_theta1 = (Gamma / (2 * π * r1)) * (1.0 - exp(-r1_2 / (r_core^2)))
        u += -v_theta1 * (dy1 / r1)
        v +=  v_theta1 * (dx1 / r1)
    end
    
    # 6. Add contribution from Vortex 2
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
    # Reset cache
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
                out_mat = cache.results[der_order][der_idx][component_idx]
                
                LinearAlgebra.mul!(
                    view(out_mat, :, J),
                    basis_val,
                    extraction_coefficients
                )
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

    # Write into pre-allocated buffers — zero allocation
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
        u_mat[i, 1] = vel[2]
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
    pid_batch = Int[]
    max_batch = 1024

    for eid in 1:num_elements
        empty!(can_batch_x)
        empty!(can_batch_y)
        empty!(pid_batch)

        pid = probes.head[eid]
        while pid != 0
            push!(pid_batch, pid)
            push!(can_batch_x, probes.can_x[pid])
            push!(can_batch_y, probes.can_y[pid])
            pid = probes.next[pid]
        end

        if isempty(pid_batch)
            continue
        end

        total_points = length(pid_batch)
        for chunk_start in 1:max_batch:total_points
            chunk_end = min(chunk_start + max_batch - 1, total_points)
            rng = chunk_start:chunk_end

            xs_view = @view can_batch_x[rng]
            ys_view = @view can_batch_y[rng]
            pids_view = @view pid_batch[rng]
            points = Mantis.Points.CartesianPoints((xs_view, ys_view))

            eval_out, _ = Forms.evaluate(form_expr, eid, points)
            local_vals = eval_out[1]

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

    # u_coeffs are stored in a rotated proxy convention. We first rotate with Hodge-star,
    # then L2-project to a true 1-form field (Mantis supports d on FormField, not on
    # Hodge(FormField) expressions), and finally compute scalar vorticity as *d(u_phys_form).
    u_form = Forms.build_form_field(d.R1, u_coeffs)
    u_phys_expr = ★(u_form)
    u_phys_form = Assemblers.solve_L2_projection(d.R1, u_phys_expr, d.dΩ)
    ω_form = ★(Forms.d(u_phys_form))
    ω_vals = evaluate_scalar_form_at_probes(ω_form, probes, d)

    @inbounds for i in 1:num_points
        raw_vel, _ = probe_field_at_point(probes.x[i], probes.y[i], u_coeffs, d, viz_cache)

        # Convert from stored flux ordering to physical velocity components.
        u = -raw_vel[2]
        v =  raw_vel[1]

        uvω_mat[i, 1] = u
        uvω_mat[i, 2] = v
        uvω_mat[i, 3] = ω_vals[i]
    end

    return uvω_mat
end

function warmup_evaluation_memo!(d::Domain)
    # Populate memoized derivative-index paths on a single thread before threaded probe calls.
    cache = EvaluationCache(evaluation_cache_size(d))
    points = Mantis.Points.CartesianPoints(([0.5], [0.5]))
    evaluate_fast!(cache, d.R1.fem_space, 1, points, 1)
    return nothing
end

function export_particles_to_vtk(particles::Particles, output_path::String)
    n_particles = length(particles.x)
    
    # Create 3D points (add z=0 for 2D particles)
    points = zeros(3, n_particles)
    points[1, :] .= particles.x
    points[2, :] .= particles.y
    # points[3, :] stays zero (2D in xy-plane)
    
    # Create cells: each particle is a single-point cell (VTK_VERTEX)
    cells = [MeshCell(VTKCellTypes.VTK_VERTEX, vec([i])) for i in 1:n_particles]
    
    # Create VTK grid
    vtkfile = vtk_grid(output_path, points, cells)
    
    # Add point data: impulse components and speed
    u_part = copy(particles.mx)
    v_part = copy(particles.my)
    speed_part = sqrt.(u_part.^2 .+ v_part.^2)
    
    vtkfile["mx", VTKPointData()] = u_part
    vtkfile["my", VTKPointData()] = v_part
    vtkfile["speed", VTKPointData()] = speed_part
    vtkfile["volume", VTKPointData()] = particles.volume
    
    # Write the file
    outfiles = vtk_save(vtkfile)
    return outfiles
end

# ==============================================================================
#                            GRID OPERATIONS & SOLVERS
# ==============================================================================

function build_B_matrix(p::Particles, d::Domain)
    fes      = d.R1.fem_space
    num_p    = length(p.x)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    cache    = EvaluationCache(evaluation_cache_size(d))

    # Upper-bound NNZ: every particle contributes at most 2 × eval_cache_size
    # non-zeros (one x-row + one y-row, each with at most eval_cache_size entries).
    # This avoids a separate counting pass while still pre-allocating exactly
    # enough memory in the common case.
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
        2num_p,
        num_dofs,
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

function solve_grid_velocity_lsqr(
        B::SparseMatrixCSC{Float64,Int},
        p::Particles,
        V_p_buf::AbstractVector{Float64};
        atol::Float64=1e-8,
        btol::Float64=1e-8,
    )
    n = 2 * length(p.x)
    V_p = @view V_p_buf[1:n]
    build_lsqr_rhs!(V_p, p)
    return lsqr(B, V_p; atol=atol, btol=btol)
end

# Backward-compatible overload (allocates V_p — kept for call sites outside the
# main step loop where allocation cost is negligible).
function solve_grid_velocity_lsqr(B::AbstractMatrix, p::Particles)
    num_particles = length(p.x)
    V_p = Vector{Float64}(undef, 2 * num_particles)
    build_lsqr_rhs!(V_p, p)
    return lsqr(B, V_p; atol=1e-8, btol=1e-8)
end

function project_and_get_pressure(dom::Domain, v_h::V) where {V}
    f = d(v_h) # Exterior derivative
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(dom.R1, dom.R2, f, dom.dΩ)
    div_free_coeffs = v_h.coefficients - u_corr.coefficients
    return div_free_coeffs, u_corr.coefficients
end

function assemble_1form_mass_matrix(R1_space::F, dΩ::Q) where {F, Q}
    # Assemble the exact, stationary mass matrix for the 1-form space.
    # This is the Galerkin mass matrix: M[i,j] = ∫(φ_i ∧ ★φ_j)
    # where φ_i and φ_j are 1-form basis functions.
    weak_form_inputs = Assemblers.WeakFormInputs(R1_space)
    vᵏ = Assemblers.get_test_form(weak_form_inputs)
    uᵏ = Assemblers.get_trial_form(weak_form_inputs)
    
    # Build the bilinear form: (v, u) = ∫(v ∧ ★(u))
    A = ∫(vᵏ ∧ ★(uᵏ), dΩ)
    
    lhs_expressions = ((A,),)  # Single 1×1 block
    rhs_expressions = ((0,),)  # Zero RHS (we only care about LHS matrix)
    
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
            # Compute the scalar projection coefficient once (a single dot product).
            raw_diff_dot = dot(f_from_particles .- f_original, circulations_original)
            scale = raw_diff_dot / original_energy  # scalar — no vector allocation

            # Single fused loop: compute corrected diff and accumulate into f_next.
            @inbounds for i in eachindex(f_next)
                fd = (f_from_particles[i] - f_original[i]) - scale * f_midpoint[i]
                f_next[i] = f_original[i] + fd
            end
            return nothing
        end
    end

    # No projection branch: f_next = f_original + (f_from_particles - f_original)
    #                              = f_from_particles
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

function apply_pressure_correction!(
        p::Particles,
        f_projected::AbstractVector{Tp},
        f_pre_projection::AbstractVector{Tq},
        d::Domain;
        delta_scale::Float64=1.0,
    ) where {Tp<:Real, Tq<:Real}
    n_thread_slots = Threads.maxthreadid()
    cache_size = evaluation_cache_size(d)
    thread_caches = [EvaluationCache(cache_size) for _ in 1:n_thread_slots]
    return apply_pressure_correction!(
        p,
        f_projected,
        f_pre_projection,
        d,
        thread_caches;
        delta_scale=delta_scale,
    )
end

function integrate_form_expression(expr, dom::Domain)
    total = 0.0
    for element_id in 1:Quadrature.get_num_base_elements(dom.dΩ)
        total += sum(Forms.evaluate(expr, element_id)[1])
    end

    return total
end

function compute_conservation_diagnostics(u_coeffs::AbstractVector{T}, dom::Domain) where {T<:Real}
    u_h = Forms.build_form_field(dom.R1, u_coeffs; label="u_h")
    u_phys_expr = ★(u_h)
    u_phys_form = Assemblers.solve_L2_projection(dom.R1, u_phys_expr, dom.dΩ)
    d_u_phys = Forms.d(u_phys_form)
    ω_h = ★(d_u_phys)

    energy = 0.5 * integrate_form_expression(∫(u_h ∧ ★(u_h), dom.dΩ), dom)
    circulation = integrate_form_expression(∫(d_u_phys, dom.dΩ), dom)
    enstrophy = 0.5 * integrate_form_expression(∫(ω_h ∧ ★(ω_h), dom.dΩ), dom)

    return (; energy, circulation, enstrophy)
end

function print_conservation_diagnostics(step::Int, time_value::Real, diagnostics)
    @printf(
        "  Diagnostics [step=%d, t=%.6f]: energy=%.12e, circulation=%.12e, enstrophy=%.12e\n",
        step,
        time_value,
        diagnostics.energy,
        diagnostics.circulation,
        diagnostics.enstrophy
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
        x::T,
        y::T,
        u_coeffs::AbstractVector{U},
        d::Domain,
        cache::EvaluationCache,
    ) where {T<:AbstractFloat, U<:Real}
    raw_vel, raw_grad = probe_field_at_point(x, y, u_coeffs, d, cache)

    # Convert proxy ordering [v, -u] back to physical [u, v] and ∇v.
    u = -raw_vel[2]
    v = raw_vel[1]

    ux = -raw_grad[2, 1]
    uy = -raw_grad[2, 2]
    vx = raw_grad[1, 1]
    vy = raw_grad[1, 2]

    vel = SVector{2, T}(u, v)
    jac = SMatrix{2, 2, T, 4}(ux, vx, uy, vy)
    return vel, jac
end

@inline function clamp_pullback(
        pullback::SMatrix{2, 2, T, 4};
        tol::T,
        max_iter::Int,
    ) where {T<:AbstractFloat}

    # Guard 1: pullback itself must be finite and non-singular.
    d_pb = det(pullback)
    (!isfinite(d_pb) || abs(d_pb) < eps(T)) && return pullback

    # Guard 2: PᵀP must be positive-definite before we invert it.
    Pt    = transpose(pullback)
    PtP   = Pt * pullback
    d_PtP = det(PtP)
    (!isfinite(d_PtP) || d_PtP < eps(T)) && return pullback

    # All 2×2 inversions below are closed-form — zero allocation, no exceptions.
    inv_Pt   = inv(Pt)
    tr_mixed = tr(inv(PtP))
    (!isfinite(tr_mixed) || abs(tr_mixed) < eps(T)) && return pullback

    prev_mu         = zero(T)
    used_mu         = zero(T)
    output_pullback = pullback   # = pullback + 0 * inv_Pt
    mixed_det       = d_pb       # det(pullback + 0 * inv_Pt) = det(pullback)

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

    v1, J1 = physical_velocity_and_jacobian(pos[1], pos[2], u_coeffs, d, cache)
    Pdot1 = -(transpose(J1) * input_pullback)

    midp_raw = pos + (dt * 0.5) * v1
    mx, my = apply_position_policy(midp_raw[1], midp_raw[2], Lx, Ly, mode, pos_tol)
    midp = SVector{2, T}(mx, my)
    midP = input_pullback + (dt * 0.5) * Pdot1

    v2, J2 = physical_velocity_and_jacobian(midp[1], midp[2], u_coeffs, d, cache)
    Pdot2 = -(transpose(J2) * midP)

    pullback = input_pullback + dt * Pdot2
    pullback = clamp_pullback(pullback; tol=pullback_tol, max_iter=pullback_max_iter)

    pos_raw = pos + dt * v2
    px, py = apply_position_policy(pos_raw[1], pos_raw[2], Lx, Ly, mode, pos_tol)

    return SVector{2, T}(px, py), pullback
end

function advect_particles_pullback_rk4!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T},
        thread_caches::Vector{EvaluationCache};
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

    max_err_per_thread = fill(zero(T), n_thread_slots)

    Threads.@threads :static for i in 1:num_p
        tid   = Threads.threadid()
        cache = thread_caches[tid]

        sx, sy = apply_position_policy(x0[i], y0[i], Lx, Ly, position_mode, position_tol)
        pos    = SVector{2, T}(sx, sy)
        input_pullback = SMatrix{2, 2, T, 4}(one(T), zero(T), zero(T), one(T))

        pos_new, pullback = pullback_rk4_step(
            pos, input_pullback, dt, u_coeffs, d, cache;
            mode=position_mode,
            pos_tol=position_tol,
            pullback_tol=pullback_tol,
            pullback_max_iter=pullback_max_iter,
        )

        m_old = SVector{2, T}(mx0[i], my0[i])
        m_new = pullback * m_old

        ox, oy     = x_out[i], y_out[i]
        ex         = abs(pos_new[1] - ox)
        ey         = abs(pos_new[2] - oy)
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
    end

    return maximum(@view max_err_per_thread[1:n_thread_slots])
end

function advect_particles_pullback_rk2!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T},
        thread_caches::Vector{EvaluationCache};
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

    max_err_per_thread = fill(zero(T), n_thread_slots)

    Threads.@threads :static for i in 1:num_p
        tid   = Threads.threadid()
        cache = thread_caches[tid]

        sx, sy = apply_position_policy(x0[i], y0[i], Lx, Ly, position_mode, position_tol)
        pos    = SVector{2, T}(sx, sy)
        input_pullback = SMatrix{2, 2, T, 4}(one(T), zero(T), zero(T), one(T))

        pos_new, pullback = pullback_rk2_step(
            pos, input_pullback, dt, u_coeffs, d, cache;
            mode=position_mode,
            pos_tol=position_tol,
            pullback_tol=pullback_tol,
            pullback_max_iter=pullback_max_iter,
        )

        m_old = SVector{2, T}(mx0[i], my0[i])
        m_new = pullback * m_old

        ox, oy    = x_out[i], y_out[i]
        ex        = abs(pos_new[1] - ox)
        ey        = abs(pos_new[2] - oy)
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
    end

    return maximum(@view max_err_per_thread[1:n_thread_slots])
end

function advect_particles_pullback_rk4!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T};
        kwargs...,
    ) where {T<:AbstractFloat, U<:Real}
    n_thread_slots = Threads.maxthreadid()
    thread_caches  = [EvaluationCache(evaluation_cache_size(d)) for _ in 1:n_thread_slots]
    return advect_particles_pullback_rk4!(
        p, x0, y0, mx0, my0, u_coeffs, d, dt,
        x_out, y_out, mx_out, my_out,
        thread_caches; kwargs...,
    )
end

function advect_particles_pullback_rk2!(
        p::Particles,
        x0::AbstractVector{T}, y0::AbstractVector{T},
        mx0::AbstractVector{T}, my0::AbstractVector{T},
        u_coeffs::AbstractVector{U},
        d::Domain, dt::T,
        x_out::AbstractVector{T}, y_out::AbstractVector{T},
        mx_out::AbstractVector{T}, my_out::AbstractVector{T};
        kwargs...,
    ) where {T<:AbstractFloat, U<:Real}
    n_thread_slots = Threads.maxthreadid()
    thread_caches  = [EvaluationCache(evaluation_cache_size(d)) for _ in 1:n_thread_slots]
    return advect_particles_pullback_rk2!(
        p, x0, y0, mx0, my0, u_coeffs, d, dt,
        x_out, y_out, mx_out, my_out,
        thread_caches; kwargs...,
    )
end

function coadjoint_step!(p::Particles, d::Domain)
    step_t0 = time()

    t0 = time()
    B = build_B_matrix(p, d)
    t_build_B_ms = time() - t0

    t0 = time()
    u_grid_n = solve_grid_velocity_lsqr(B, p)
    t_solve_raw_ms = time() - t0

    t0 = time()
    u_grid_n_h = Forms.build_form_field(d.R1, u_grid_n)
    u_div_free_coeffs_n, _ = project_and_get_pressure(d, u_grid_n_h)
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
        p::Particles,
        d::Domain,
        dt::Float64,
        f_n::AbstractVector{Tf},
        buf::SimulationBuffers,
        mass_matrix_1form::AbstractMatrix{Float64};
        min_particles_per_element::Int=10,
        max_particles_per_element::Int=25,
        min_particles_per_quarter::Int=3,
    ) where {Tf<:Real}

    step_t0       = time()
    thread_caches = buf.thread_caches

    copy!(buf.f_n_saved, f_n)
    f_n_saved = buf.f_n_saved

    t0 = time()
    added_start, removed_start = enforce_min_particles_per_element!(
        p, d,
        min_particles_per_element, max_particles_per_element, min_particles_per_quarter,
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
    @inbounds for i in 1:num_p
        x_np1[i]  = x_n[i];   y_np1[i]  = y_n[i]
        mx_np1[i] = mx_n[i];  my_np1[i] = my_n[i]
    end

    f_star     = buf.f_star
    f_np1      = buf.f_np1
    f_np1_raw  = buf.f_np1_raw
    f_np1_proj = buf.f_np1_proj

    copy!(f_star, f_n_saved)

    max_iter = 5
    tol      = 1e-5

    B_np1 = build_B_matrix(p, d)

    for iter in 1:max_iter
        iter_t0 = time()

        # 1. Advect n-state particles using current midpoint f_star.
        t0  = time()
        err = advect_particles_pullback_rk2!(
            p,
            @view(x_n[1:num_p]),  @view(y_n[1:num_p]),
            @view(mx_n[1:num_p]), @view(my_n[1:num_p]),
            f_star, d, dt,
            @view(x_np1[1:num_p]),  @view(y_np1[1:num_p]),
            @view(mx_np1[1:num_p]), @view(my_np1[1:num_p]),
            thread_caches,
        )
        t_advect = time() - t0

        # 2. Copy only (n+1) POSITIONS into p — impulses stay at n-state
        #    throughout the iteration so that the P2G RHS is stable.
        t0 = time()
        @inbounds for i in 1:num_p
            p.x[i] = x_np1[i]
            p.y[i] = y_np1[i]
        end
        particle_sorter!(p, d)
        t_sort = time() - t0

        # 3. Build B at (n+1) positions.
        t0    = time()
        B_np1 = build_B_matrix(p, d)
        t_build_B = time() - t0

        # 4. P2G: maps n-state impulses (p.mx/my) at (n+1) positions.
        t0      = time()
        raw_sol = solve_grid_velocity_lsqr(B_np1, p, buf.V_p)
        copy!(f_np1_raw, raw_sol)
        t_solve = time() - t0

        # 5. Energy correction using exact, stationary mass matrix.
        t0         = time()
        star_apply = v -> mass_matrix_1form * v  # <-- NOW USES EXACT GRID MASS MATRIX
        apply_energy_correction!(
            f_np1, f_n_saved, f_np1_raw, f_star;
            star_apply=star_apply,
            tol=1e-9,
            project_non_orthogonal=true,
        )
        t_energy = time() - t0

        # 6. Pressure projection.
        t0             = time()
        f_np1_h        = Forms.build_form_field(d.R1, f_np1)
        proj_result, _ = project_and_get_pressure(d, f_np1_h)
        copy!(f_np1_proj, proj_result)
        @. f_star = 0.5 * (f_n_saved + f_np1_proj)
        t_project = time() - t0

        t_iter = time() - iter_t0
        println("  Iter $iter: Pos Change = $err")
        println(
            "    Timings [s]: advect=$(round(t_advect,   digits=2)), " *
            "sort=$(round(t_sort,         digits=2)), " *
            "build_B=$(round(t_build_B,   digits=2)), " *
            "solve_raw=$(round(t_solve,   digits=2)), " *
            "project=$(round(t_project,   digits=2)), " *
            "energy=$(round(t_energy,     digits=2)), " *
            "iter_total=$(round(t_iter,   digits=2))",
        )

        err < tol && break
    end

    # Iteration converged. Now commit the final advected impulses into p
    # before the pressure correction acts on them.
    @inbounds for i in 1:num_p
        p.mx[i] = mx_np1[i]
        p.my[i] = my_np1[i]
    end

    # 7. Pressure feedback to particle impulses.
    t0 = time()
    apply_pressure_correction!(
        p, f_np1_proj, f_np1, d, thread_caches; delta_scale=1.0,
    )
    t_pressure = time() - t0

    # 8. Final sort.
    t0 = time()
    particle_sorter!(p, d)
    t_final_sort = time() - t0

    # 9. Particle rebalance after solve.
    t0 = time()
    added_end, removed_end = enforce_min_particles_per_element!(
        p, d,
        min_particles_per_element, max_particles_per_element, min_particles_per_quarter,
        f_np1_proj, buf,
    )
    t_seed_end = time() - t0
    if added_end > 0 || removed_end > 0
        println("  Particle rebalance after solve: +$added_end / -$removed_end")
    end

    t_total = time() - step_t0
    println(
        "  Post Timings [s]: " *
        "pressure_feedback=$(round(t_pressure,      digits=2)), " *
        "final_sort=$(round(t_final_sort,            digits=2)), " *
        "final_rebalance=$(round(t_seed_end,          digits=2)), " *
        "step_total=$(round(t_total,                 digits=2))",
    )

    return f_np1_proj
end

function maybe_clear_memo_tables!(step::Int, clear_every::Int, d::Domain)
    if clear_every > 0 && (step % clear_every == 0)
        Memoization.empty_all_caches!()
        warmup_evaluation_memo!(d)
        # println("  Cleared memoization caches at step $step")
    end
end

function compute_cfl_dt(p::Particles, d::Domain, target_cfl::T) where {T<:AbstractFloat}
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx
    dy = Ly / ny

    max_advective_rate = zero(T)
    @inbounds for i in 1:length(p.mx)
        # Initial impulse equals velocity in this Cartesian setup.
        u = p.mx[i]
        v = p.my[i]
        local_rate = abs(u) / dx + abs(v) / dy
        if local_rate > max_advective_rate
            max_advective_rate = local_rate
        end
    end

    if max_advective_rate <= eps(Float64)
        return Inf
    end

    return target_cfl / max_advective_rate
end

# ==============================================================================
#                                   MAIN
# ==============================================================================

function main()
    println("Initializing Domain and Particles...")

    LinearAlgebra.BLAS.set_num_threads(1)

    # Configuration
    nel = (64, 64)
    p   = (2, 2)
    k   = (1, 1)

    num_particles = nel[1] * nel[2] * 20

    # 1. Generate domain and particles
    domain    = GenerateDomain(nel, p, k)
    particles = generate_particles(num_particles, domain, :convecting)
    particle_sorter!(particles, domain)

    # Time-stepping parameters
    target_cfl = 0.5
    T_final    = 1.0
    dt_cfl     = compute_cfl_dt(particles, domain, target_cfl)
    if !isfinite(dt_cfl)
        n_steps = 1;  dt = T_final
    else
        n_steps = max(1, ceil(Int, T_final / dt_cfl))
        dt      = T_final / n_steps
    end

    clear_memo_every = 1
    reset_every      = 20

    println("Starting Time Integration (T_final=$T_final, dt=$dt, steps=$n_steps)...")

    output_dir         = "coflip_output";       mkpath(output_dir)
    particle_output_dir = "particle_output";    mkpath(particle_output_dir)

    # ---- t = 0 snapshot ----
    println("Saving true t=0 snapshot...")
    u_coeffs = coadjoint_step!(particles, domain)

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
            u_phys_form,
        )
    end

    d_u_phys = Forms.d(u_phys_form)
    ω_h      = ★(d_u_phys)
    Plot.export_form_fields_to_vtk((u_form,), "u_h_0000"; output_directory_tree=["coflip_output"])
    Plot.export_form_fields_to_vtk((ω_h,),    "w_h_0000"; output_directory_tree=["coflip_output"])
    export_particles_to_vtk(particles, joinpath(particle_output_dir, "particles_0000"))
    println("  Saved step 0 particles to VTK")

    warmup_evaluation_memo!(domain)

    # ---- Assemble exact 1-form mass matrix (precomputed, stationary) ----
    println("Assembling exact 1-form mass matrix...")
    M_1form = assemble_1form_mass_matrix(domain.R1, domain.dΩ)
    println("  Mass matrix assembled: $(size(M_1form, 1)) × $(size(M_1form, 2))")

    # ---- Allocate SimulationBuffers once before the loop ----
    ndofs        = length(u_coeffs)
    num_elements = prod(domain.nel)
    sim_buf      = SimulationBuffers(num_particles, ndofs, num_elements, domain)

    # ---- Time loop ----
    for step in 1:n_steps
        println("Step $step / $n_steps (t = $(round(step * dt, digits=3)))...")

        # FIX 3: pass sim_buf — no internal copy/similar allocations
        u_coeffs = step_co_flip!(particles, domain, dt, u_coeffs, sim_buf, M_1form)
        u_form      = Forms.build_form_field(domain.R1, u_coeffs; label="u_h")
        u_phys_expr = ★(u_form)
        u_phys_form = Assemblers.solve_L2_projection(domain.R1, u_phys_expr, domain.dΩ)
        
        if step % reset_every == 0
            particle_sorter!(particles, domain)
            @inbounds for pid in eachindex(particles.x)
                eid = particles.elem_ids[pid]
                ej  = ((eid - 1) ÷ nx) + 1
                ei  = eid - (ej - 1) * nx
                set_g2p_velocity(
                    particles, pid,
                    particles.x[pid], particles.y[pid],
                    eid, (ei - 1) * dx, (ej - 1) * dy, dx, dy,
                    u_phys_form,
                )
            end
            println("  Periodic G2P reset at step $step")
        end

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

        # FIX 1: clear_memo_every = 0 means this branch is never taken.
        maybe_clear_memo_tables!(step, clear_memo_every, domain)
    end

    println("\nSimulation complete! Step files written to '$(output_dir)'.")
end

function plot_flow()
    x = range(0, 1, length=100)
    y = range(0, 1, length=100)

    Lx = 1.
    Ly = 1.
    # Plot the flow_decay field
    u_vals = zeros(length(x), length(y))
    v_vals = zeros(length(x), length(y))

    for i in eachindex(x)
        for j in eachindex(y)
            u_vals[i, j], v_vals[i, j] = flow_decay(x[i], y[j], Lx, Ly)
        end
    end

    log_vel_mag = sqrt.(u_vals.^2 .+ v_vals.^2)

    heatmap(
        x, y, log_vel_mag',
        aspect_ratio = :equal,
        c = :viridis,
        title = "flow_decay velocity magnitude |u|",
        xlims = (0, Lx), ylims = (0, Ly)
    )
end

main()


