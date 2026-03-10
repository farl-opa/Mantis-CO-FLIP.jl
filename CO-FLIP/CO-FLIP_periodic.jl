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
    
    function EvaluationCache(max_basis_size::Int)
        v0 = [[zeros(1, max_basis_size) for _ in 1:2]] 
        v1 = [[zeros(1, max_basis_size) for _ in 1:2] for _ in 1:2] 
        res = Vector{Vector{Vector{Matrix{Float64}}}}(undef, 2)
        res[1] = v0; res[2] = v1
        return new(zeros(1, max_basis_size), res)
    end
end

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
    
    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k, box_size)
end

# ==============================================================================
#                          PARTICLE GENERATION & SORTING
# ==============================================================================

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
        u, v = 0.0, 0.0
        
        if flow_type == :tg
            u, v = flow_taylor_green(px, py, Lx, Ly)
        elseif flow_type == :vortex
            u, v = flow_lamb_oseen(px, py, Lx, Ly)
        elseif flow_type == :gyre
            u, v = flow_double_gyre(px, py, Lx, Ly)
        elseif flow_type == :decay
            u, v = flow_decay(px, py, Lx, Ly)
        elseif flow_type == :convecting
            u, v = flow_convecting_vortex(px, py, Lx, Ly)
        elseif flow_type == :merging
            u, v = flow_merging_vortices(px, py, Lx, Ly)
        else
            error("Unknown flow type: $flow_type")
        end
        
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

# ==============================================================================
#                             FLOW DEFINITIONS
# ==============================================================================
# ------------------------------------------------------------------------------
# 1. Taylor-Green Vortex (Periodic)
# ------------------------------------------------------------------------------
function flow_taylor_green(x::Float64, y::Float64, Lx::Float64, Ly::Float64)
    U0 = 1.0
    # Wavenumbers optimized for [0, Lx] x [0, Ly]
    kx = 2 * π * x / Lx
    ky = 2 * π * y / Ly
    
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
    # 1. Background Convection Velocity (moves the flow bottom to top)
    U_bg = 0.0
    V_bg = 2.0  # Adjust this to change how fast it travels upwards

    # 2. Initial Vortex Center (Start in the lower half to watch it rise)
    cx = Lx / 2.0
    cy = Ly / 4.0 
    
    # 3. Vortex Parameters
    Gamma = 2.0                # Circulation strength
    r_core = min(Lx, Ly) / 8.0  # Radius where max velocity occurs
    
    dx = x - cx
    dy = y - cy
    r2 = dx^2 + dy^2
    r  = sqrt(r2)
    
    # Avoid singularity at center
    if r < 1e-8
        return U_bg, V_bg
    end

    # 4. Tangential velocity using a sharp exponential decay
    # This ensures velocity perturbations are ~0 at the boundaries
    v_theta = (Gamma / (2 * π * r)) * (1.0 - exp(-r2 / (r_core^2)))
    
    # 5. Convert polar to Cartesian and superimpose background flow
    # u_total = u_bg + u_vortex
    # v_total = v_bg + v_vortex
    u = U_bg - v_theta * (dy / r)
    v = V_bg + v_theta * (dx / r)
    
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

function evaluate_fast!(cache::EvaluationCache, space, element_id, xi, nderivatives)
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

function probe_field_at_point(x::Float64, y::Float64, u_coeffs, d::Domain, cache::EvaluationCache)
    Lx, Ly = d.box_size
    nx, ny = d.nel
    
    x_wrapped = mod(x, Lx)
    y_wrapped = mod(y, Ly)
    
    dx = Lx / nx; dy = Ly / ny
    inv_dx = 1.0 / dx; inv_dy = 1.0 / dy
    
    ei = clamp(floor(Int, x_wrapped * inv_dx) + 1, 1, nx)
    ej = clamp(floor(Int, y_wrapped * inv_dy) + 1, 1, ny)
    elem_idx = (ej - 1) * nx + ei
    
    xi  = (x_wrapped - (ei - 1) * dx) * inv_dx
    eta = (y_wrapped - (ej - 1) * dy) * inv_dy
    
    points = Mantis.Points.CartesianPoints(([xi], [eta]))
    
    eval_out = evaluate_fast!(cache, d.R1.fem_space, elem_idx, points, 1)
    dof_indices = FunctionSpaces.get_basis_indices(d.R1.fem_space, elem_idx)
    local_coeffs = @view u_coeffs[dof_indices]
    n_loc = length(local_coeffs)

    # --- Velocity (Order 0) ---
    val_x = @view eval_out[1][1][1][1:n_loc]
    val_y = @view eval_out[1][1][2][1:n_loc]
    
    u = dot(val_x, local_coeffs)
    v = dot(val_y, local_coeffs)
    
    # --- Gradient (Order 1) ---
    dxi_u  = @view eval_out[2][1][1][1:n_loc]
    deta_u = @view eval_out[2][2][1][1:n_loc]
    dxi_v  = @view eval_out[2][1][2][1:n_loc]
    deta_v = @view eval_out[2][2][2][1:n_loc]
    
    du_dx = dot(dxi_u,  local_coeffs) * inv_dx
    du_dy = dot(deta_u, local_coeffs) * inv_dy
    dv_dx = dot(dxi_v,  local_coeffs) * inv_dx
    dv_dy = dot(deta_v, local_coeffs) * inv_dy
    
    return SVector{2}(u, v), SMatrix{2,2}(du_dx, dv_dx, du_dy, dv_dy)
end

function evaluate_field_at_probes(u_coeffs, probes::Particles, d::Domain)
    num_points = length(probes.x)
    u_mat = zeros(Float64, num_points, 2)
    viz_cache = EvaluationCache(32)
    
    @inbounds for i in 1:num_points
        vel, _ = probe_field_at_point(probes.x[i], probes.y[i], u_coeffs, d, viz_cache)
        u_mat[i, 1] = vel[2]
        u_mat[i, 2] = -vel[1]
    end
    return u_mat
end

function evaluate_velocity_and_vorticity_at_probes(u_coeffs, probes::Particles, d::Domain)
    num_points = length(probes.x)
    uvω_mat = zeros(Float64, num_points, 3)
    viz_cache = EvaluationCache(32)

    @inbounds for i in 1:num_points
        raw_vel, raw_grad = probe_field_at_point(probes.x[i], probes.y[i], u_coeffs, d, viz_cache)

        # Convert from stored flux ordering to physical velocity components.
        u = -raw_vel[2]
        v =  raw_vel[1]

        # Physical gradients after undoing component swap in the 1-form basis.
        dv_dx =  raw_grad[1,1]
        du_dy = -raw_grad[2,2]

        # 2D scalar vorticity: ω = ∂v/∂x - ∂u/∂y.
        ω = dv_dx - du_dy

        uvω_mat[i, 1] = u
        uvω_mat[i, 2] = v
        uvω_mat[i, 3] = ω
    end

    return uvω_mat
end

# ==============================================================================
#                            GRID OPERATIONS & SOLVERS
# ==============================================================================

function build_B_matrix(p::Particles, d::Domain)
    fes = d.R1.fem_space 
    num_particles = length(p.x)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    
    estimated_nnz = num_particles * 16 
    
    I_idx = Vector{Int}(); sizehint!(I_idx, estimated_nnz)
    J_idx = Vector{Int}(); sizehint!(J_idx, estimated_nnz)
    V_val = Vector{Float64}(); sizehint!(V_val, estimated_nnz)
    
    num_elements = prod(d.nel)
    
    batch_xs = Float64[]
    batch_ys = Float64[]
    batch_pids = Int[]
    MAX_BATCH = 1024 

    for eid in 1:num_elements
        # Collect all particles in this element
        empty!(batch_xs); empty!(batch_ys); empty!(batch_pids)
        
        pid = p.head[eid]
        while pid != 0
            push!(batch_pids, pid)
            push!(batch_xs, p.can_x[pid])
            push!(batch_ys, p.can_y[pid])
            pid = p.next[pid]
        end
        
        if isempty(batch_pids) continue end

        # Process in chunks
        total_p_in_elem = length(batch_pids)
        
        for chunk_start in 1:MAX_BATCH:total_p_in_elem
            chunk_end = min(chunk_start + MAX_BATCH - 1, total_p_in_elem)
            rng = chunk_start:chunk_end
            
            xs_view = @view batch_xs[rng]
            ys_view = @view batch_ys[rng]
            pids_view = @view batch_pids[rng]
            
            points_wrapped = Mantis.Points.CartesianPoints((xs_view, ys_view))
            eval_out, dof_indices = FunctionSpaces.evaluate(fes, eid, points_wrapped, 0)
            
            vals_x_matrix = eval_out[1][1][1] 
            vals_y_matrix = eval_out[1][1][2] 
            num_local_basis = size(vals_x_matrix, 2)
            
            # Assemble Sparse Entries
            for (i, global_pid) in enumerate(pids_view)
                w = sqrt(p.volume[global_pid])
                
                for k in 1:num_local_basis
                    val_x = vals_x_matrix[i, k] * w
                    val_y = vals_y_matrix[i, k] * w
                    
                    if abs(val_x) > 1e-15
                        push!(I_idx, 2 * global_pid - 1)
                        push!(J_idx, dof_indices[k])
                        push!(V_val, val_x)
                    end
                    
                    if abs(val_y) > 1e-15
                        push!(I_idx, 2 * global_pid)
                        push!(J_idx, dof_indices[k])
                        push!(V_val, val_y)
                    end
                end
            end
        end
    end
    
    return sparse(I_idx, J_idx, V_val, 2 * num_particles, num_dofs)
end

function solve_grid_velocity_lsqr(B, p)

    num_particles = length(p.x)

    V_p = Vector{Float64}(undef, 2*num_particles)
    @inbounds for i in 1:num_particles
        w = sqrt(p.volume[i])
        V_p[2*i-1] =  p.my[i] * w
        V_p[2*i]   = -p.mx[i] * w
    end

    function B_mul!(y, x)
        mul!(y, B, x)
    end

    function Bt_mul!(y, x)
        mul!(y, B', x)
    end

    B_map = LinearMap(B_mul!, Bt_mul!, size(B,1), size(B,2); ismutating=true)

    result = lsqr(
        B_map,
        V_p;
        atol = 1e-6,
        btol = 1e-6,
        conlim = 1e8,
        maxiter = 5000
    )

    return result
end

function project_and_get_pressure(dom::Domain, v_h)
    f = d(v_h) # Exterior derivative
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(dom.R1, dom.R2, f, dom.dΩ)
    div_free_coeffs = v_h.coefficients - u_corr.coefficients
    return div_free_coeffs, u_corr.coefficients
end

# ==============================================================================
#                        PHYSICS & ENERGY CORRECTION
# ==============================================================================

function compute_metric_inner_product(u, v, B)
    # Computes <u, v>_B = u' * B' * B * v
    p_u = B * u
    p_v = B * v
    return dot(p_u, p_v)
end

function apply_energy_correction!(f_next, f_curr, f_star, B)
    delta_f = f_next .- f_curr
    
    # Denominator: |f_star|^2_B
    norm_sq = compute_metric_inner_product(f_star, f_star, B)
    
    if norm_sq < 1e-10
        return
    end
    
    # Numerator: <f_star, delta_f>_B
    projection_overlap = compute_metric_inner_product(f_star, delta_f, B)
    scalar = projection_overlap / norm_sq
    scalar = clamp(scalar, -2.0, 2.0)
    
    # P_perp(delta_f) = delta_f - scalar * f_star
    @. f_next = f_curr + (delta_f - scalar * f_star)
end

function apply_pressure_correction!(p::Particles, tau_coeffs, d::Domain)
    n_threads = Threads.nthreads()
    thread_caches = [EvaluationCache(32) for _ in 1:n_threads]

    Threads.@threads for i in 1:length(p.x)
        tid = Threads.threadid()
        if tid > n_threads
            tid = mod1(tid, n_threads)
        end
        cache = thread_caches[tid]
        
        val, _ = probe_field_at_point(p.x[i], p.y[i], tau_coeffs, d, cache)
        
        p.my[i] += val[1]       
        p.mx[i] -= val[2]       
    end
end

# ==============================================================================
#                          TIME INTEGRATION & STEPPING
# ==============================================================================

function ode_rhs(state, u_coeffs, d, cache)
    x, y, mx, my = state
    
    # Probe Flux field (swapped components)
    raw_vel, raw_grad = probe_field_at_point(x, y, u_coeffs, d, cache)
    
    # Un-swap to get Physical Velocity
    u = -raw_vel[2]
    v =  raw_vel[1]
    
    # Un-swap Gradients
    ux = -raw_grad[2,1]
    vx =  raw_grad[1,1]
    uy = -raw_grad[2,2]
    vy =  raw_grad[1,2]
    
    # Advect Position
    dx_dt = u
    dy_dt = v
    
    # Advect Impulse (Covector Advection): d/dt(m) = - (∇v)ᵀ m
    dmx_dt = -(ux * mx + vx * my)
    dmy_dt = -(uy * mx + vy * my)
    
    return SVector{4}(dx_dt, dy_dt, dmx_dt, dmy_dt)
end

function advect_particles_rk2!(p, x0, y0, mx0, my0, u_coeffs, d, dt,
                               x_out, y_out, mx_out, my_out)

    num_p = length(x0)
    Lx, Ly = d.box_size
    max_err = Threads.Atomic{Float64}(0.0)
    
    n_threads = Threads.nthreads()
    thread_caches = [EvaluationCache(32) for _ in 1:n_threads]
    
    Threads.@threads for i in 1:num_p
        
        tid = Threads.threadid()
        if tid > length(thread_caches); tid = 1; end
        cache = thread_caches[tid]
        
        state = SVector{4}(x0[i], y0[i], mx0[i], my0[i])
        
        # --- RK2 (Midpoint Method) ---
        
        k1 = ode_rhs(state, u_coeffs, d, cache)
        
        midpoint_state = state + 0.5 * dt * k1
        
        k2 = ode_rhs(midpoint_state, u_coeffs, d, cache)
        
        new_state = state + dt * k2
        
        # --- Error Calculation ---
        
        nx_raw, ny_raw = new_state[1], new_state[2]
        ox, oy = x_out[i], y_out[i]
        
        dx = abs(nx_raw - ox)
        dy = abs(ny_raw - oy)
        
        if dx > 0.5 * Lx; dx = Lx - dx; end
        if dy > 0.5 * Ly; dy = Ly - dy; end
        
        Threads.atomic_max!(max_err, max(dx, dy))
        
        x_out[i]  = mod(nx_raw, Lx)
        y_out[i]  = mod(ny_raw, Ly)
        mx_out[i] = new_state[3]
        my_out[i] = new_state[4]
    end
    
    return max_err[]
end

function advect_particles_rk4!(p, x0, y0, mx0, my0, u_coeffs, d, dt, x_out, y_out, mx_out, my_out)
    num_p = length(x0)
    Lx, Ly = d.box_size
    max_err = Threads.Atomic{Float64}(0.0)
    
    n_threads = Threads.nthreads()
    thread_caches = [EvaluationCache(32) for _ in 1:n_threads]
    
    Threads.@threads for i in 1:num_p
        tid = Threads.threadid()
        if tid > length(thread_caches); tid = 1; end
        cache = thread_caches[tid]
        
        state = SVector{4}(x0[i], y0[i], mx0[i], my0[i])
        
        k1 = ode_rhs(state, u_coeffs, d, cache)
        k2 = ode_rhs(state + 0.5 * dt * k1, u_coeffs, d, cache)
        k3 = ode_rhs(state + 0.5 * dt * k2, u_coeffs, d, cache)
        k4 = ode_rhs(state + dt * k3, u_coeffs, d, cache)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Error Calculation
        nx_raw, ny_raw = new_state[1], new_state[2]
        ox, oy = x_out[i], y_out[i]
        
        dx = abs(nx_raw - ox)
        dy = abs(ny_raw - oy)
        
        if dx > 0.5 * Lx; dx = Lx - dx; end
        if dy > 0.5 * Ly; dy = Ly - dy; end
        
        Threads.atomic_max!(max_err, max(dx, dy))
        
        x_out[i]  = mod(nx_raw, Lx)
        y_out[i]  = mod(ny_raw, Ly)
        mx_out[i] = new_state[3]
        my_out[i] = new_state[4]
    end
    
    return max_err[]
end

function coadjoint_step!(p::Particles, d::Domain)
    B = build_B_matrix(p, d)
    u_grid_n = solve_grid_velocity_lsqr(B, p)
    u_grid_n_h = Forms.build_form_field(d.R1, u_grid_n)
    u_div_free_coeffs_n, _ = project_and_get_pressure(d, u_grid_n_h)
    return u_div_free_coeffs_n
end

function step_co_flip!(p::Particles, d::Domain, dt::Float64)
    step_t0 = time_ns()

    x_n  = copy(p.x)
    y_n  = copy(p.y)
    mx_n = copy(p.mx)
    my_n = copy(p.my)
    
    f_n = coadjoint_step!(p, d)
    
    # Initialize Fixed Point variables
    f_star = copy(f_n)
    x_np1 = copy(x_n)
    y_np1 = copy(y_n)
    mx_np1 = copy(mx_n)
    my_np1 = copy(my_n)
    
    f_np1_raw = similar(f_n)
    f_np1_proj = similar(f_n)
    
    max_iter = 5
    tol = 0.005
    
    for iter in 1:max_iter
        iter_t0 = time_ns()

        # Advect Particles
        t0 = time_ns()
        err = advect_particles_rk2!(p, x_n, y_n, mx_n, my_n, f_star, d, dt, x_np1, y_np1, mx_np1, my_np1)
        t_advect_ms = (time_ns() - t0) / 1.0e6
        
        # Project new particles to grid
        t0 = time_ns()
        p.x, p.y = x_np1, y_np1
        # p.mx, p.my = mx_np1, my_np1
        particle_sorter!(p, d)
        t_sort_ms = (time_ns() - t0) / 1.0e6
        
        t0 = time_ns()
        B_np1 = build_B_matrix(p, d) 
        t_build_B_ms = (time_ns() - t0) / 1.0e6
        
        # Solve Raw velocity from Particles
        t0 = time_ns()
        f_np1_raw = solve_grid_velocity_lsqr(B_np1, p)
        t_solve_raw_ms = (time_ns() - t0) / 1.0e6
        
        # Project to get Divergence-Free part
        t0 = time_ns()
        f_np1_raw_h = Forms.build_form_field(d.R1, f_np1_raw)
        f_np1_proj, _ = project_and_get_pressure(d, f_np1_raw_h)
        t_project_ms = (time_ns() - t0) / 1.0e6
        
        # Energy Correction
        t0 = time_ns()
        f_star_new = 0.5 .* (f_n .+ f_np1_proj)
        apply_energy_correction!(f_np1_proj, f_n, f_star_new, B_np1)
        t_energy_ms = (time_ns() - t0) / 1.0e6
        
        
        @. f_star = 0.5 * (f_n + f_np1_proj)
        t_iter_ms = (time_ns() - iter_t0) / 1.0e6
        
        println("  Iter $iter: Pos Change = $err")
        println(
            "    Timings [ms]: advect=$(round(t_advect_ms, digits=2)), sort=$(round(t_sort_ms, digits=2)), build_B=$(round(t_build_B_ms, digits=2)), solve_raw=$(round(t_solve_raw_ms, digits=2)), project=$(round(t_project_ms, digits=2)), energy=$(round(t_energy_ms, digits=2)), iter_total=$(round(t_iter_ms, digits=2))"
        )
        if err < tol
            break
        end
    end
    
    # Apply Pressure Feedback
    t0 = time_ns()
    tau_coeffs = f_np1_proj .- f_np1_raw
    apply_pressure_correction!(p, tau_coeffs, d)
    t_pressure_ms = (time_ns() - t0) / 1.0e6

    t0 = time_ns()
    particle_sorter!(p, d)
    t_final_sort_ms = (time_ns() - t0) / 1.0e6

    t_total_ms = (time_ns() - step_t0) / 1.0e6
    println(
        "  Post Timings [ms]: pressure_feedback=$(round(t_pressure_ms, digits=2)), final_sort=$(round(t_final_sort_ms, digits=2)), step_total=$(round(t_total_ms, digits=2))"
    )
end

function maybe_clear_memo_tables!(step::Int, clear_every::Int)
    if clear_every > 0 && (step % clear_every == 0)
        Memoization.empty_all_caches!()
        # println("  Cleared memoization caches at step $step")
    end
end

function estimate_cfl_number(p::Particles, d::Domain, dt::Float64)
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx
    dy = Ly / ny

    max_advective_rate = 0.0
    @inbounds for i in 1:length(p.mx)
        # Initial impulse equals velocity in this Cartesian setup.
        u = p.mx[i]
        v = p.my[i]
        local_rate = abs(u) / dx + abs(v) / dy
        if local_rate > max_advective_rate
            max_advective_rate = local_rate
        end
    end

    return dt * max_advective_rate
end

function compute_cfl_dt(p::Particles, d::Domain, target_cfl::Float64)
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx
    dy = Ly / ny

    max_advective_rate = 0.0
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

    # Avoid nested thread oversubscription between Julia threads and BLAS threads.
    LinearAlgebra.BLAS.set_num_threads(1)

    # Configuration
    nel = (80, 80)
    p = (2, 2)
    k = (1, 1)
    
    # Particles per cell (PPC) = 16
    num_particles = nel[1] * nel[2] * 20

    # 1. Generate the Domain
    domain = GenerateDomain(nel, p, k)
        
    # 2. Generate and Sort Particles
    particles = generate_particles(num_particles, domain, :merging)
    particle_sorter!(particles, domain)

    println("Initializing Visualization Grid...")
    nx_plot, ny_plot = 100, 100
    x_range = range(0, domain.box_size[1], length=nx_plot)
    y_range = range(0, domain.box_size[2], length=ny_plot)
    
    grid_x = [x for y in y_range, x in x_range] |> vec
    grid_y = [y for y in y_range, x in x_range] |> vec
    n_grid = length(grid_x)
    
    grid_probes = Particles(
        grid_x, grid_y, 
        zeros(n_grid), zeros(n_grid), zeros(n_grid), 
        zeros(n_grid), zeros(n_grid), Int[], Int[], Int[]
    )

    # Time-stepping parameters
    """
    target_cfl = 0.49
    T_final = 1.0
    dt_cfl = compute_cfl_dt(particles, domain, target_cfl)
    if !isfinite(dt_cfl)
        n_steps = 1
        dt = T_final
    else
        n_steps = max(1, ceil(Int, T_final / dt_cfl))
        dt = T_final / n_steps
    end
    cfl_used = estimate_cfl_number(particles, domain, dt)
    """
    T_final = 1.0
    dt = 0.001
    n_steps = ceil(Int, T_final / dt)
    clear_memo_every = 1
    
    println("Starting Time Integration (T_final=$T_final, dt=$dt, steps=$n_steps)...")

    
    output_dir = "coflip_output"
    mkpath(output_dir)

    metadata_path = joinpath(output_dir, "metadata.txt")
    open(metadata_path, "w") do io
        println(io, "# CO-FLIP output metadata")
        println(io, "nx_plot $nx_plot")
        println(io, "ny_plot $ny_plot")
        println(io, "n_steps $n_steps")
        println(io, "dt $dt")
        println(io, "Lx $(domain.box_size[1])")
        println(io, "Ly $(domain.box_size[2])")
        println(io, "has_vorticity 1")
    end

    # Warmup run on a copied particle state to reduce first-step compilation noise.
    println("Running warmup step...")
    warmup_particles = Particles(
        copy(particles.x),
        copy(particles.y),
        copy(particles.mx),
        copy(particles.my),
        copy(particles.volume),
        copy(particles.can_x),
        copy(particles.can_y),
        copy(particles.head),
        copy(particles.next),
        copy(particles.elem_ids),
    )
    particle_sorter!(warmup_particles, domain)
    step_co_flip!(warmup_particles, domain, dt)
    println("Warmup done. Starting measured steps...")

    for step in 1:n_steps
        println("Step $step / $n_steps (t = $(round(step*dt, digits=3)))...")

        # 3. Advance the CO-FLIP state
        step_co_flip!(particles, domain, dt)
        
        # 4. Visualization
        # Re-evaluate the velocity field to plot the results
        u_coeffs = coadjoint_step!(particles, domain)
        
        uvω_grid = evaluate_velocity_and_vorticity_at_probes(u_coeffs, grid_probes, domain)
        vel_mag = sqrt.(uvω_grid[:, 1].^2 .+ uvω_grid[:, 2].^2)

        # Columns: x, y, u, v, |u|, ω
        step_matrix = hcat(grid_x, grid_y, uvω_grid[:, 1], uvω_grid[:, 2], vel_mag, uvω_grid[:, 3])
        step_file = joinpath(output_dir, @sprintf("step_%04d.txt", step))
        writedlm(step_file, step_matrix)

        println("  Saved $(step_file)")
        maybe_clear_memo_tables!(step, clear_memo_every)
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

