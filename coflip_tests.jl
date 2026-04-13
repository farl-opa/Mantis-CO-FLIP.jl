using Mantis
using Random
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using StaticArrays

# CO-FLIP OPERATORS

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

# domain struct
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

# Implementation helper: choose a stable floating output type for probing.
probe_output_eltype(::Type{T}) where {T<:Real} = promote_type(Float64, T)

# Implementation helper: expose cached local basis size used in evaluations.
evaluation_cache_size(d::Domain) = d.eval_cache_size

# particles struct
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

# omplementation evaluation buffers reused to avoid per-particle allocations in tests.
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

# domain generation
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

# particle generation
function generate_particles_strat(num_particles::Int, domain::Domain, flow_type::Symbol=:tg)
    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    dx = Lx / nx
    dy = Ly / ny
    
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

    # Distribute particles evenly across grid cells and sample uniformly inside each cell.
    # This keeps the particle cloud stratified with respect to the mesh geometry.
    cell_order = randperm(num_elements)
    base_per_cell = div(num_particles, num_elements)
    extra_particles = rem(num_particles, num_elements)

    particle_idx = 1
    for (cell_rank, eid) in enumerate(cell_order)
        cell_particles = base_per_cell + (cell_rank <= extra_particles ? 1 : 0)
        if cell_particles == 0
            continue
        end

        ei = mod1(eid, nx)
        ej = cld(eid, nx)
        cell_x0 = (ei - 1) * dx
        cell_y0 = (ej - 1) * dy

        for _ in 1:cell_particles
            px = cell_x0 + rand() * dx
            py = cell_y0 + rand() * dy
            x[particle_idx] = px
            y[particle_idx] = py
            elem_ids[particle_idx] = eid

            # 2. Select Flow Type
            u, v = initial_velocity(flow_type, px, py, Lx, Ly)

            # 3. Assign Impulse
            # In this metric (Cartesian), impulse (covector) components = velocity components
            mx[particle_idx] = v
            my[particle_idx] = -u
            particle_idx += 1
        end
    end
    
    # Estimate particle volume (Area / N) for integration weights
    # This helps the least-squares P2G solver converge better
    particle_vol = (Lx * Ly) / num_particles
    fill!(vol, particle_vol)
    
    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids)
end

function generate_particles(num_particles::Int, domain::Domain, flow_type::Symbol=:tg)
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
        mx[i] = v
        my[i] = -u
    end
    
    # Estimate particle volume (Area / N) for integration weights
    # This helps the least-squares P2G solver converge better
    particle_vol = (Lx * Ly) / num_particles
    fill!(vol, particle_vol)
    
    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids)
end

# particle sorting
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

# building the matrix for the LS solve
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
            vy = -vals_x[k] * w
            vx = vals_y[k] * w

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

# Build the weighted right-hand side y used by normal equations B'B x = B'y.
function build_lsqr_rhs!(V_p::AbstractVector{Float64}, p::Particles)
    @inbounds for i in 1:length(p.x)
        w = sqrt(p.volume[i])
        V_p[2i-1] =  p.my[i] * w
        V_p[2i]   = -p.mx[i] * w
    end
    return V_p
end


# Generic matrix overload using CG on the normal equations.
function solve_grid_velocity_lsqr(
        B::AbstractMatrix,
        p::Particles;
        atol::Float64=1e-10,
        btol::Float64=1e-10,
        maxiter::Int=max(size(B)...),
    )
    num_particles = length(p.x)
    y = Vector{Float64}(undef, 2 * num_particles)
    build_lsqr_rhs!(y, p)

    # Solve min ||B*x - y||_2 directly on the rectangular system.
    return lsqr(B, y; atol=atol, btol=btol, maxiter=maxiter)
end

function project_and_get_pressure(dom::Domain, v_h::V) where {V}
    f = d(v_h) # Exterior derivative
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(dom.R1, dom.R2, f, dom.dΩ)
    div_free_coeffs = v_h.coefficients - u_corr.coefficients
    return div_free_coeffs, u_corr.coefficients
end



# TESTING FUNCTIONS (These are not from the CO-FLIP implementation)

# Generate form from analytical field
function analytical_form(dom::Domain, flow_type::Symbol)
    lx, ly = dom.box_size
    wx = 2π / lx
    wy = 2π / ly

    if flow_type == :tg
        function proxy_solution_tg(x::Matrix{Float64})
            xs = @view x[:, 1]
            ys = @view x[:, 2]
            u = @. sin(wx * xs) * cos(wy * ys)
            v = @. -cos(wx * xs) * sin(wy * ys)
            return [v, -u]
        end
        return Forms.AnalyticalFormField(1, proxy_solution_tg, dom.geo, "tg_proxy")
    
    elseif flow_type == :vortex
        function proxy_solution_vor(x::Matrix{Float64})
            xs = @view x[:, 1]
            ys = @view x[:, 2]
            u = zeros(size(xs))
            v = zeros(size(ys))
            for i in eachindex(xs)
                u[i], v[i] = flow_lamb_oseen(xs[i], ys[i], lx, ly)
            end
            return [v, -u]
        end
        return Forms.AnalyticalFormField(1, proxy_solution_vor, dom.geo, "vortex_proxy")
    
    elseif flow_type == :gyre
        function proxy_solution_gyr(x::Matrix{Float64})
            xs = @view x[:, 1]
            ys = @view x[:, 2]
            u = zeros(size(xs))
            v = zeros(size(ys))
            for i in eachindex(xs)
                u[i], v[i] = flow_double_gyre(xs[i], ys[i], lx, ly)
            end
            return [v, -u]
        end
        return Forms.AnalyticalFormField(1, proxy_solution_gyr, dom.geo, "gyre_proxy")
    
    elseif flow_type == :decay
        function proxy_solution_dec(x::Matrix{Float64})
            xs = @view x[:, 1]
            ys = @view x[:, 2]
            u = zeros(size(xs))
            v = zeros(size(ys))
            for i in eachindex(xs)
                u[i], v[i] = flow_decay(xs[i], ys[i], lx, ly)
            end
            return [v, -u]
        end
        return Forms.AnalyticalFormField(1, proxy_soluproxy_solution_dection, dom.geo, "decay_proxy")
    
    elseif flow_type == :convecting
        function proxy_solution_con(x::Matrix{Float64})
            xs = @view x[:, 1]
            ys = @view x[:, 2]
            u = zeros(size(xs))
            v = zeros(size(ys))
            for i in eachindex(xs)
                u[i], v[i] = flow_convecting_vortex(xs[i], ys[i], lx, ly)
            end
            return [v, -u]
        end
        return Forms.AnalyticalFormField(1, proxy_solution_con, dom.geo, "convecting_proxy")
    
    elseif flow_type == :merging
        function proxy_solution_mer(x::Matrix{Float64})
            xs = @view x[:, 1]
            ys = @view x[:, 2]
            u = zeros(size(xs))
            v = zeros(size(ys))
            for i in eachindex(xs)
                u[i], v[i] = flow_merging_vortices(xs[i], ys[i], lx, ly)
            end
            return [v, -u]
        end
        return Forms.AnalyticalFormField(1, proxy_solution_mer, dom.geo, "merging_proxy")
    
    else
        error("Unknown flow type: $flow_type")
    end
end


# Main tester function.
function run_p2g_convergence_mesh(; levels=((16, 16), (32, 32), (64, 64), (128, 128)), p=(3, 3), k=(2, 2), ppc=20)
    rows = Vector{Vector{Float64}}()

    for nel in levels
        dom = GenerateDomain(nel, p, k)

        # generate original form
        proxy_form = analytical_form(dom, :tg)
        # L2 project it
        ref_form = Assemblers.solve_L2_projection(dom.R1, proxy_form, dom.dΩ)

        
        Plot.plot(proxy_form; vtk_filename="original_form")
        Plot.plot(ref_form; vtk_filename="L2_projected_form")
        Plot.plot(ref_form - proxy_form; vtk_filename="convergence_L2_proj")
        

        # generate, sort and assign velocity to particles
        nparticles = nel[1] * nel[2] * ppc
        particles = generate_particles(nparticles, dom)
        particle_sorter!(particles, dom)

        # build and solve LS
        b = build_B_matrix(particles, dom)
        rec_coeffs = solve_grid_velocity_lsqr(b, particles)


        # reconstruct form and compute error
        rec_form = Forms.build_form_field(dom.R1, rec_coeffs)
        p2g_err = Analysis.compute_error_total(rec_form, proxy_form, dom.dΩ, "L2")
        l2_err = Analysis.compute_error_total(ref_form, proxy_form, dom.dΩ, "L2")

        # project and get projected error
        projected_coeffs, _ = project_and_get_pressure(dom, rec_form)
        projected_form = Forms.build_form_field(dom.R1, projected_coeffs)
        proj_err = Analysis.compute_error_total(projected_form, proxy_form, dom.dΩ, "L2")

        
        Plot.plot(rec_form; vtk_filename="recuperated_form")
        Plot.plot(rec_form - proxy_form; vtk_filename="p2g_convergence")
        Plot.plot(projected_form; vtk_filename="projected_form")
        Plot.plot(projected_form - proxy_form; vtk_filename="projection_convergence")

        

        push!(rows, [nel[1], nel[2], nparticles, p2g_err, l2_err, proj_err])
    end

    return rows
end

function write_table(path::String, header::String, rows::Vector{Vector{Float64}})
    open(path, "w") do io
        println(io, header)
        for row in rows
            println(io, join(row, " "))
        end
    end
end

function main()

    rows_mesh = run_p2g_convergence_mesh()

    mkpath(joinpath(@__DIR__, "tests_output"))

    write_table(
        joinpath(@__DIR__, "tests_output", "coflip_p2g_mesh_convergence.txt"),
        "# nel_x nel_y nparticles p2g_err l2_err proj_err",
        rows_mesh
    )

    println("P2G convergence tests completed.")
    println("Wrote: tests_output/coflip_p2g_mesh_convergence.txt")
end


main()
