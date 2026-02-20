using Mantis
using Random
using LinearAlgebra
using CairoMakie
using Colors
using SparseArrays
using IterativeSolvers
using LinearMaps

# ==============================================================================
#                               DATA STRUCTURES
# ==============================================================================

struct Domain
    R0::Any
    R1::Any
    R2::Any
    geo::Any
    dΩ::Any

    nel::NTuple{2,Int}
    p::NTuple{2,Int}
    k::NTuple{2,Int}
    box_size::NTuple{2,Float64}
end

mutable struct Particles
    position::Vector{Tuple{Float64, Float64}}
    impulse::Vector{Tuple{Float64, Float64}}
    particles_in_elements::Vector{Vector{Int}}
    canonical_positions::Vector{Tuple{Float64, Float64}}
end

# ==============================================================================
#                           DOMAIN & PARTICLE SETUP
# ==============================================================================

function GenerateDomain(nel::NTuple{2,Int}, p::NTuple{2,Int}, k::NTuple{2,Int})
    starting_point = (0.0, 0.0)
    box_size       = (1.0, 1.0)
    nq_assembly = p .+ 1
    nq_error    = nq_assembly .* 2
    ∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(Quadrature.gauss_legendre, nq_assembly, nq_error)
    dΩ = Quadrature.StandardQuadrature(∫ₐ, prod(nel))
    R = Forms.create_tensor_product_bspline_de_rham_complex(starting_point, box_size, nel, p, k)
    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k, box_size)
end

function generate_particles(num_particles::Int, domain::Domain)
    positions = Vector{Tuple{Float64, Float64}}()
    impulses = Vector{Tuple{Float64, Float64}}()
    Lx, Ly = domain.box_size
    U0 = 1.0 

    for _ in 1:num_particles
        x = rand() * Lx
        y = rand() * Ly
        push!(positions, (x, y))
        
        # Taylor-Green Vortex Initial Condition
        kx = 2 * π * x / Lx
        ky = 2 * π * y / Ly
        px =  U0 * sin(kx) * cos(ky)
        py = -U0 * cos(kx) * sin(ky)
        push!(impulses, (px, py))
    end
    
    return Particles(positions, impulses, Vector{Vector{Int}}(), Vector{Tuple{Float64, Float64}}())
end

function particle_sorter!(particles::Particles, domain::Domain)
    nel = domain.nel
    num_elements = nel[1] * nel[2]
    particles.particles_in_elements = [Vector{Int}() for _ in 1:num_elements]
    empty!(particles.canonical_positions)
    
    elem_width_x = domain.box_size[1] / nel[1]
    elem_width_y = domain.box_size[2] / nel[2]
    
    for (idx, (x, y)) in enumerate(particles.position)
        x_wrapped = mod(x, domain.box_size[1])
        y_wrapped = mod(y, domain.box_size[2])
        particles.position[idx] = (x_wrapped, y_wrapped)

        elem_i = clamp(Int(floor(x_wrapped / elem_width_x)) + 1, 1, nel[1])
        elem_j = clamp(Int(floor(y_wrapped / elem_width_y)) + 1, 1, nel[2])
        element_num = (elem_j - 1) * nel[1] + elem_i
        
        push!(particles.particles_in_elements[element_num], idx)
        
        norm_x = (x_wrapped - (elem_i - 1) * elem_width_x) / elem_width_x
        norm_y = (y_wrapped - (elem_j - 1) * elem_width_y) / elem_width_y
        push!(particles.canonical_positions, (norm_x, norm_y))
    end
end

# ==============================================================================
#                           CORE CO-FLIP METHODS
# ==============================================================================

function build_B_matrix(particles::Particles, domain::Domain)
    form_space = domain.R1
    fes = form_space.fem_space 
    num_particles = length(particles.position)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    
    estimated_nnz = num_particles * 16 
    I_idx = Vector{Int}(); sizehint!(I_idx, estimated_nnz)
    J_idx = Vector{Int}(); sizehint!(J_idx, estimated_nnz)
    V_val = Vector{Float64}(); sizehint!(V_val, estimated_nnz)

    for elem_idx in eachindex(particles.particles_in_elements)
        p_indices = particles.particles_in_elements[elem_idx]
        if isempty(p_indices) continue end

        xs = [particles.canonical_positions[pid][1] for pid in p_indices]
        ys = [particles.canonical_positions[pid][2] for pid in p_indices]
        points_wrapped = Mantis.Points.CartesianPoints((xs, ys))
        
        # Note: B-matrix only needs values
        eval_out, dof_indices = FunctionSpaces.evaluate(fes, elem_idx, points_wrapped, 0)
        
        vals_x_matrix = eval_out[1][1][1] 
        vals_y_matrix = eval_out[1][1][2] 
        num_local_basis = size(vals_x_matrix, 2)
        
        for (local_pid_idx, global_pid) in enumerate(p_indices)
            for k in 1:num_local_basis
                val_x = vals_x_matrix[local_pid_idx, k]
                if abs(val_x) > 1e-15
                    push!(I_idx, 2 * global_pid - 1); push!(J_idx, dof_indices[k]); push!(V_val, val_x)
                end
                val_y = vals_y_matrix[local_pid_idx, k]
                if abs(val_y) > 1e-15
                    push!(I_idx, 2 * global_pid); push!(J_idx, dof_indices[k]); push!(V_val, val_y)
                end
            end
        end
    end
    return sparse(I_idx, J_idx, V_val, 2 * num_particles, num_dofs)
end

function transfer_particles_to_grid_pcg(particles::Particles, domain::Domain)
    B = build_B_matrix(particles, domain)
    num_particles = length(particles.position)
    
    V_p = zeros(Float64, 2 * num_particles)
    for i in 1:num_particles
        vx, vy = particles.impulse[i]
        V_p[2*i - 1] = vy
        V_p[2*i]     = -vx
    end

    b_vec = B' * V_p
    function calc_Ax(x) return B' * (B * x) end
    num_dofs = size(B, 2)
    A_map = LinearMap(calc_Ax, num_dofs; ismutating=false, issymmetric=true, isposdef=true)

    diag_vals = zeros(Float64, num_dofs)
    vals = nonzeros(B)
    for j = 1:size(B, 2)
        sum_sq = 0.0
        for k in nzrange(B, j)
             sum_sq += vals[k]^2
        end
        diag_vals[j] = sum_sq + 1e-8
    end
    P = Diagonal(1.0 ./ diag_vals) 

    u_sol = cg(A_map, b_vec; Pl=P, reltol=1e-8, maxiter=500)
    return u_sol
end

function project(hp::Domain, v_h)
    f = d(v_h) 
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(hp.R1, hp.R2, f, hp.dΩ)
    return v_h.coefficients - u_corr.coefficients
end

# ==============================================================================
#                           RK4 ADVECTION & GRADIENTS
# ==============================================================================

"""
    probe_field_at_point(x, y, u_coeffs, domain)

Evaluates velocity AND computes Gradient using Exact B-Spline Derivatives.
"""
function probe_field_at_point(x::Float64, y::Float64, u_coeffs, domain::Domain)
    # 1. Wrap (Periodic)
    Lx, Ly = domain.box_size
    nx, ny = domain.nel
    x = mod(x, Lx)
    y = mod(y, Ly)
    
    # 2. Find Element
    dx = Lx / nx
    dy = Ly / ny
    ei = clamp(floor(Int, x / dx) + 1, 1, nx)
    ej = clamp(floor(Int, y / dy) + 1, 1, ny)
    elem_idx = (ej - 1) * nx + ei
    
    # 3. Canonical Coords
    xi = (x - (ei - 1) * dx) / dx
    eta = (y - (ej - 1) * dy) / dy
    
    # 4. Evaluate using Mantis
    fes = domain.R1.fem_space
    points = Mantis.Points.CartesianPoints(([xi], [eta]))
    
    # !!! KEY FIX: nderivatives=1 !!!
    eval_out, dof_indices = FunctionSpaces.evaluate(fes, elem_idx, points, 1)
    
    local_coeffs = u_coeffs[dof_indices]
    
    # --- VELOCITY (Order 0, Index 1) ---
    # eval_out[Order=1][Dir=1][Component]
    val_x_basis = eval_out[1][1][1] # Basis X
    val_y_basis = eval_out[1][1][2] # Basis Y
    
    u = (val_x_basis * local_coeffs)[1]
    v = (val_y_basis * local_coeffs)[1]
    
    # --- GRADIENT (Order 1, Index 2) ---
    # Structure: eval_out[2][Dir][Component]
    # Dir 1 = d/dx, Dir 2 = d/dy
    
    # d/dx (Direction 1)
    d_dx_basis_x = eval_out[2][1][1]
    d_dx_basis_y = eval_out[2][1][2]
    
    # d/dy (Direction 2)
    d_dy_basis_x = eval_out[2][2][1]
    d_dy_basis_y = eval_out[2][2][2]
    
    # Compute dot products
    du_dx = (d_dx_basis_x * local_coeffs)[1]
    dv_dx = (d_dx_basis_y * local_coeffs)[1]
    
    du_dy = (d_dy_basis_x * local_coeffs)[1]
    dv_dy = (d_dy_basis_y * local_coeffs)[1]
    
    grad = [du_dx du_dy; dv_dx dv_dy]
    return (u, v), grad
end

function rk4_advect_coadjoint!(particles::Particles, u_coeffs, domain::Domain, dt::Float64)
    
    function get_derivative(pos, impulse)
        # Now uses exact derivatives!
        vel, grad = probe_field_at_point(pos[1], pos[2], u_coeffs, domain)
        
        # d_pos/dt = u
        dx_dt = vel
        
        # d_m/dt = - (∇u)ᵀ m  (Coadjoint Action)
        m_vec = [impulse[1], impulse[2]]
        dm_dt_vec = - (grad' * m_vec)
        dm_dt = (dm_dt_vec[1], dm_dt_vec[2])
        
        return dx_dt, dm_dt
    end

    for i in 1:length(particles.position)
        x0 = particles.position[i]
        m0 = particles.impulse[i]
        
        # RK4
        k1_x, k1_m = get_derivative(x0, m0)
        
        x1 = (x0[1] + 0.5 * dt * k1_x[1], x0[2] + 0.5 * dt * k1_x[2])
        m1 = (m0[1] + 0.5 * dt * k1_m[1], m0[2] + 0.5 * dt * k1_m[2])
        k2_x, k2_m = get_derivative(x1, m1)
        
        x2 = (x0[1] + 0.5 * dt * k2_x[1], x0[2] + 0.5 * dt * k2_x[2])
        m2 = (m0[1] + 0.5 * dt * k2_m[1], m0[2] + 0.5 * dt * k2_m[2])
        k3_x, k3_m = get_derivative(x2, m2)
        
        x3 = (x0[1] + dt * k3_x[1], x0[2] + dt * k3_x[2])
        m3 = (m0[1] + dt * k3_m[1], m0[2] + dt * k3_m[2])
        k4_x, k4_m = get_derivative(x3, m3)
        
        new_x = x0[1] + (dt / 6.0) * (k1_x[1] + 2*k2_x[1] + 2*k3_x[1] + k4_x[1])
        new_y = x0[2] + (dt / 6.0) * (k1_x[2] + 2*k2_x[2] + 2*k3_x[2] + k4_x[2])
        
        new_mx = m0[1] + (dt / 6.0) * (k1_m[1] + 2*k2_m[1] + 2*k3_m[1] + k4_m[1])
        new_my = m0[2] + (dt / 6.0) * (k1_m[2] + 2*k2_m[2] + 2*k3_m[2] + k4_m[2])
        
        particles.position[i] = (new_x, new_y)
        particles.impulse[i]  = (new_mx, new_my)
    end
end

function coadjoint_step!(particles::Particles, domain::Domain, dt::Float64)
    # 1. P2G
    u_star = transfer_particles_to_grid_pcg(particles, domain)
    
    # 2. Projection
    u_star_h = Forms.build_form_field(domain.R1, u_star; label="u_star")
    u_div_free_coeffs = project(domain, u_star_h)
    
    # 3. RK4 Advection (Position + Coadjoint Impulse)
    rk4_advect_coadjoint!(particles, u_div_free_coeffs, domain, dt)
    
    # 4. Resort
    particle_sorter!(particles, domain)
    
    return u_div_free_coeffs
end

function evaluate_field_at_probes(u_coeffs, probes::Particles, domain::Domain)
    num_points = length(probes.position)
    u_mat = zeros(Float64, num_points, 2)
    for i in 1:num_points
        px, py = probes.position[i]
        # We only need velocity here, discard gradient
        (u, v), _ = probe_field_at_point(px, py, u_coeffs, domain)
        u_mat[i, 1] = u
        u_mat[i, 2] = v
    end
    return u_mat
end

# ==============================================================================
#                               MAIN LOOP
# ==============================================================================

function main()
    println("Initializing Domain and Particles...")
    nel = (32, 32)
    p = (2, 2)
    k = (1, 1)
    num_particles = nel[1]*nel[1]*20

    domain = GenerateDomain(nel, p, k)
    particles = generate_particles(num_particles, domain)
    particle_sorter!(particles, domain)

    println("Initializing Visualization Grid...")
    nx_plot, ny_plot = 100, 100
    x_range = range(0, domain.box_size[1], length=nx_plot)
    y_range = range(0, domain.box_size[2], length=ny_plot)
    grid_flat = [ (x, y) for y in y_range, x in x_range ]
    
    grid_probes = Particles(vec(grid_flat), [(0.0,0.0) for _ in 1:length(grid_flat)], Vector{Vector{Int}}(), Vector{Tuple{Float64, Float64}}())

    fig = Figure(size=(800, 700))
    ax = Axis(fig[1,1], title="CO-FLIP: Taylor-Green Vortex (Velocity Magnitude)", aspect=DataAspect())
    mag_node = Observable(zeros(nx_plot, ny_plot))
    hm = heatmap!(ax, x_range, y_range, mag_node, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label = "|u|")

    dt = 0.01
    num_steps = 50
    
    println("Starting Simulation & Recording...")
    
    record(fig, "co_flip_simulation.mp4", 1:num_steps; framerate=20) do step
        if step % 10 == 0
            println("  Step $step / $num_steps")
        end
        
        u_coeffs = coadjoint_step!(particles, domain, dt)
        
        u_grid = evaluate_field_at_probes(u_coeffs, grid_probes, domain)
        mag = reshape(sqrt.(u_grid[:,1].^2 .+ u_grid[:,2].^2), nx_plot, ny_plot)
        mag_node[] = mag
    end

    println("\nAnimation saved to 'co_flip_simulation.mp4'.")
end

main()