using Mantis
using Random
using LinearAlgebra
using Plots
using SparseArrays
using IterativeSolvers
using LinearMaps
using StaticArrays

gr()

# ==============================================================================
#                               DATA STRUCTURES
# ==============================================================================
# OPTIMIZATION 1: Parametric Types for Domain

# This tells Julia exactly what types are inside, avoiding dynamic dispatch.

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

# OPTIMIZATION 2: Struct of Arrays (SoA) for Particles
# Separate vectors for x, y, mx, my are much faster for the CPU cache.

mutable struct Particles
    # Position
    x::Vector{Float64}
    y::Vector{Float64}
    # Impulse (Covector)
    mx::Vector{Float64}
    my::Vector{Float64}
    # Properties
    volume::Vector{Float64}
    # Pre-computed Canonical Coordinates (for B-spline eval)
    can_x::Vector{Float64}
    can_y::Vector{Float64}
    # OPTIMIZATION 3: Flat Linked-List for Spatial Sorting
    # head[element_index] -> first_particle_index
    # next[particle_index] -> next_particle_index
    head::Vector{Int}
    next::Vector{Int}
    # Track which element a particle is in
    elem_ids::Vector{Int}
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
    # Julia automatically infers the types F0, F1, etc.
    return Domain(R[1], R[2], R[3], Forms.get_geometry(R[2]), dΩ, nel, p, k, box_size)
end

function generate_particles(num_particles::Int, domain::Domain)
    Lx, Ly = domain.box_size
    U0 = 1.0
    # Initialize SoA vectors
    x  = Vector{Float64}(undef, num_particles)
    y  = Vector{Float64}(undef, num_particles)
    mx = Vector{Float64}(undef, num_particles)
    my = Vector{Float64}(undef, num_particles)
    vol = ones(Float64, num_particles) # Default volume 1.0
    can_x = zeros(Float64, num_particles)
    can_y = zeros(Float64, num_particles)
    num_elements = prod(domain.nel)
    head = zeros(Int, num_elements)
    next = zeros(Int, num_particles)
    elem_ids = zeros(Int, num_particles)



    for i in 1:num_particles
        # Random Position
        px = rand() * Lx
        py = rand() * Ly
        x[i] = px
        y[i] = py
        # Taylor-Green Vortex Initial Condition
        kx = 2 * π * px / Lx
        ky = 2 * π * py / Ly
        # Impulse
        mx[i] =  U0 * sin(kx) * cos(ky)
        my[i] = -U0 * cos(kx) * sin(ky)
    end
    return Particles(x, y, mx, my, vol, can_x, can_y, head, next, elem_ids)
end

function particle_sorter!(p::Particles, d::Domain)
    # Reset linked list heads
    fill!(p.head, 0)
    nx, ny = d.nel
    Lx, Ly = d.box_size
    dx = Lx / nx
    dy = Ly / ny
    # Pre-calculate inverse width for speed
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    @inbounds for i in 1:length(p.x)
        # Wrap positions (Periodic)
        p.x[i] = mod(p.x[i], Lx)
        p.y[i] = mod(p.y[i], Ly)
        # Find element index
        # 1-based indexing
        ei = clamp(floor(Int, p.x[i] * inv_dx) + 1, 1, nx)
        ej = clamp(floor(Int, p.y[i] * inv_dy) + 1, 1, ny)
        eid = (ej - 1) * nx + ei
        p.elem_ids[i] = eid
        # Canonical coordinates [-1, 1] or [0, 1] depending on basis?
        # Usually B-splines in Mantis are on [0,1] knot spans.
        p.can_x[i] = (p.x[i] - (ei - 1) * dx) * inv_dx
        p.can_y[i] = (p.y[i] - (ej - 1) * dy) * inv_dy
        # Linked List Insertion
        p.next[i] = p.head[eid]
        p.head[eid] = i
    end
end

# ==============================================================================
#                           CORE CO-FLIP METHODS
# ==============================================================================

function build_B_matrix(p::Particles, d::Domain)
    fes = d.R1.fem_space
    num_particles = length(p.x)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    # Heuristic for pre-allocation
    estimated_nnz = num_particles * 16
    # We use dynamic vectors but push! is amortized O(1)
    I_idx = Vector{Int}(); sizehint!(I_idx, estimated_nnz)
    J_idx = Vector{Int}(); sizehint!(J_idx, estimated_nnz)
    V_val = Vector{Float64}(); sizehint!(V_val, estimated_nnz)

    # Reusable buffers for batch evaluation
    # Max particles per element is roughly total/elements * safety_factor
    # But for safety we just resize inside the loop or use a reasonable buffer
    # Iterate over elements using the Linked List
    num_elements = prod(d.nel)
    # These buffers prevent allocations inside the loop
    batch_xs = Float64[]
    batch_ys = Float64[]
    batch_pids = Int[]
    for eid in 1:num_elements
        # Collect all particles in this element
        empty!(batch_xs)
        empty!(batch_ys)
        empty!(batch_pids)
        pid = p.head[eid]
        while pid != 0
            push!(batch_pids, pid)
            push!(batch_xs, p.can_x[pid])
            push!(batch_ys, p.can_y[pid])
            pid = p.next[pid]
        end
        if isempty(batch_pids) continue end

        # Batch Evaluate Basis
        points_wrapped = Mantis.Points.CartesianPoints((batch_xs, batch_ys))
        # eval_out: [Order][Direction][Component]
        # Order 0 = Values
        eval_out, dof_indices = FunctionSpaces.evaluate(fes, eid, points_wrapped, 0)
        vals_x_matrix = eval_out[1][1][1]
        vals_y_matrix = eval_out[1][1][2]
        num_local_basis = size(vals_x_matrix, 2)
        for (i, global_pid) in enumerate(batch_pids)
            # Volume weighting
            w = sqrt(p.volume[global_pid])
            for k in 1:num_local_basis
                val_x = vals_x_matrix[i, k] * w
                val_y = vals_y_matrix[i, k] * w
                # Thresholding
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
    return sparse(I_idx, J_idx, V_val, 2 * num_particles, num_dofs)

end

function transfer_particles_to_grid_pcg(p::Particles, d::Domain)
    B = build_B_matrix(p, d)
    num_particles = length(p.x)
    # Build RHS V_p
    V_p = zeros(Float64, 2 * num_particles)
    @inbounds for i in 1:num_particles
        w = sqrt(p.volume[i])
        # Impulse is a covector: (mx, my)
        # Mapping impulse to flux usually involves a rotation or metric scaling
        V_p[2*i - 1] = p.my[i] * w
        V_p[2*i]     = -p.mx[i] * w
    end
    b_vec = B' * V_p
    function calc_Ax(x) return B' * (B * x) end
    num_dofs = size(B, 2)
    A_map = LinearMap(calc_Ax, num_dofs; ismutating=false, issymmetric=true, isposdef=true)
    # Jacobi Preconditioner (Diagonal of B'B)
    diag_vals = zeros(Float64, num_dofs)
    # Iterate nonzeros of B efficiently
    rows = rowvals(B)
    vals = nonzeros(B)
    n = size(B, 2)
    # Column-wise iteration is native for CSC matrices
    @inbounds for j in 1:n
        sum_sq = 0.0
        for k in nzrange(B, j)
            val = vals[k]
            sum_sq += val^2
        end
        diag_vals[j] = sum_sq + 1e-8
    end
    P = Diagonal(1.0 ./ diag_vals)
    u_sol = cg(A_map, b_vec; Pl=P, reltol=1e-8, maxiter=500)
    return u_sol
end

# ==============================================================================
#                           RK4 ADVECTION (Optimized)
# ==============================================================================

# OPTIMIZATION 4: Zero-Allocation Probe
# Returns SVector (StaticArrays) instead of Tuple/Array
# This creates the result on the CPU stack.
function probe_field_at_point(x::Float64, y::Float64, u_coeffs, d::Domain)
    Lx, Ly = d.box_size
    nx, ny = d.nel
    x = mod(x, Lx)
    y = mod(y, Ly)
    dx = Lx / nx
    dy = Ly / ny
    ei = clamp(floor(Int, x / dx) + 1, 1, nx)
    ej = clamp(floor(Int, y / dy) + 1, 1, ny)
    elem_idx = (ej - 1) * nx + ei
    xi = (x - (ei - 1) * dx) / dx
    eta = (y - (ej - 1) * dy) / dy
    # Mantis evaluation
    # Note: Creating CartesianPoints here does allocate a tiny bit,
    # but much less than the full arrays before.
    points = Mantis.Points.CartesianPoints(([xi], [eta]))
    # Evaluate with derivatives (1)
    eval_out, dof_indices = FunctionSpaces.evaluate(d.R1.fem_space, elem_idx, points, 1)
    # Get local coeffs
    local_coeffs = @view u_coeffs[dof_indices]
    # --- Velocity ---
    val_x = eval_out[1][1][1]
    val_y = eval_out[1][1][2]
    # Dot product with local coeffs
    u = dot(val_x, local_coeffs)
    v = dot(val_y, local_coeffs)
    # --- Gradient ---
    # Dir 1 (d/dx)
    d_dx_x = eval_out[2][1][1]
    d_dx_y = eval_out[2][1][2]
    # Dir 2 (d/dy)
    d_dy_x = eval_out[2][2][1]
    d_dy_y = eval_out[2][2][2]
    du_dx = dot(d_dx_x, local_coeffs)
    dv_dx = dot(d_dx_y, local_coeffs)
    du_dy = dot(d_dy_x, local_coeffs)
    dv_dy = dot(d_dy_y, local_coeffs)
    # Return stack-allocated static arrays
    return SVector{2}(u, v), SMatrix{2,2}(du_dx, dv_dx, du_dy, dv_dy)
end

function rk4_advect_coadjoint!(p::Particles, u_coeffs, d::Domain, dt::Float64)
    # Closure for derivative calculation
    # Uses StaticArrays for everything
    function get_derivative(pos::SVector{2, Float64}, impulse::SVector{2, Float64})
        vel, grad = probe_field_at_point(pos[1], pos[2], u_coeffs, d)
        # d_pos/dt = u
        dx_dt = vel
        # d_m/dt = - (∇u)ᵀ m
        # StaticArrays handles the transpose and multiply efficiently
        dm_dt = - (grad' * impulse)
        return dx_dt, dm_dt
    end
    @inbounds for i in 1:length(p.x)
        # Load into registers (SVector)
        x0 = SVector{2}(p.x[i], p.y[i])
        m0 = SVector{2}(p.mx[i], p.my[i])
        # RK4 Steps
        k1_x, k1_m = get_derivative(x0, m0)
        x1 = x0 + 0.5 * dt * k1_x
        m1 = m0 + 0.5 * dt * k1_m
        k2_x, k2_m = get_derivative(x1, m1)
        x2 = x0 + 0.5 * dt * k2_x
        m2 = m0 + 0.5 * dt * k2_m
        k3_x, k3_m = get_derivative(x2, m2)
        x3 = x0 + dt * k3_x
        m3 = m0 + dt * k3_m
        k4_x, k4_m = get_derivative(x3, m3)
        # Final Update
        xn = x0 + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        mn = m0 + (dt / 6.0) * (k1_m + 2*k2_m + 2*k3_m + k4_m)
        # Write back to SoA
        p.x[i]  = xn[1]
        p.y[i]  = xn[2]
        p.mx[i] = mn[1]
        p.my[i] = mn[2]
    end
end

function coadjoint_step!(p::Particles, d::Domain, dt::Float64)
    # 1. Transfer P2G
    t1 = time()
    u_grid_n = transfer_particles_to_grid_pcg(p, d)
    println("  P2G: $(time() - t1) s")
    # 2. Pressure Projection
    t2 = time()
    u_grid_n_h = Forms.build_form_field(d.R1, u_grid_n)
    u_div_free_coeffs_n, pressure_grad_coeffs = project_and_get_pressure(d, u_grid_n_h)
    println("  Pressure Projection: $(time() - t2) s")
    # 3. Fixed Point Iteration
    t3 = time()
    p_next = deepcopy(p)
    v_advect_coeffs = u_div_free_coeffs_n
    max_iter = 3
    for iter in 1:max_iter
        # Reset p_next to state n
        p_next.x .= p.x
        p_next.y .= p.y
        p_next.mx .= p.mx
        p_next.my .= p.my
        # Advect
        rk4_advect_coadjoint!(p_next, v_advect_coeffs, d, dt)
        # P2G next
        particle_sorter!(p_next, d)
        u_grid_next = transfer_particles_to_grid_pcg(p_next, d)
        # Project
        u_grid_next_h = Forms.build_form_field(d.R1, u_grid_next)
        u_div_free_coeffs_next, _ = project_and_get_pressure(d, u_grid_next_h)
        # Average
        v_advect_coeffs = 0.5 .* (u_div_free_coeffs_n .+ u_div_free_coeffs_next)
    end
    println("  Fixed Point Iteration: $(time() - t3) s")
    # 4. Pressure Feedback
    t4 = time()
    apply_pressure_force!(p_next, pressure_grad_coeffs, d)
    println("  Pressure Feedback: $(time() - t4) s")
    # 5. Finalize
    p.x .= p_next.x
    p.y .= p_next.y
    p.mx .= p_next.mx
    p.my .= p_next.my
    p.head .= p_next.head
    p.next .= p_next.next
    return v_advect_coeffs
end

# Helper to project
function project_and_get_pressure(dom::Domain, v_h)
    f = d(v_h) # Exterior derivative
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(dom.R1, dom.R2, f, dom.dΩ)
    div_free_coeffs = v_h.coefficients - u_corr.coefficients
    return div_free_coeffs, u_corr.coefficients
end

function apply_pressure_force!(p::Particles, pressure_grad_coeffs, d::Domain)
    @inbounds for i in 1:length(p.x)
        px, py = p.x[i], p.y[i]
        # Probe field (ignore gradient output)
        grad_p, _ = probe_field_at_point(px, py, pressure_grad_coeffs, d)
        # Update impulse
        p.mx[i] -= grad_p[1]
        p.my[i] -= grad_p[2]
    end
end

function evaluate_field_at_probes(u_coeffs, probes::Particles, d::Domain)
    num_points = length(probes.x)
    u_mat = zeros(Float64, num_points, 2)
    @inbounds for i in 1:num_points
        vel, _ = probe_field_at_point(probes.x[i], probes.y[i], u_coeffs, d)
        u_mat[i, 1] = vel[1]
        u_mat[i, 2] = vel[2]
    end
    return u_mat
end

# ==============================================================================
#                                   MAIN
# ==============================================================================

function main()
    println("Initializing Domain and Particles...")
    nel = (16, 16)
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
    grid_x = Float64[]
    grid_y = Float64[]
    for y in y_range, x in x_range
        push!(grid_x, x)
        push!(grid_y, y)
    end
    n_grid = length(grid_x)
    grid_probes = Particles(grid_x, grid_y, zeros(n_grid), zeros(n_grid), zeros(n_grid), zeros(n_grid), zeros(n_grid), Int[], Int[], Int[])
    dt = 0.01
    num_steps = 10
    println("Starting Simulation & Recording...")
    # Plots.jl @animate macro handles the loop and frame capture
    anim = @animate for step in 1:num_steps
        if step % 1 == 0
            println("Step $step / $num_steps")
        end
        u_coeffs = coadjoint_step!(particles, domain, dt)
        # Viz Update
        u_grid = evaluate_field_at_probes(u_coeffs, grid_probes, domain)
        mag = reshape(sqrt.(u_grid[:,1].^2 .+ u_grid[:,2].^2), nx_plot, ny_plot)
        # Plotting the frame
        heatmap(
            x_range,
            y_range,
            mag', # Note: Transpose often needed depending on how reshape filled the array
            aspect_ratio = :equal,
            c = :viridis,
            title = "CO-FLIP Step $step: |u|",
            xlims = (0, 1),
            ylims = (0, 1),
            clims = (0, 1.5) # Fix colorbar range to stop flickering
        )
    end
    println("Saving animation...")
    mp4(anim, "co_flip_simulation_plots.mp4", fps=20)
    println("\nAnimation saved to 'co_flip_simulation_plots.mp4'.")
end

main()