using Mantis
using Random
using LinearAlgebra
using Plots
using SparseArrays
using IterativeSolvers
using LinearMaps
using StaticArrays
using CUDA

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
    num_dofs = size(B, 2)
    
    # Build RHS V_p
    V_p = Vector{Float64}(undef, 2 * num_particles)
    @inbounds for i in 1:num_particles
        w = sqrt(p.volume[i])
        V_p[2*i - 1] = p.my[i] * w
        V_p[2*i]     = -p.mx[i] * w
    end

    # Efficient Mat-Vec for RHS
    # b_vec = B' * V_p
    b_vec = Vector{Float64}(undef, num_dofs)
    mul!(b_vec, B', V_p)
    
    # --- Optimization: Pre-allocate Scratch Buffer for CG ---
    # We need a buffer of size (2 * num_particles) to store (B * x)
    temp_buffer = Vector{Float64}(undef, 2 * num_particles)

    # Mutating function for LinearMap
    # y = A * x  =>  y = B' * (B * x)
    function calc_Ax!(y, x)
        # 1. temp_buffer = B * x
        mul!(temp_buffer, B, x)
        # 2. y = B' * temp_buffer
        mul!(y, B', temp_buffer)
        return y
    end

    # Define LinearMap as Mutating
    A_map = LinearMap(calc_Ax!, num_dofs; 
                      ismutating=true, issymmetric=true, isposdef=true)

    # Jacobi Preconditioner construction
    # (This part was mostly fine, just ensured types)
    diag_vals = zeros(Float64, num_dofs)
    rows = rowvals(B)
    vals = nonzeros(B)
    
    @inbounds for j in 1:num_dofs
        sum_sq = 0.0
        for k in nzrange(B, j)
            val = vals[k]
            sum_sq += val^2
        end
        diag_vals[j] = sum_sq + 1e-8
    end
    
    P = Diagonal(1.0 ./ diag_vals) 

    # Run CG (allocations now drastically reduced)
    u_sol = cg(A_map, b_vec; Pl=P, reltol=1e-8, maxiter=1000)
    
    return u_sol
end

function project_and_get_pressure(dom::Domain, v_h)
    f = d(v_h) # Exterior derivative
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(dom.R1, dom.R2, f, dom.dΩ)
    div_free_coeffs = v_h.coefficients - u_corr.coefficients
    return div_free_coeffs, u_corr.coefficients
end

function coadjoint_step!(p::Particles, d::Domain, dt::Float64)
    # 1. Transfer P2G
    # t1 = time()
    u_grid_n = transfer_particles_to_grid_pcg(p, d)
    # println("  P2G: $(time() - t1) s")
    
    # 2. Pressure Projection
    # t2 = time()
    u_grid_n_h = Forms.build_form_field(d.R1, u_grid_n)
    u_div_free_coeffs_n, _ = project_and_get_pressure(d, u_grid_n_h)
    # println("  Pressure Projection: $(time() - t2) s")
    
    return u_div_free_coeffs_n
end

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
    nel = (150, 150)
    p = (2, 2)
    k = (1, 1)
    num_particles = nel[1]*nel[1]*16

    domain = GenerateDomain(nel, p, k)
    particles = generate_particles(num_particles, domain)
    particle_sorter!(particles, domain)

    println("Initializing Visualization Grid...")
    nx_plot, ny_plot = nel[1], nel[2]
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
    
    println("Running single coadjoint step...")
    u_coeffs = coadjoint_step!(particles, domain, dt)

    # Viz: Reconstructed velocity
    u_grid = evaluate_field_at_probes(u_coeffs, grid_probes, domain)
    mag_reconstructed = reshape(sqrt.(u_grid[:,1].^2 .+ u_grid[:,2].^2), nx_plot, ny_plot)
    
    # Exact Taylor-Green vortex solution
    U0 = 1.0
    Lx, Ly = domain.box_size
    mag_exact = zeros(nx_plot, ny_plot)
    @inbounds for j in 1:ny_plot
        for i in 1:nx_plot
            x = x_range[i]
            y = y_range[j]
            kx = 2 * π * x / Lx
            ky = 2 * π * y / Ly
            u_x = U0 * sin(kx) * cos(ky)
            u_y = -U0 * cos(kx) * sin(ky)
            mag_exact[i, j] = sqrt(u_x^2 + u_y^2)
        end
    end
    
    # Create side-by-side plots
    plt1 = heatmap(
        x_range, 
        y_range, 
        mag_reconstructed',
        aspect_ratio = :equal,
        c = :viridis,
        title = "CO-FLIP: |u| (Reconstructed)",
        xlims = (0, 1),
        ylims = (0, 1),
        clims = (0, 1.2)
    )
    
    plt2 = heatmap(
        x_range, 
        y_range, 
        mag_exact',
        aspect_ratio = :equal,
        c = :viridis,
        title = "Taylor-Green Vortex: |u| (Exact)",
        xlims = (0, 1),
        ylims = (0, 1),
        clims = (0, 1.2)
    )
    
    plt = plot(plt1, plt2, layout = (1, 2), size = (1200, 500))
    
    println("Saving plot...")
    savefig(plt, "co_flip_comparison.png")
    println("\nComparison plot saved to 'co_flip_comparison.png'.")
end

main()