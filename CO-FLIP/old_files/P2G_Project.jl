using Mantis
using Random
using LinearAlgebra
using CairoMakie
using Colors
using SparseArrays
using IterativeSolvers
using LinearMaps

"""
    DomainGenerator(nel, p, k)

Create a Domain with the necessary information for particle sorting and projection.
"""
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
    Ψ::Vector{Matrix{Float64}}
    particles_in_elements::Vector{Vector{Int}}
    canonical_positions::Vector{Tuple{Float64, Float64}}
end

function update_particles!(new_position::Vector{Tuple{Float64, Float64}}, new_impulse::Vector{Tuple{Float64, Float64}}, new_Ψ::Vector{Matrix{Float64}}, particles::Particles, domain::Domain)
    particles.position = new_position
    particles.impulse = new_impulse
    particles.Ψ = new_Ψ
    particle_sorter(particles, domain)
end

# -------------------------------------------------------------
#                      Constructor
# -------------------------------------------------------------
function GenerateDomain(nel::NTuple{2,Int},
                        p::NTuple{2,Int},
                        k::NTuple{2,Int})

    # mesh description
    starting_point = (0.0, 0.0)
    box_size       = (1.0, 1.0)

    # quadrature rules
    nq_assembly = p .+ 1
    nq_error    = nq_assembly .* 2

    ∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
        Quadrature.gauss_legendre, nq_assembly, nq_error
    )

    dΩ = Quadrature.StandardQuadrature(∫ₐ, prod(nel))

    # de Rham complex of B-splines
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_point, box_size, nel, p, k
    )
    R0, R1, R2 = R[1], R[2], R[3]

    # geometry of 1-forms (needed for evaluation)
    geo = Forms.get_geometry(R1)

    return Domain(R0, R1, R2, geo, dΩ, nel, p, k, box_size)
end

"""
    project(hp, v_h)

Given a discrete 1-form field `v_h`, return its divergence-free
component by removing the exact part δϕₕ obtained from the
Hodge–Laplace problem.
"""
function project(hp::Domain, v_h)

    # 1. Compute discrete divergence
    f = d(v_h)  

    # 3. Solve Hodge-Laplacian for δϕ_h
    u_corr, ϕ_corr = Assemblers.solve_volume_form_hodge_laplacian(hp.R1, hp.R2, f, hp.dΩ)

    # 4. Subtract non-div-free part
    return v_h.coefficients - u_corr.coefficients
end

"""
    generate_particles(num_particles::Int, domain::Domain)

Generates random particles within the domain and assigns them impulses 
based on a Taylor-Green Vortex flow field.
"""
function generate_particles(num_particles::Int, domain::Domain)
    positions = Vector{Tuple{Float64, Float64}}()
    impulses = Vector{Tuple{Float64, Float64}}()
    psis = Vector{Matrix{Float64}}()
    
    # Initialize empty vectors for the sorting fields
    particles_in_elements = Vector{Vector{Int}}()
    canonical_positions = Vector{Tuple{Float64, Float64}}()
    
    box_size = domain.box_size
    Lx = box_size[1]
    Ly = box_size[2]
    
    # Reference velocity magnitude
    U0 = 1.0 

    for _ in 1:num_particles
        # 1. Random Position
        x = rand() * Lx
        y = rand() * Ly
        push!(positions, (x, y))
        
        # 2. Assign Impulse based on Taylor-Green Vortex
        #    u =  sin(2πx) cos(2πy)
        #    v = -cos(2πx) sin(2πy)
        
        kx = 2 * π * x / Lx
        ky = 2 * π * y / Ly
        
        px =  U0 * sin(kx) * cos(ky)
        py = -U0 * cos(kx) * sin(ky)
        
        push!(impulses, (px, py))
        
        # 3. Identity Matrix for Psis
        push!(psis, Matrix{Float64}(I, 2, 2))
    end
    
    return Particles(positions, impulses, psis, particles_in_elements, canonical_positions)
end

"""
    particle_sorter(particles::Particles, domain::Domain)

Assigns each particle to its corresponding element and normalized coordinates.
"""
function particle_sorter(particles::Particles, domain::Domain)
    nel = domain.nel
    num_elements = nel[1] * nel[2]
    particles_in_elements = [Vector{Int}() for _ in 1:num_elements]
    canonical_positions = Vector{Tuple{Float64, Float64}}()
    
    elem_width_x = domain.box_size[1] / nel[1]
    elem_width_y = domain.box_size[2] / nel[2]
    
    for (idx, (x, y)) in enumerate(particles.position)
        # Determine the element index based on coordinates (using floor and clamping)
        elem_i = clamp(Int(floor(x / elem_width_x)) + 1, 1, nel[1])
        elem_j = clamp(Int(floor(y / elem_width_y)) + 1, 1, nel[2])
        element_num = (elem_j - 1) * nel[1] + elem_i
        
        # Add particle index to the corresponding element
        push!(particles_in_elements[element_num], idx)
        
        # Calculate normalized coordinates within the element
        norm_x = (x - (elem_i - 1) * elem_width_x) / elem_width_x
        norm_y = (y - (elem_j - 1) * elem_width_y) / elem_width_y
        push!(canonical_positions, (norm_x, norm_y))
    end

    particles.particles_in_elements = particles_in_elements
    particles.canonical_positions = canonical_positions

end

using SparseArrays

"""
    build_LS_matrix(particles::Particles, domain::Domain)

Constructs the sparse least-squares evaluation matrix for projecting particle data 
onto the R1 (1-form/velocity) finite element space.

The matrix B has dimensions (2 * num_particles, num_dofs).
- Rows 2i-1 and 2i correspond to the x and y components of particle i.
- Columns correspond to the global degrees of freedom in the 1-form space.

This function iterates element-by-element to leverage local support of B-splines.
"""
function build_B_matrix(particles::Particles, domain::Domain)
    form_space = domain.R1
    fes = form_space.fem_space 
    
    num_particles = length(particles.position)
    num_dofs = FunctionSpaces.get_num_basis(fes)
    
    # Estimate non-zeros (approx 16 per particle for 2D cubic/quadratic)
    estimated_nnz = num_particles * 16 
    I_idx = Vector{Int}()
    J_idx = Vector{Int}()
    V_val = Vector{Float64}()
    sizehint!(I_idx, estimated_nnz)
    sizehint!(J_idx, estimated_nnz)
    sizehint!(V_val, estimated_nnz)

    for elem_idx in eachindex(particles.particles_in_elements)
        p_indices = particles.particles_in_elements[elem_idx]
        if isempty(p_indices) continue end

        # 1. Wrap Points
        xs = [particles.canonical_positions[pid][1] for pid in p_indices]
        ys = [particles.canonical_positions[pid][2] for pid in p_indices]
        points_wrapped = Mantis.Points.CartesianPoints((xs, ys))
        
        # 2. Evaluate
        # Returns: evaluations, basis_indices
        # evaluations structure: [Order][Derivative][Component]
        eval_out, dof_indices = FunctionSpaces.evaluate(fes, elem_idx, points_wrapped)
        
        # Access 0-th order (index 1), 1st derivative-index (index 1)
        vals_x_matrix = eval_out[1][1][1] # Component 1 (X)
        vals_y_matrix = eval_out[1][1][2] # Component 2 (Y)
        
        num_local_basis = size(vals_x_matrix, 2) # Number of basis functions (columns)
        
        # 3. Build Triplets
        for (local_pid_idx, global_pid) in enumerate(p_indices)
            for k in 1:num_local_basis
                
                # Get the global Degree of Freedom index for this column
                dof_id = dof_indices[k]
                
                # --- X-Momentum Equation ---
                val_x = vals_x_matrix[local_pid_idx, k]
                if abs(val_x) > 1e-15
                    push!(I_idx, 2 * global_pid - 1) # Row: Particle X-velocity
                    push!(J_idx, dof_id)             # Col: Global DOF
                    push!(V_val, val_x)
                end
                
                # --- Y-Momentum Equation ---
                val_y = vals_y_matrix[local_pid_idx, k]
                if abs(val_y) > 1e-15
                    push!(I_idx, 2 * global_pid)     # Row: Particle Y-velocity
                    push!(J_idx, dof_id)             # Col: Global DOF
                    push!(V_val, val_y)
                end
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, 2 * num_particles, num_dofs)
end

"""
    evaluate_field(coefficients, domain, grid_probes)

Helper function to evaluate a 1-form field (defined by coefficients) onto the 
visualization grid. Returns a (N_grid x 2) matrix of velocities.
"""
function evaluate_field(coefficients, domain, grid_probes)
    fes = domain.R1.fem_space 
    num_points = length(grid_probes.position)
    u_evaluated = zeros(Float64, num_points, 2)
    
    # Pre-fetch evaluations to avoid repeated overhead
    # In a real app, you might cache this 'eval_out' since the grid doesn't change
    for elem_idx in eachindex(grid_probes.particles_in_elements)
        probe_indices = grid_probes.particles_in_elements[elem_idx]
        if isempty(probe_indices) continue end

        xs_local = [grid_probes.canonical_positions[pid][1] for pid in probe_indices]
        ys_local = [grid_probes.canonical_positions[pid][2] for pid in probe_indices]
        points_wrapped = Mantis.Points.CartesianPoints((xs_local, ys_local))

        eval_out, dof_indices = FunctionSpaces.evaluate(fes, elem_idx, points_wrapped)
        
        # Extract matrices for X and Y components
        # eval_out[Order=1][Deriv=1][Component=1 or 2]
        vals_x_matrix = eval_out[1][1][1] 
        vals_y_matrix = eval_out[1][1][2] 
        
        # Get local coefficients
        local_coeffs = coefficients[dof_indices]

        # Compute field values: Basis * Coeffs
        vel_x = vals_x_matrix * local_coeffs
        vel_y = vals_y_matrix * local_coeffs

        for (i, global_pid) in enumerate(probe_indices)
            u_evaluated[global_pid, 1] = vel_x[i]
            u_evaluated[global_pid, 2] = vel_y[i]
        end
    end
    return u_evaluated
end

function transfer_particles_to_grid_pcg(particles::Particles, domain::Domain)
    B = build_B_matrix(particles, domain)
    num_particles = length(particles.position)
    
    # 1. Build RHS Vector V_p
    V_p = zeros(Float64, 2 * num_particles)
    for i in 1:num_particles
        vx, vy = particles.impulse[i]
        V_p[2*i - 1] = vy
        V_p[2*i]     = -vx
    end

    b_vec = B' * V_p
    
    # 2. Linear Map for Matrix-free PCG
    function calc_Ax(x)
        return B' * (B * x)
    end
    
    num_dofs = size(B, 2)
    # Note: issymmetric=true is required for CG
    A_map = LinearMap(calc_Ax, num_dofs; ismutating=false, issymmetric=true, isposdef=true)

    # 3. Jacobi Preconditioner
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

    # 4. Solve (Using reltol for newer IterativeSolvers)
    u_sol = cg(A_map, b_vec; Pl=P, reltol=1e-8, maxiter=500)
    
    return u_sol
end

# ==============================================================================
#                               MAIN SCRIPT
# ==============================================================================

# 1. Setup Parameters (Robust settings)
nel = (70, 70) # Finer mesh for better accuracy
p = (2, 2)
k = (1, 1)        # Symmetric regularity
num_particles = nel[1]*nel[1]*20 # High count to prevent starvation

# 2. Create Domain & Particles
println("Creating Domain and Particles...")
domain = GenerateDomain(nel, p, k)
particles = generate_particles(num_particles, domain)
particle_sorter(particles, domain)

# 3. Create Visualization Grid & Sort It
println("Creating Visualization Grid...")
nx_plot, ny_plot = 100, 100
x_range = range(0, domain.box_size[1], length=nx_plot)
y_range = range(0, domain.box_size[2], length=ny_plot)
grid_flat = [ (x, y) for y in y_range, x in x_range ]
grid_flat = vec(grid_flat)

# Create dummy particles for the grid so we can use the sorter
grid_probes = Particles(
    grid_flat, 
    [(0.0,0.0) for _ in 1:length(grid_flat)], 
    [Matrix{Float64}(I,2,2) for _ in 1:length(grid_flat)],
    Vector{Vector{Int}}(), Vector{Tuple{Float64, Float64}}()
)
particle_sorter(grid_probes, domain)

# 4. Least Squares Approximation
println("Least Squares Approximation...")

u_ls = transfer_particles_to_grid_pcg(particles, domain)

println("LS Reconstruction Complete.")

# ==============================================================================
#                           PROJECTION STEP
# ==============================================================================
println("\nProjecting to Divergence-Free Space...")

u_ls_h = Forms.build_form_field(domain.R1, u_ls; label="u_ls")
#Mantis.Plot.export_form_fields_to_vtk((u_ls_h,), "U Before Projection")

u_projected = project(domain, u_ls_h)
#Mantis.Plot.export_form_fields_to_vtk((u_projected,), "U Projected Divergence Free")
println("Projection Complete.")


# ==============================================================================
#                               EVALUATION
# ==============================================================================

# 1. Evaluate LS Reconstruction
println("Evaluating fields...")
u_eval_ls = evaluate_field(u_ls, domain, grid_probes)

# 2. Evaluate Projected (Div-Free) Field
u_eval_proj = evaluate_field(u_projected, domain, grid_probes)

# 3. Compute Exact Solution
u_eval_exact = zeros(Float64, length(grid_flat), 2)
U0 = 1.0
Lx, Ly = domain.box_size
for (i, (x, y)) in enumerate(grid_flat)
    kx = 2 * π * x / Lx
    ky = 2 * π * y / Ly
    u_eval_exact[i, 1] =  U0 * sin(kx) * cos(ky)
    u_eval_exact[i, 2] = -U0 * cos(kx) * sin(ky)
end

# 4. Compute Magnitudes for Plotting (Reshaping to Matrix)
get_mag_mat = (vec_field) -> reshape(sqrt.(vec_field[:,1].^2 .+ vec_field[:,2].^2), nx_plot, ny_plot)

mag_ls   = get_mag_mat(u_eval_ls)
mag_proj = get_mag_mat(u_eval_proj)
mag_exact= get_mag_mat(u_eval_exact)

# Compute Error (Using the Projected field vs Exact)
diff_vec = u_eval_proj - u_eval_exact
mag_error = get_mag_mat(diff_vec)

l2_err = sqrt(sum(diff_vec.^2)/length(diff_vec))
println("Projected L2 Error: $l2_err")


# ==============================================================================
#                               PLOTTING
# ==============================================================================
println("Plotting...")

fig = Figure(size = (1100, 350))

# 1. Reconstructed (LS)
ax1 = Axis(fig[1, 1], title = "1. Reconstructed (LS)", aspect=DataAspect())
hm1 = heatmap!(ax1, x_range, y_range, mag_ls, colormap = :plasma)
Colorbar(fig[1, 2], hm1, label = "|u_ls|")

# 2. Projected (Div-Free)

ax2 = Axis(fig[1, 3], title = "2. Projected (Div-Free)", aspect=DataAspect())
hm2 = heatmap!(ax2, x_range, y_range, mag_proj, colormap = :plasma, colorrange=hm1.attributes.colorrange)
Colorbar(fig[1, 4], hm2, label = "|u_proj|")

# 3. Exact
ax3 = Axis(fig[1, 5], title = "3. Exact", aspect=DataAspect())
hm3 = heatmap!(ax3, x_range, y_range, mag_exact, colormap = :plasma, colorrange=hm1.attributes.colorrange)
Colorbar(fig[1, 6], hm3, label = "|u_exact|")


save("comparison_projection.png", fig)
display(fig)