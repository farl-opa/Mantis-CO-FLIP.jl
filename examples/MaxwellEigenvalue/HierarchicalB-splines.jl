module HierarchicalBsplines

import Mantis

# Refer to the following file for method and variable definitions
include("../HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh 
starting_point = (0.0, 0.0)
box_size = (fpi, fpi) #(π, π)
num_elements = (2, 2) .^ 3 # Ininital mesh size.

# B-spline parameters
p = (2, 2) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Hierarchical parameters.
truncate = true # true = THB, false = HB
simplified = false
num_steps = 2 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.
θ = 0.2 # Dorfler parameter.
Lchains = false # Decide if Lchains are added to fix inexact refinements.
eigenfunc = 1 # Eigenfunction to use for adaptive refinement.

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Mantis.Quadrature.get_quadrature_rules(
    Mantis.Quadrature.gauss_legendre, nq_assembly, nq_error
)

# Number of eigenvalues to compute
num_eig = 5 

verbose = true # Set to true for problem information.
export_vtk = true # Set to true to export the computed eigenfunctions.

############################################################################################
#                                       Run problem                                        #
############################################################################################
# Hierarchical de Rham complex 
ℌ = Mantis.Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate
)
# Geometry
# TODO: ⊞ = Mantis.Forms.get_geometry(ℌ) (from forms overhaul)
⊞ = Mantis.Forms.get_geometry(ℌ[1])

# Solve problem
ωₕ², uₕ= Mantis.Assemblers.solve_maxwell_eig(
    ℌ, num_steps, θ, Lchains, ∫ₐ, ∫ₑ, eigenfunc, num_eig; verbose
)

############################################################################################
#                                      Solution data                                       #
############################################################################################
# Print eigenvalues

if verbose
    # Exact eigenvalues
    ω² = Mantis.Assemblers.get_maxwell_eig(num_eig, ⊞, box_size)[1]

    println("Printing first $(num_eig) exact and computed eigenvalues...")
    println("i    ω²[i]      ωₕ²[i]     (ωₕ²[i] - ω²[i])^2")
    println("--   --------   --------   --------")
    for i in 1:num_eig
        @printf "%02.f" i
        @printf "   %08.5f" ω²[i]
        @printf "   %08.5f" ωₕ²[i]
        @printf "   %08.5f\n" (ωₕ²[i] - ω²[i])^2
    end
end

if export_vtk
    println("Exporting computed eigenfunctions to VTK...")
    hier_num_elements = Mantis.Geometry.get_num_elements(Mantis.Forms.get_geometry(uₕ[1]))
    file_base_name = "MaxwellEigenvalueHBsplines-computed-p=$(p)-k=$(k)-nels=$(hier_num_elements)"
    labels = Vector{String}(undef, num_eig)
    for i in 1:num_eig
        labels[i] = uₕ[i].label
    end

    Mantis.Plot.export_form_fields_to_vtk(uₕ, labels, file_base_name)
end

end
