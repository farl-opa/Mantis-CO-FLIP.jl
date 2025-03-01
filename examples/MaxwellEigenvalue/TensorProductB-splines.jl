module TensorProductBsplines

import Mantis

# Refer to the following file for method and variable definitions
include("../HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh 
starting_point = (0.0, 0.0)
box_size = (fpi, fpi) #(π, π)
num_elements = (2, 2) .^ 3

# B-spline parameters
p = (3, 3) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = get_quadrature_rules(nq_assembly, nq_error)

# Number of eigenvalues to compute
num_eig = 20

verbose = true # Set to true for problem information.
export_vtk = false # Set to true to export the computed eigenfunctions.

############################################################################################
#                                       Run problem                                        #
############################################################################################
# de Rham complex 
ℜ = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point, box_size, num_elements, p, k
)
# Only 0- and 1-forms are needed for the Maxwell eigenvalue problem.
ℜ⁰, ℜ¹  = ℜ[1], ℜ[2]

# Geometry
# TODO: ⊞ = Mantis.Forms.get_geometry(ℜ) (from forms overhaul)
⊞ = Mantis.Forms.get_geometry(ℜ⁰)

# Solve problem
ωₕ², uₕ = solve_maxwell_eig(ℜ⁰, ℜ¹, ∫ₐ, num_eig; verbose)

############################################################################################
#                                      Solution data                                       #
############################################################################################
# Print eigenvalues

if verbose
    # Exact eigenvalues
    num_eig = 20
    ω² = get_maxwell_eig(num_eig, ⊞)[1]

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

    file_base_name = "MaxwellEigenvalueTPBsplines-computed-p=$(p)-k=$(k)-nels=$(num_elements)"
    labels = Vector{String}(undef, num_eig)
    for i in 1:num_eig
        labels[i] = uₕ[i].label
    end

    Mantis.Plot.export_form_fields_to_vtk(uₕ, labels, file_base_name)
end

end
