module TensorProductBsplines

using Mantis

# Refer to the following file for method and variable definitions
include("../HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (2, 2) .^ 7

# B-spline parameters
p = (3, 3) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
dΩₐ = Quadrature.StandardQuadrature(∫ₐ, prod(num_elements))

verbose = true # Set to true for problem information.
export_vtk = false # Set to true to export the computed solutions.

############################################################################################
#                                       Run problem                                        #
############################################################################################
# de Rham complex
ℜ = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point, box_size, num_elements, p, k
)
# Only 0- and 1-forms are needed for the 1-form Hodge Laplacian.
ℜ⁰, ℜ¹  = ℜ[1], ℜ[2]

# Geometry
⊞ = Forms.get_geometry(ℜ⁰)

# Forcing function and analytical solutions.
u, δu, fₑ = sinusoidal_data(1, ⊞)

# Solve problem
uₕ, δuₕ = Assemblers.solve_one_form_hodge_laplacian(ℜ⁰, ℜ¹, fₑ, dΩₐ)

############################################################################################
#                                      Solution data                                       #
############################################################################################
if export_vtk
    println("Exporting computed solutions to VTK...")

    compt_file_name = "1-Form-HodgeLaplacian-TPBsplines-computed-p=$(p)-k=$(k)-nels=$(num_elements)"
    exact_file_name = "1-Form-HodgeLaplacian-TPBsplines-exact-p=$(p)-k=$(k)-nels=$(num_elements)"

    Plot.export_form_fields_to_vtk((uₕ, δuₕ), compt_file_name)
    Plot.export_form_fields_to_vtk((u, δu), exact_file_name)
end

end
