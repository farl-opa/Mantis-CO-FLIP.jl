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
export_vtk = true # Set to true to export the computed solutions.

############################################################################################
#                                       Run problem                                        #
############################################################################################
# de Rham complex
ℜ = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point, box_size, num_elements, p, k
)
# Only 0- and 1-forms are needed for the 1-form Hodge Laplacian.
ℜ0, ℜ1, ℜ2  = ℜ[1], ℜ[2], ℜ[3] #for Poisson use 1 and 2 forms

# Geometry
⊞ = Forms.get_geometry(ℜ1)

# Build Psi exact
form_rank = 0
manifold_dim = 2
n_form_components = binomial(manifold_dim, form_rank)

function psi_expression(x::Matrix{Float64})
    v = sin.(2pi .* x[1,:]) .* sin.(2pi .* x[2,:])
    return repeat([v], n_form_components)
end

function sinusoidal_solution(
    form_rank::Int, geo::Geometry.AbstractGeometry{manifold_dim}
) where {manifold_dim}
    n_form_components = binomial(manifold_dim, form_rank)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # ∀i ∈ {1, 2, ..., n}: uᵢ = sin(ωx¹)sin(ωx²)...sin(ωxⁿ)
        y = @. sin(ω * x)
        return repeat([vec(prod(y; dims=2))], n_form_components)
    end
    return Mantis.Forms.AnalyticalFormField(form_rank, my_sol, geo, "f")
end


Ψ_exact = sinusoidal_solution(0, ⊞) #Mantis.Forms.AnalyticalFormField(0, psi_expression, ⊞, "psi")

# Solve L2
Ψ_h = Mantis.Assemblers.solve_L2_projection(ℜ0, Ψ_exact, dΩₐ)

# Forcing function and analytical solutions.
v, δu, fₑ = sinusoidal_data(2, ⊞) #take a look at how this is built

# Solve problem
vₕ, δuₕ = Assemblers.solve_volume_form_hodge_laplacian(ℜ1, ℜ2, fₑ, dΩₐ) #solve volume  n form

u_h = d(Ψ_h) + vₕ

f2 = d(u_h)

# println(typeof(f2))

phi, v_prime = Assemblers.solve_volume_form_hodge_laplacian(ℜ1, ℜ2, f2, dΩₐ)

############################################################################################
#                                      Solution data                                       #
############################################################################################
if export_vtk
    println("Exporting computed solutions to VTK...")

    compt_file_name = "1-Form-Poisson-computed-p=$(p)-k=$(k)-nels=$(num_elements)"
    exact_file_name = "1-Form-Poisson-exact-p=$(p)-k=$(k)-nels=$(num_elements)"

    # Plot.export_form_fields_to_vtk((uₕ, δuₕ), compt_file_name)
    # Plot.export_form_fields_to_vtk((u, δu), exact_file_name)
    # Plot.export_form_fields_to_vtk((Ψ_h,), "L2-projection-psi-p=$(p)-k=$(k)-nels=$(num_elements)")
    Plot.export_form_fields_to_vtk((phi, v_prime), "Divergence part solution")
end

end
