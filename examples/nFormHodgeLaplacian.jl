import Mantis
import SparseArrays

include("HelperFunctions.jl")

function stokes()
    # ...
end

function n_form_mixed_laplacian(X, Y, ∫, fₑ)
    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputsMixed(fₑ, X, Y, X, Y, ∫)
    
    # assemble all matrices
    weak_form = Mantis.Assemblers.poisson_mixed
    global_assembler = Mantis.Assemblers.Assembler(Dict{Int, Float64}())
    A, b = global_assembler(weak_form, weak_form_inputs)

    # solve for coefficients of solution
    sol = A \ b

    # create solution as forms and return
    σₕ = Mantis.Forms.FormField(X, "σ")
    uₕ = Mantis.Forms.FormField(Y, "u")
    σₕ.coefficients .= sol[1:Mantis.Forms.get_num_basis(X)]
    uₕ.coefficients .= sol[Mantis.Forms.get_num_basis(X)+1:end]
    
    return σₕ, uₕ
end

# PROBLEM PARAMETERS -------------------------------------------------------------------
# mesh type
mesh_type = "curvilinear" # ∈ {"curvilinear", "cartesian"}
# number of elements in each direction
num_el = (2,2) .* 2^5
# polynomial degree of the zero-form space in each direction
p⁰ = (2, 2)
# length of the domain in each direction
L = (1.0, 1.0)
# type of section space to use in each direction
section_space_type = "bernstein" # ∈ {"bernstein", "trigonometric", "legendre"}
θ = 2*pi ./ num_el

# GEOMETRY, SPACES & QUADRATURE -------------------------------------------------------------------
println("Setting up geometry, function spaces and quadrature rules...")
# geometry
if mesh_type == "cartesian"
    □ = unit_cube_cartesian(num_el)
else
    □ = unit_square_curvilinear(num_el)
end

# function spaces
_, X, Y, ∫ₐ, ∫ₑ = tensor_product_de_rham_complex(□, p⁰, num_el, L, section_space_type, θ)

# EXACT SOLUTION -------------------------------------------------------------------
# exact solution for the problem
uₑ, σₑ, fₑ = sinusoidal_solution(2, □)

# SOLVE PROBLEM -------------------------------------------------------------------
println("Solving the problem...")
σₕ, uₕ = n_form_mixed_laplacian(X, Y, ∫ₐ, fₑ)

# COMPUTE ERROR -------------------------------------------------------------------
println("Computing error...")
error_σ = L2_norm(σₕ - σₑ, ∫ₑ)
error_u = L2_norm(uₕ - uₑ, ∫ₑ)
println("Error in σ: ", error_σ)
println("Error in u: ", error_u)

# VISUALIZE SOLUTION -------------------------------------------------------------------
# println("Visualizing the solution...")
# visualize_solution((σₕ, uₕ, σₑ, uₑ), ("σh", "uh", "σe", "ue"), "nFormHodgeLaplacian_$section_space_type _$mesh_type", □, 1, 4)

println("...done!")