import Mantis
import SparseArrays

include("HelperFunctions.jl")

@doc raw"""
    L2_projection(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Weak form for the computation of the ``L^2``-projection on the given element. The associated weak formulation is:

For given ``f^k \in L^2 \Lambda^k (\Omega)``, find ``\phi^k \in L^2 \Lambda^k (\Omega)`` such that 
```math
\int_{\Omega} \phi^k \wedge \star \varphi^k = -\int_{\Omega} f^k \wedge \star \varphi^k \quad \forall \ \varphi^k \in L^2 \Lambda^k (\Omega)
```
"""
function L2_projection(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    Forms = Mantis.Forms

    # The l.h.s. is the inner product between the test and trial functions.
    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(inputs.space_test[1], inputs.space_trial[1], element_id, inputs.quad_rule)

    # The r.h.s. is the inner product between the test and forcing functions.
    b_row_idx, _, b_elem = Forms.evaluate_inner_product(inputs.space_test[1], inputs.forcing[1], element_id, inputs.quad_rule)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # column_indices, values for the vector part.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end

function run_L2_projection(X, ∫, fₑ)
    # create mixed form space
    V = Mantis.Forms.MixedFormSpace((X,))
    F = Mantis.Forms.MixedFormField((fₑ,))

    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(F, V, ∫)

    # assemble all matrices
    A, b = Mantis.Assemblers.assemble(L2_projection, weak_form_inputs, Dict{Int, Float64}())

    # solve for coefficients of solution
    sol = A \ b

    # create solution as form fields and return
    V_fields = Mantis.Forms.build_form_fields(V, sol; labels=("u",))
    
    return V_fields[1]
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
W, _, _, ∫ₐ, ∫ₑ = tensor_product_de_rham_complex(□, p⁰, num_el, L, section_space_type, θ)

# EXACT SOLUTION -------------------------------------------------------------------
# exact solution for the problem
uₑ, _, _ = sinusoidal_solution(0, □)

# SOLVE PROBLEM -------------------------------------------------------------------
println("Solving the problem...")
uₕ = run_L2_projection(W, ∫ₐ, uₑ)

# COMPUTE ERROR -------------------------------------------------------------------
println("Computing error...")
error_u = L2_norm(uₕ - uₑ, ∫ₑ)
println("Error in u: ", error_u)

# VISUALIZE SOLUTION -------------------------------------------------------------------
# println("Visualizing the solution...")
# visualize_solution((uₕ, uₑ), ("uh", "ue"), "L2projection_$section_space_type _$mesh_type", □, 1, 4)

println("...done!")