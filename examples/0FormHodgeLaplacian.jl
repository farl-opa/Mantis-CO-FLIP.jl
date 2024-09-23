import Mantis
import SparseArrays

include("HelperFunctions.jl")

@doc raw"""
    zero_form_Hodge_Laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Function for computing the weak form of the 0-form Hodge Laplacian on the given element. The associated weak formulation is:

For given ``f^0 \in L^2 \Lambda^n (\Omega)``, find ``\phi^0 \in H^1_0 \Lambda^n (\Omega)`` such that 
```math
\int_{\Omega} d \phi^0 \wedge \star d \varphi^0 = -\int_{\Omega} f^0 \wedge \star \varphi^0 \quad \forall \ \varphi^0 \in H^1_0 \Lambda^0 (\Omega)
```
"""
function zero_form_Hodge_Laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    Forms = Mantis.Forms

    # The inner product will be between the exterior derivative of the 
    # trial zero form with the exterior derivative of the test zero 
    # form, so we compute those first.
    dtrial = Forms.exterior_derivative(inputs.space_trial[1])
    dtest = Forms.exterior_derivative(inputs.space_test[1])

    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(dtest, dtrial, element_id, inputs.quad_rule)
    
    # The linear form is the inner product between the trial form and 
    # the forcing function which is a form of an appropriate rank.
    b_row_idx, _, b_elem = Forms.evaluate_inner_product(inputs.space_test[1], inputs.forcing[1], element_id, inputs.quad_rule)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end

function run_zero_form_Hodge_Laplacian(X, ∫, fₑ)
    # create mixed form space
    V = Mantis.Forms.MixedFormSpace((X,))
    F = Mantis.Forms.MixedFormField((fₑ,))

    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(F, V, ∫)

    # boundary conditions
    dof_partition = Mantis.FunctionSpaces.get_dof_partition(W.fem_space[1])
    bc_inds = vcat([dof_partition[1][i] for i in setdiff(1:9, 5)]...)
    bc_vals = zeros(Float64,length(bc_inds))

    # assemble all matrices
    A, b = Mantis.Assemblers.assemble(zero_form_Hodge_Laplacian, weak_form_inputs, Dict(zip(bc_inds, bc_vals)))

    # solve for coefficients of solution
    sol = A \ b

    # create solution as form fields and return
    V_fields = Mantis.Forms.build_form_fields(V, sol; labels=("u",))
    
    return V_fields[1]
end

# PROBLEM PARAMETERS -------------------------------------------------------------------
# mesh type
mesh_type = "cartesian" # ∈ {"curvilinear", "cartesian"}
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
uₑ, σₑ, fₑ = sinusoidal_solution(0, □)

# SOLVE PROBLEM -------------------------------------------------------------------
println("Solving the problem...")
uₕ = run_zero_form_Hodge_Laplacian(W, ∫ₐ, fₑ)

# COMPUTE ERROR -------------------------------------------------------------------
println("Computing error...")
error_u = L2_norm(uₕ - uₑ, ∫ₑ)
println("Error in u: ", error_u)

# VISUALIZE SOLUTION -------------------------------------------------------------------
# println("Visualizing the solution...")
# visualize_solution((uₕ, uₑ), ("uh", "ue"), "0FormHodgeLaplacian_$section_space_type _$mesh_type", □, 1, 4)

println("...done!")

