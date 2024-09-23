import Mantis
import SparseArrays

include("HelperFunctions.jl")

@doc raw"""
    volume_form_Hodge_Laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 2, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Bilinear form for the computation of the volume form Hodge Laplacian on the given element.

This function computes the contribution of the given element of both the 
bilinear and linear form for the volume form Hodge Laplace problem. The associated weak 
formulation is:

For given ``f^n \in L^2 \Lambda^n (\Omega)``, find ``u^{n-1} \in H(div, \Omega) \Lambda^{n-1} (\Omega)`` and ``\phi^n \in L^2 \Lambda^n (\Omega)`` such that 
```math
\begin{gather}
\langle \varepsilon^{n-1}, u^{n-1} \rangle - \langle d \varepsilon^{n-1}, \phi^n \rangle = 0 \quad \forall \ \varepsilon^{n-1} \in H(div, \Omega) \Lambda^{n-1} (\Omega) \\
\langle \varepsilon^n, d u^{n-1} \rangle = -\langle \varepsilon^n f^n \rangle \quad \forall \ \varepsilon^n \in L^2 \Lambda^n (\Omega)
\end{gather}
```
"""
function volume_form_Hodge_Laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 2, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    Forms = Mantis.Forms

    # Left hand side.
    # <ε¹, u¹>
    A_row_idx_11, A_col_idx_11, A_elem_11 = Forms.evaluate_inner_product(inputs.space_test[1], inputs.space_trial[1], element_id, inputs.quad_rule)
    
    # <dε¹, ϕ²>
    A_row_idx_12, A_col_idx_12, A_elem_12 = Forms.evaluate_inner_product(Forms.exterior_derivative(inputs.space_test[1]), inputs.space_trial[2], element_id, inputs.quad_rule)
    
    # <ε², du¹>
    A_row_idx_21, A_col_idx_21, A_elem_21 = Forms.evaluate_inner_product(inputs.space_test[2], Forms.exterior_derivative(inputs.space_trial[1]), element_id, inputs.quad_rule)

    # The remain term, A22, is zero, so not computed.

    # Add offsets.
    A_row_idx_21 .+= Forms.get_num_basis(inputs.space_test[1])

    A_col_idx_12 .+= Forms.get_num_basis(inputs.space_trial[1])

    # Put all variables together.
    A_row_idx = vcat(A_row_idx_11, A_row_idx_12, A_row_idx_21)
    A_col_idx = vcat(A_col_idx_11, A_col_idx_12, A_col_idx_21)
    A_elem = vcat(A_elem_11, A_elem_12, A_elem_21)


    # Right hand side. Only the second part is non-zero.
    # <ε², f²>
    b_row_idx, _, b_elem = Forms.evaluate_inner_product(inputs.space_test[2], inputs.forcing[2], element_id, inputs.quad_rule)
    b_elem .*= -1.0
    
    b_row_idx .+= Forms.get_num_basis(inputs.space_test[1])

    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end

function run_volume_form_Hodge_Laplacian(X, Y, ∫, fₑ)
    # create mixed form space
    V = Mantis.Forms.MixedFormSpace((X,Y))
    F = Mantis.Forms.MixedFormField((nothing,fₑ))

    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(F, V, ∫)

    # assemble all matrices
    A, b = Mantis.Assemblers.assemble(volume_form_Hodge_Laplacian, weak_form_inputs, Dict{Int, Float64}())

    # solve for coefficients of solution
    sol = A \ b

    # create solution as form fields and return
    V_fields = Mantis.Forms.build_form_fields(V, sol; labels=("σ", "u"))
    σₕ = V_fields[1]
    uₕ = V_fields[2]
    
    return σₕ, uₕ
end

# PROBLEM PARAMETERS -------------------------------------------------------------------
# mesh type
mesh_type = "curvilinear" # ∈ {"curvilinear", "cartesian"}
# number of elements in each direction
num_el = (2,2) .* 2^4
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
σₕ, uₕ = run_volume_form_Hodge_Laplacian(X, Y, ∫ₐ, fₑ)

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