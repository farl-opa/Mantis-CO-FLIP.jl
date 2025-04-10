############################################################################################
#                              n-form Hodge Laplacian                                      #
############################################################################################

@doc raw"""
    volume_form_hodge_laplacian(inputs::WeakFormInputs{manifold_dim, 2, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Function for assembling the weak form of the n-form Hodge Laplacian on the given element. The associated weak formulation is:

Given ``f^n \in L^2 \Lambda^n (\Omega)``, find ``u^{n-1}_h \in X^{n-1}`` and ``\phi^n \in X^n`` such that
```math
\begin{gather}
\langle \varepsilon^{n-1}_h, u^{n-1}_h \rangle - \langle d \varepsilon^{n-1}_h, \phi^n_h \rangle = 0 \quad \forall \ \varepsilon^{n-1}_h \in X^{n-1} \\
\langle \varepsilon^n_h, d u^{n-1}_h \rangle = -\langle \varepsilon^n_h, f^n \rangle \quad \forall \ \varepsilon^n_h \in X^n
\end{gather}
```
"""
function volume_form_hodge_laplacian(inputs::WeakFormInputs, element_id)

    trial_forms = get_trial_forms(inputs)
    test_forms = get_test_forms(inputs)
    forcing = get_forcing(inputs)
    ∫ = get_global_quadrature_rule(inputs)

    # First equation: <ε¹, u¹> - <dε¹, ϕ²> = 0
    A_11 = Forms.Wedge(test_forms[1], Forms.Hodge(trial_forms[1]))
    A_12 = Forms.Wedge(Forms.ExteriorDerivative(test_forms[1]), Forms.Hodge(trial_forms[2]))
    # Second equation: <ε², du¹> = <ε², f²>
    A_21 = Forms.Wedge(test_forms[2], Forms.Hodge(Forms.ExteriorDerivative(trial_forms[1])))
    b_21 = Forms.Wedge(test_forms[2], Forms.Hodge(forcing[1]))

    # compute contributions
    A_elem_11, A_idx_11 = Analysis.integrate(∫, element_id, A_11)
    A_elem_12, A_idx_12 = Analysis.integrate(∫, element_id, A_12)
    A_elem_21, A_idx_21 = Analysis.integrate(∫, element_id, A_21)
    b_elem_21, b_idx_21 = Analysis.integrate(∫, element_id, b_21)

    # Add offsets
    A_idx_21[1] .+= Forms.get_num_basis(test_forms[1])
    A_idx_12[2] .+= Forms.get_num_basis(trial_forms[1])
    b_idx_21[1] .+= Forms.get_num_basis(test_forms[1])

    # Put all variables together.
    A_row_idx = vcat(A_idx_11[1], A_idx_12[1], A_idx_21[1])
    A_col_idx = vcat(A_idx_11[2], A_idx_12[2], A_idx_21[2])
    A_elem = vcat(A_elem_11, -A_elem_12, A_elem_21)

    # The output should be the contribution to the left-hand-side matrix
    # A and right-hand-side vector b. The outputs are tuples of
    # row_indices, column_indices, values for the matrix part and
    # row_indices, values for the vector part. For this case, no shifts
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_idx_21[1], b_elem_21)
end

function solve_volume_form_hodge_laplacian(∫, Xⁿ⁻¹, Xⁿ, fₑ)
    # inputs for the mixed weak form
    weak_form_inputs = WeakFormInputs(∫, (Xⁿ⁻¹, Xⁿ), (fₑ,))

    # assemble all matrices
    A, b = assemble(volume_form_hodge_laplacian, weak_form_inputs, Dict{Int, Float64}())

    # solve for coefficients of solution
    sol = A \ b

    # create solution as form fields and return
    σₕ, uₕ = Forms.build_form_fields((Xⁿ⁻¹, Xⁿ), sol; labels=("uₕ", "ϕₕ"))

    # return the field
    return σₕ, uₕ#, LinearAlgebra.cond(Array(A))
end
