############################################################################################
#                              0-form Hodge Laplacian                                      #
############################################################################################

@doc raw"""
    zero_form_hodge_laplacian(inputs::WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Function for assembling the weak form of the 0-form Hodge Laplacian on the given element. The associated weak formulation is:

For given ``f^0 \in L^2 \Lambda^0 (\Omega)``, find ``\phi^0_h \in X^0`` such that
```math
\int_{\Omega} d \phi^0_h \wedge \star d \varphi^0_h = -\int_{\Omega} f^0 \wedge \star \varphi^0_h \quad \forall \ \varphi^0_h \in X^0\;,
```
where ``X`` is the discrete de Rham complex, and such that ``\phi^0_h`` satisfies zero Dirichlet boundary conditions.
"""
function zero_form_hodge_laplacian(inputs::AbstractInputs, element_id)

    trial_forms = get_trial_forms(inputs)
    test_forms = get_test_forms(inputs)
    forcing = get_forcing(inputs)
    ∫ = get_global_quadrature_rule(inputs)

    # LHS = (dv, du)
    A = Forms.wedge(
        Forms.exterior_derivative(test_forms[1]),
        Forms.hodge(Forms.exterior_derivative(trial_forms[1]))
    )
    # RHS = (v, f)
    b = Forms.wedge(test_forms[1], Forms.hodge(forcing[1]))

    # compute contributions
    A_elem, A_idx = Analysis.integrate(
        ∫, element_id, A
    )
    b_elem, b_idx = Analysis.integrate(
        ∫, element_id, b
    )

    # The output should be the contribution to the left-hand-side matrix
    # A and right-hand-side vector b. The outputs are tuples of
    # row_indices, column_indices, values for the matrix part and
    # row_indices, values for the vector part. For this case, no shifts
    # or offsets are needed.
    return (A_idx[1], A_idx[2], A_elem), (b_idx[1], b_elem)

end

function solve_zero_form_hodge_laplacian(∫, X⁰, fₑ)
    # inputs for the mixed weak form
    weak_form_inputs = WeakFormInputs(∫, X⁰, fₑ)

    # homogeneous boundary conditions
    bc = Forms.zero_trace_boundary_conditions(X⁰) # TODO: generalize to non-zero bcs!

    # assemble all matrices
    A, b = assemble(zero_form_hodge_laplacian, weak_form_inputs, bc)

    # solve for coefficients of solution
    sol = A \ b

    # create the form field from the solution coefficients
    uₕ = Forms.build_form_field(X⁰, sol)

    return uₕ
end
