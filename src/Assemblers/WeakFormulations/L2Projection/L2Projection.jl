############################################################################################
#                                     Global L2 projection                                 #
############################################################################################

@doc raw"""
    L2_projection(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Weak form for the computation of the ``L^2``-projection on the given element. The associated weak formulation is:

For given ``f^k \in L^2 \Lambda^k (\Omega)``, find ``\phi^k_h \in X^k`` such that
```math
\int_{\Omega} \phi^k_h \wedge \star \varphi^k_h = -\int_{\Omega} f^k \wedge \star \varphi^k_h \quad \forall \ \varphi^k_h \in X^k\;,
```
where ``X`` is the discrete de Rham complex.
"""
function L2_projection(inputs::AbstractInputs, element_id)

    trial_forms = get_trial_forms(inputs)
    test_forms = get_test_forms(inputs)
    forcing = get_forcing(inputs)
    ∫ = get_global_quadrature_rule(inputs)

    # The l.h.s. is the inner product between the test and trial functions.
    A = Forms.Wedge(test_forms[1], Forms.Hodge(trial_forms[1]))
    # The r.h.s. is the inner product between the test and forcing functions.
    b = Forms.Wedge(test_forms[1], Forms.Hodge(forcing[1]))

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
    # column_indices, values for the vector part.
    return (A_idx[1], A_idx[2], A_elem), (b_idx[1], b_elem)

end

function solve_L2_projection(∫, Xᵏ, fₑ)
    # inputs for the mixed weak form
    weak_form_inputs = WeakFormInputs(∫, Xᵏ, fₑ)

    # assemble all matrices
    A, b = assemble(L2_projection, weak_form_inputs, Dict{Int, Float64}())

    # solve for coefficients of solution
    sol = A \ b

    # create the form field from the solution coefficients
    fₕ = Forms.build_form_field(Xᵏ, sol; label="fₕ")

    return fₕ
end
