############################################################################################
#                              n-form Hodge Laplacian                                      #
############################################################################################

"""
    n_form_hodge_laplacian(inputs::WeakFormInputs)

Function for assembling the weak form of the n-form Hodge Laplacian problem.

# Arguments
- `inputs::WeakFormInputs`: The inputs for the weak form assembly, including test, trial and
    forcing terms.

# Returns
- `lhs_expressions<:NTuple{num_lhs_rows, NTuple{num_lhs_cols, AbstractRealValuedOperator}}`:
    The left-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the left-hand side matrix.
- `rhs_expressions<:NTuple{num_rhs_rows, NTuple{num_rhs_cols, AbstractRealValuedOperator}}`:
    The right-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the right-hand side matrix.
"""
function n_form_hodge_laplacian(inputs::WeakFormInputs)
    ϵ¹, ε² = get_test_forms(inputs)
    u¹, ϕ² = get_trial_forms(inputs)
    f² = get_forcing(inputs)
    A_11 = ∫(ϵ¹ ∧ ★(u¹)) 
    A_12 = -∫(d(ϵ¹) ∧ ★(ϕ²))
    A_21 = ∫(ε² ∧ ★(d(u¹)))
    lhs_expressions = ((A_11, A_12), (A_21, 0))
    b_21 = ∫(ε² ∧ ★(f²))
    rhs_expressions = ((0,), (b_21,))

    return lhs_expressions, rhs_expressions
end

"""
    solve_volume_form_hodge_laplacian(Xⁿ⁻¹, Xⁿ, fₑ, Σ)

Returns the solution of the weak form of the n-form Hodge Laplacian.

# Arguments
- `Xⁿ⁻¹`: The (n-1)-form space to use as trial and test space.
- `Xⁿ`: The n-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `Σ`: The quadrature rule to use for the assembly.

# Returns
- `u¹ₕ`: The (n-1)-form solution of the weak-formulation.
- `ϕ²ₕ`: The n-form solution of the weak-formulation.
"""
function solve_volume_form_hodge_laplacian(Xⁿ⁻¹, Xⁿ, fₑ, Σ)
    weak_form_inputs = WeakFormInputs((Xⁿ⁻¹, Xⁿ), (fₑ,))
    weak_form = WeakForm(weak_form_inputs, n_form_hodge_laplacian)
    A, b = assemble(weak_form, Σ)
    sol = vec(A \ b)
    u¹ₕ, ϕ²ₕ = Forms.build_form_fields((Xⁿ⁻¹, Xⁿ), sol; labels=("u¹ₕ", "ϕ²ₕ"))

    return u¹ₕ, ϕ²ₕ
end
