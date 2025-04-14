############################################################################################
#                                     Global L2 projection                                 #
############################################################################################

"""
    L2_projection(inputs::AbstractInputs)

Function to compute the L2 projection of a function onto a discrete form space.

# Arguments
- `inputs::AbstractInputs`: The inputs for the weak form assembly, including test, trial and
    forcing terms.

# Returns
- `lhs_expression<:NTuple{num_lhs_rows, NTuple{num_lhs_cols, AbstractRealValuedOperator}}`:
    The left-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the left-hand side matrix.
- `rhs_expression<:NTuple{num_rhs_rows, NTuple{num_rhs_cols, AbstractRealValuedOperator}}`:
    The right-hand side of the weak form, which is a tuple of tuples contain all the blocks
    of the right-hand side matrix.
"""
function L2_projection(inputs::AbstractInputs)
    vᵏ = get_test_form(inputs)
    uᵏ = get_trial_form(inputs)
    fᵏ = get_forcing(inputs)
    A = ∫(vᵏ ∧ ★(uᵏ))
    lhs_expression = ((A,),)
    b = ∫(vᵏ ∧ ★(fᵏ))
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end

"""
    solve_L2_projection(Xᵏ, fₑ, Σ)

Returns the solution of the weak form of the L2 projection.

# Arguments
- `Xᵏ`: The k-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `Σ`: The quadrature rule to use for the assembly.

# Returns
- `fₕ::FormField`: The projection of `fₑ` onto `Xᵏ`.
"""
function solve_L2_projection(Xᵏ, fₑ, Σ)
    weak_form_inputs = WeakFormInputs(Xᵏ, fₑ)
    weak_form = WeakForm(weak_form_inputs, L2_projection)
    A, b = assemble(weak_form, Σ)
    sol = vec(A \ b)
    fₕ = Forms.build_form_field(Xᵏ, sol; label="fₕ")

    return fₕ
end
