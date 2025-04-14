############################################################################################
#                                  0-form Hodge Laplacian                                  #
############################################################################################

"""
    zero_form_hodge_laplacian(inputs::AbstractInputs)

Function for assembling the weak form of the 0-form Hodge Laplacian.

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
function zero_form_hodge_laplacian(inputs::AbstractInputs)
    v⁰ = Assemblers.get_test_form(inputs)
    u⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)
    A = ∫(d(v⁰) ∧ ★(d(u⁰)))
    lhs_expression = ((A,),)
    b = ∫(v⁰ ∧ ★(f⁰))
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end

"""
    solve_zero_form_hodge_laplacian(X⁰, fₑ, Σ)

Returns the solution of the weak form of the 0-form Hodge Laplacian.

# Arguments
- `X⁰`: The 0-form space to use as trial and test space.
- `fₑ`: The forcing term to use for the right-hand side of the weak formulation.
- `Σ`: The quadrature rule to use for the assembly.

# Returns
- `::Forms.FormField`: The solution of the weak-formulation.
"""
function solve_zero_form_hodge_laplacian(X⁰, fₑ, Σ)
    weak_form_inputs = WeakFormInputs(X⁰, fₑ)
    weak_form = WeakForm(weak_form_inputs, zero_form_hodge_laplacian)
    # homogeneous boundary conditions
    bc = Forms.zero_trace_boundary_conditions(X⁰) # TODO: generalize to non-zero bcs!
    # assemble all matrices
    A, b = assemble(weak_form, Σ, bc)
    # solve for coefficients of solution
    sol = vec(A \ b)
    # create the form field from the solution coefficients
    uₕ = Forms.build_form_field(X⁰, sol)

    return uₕ
end
