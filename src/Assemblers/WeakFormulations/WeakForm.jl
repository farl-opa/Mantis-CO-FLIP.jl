############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    WeakForm{manifold_dim, LHS, RHS, I}

Structure representing the weak-formulation of a continuous Petrov-Galerking method. Both
the left and right-hand sides of the formulation are given as blocks of real-valued
operators; these are defined from a set of inputs holding the test, trial and forcing terms,
and a constructor method that defines where the blocks are placed.

# Fields
- `lhs_expressions::LHS`: The left-hand side blocks of the weak-formulation.
- `rhs_expressions::RHS`: The right-hand side blocks of the weak-formulation.
- `inputs::I`: The inputs for the weak-formulation, which include the test and trial spaces,
    and forcing terms.
# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the weak-formulation is defined.
- `LHS`: The type of the left-hand side expressions. Each row-column entry should be a
    subtype of  `AbstractRealValuedOperator` or `0`.
- `RHS`: The type of the right-hand side expressions. Each row-column entry should be a
    subtype of  `AbstractRealValuedOperator` or `0`.
- `I`: The type of the inputs. It should be a subtype of `WeakFormInputs{manifold_dim}`.
# Inner constructors
- `WeakForm(inputs::I, constructor::F)`: Creates a new `WeakForm` instance with the given
    inputs and constructor function. The constructor function is used to generate the
    left-hand side and right-hand side blocks of real-valued operators.
"""
struct WeakForm{manifold_dim, LHS, RHS, I}
    lhs_expressions::LHS
    rhs_expressions::RHS
    inputs::I
    function WeakForm(
        lhs_expressions::LHS, rhs_expressions::RHS, inputs::I
    ) where {
        lhs_num_rows,
        lhs_num_cols,
        rhs_num_rows,
        rhs_num_cols,
        LHS <: NTuple{
            lhs_num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        RHS <: NTuple{
            rhs_num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        I <: WeakFormInputs,
    }
        if lhs_num_rows != rhs_num_rows
            throw(
                ArgumentError(
                    """"The number of rows on the left-hand side must match the number \
                    of rows on the right-hand side. The left-hand side has $(lhs_num_rows) \
                    rows and the right-hand side has $(rhs_num_rows) rows."""
                ),
            )
        end

        num_test = get_num_test(inputs)
        num_trial = get_num_trial(inputs)
        if num_test != lhs_num_rows
            throw(
                ArgumentError(
                    """The number of rows on the left-hand side must match the number \
                    of test forms. The left-hand side has $(lhs_num_rows) rows and the \
                    inputs have $(num_test) test spaces."""
                ),
            )
        end

        if num_trial != lhs_num_cols
            throw(
                ArgumentError(
                    """The number of columns on the left-hand side must match the number \
                    of trial forms. The left-hand side has $(lhs_num_cols) columns and \
                    the inputs have $(num_trial) trial spaces."""
                ),
            )
        end

        manifold_dim = 0
        for row in 1:lhs_num_rows
            for col in 1:lhs_num_cols
                expression_type = typeof(lhs_expressions[row][col])
                if expression_type <: Forms.AbstractRealValuedOperator
                    manifold_dim = max(
                        manifold_dim, Forms.get_manifold_dim(lhs_expressions[row][col])
                    )
                end

                if expression_type == Int && lhs_expressions[row][col] != 0
                    throw(
                        ArgumentError(
                            """The only supported integer value for the entry of a \
                            weak-form is 0. The entry at row $(row), column $(col) of \
                            the left-hand side has value $(lhs_expressions[row][col])."""
                        ),
                    )
                end
            end
        end

        for row in 1:rhs_num_rows
            for col in 1:rhs_num_cols
                expression_type = typeof(rhs_expressions[row][col])
                if expression_type <: Forms.AbstractRealValuedOperator
                    manifold_dim = max(
                        manifold_dim, Forms.get_manifold_dim(rhs_expressions[row][col])
                    )
                end

                if expression_type == Int && rhs_expressions[row][col] != 0
                    throw(
                        ArgumentError(
                            """The only supported integer value for the entry of a \
                            weak-form is 0. The entry at row $(row), column $(col) of \
                            the right-hand side has value $(rhs_expressions[row][col])."""
                        ),
                    )
                end
            end
        end

        return new{manifold_dim, LHS, RHS, I}(lhs_expressions, rhs_expressions, inputs)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_lhs_expressions(wf::WeakForm) = wf.lhs_expressions
get_rhs_expressions(wf::WeakForm) = wf.rhs_expressions
get_inputs(wf::WeakForm) = wf.inputs
get_test_forms(wf::WeakForm) = get_test_forms(get_inputs(wf))
get_trial_forms(wf::WeakForm) = get_trial_forms(get_inputs(wf))
get_forcings(wf::WeakForm) = get_forcings(get_inputs(wf))
get_forcing(wf::WeakForm, id::Int=1) = get_forcing(get_inputs(wf), id)
get_test_sizes(wf::WeakForm) = Forms.get_num_basis.(get_test_forms(wf))
get_trial_sizes(wf::WeakForm) = Forms.get_num_basis.(get_trial_forms(wf))
get_test_offsets(wf::WeakForm) = [0; cumsum(get_test_sizes(wf)[1:(end - 1)])...]
get_trial_offsets(wf::WeakForm) = [0; cumsum(get_trial_sizes(wf)[1:(end - 1)])...]
get_test_size(wf::WeakForm) = sum(get_test_sizes(wf))
get_trial_size(wf::WeakForm) = sum(get_trial_sizes(wf))
get_test_max_local_dim(wf::WeakForm) = sum(Forms.get_max_local_dim.(get_test_forms(wf)))
get_trial_max_local_dim(wf::WeakForm) = sum(Forms.get_max_local_dim.(get_trial_forms(wf)))
get_num_rhs_cols(wf::WeakForm) = length(get_rhs_expressions(wf)[1])
get_lhs_size(wf::WeakForm) = get_test_size(wf), get_trial_size(wf)

function get_rhs_size(wf::WeakForm)
    if isnothing(get_forcing(wf))
        return get_test_size(wf), get_trial_size(wf)
    end

    return get_test_size(wf), 1
end

"""
    get_num_lhs_blocks(wf::WeakForm)

Returns a tuple containing the number of row and column blocks, respectively, for the
left-hand side of the weak-formulation.

# Arguments
- `wf::WeakForm`: The weak-formulation for which the number of blocks is to be determined.

# Returns
- `Tuple(Int, Int)`: The number of row and column blocks for the left-hand side of the weak
    formulation.
"""
function get_num_lhs_blocks(wf::WeakForm)
    inputs = get_inputs(wf)

    return (get_num_test(inputs), get_num_trial(inputs))
end

"""
    get_num_rhs_blocks(wf::WeakForm)

Returns a tuple containing the number of row and column blocks, respectively, for the
right-hand side of the weak-formulation.

# Arguments
- `wf::WeakForm`: The weak-formulation for which the number of blocks is to be determined.

# Returns
- `Tuple(Int, Int)`: The number of row and column blocks for the right-hand side of the weak
    formulation.
"""
function get_num_rhs_blocks(wf::WeakForm)
    inputs = get_inputs(wf)

    return (get_num_test(inputs), get_num_rhs_cols(wf))
end

"""
    get_estimated_nnz_per_elem(wf::WeakForm)

Returns the estimated number of non-zero entries per element for the left- and right-hand
sides of the weak-formulation.

# Arguments
- `wf::WeakForm`: The weak-formulation for which the estimated number of non-zero entries is
    to be determined.

# Returns
- `Tupe(Int, Int)`: The estimated number of non-zero entries per element for the left-hand
    side and right-hand side of the weak-formulation, respectively.
"""
function get_estimated_nnz_per_elem(wf::WeakForm)
    left_hand_nnz = 0
    for lhs_row in get_lhs_expressions(wf)
        for expression in lhs_row
            if expression == 0
                continue
            end

            left_hand_nnz += Forms.get_estimated_nnz_per_elem(expression)
        end
    end

    right_hand_nnz = 0
    for rhs_row in get_rhs_expressions(wf)
        for expression in rhs_row
            if expression == 0
                continue
            end

            right_hand_nnz += Forms.get_estimated_nnz_per_elem(expression)
        end
    end

    return (left_hand_nnz, right_hand_nnz)
end

"""
    get_num_elements(wf::WeakForm)

Returns the number of elements over which the discrete weak-formulation is defined.

# Arguments
- `wf::WeakForm`: The weak-formulation for which the number of elements is to be determined.

# Returns
- `::Int`: The number of elements.
"""
function get_num_elements(wf::WeakForm)
    for lhs_row in get_lhs_expressions(wf)
        for expression in lhs_row
            if expression == 0
                continue
            end

            return Forms.get_num_elements(expression)
        end
    end

    throw(ArgumentError("No elements found in the left-hand side of the weak-formulation."))
end

"""
    get_num_evaluation_elements(wf::WeakForm)

Returns the maximum number of elements over which the weak form blocks are evaluated. This
is the max over all lhs and rhs expression blocks.

# Arguments
- `wf::WeakForm`: The weak-formulation for which the number of quadrature elements is to be
    determined.

# Returns
- `::Int`: The number of quadrature elements.
"""
function get_num_evaluation_elements(wf::WeakForm)
    num_eval_elements = 0
    for lhs_row in get_lhs_expressions(wf)
        for expression in lhs_row
            if expression == 0
                continue
            end

            num_eval_elements = max(
                num_eval_elements, Forms.get_num_evaluation_elements(expression)
            )
        end
    end

    for rhs_row in get_rhs_expressions(wf)
        for expression in rhs_row
            if expression == 0
                continue
            end

            num_eval_elements = max(
                num_eval_elements, Forms.get_num_evaluation_elements(expression)
            )
        end
    end

    return num_eval_elements
end
