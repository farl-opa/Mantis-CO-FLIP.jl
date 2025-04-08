############################################################################################
#                                        Structure                                         #
############################################################################################

struct WeakForm{manifold_dim, LHS_E, RHS_E, LHS_O, RHS_O}
    lhs_expressions::LHS_E
    rhs_expressions::RHS_E
    lhs_offsets::LHS_O
    rhs_offsets::RHS_O
    inputs::WeakFormInputs{manifold_dim}
    sparse_lhs::Bool
    sparse_rhs::Bool
    function WeakForm(
        inputs::I, expression::E; sparse_lhs::Bool=true, sparse_rhs::Bool=false
    ) where {manifold_dim, I <: WeakFormInputs{manifold_dim}, E <: Function}
        lhs_expressions, rhs_expressions, lhs_offsets, rhs_offsets = expression(inputs)
        num_lhs = length(lhs_expressions)
        num_rhs = length(rhs_expressions)
        LHS_E = typeof(lhs_expressions)
        RHS_E = typeof(rhs_expressions)
        LHS_O = typeof(lhs_offsets)
        RHS_O = typeof(rhs_offsets)
        if num_lhs != length(lhs_offsets[1]) || num_lhs != length(lhs_offsets[2])
            throw(
                ArgumentError(
                    "Mismatch between the given left-hand side expressions and offsets. " *
                    "The given numbers were $(num_lhs), $(length(lhs_offsets[1])) and " *
                    "$(length(lhs_offsets[2])), respectively.",
                ),
            )
        end

        if num_rhs != length(rhs_offsets[1]) || num_rhs != length(rhs_offsets[2])
            throw(
                ArgumentError(
                    "Mismatch between the given right-hand side expressions and offsets. " *
                    "The given numbers were $(num_rhs), $(length(rhs_offsets[1])) and " *
                    "$(length(rhs_offsets[2])), respectively.",
                ),
            )
        end

        if ~(LHS_E <: NTuple{num_lhs, Forms.AbstractFormExpression})
            throw(
                ArgumentError(
                    "The left-hand side expressions must be a tuple of elements of the " *
                    "type Forms.AbstractFormExpression. The given type was $(LHS_E).",
                ),
            )
        end

        if ~(RHS_E <: NTuple{num_rhs, Forms.AbstractFormExpression})
            throw(
                ArgumentError(
                    "The right-hand side expressions must be a tuple of elements of the " *
                    "type Forms.AbstractFormExpression. The given type was $(RHS_E).",
                ),
            )
        end

        if ~(LHS_O <: Tuple{NTuple{num_lhs, Int}, NTuple{num_lhs, Int}})
            throw(
                ArgumentError(
                    "The left-hand side offsets must be a tuple row and column offsets " *
                    "for each left-hand side expression. The given type was $(LHS_O) and " *
                    "the number of left-hand side expressions was $(num_lhs).",
                ),
            )
        end

        if ~(RHS_O <: Tuple{NTuple{num_rhs, Int}, NTuple{num_rhs, Int}})
            throw(
                ArgumentError(
                    "The right-hand side offsets must be a tuple row and column offsets " *
                    "for each right-hand side expression. The given type was $(RHS_O) " *
                    "and the number of right-hand side expressions was $(num_rhs).",
                ),
            )
        end

        return new{manifold_dim, LHS_E, RHS_E, LHS_O, RHS_O}(
            lhs_expressions,
            rhs_expressions,
            lhs_offsets,
            rhs_offsets,
            inputs,
            sparse_lhs,
            sparse_rhs,
        )
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_lhs_expressions(wf::WeakForm) = wf.lhs_expressions
get_rhs_expressions(wf::WeakForm) = wf.rhs_expressions
get_lhs_offsets(wf::WeakForm) = wf.lhs_offsets
get_rhs_offsets(wf::WeakForm) = wf.rhs_offsets
get_inputs(wf::WeakForm) = wf.inputs

function get_trial_forms(wf::WeakForm)
    inputs = get_inputs(wf)

    return get_trial_forms(inputs)
end

function get_test_forms(wf::WeakForm)
    inputs = get_inputs(wf)

    return get_test_forms(inputs)
end

function get_forcing(wf::WeakForm)
    inputs = get_inputs(wf)

    return get_forcing(inputs)
end

function get_trial_size(wf::WeakForm)
    inputs = get_inputs(wf)
    trial_size = 0
    for form in get_trial_forms(inputs)
        trial_size += Forms.get_num_basis(form)
    end

    return trial_size
end

function get_test_size(wf::WeakForm)
    inputs = get_inputs(wf)
    test_size = 0
    for form in get_test_forms(inputs)
        test_size += Forms.get_num_basis(form)
    end

    return test_size
end

function get_problem_size(wf::WeakForm)
    return get_trial_size(wf), get_test_size(wf)
end

function get_trial_max_local_dim(wf::WeakForm)
    trial_max_local_dim = 0
    for form in get_trial_forms(wf)
        trial_max_local_dim += Forms.get_max_local_dim(form)
    end

    return trial_max_local_dim
end

function get_test_max_local_dim(wf::WeakForm)
    test_max_local_dim = 0
    for form in get_test_forms(wf)
        test_max_local_dim += Forms.get_max_local_dim(form)
    end

    return test_max_local_dim
end

function get_estimated_nnz_per_elem(wf::WeakForm)
    trial_max_local_dim = get_trial_max_local_dim(wf)
    test_max_local_dim = get_test_max_local_dim(wf)
    left_hand_nnz = trial_max_local_dim * test_max_local_dim
    right_hand_nnz = test_max_local_dim
    if typeof(get_forcing(wf)) == Nothing
        right_hand_nnz *= trial_max_local_dim
    end

    return left_hand_nnz, right_hand_nnz
end

function get_num_elements(wf::WeakForm)
    return Geometry.get_num_elements(Forms.get_geometry(get_trial_forms(wf)))
end

############################################################################################
#                                          Checks                                          #
############################################################################################

lhs_is_sparse(wf::WeakForm) = wf.sparse_lhs
rhs_is_sparse(wf::WeakForm) = wf.sparse_rhs
