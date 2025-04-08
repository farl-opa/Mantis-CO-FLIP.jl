function assemble(
    quad_rule::Q,
    weak_form::WeakForm,
    dirichlet_bcs::Dict{Int, Float64};
    sparse_lhs::Bool=true,
    sparse_rhs::Bool=false,
) where {Q <: Quadrature.AbstractQuadratureRule}
    num_elements = get_num_elements(weak_form)
    n_dofs_trial, n_dofs_test = get_problem_size(weak_form)
    lhs_row_ids, lhs_col_ids, lhs_vals = get_pre_allocation(weak_form, "lhs")
    rhs = zeros(Float64, n_dofs_test)
    lhs_counts = 0 # Number of non-zero entries in A (Can contain duplicates)
    for elem_id in 1:num_elements
        lhs_contrib, rhs_contrib = evaluate(weak_form, elem_id)
        for idx in eachindex(lhs_contrib...)
            lhs_counts += 1
            lhs_row_ids[lhs_counts] = lhs_contrib[1][idx]
            lhs_col_ids[lhs_counts] = lhs_contrib[2][idx]
            lhs_vals[lhs_counts] = lhs_contrib[3][idx]
        end

        for idx in eachindex(rhs_contrib...)
            rhs[rhs_contrib[1][idx]] += rhs_contrib[2][idx]
        end
    end

    # WARNING: This may not work as intended if the boundary indices are 
    # duplicated in the given row and column vectors!
    if ~isempty(dirichlet_bcs)
        # Set the bc value for the rhs.
        for (bc_idx, bc_value) in pairs(dirichlet_bcs)
            rhs[bc_idx] = bc_value
        end

        # Update the A matrix to be an identity row for the given 
        # indices. As the default sparse matrices are column-based, we 
        # cannot easily set all values in a row to zero, so we have to 
        # loop over all indices and check if they are boundary indices.
        for idx in eachindex(lhs_row_ids, lhs_col_ids, lhs_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, lhs_row_ids[idx])
                if lhs_col_ids[idx] == lhs_row_ids[idx]
                    # Diagonal term, set to 1.0.
                    lhs_vals[idx] = 1.0
                else
                    # Non-diagoal term, set to 0.0.
                    lhs_vals[idx] = 0.0
                end
            end
        end
    end

    lhs = spa.sparse(
        lhs_row_ids[1:lhs_counts],
        lhs_col_ids[1:lhs_counts],
        lhs_vals[1:lhs_counts],
        n_dofs_trial,
        n_dofs_test,
    )

    return lhs, rhs
end

function get_pre_allocation(weak_form::WeakForm, side::String)
    nnz_elem = get_estimated_nnz_per_elem(weak_form)
    if side == "lhs"
        nvals = 9 * nnz_elem[1] * get_num_elements(weak_form)
    elseif side == "rhs"
        nvals = 9 * nnz_elem[2] * get_num_elements(weak_form)
    else
        throw(ArgumentError("Invalid side: $(side). Must be 'lhs' or 'rhs'."))
    end

    row_ids = Vector{Int}(undef, nvals)
    col_ids = Vector{Int}(undef, nvals)
    vals = Vector{Float64}(undef, nvals)

    return row_ids, col_ids, vals
end
