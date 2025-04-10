"""
    assemble(
        weak_form::WeakForm,
        quad_rule::Q,
        dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
        sparse_lhs::Bool=true,
        sparse_rhs::Bool=false,
    ) where {Q <: Quadrature.AbstractQuadratureRule}

Assemble the left- and right-hand sides of a discrete Petrov-Galerkin problem for the given
weak-formulation and Dirichlet boundary conditions.

# Arguments
- `weak_form::WeakForm`: The weak form to assemble.
- `quad_rule::Q`: The quadrature rule to use for the assembly.
- `dirichlet_bcs::Dict{Int, Float64}`: A dictionary containing the Dirichlet boundary
    conditions, where the key is the index of the boundary condition and the value is the
    boundary condition value.
- `sparse_lhs::Bool`: Whether to use a sparse matrix for the left-hand side.
- `sparse_rhs::Bool`: Whether to use a sparse matrix for the right-hand side.

# Returns
- `lhs::SparseMatrixCSC{Float64, Int}`: The assembled left-hand side matrix.
- `rhs::Matrix{Float64}`: The assembled right-hand side vector.
"""
function assemble(
    weak_form::WeakForm,
    quad_rule::Q,
    dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
    sparse_lhs::Bool=true,
    sparse_rhs::Bool=false,
) where {Q <: Quadrature.AbstractQuadratureRule}
    num_elements = get_num_elements(weak_form)
    n_dofs_test, n_dofs_trial = get_problem_size(weak_form)
    test_offsets = get_test_offsets(weak_form)
    trial_offsets = get_trial_offsets(weak_form)
    num_lhs_blocks = get_num_lhs_blocks(weak_form)
    num_rhs_blocks = get_num_rhs_blocks(weak_form)
    lhs_expressions = get_lhs_expressions(weak_form)
    rhs_expressions = get_rhs_expressions(weak_form)
    lhs_row_ids, lhs_col_ids, lhs_vals = get_pre_allocation(weak_form, "lhs")
    if isnothing(get_forcing(weak_form))
        rhs = zeros(n_dofs_test, 1)
    else
        rhs = zeros(n_dofs_test, n_dofs_trial)
    end

    lhs_counts = 0 # Number of non-zero entries in A (Can contain duplicates)
    for elem_id in 1:num_elements
        for block_id in CartesianIndices(num_lhs_blocks)
            if lhs_expressions[block_id[1]][block_id[2]] != 0
                block_eval, block_indices = Forms.evaluate(
                    lhs_expressions[block_id[1]][block_id[2]], elem_id, quad_rule
                )
                for idx in eachindex(block_eval)
                    lhs_counts += 1
                    lhs_row_ids[lhs_counts] =
                        block_indices[1][idx] .+ test_offsets[block_id[1]]
                    lhs_col_ids[lhs_counts] =
                        block_indices[2][idx] .+ trial_offsets[block_id[2]]
                    lhs_vals[lhs_counts] = block_eval[idx]
                end
            end
        end

        for block_id in CartesianIndices(num_rhs_blocks)
            if rhs_expressions[block_id[1]][block_id[2]] != 0
                block_eval, block_indices = Forms.evaluate(
                    rhs_expressions[block_id[1]][block_id[2]], elem_id, quad_rule
                )
                for idx in eachindex(block_eval)
                    rhs[block_indices[1][idx] .+ test_offsets[block_id[1]]] += block_eval[idx]
                end
            end
        end
    end

    if ~isempty(dirichlet_bcs)
        # Set the bc value for the rhs.
        for (bc_idx, bc_value) in pairs(dirichlet_bcs)
            rhs[bc_idx] = bc_value
        end

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

"""
    get_pre_allocation(weak_form::WeakForm, side::String)

Returns pre-allocated row, column, and value vectors for the left-hand side (lhs) or
right-hand side (rhs) matrix.

# Arguments
- `weak_form::WeakForm`: The weak form to use for the pre-allocation.
- `side::String`: The side of the matrix to pre-allocate. Must be either "lhs" or "rhs".

# Returns
- `row_ids::Vector{Int}`: The pre-allocated row indices.
- `col_ids::Vector{Int}`: The pre-allocated column indices.
- `vals::Vector{Float64}`: The pre-allocated values.
"""
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
