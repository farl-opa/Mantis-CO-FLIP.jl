"""
    assemble(
        weak_form::WeakForm,
        quad_rule::Q,
        dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
        lhs_type::Type=spa.SparseMatrixCSC{Float64, Int},
        rhs_type::Type=Matrix{Float64},
    ) where {Q <: Quadrature.AbstractQuadratureRule}

Assemble the left- and right-hand sides of a discrete Petrov-Galerkin problem for the given
weak-formulation and Dirichlet boundary conditions.

# Arguments
- `weak_form::WeakForm`: The weak form to assemble.
- `quad_rule::Q`: The quadrature rule to use for the assembly.
- `dirichlet_bcs::Dict{Int, Float64}`: A dictionary containing the Dirichlet boundary
    conditions, where the key is the index of the boundary condition and the value is the
    boundary condition value.
- `lhs_type::Type`: The type of the left-hand side matrix. Default is
    `SparseMatrixCSC{Float64, Int}`.
- `rhs_type::Type`: The type of the right-hand side matrix. Default is `Matrix{Float64}`.

# Returns
- `lhs::lhs_type`: The assembled left-hand side matrix.
- `rhs::rhs_type`: The assembled right-hand side vector.
"""
function assemble(
    weak_form::WeakForm,
    quad_rule::Q,
    dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
    lhs_type::Type=spa.SparseMatrixCSC{Float64, Int},
    rhs_type::Type=Matrix{Float64},
) where {Q <: Quadrature.AbstractGlobalQuadratureRule}
    num_lhs_blocks = get_num_lhs_blocks(weak_form)
    num_rhs_blocks = get_num_rhs_blocks(weak_form)
    lhs_expressions = get_lhs_expressions(weak_form)
    rhs_expressions = get_rhs_expressions(weak_form)
    test_offsets = get_test_offsets(weak_form)
    trial_offsets = get_trial_offsets(weak_form)
    lhs_rows, lhs_cols, lhs_vals = get_pre_allocation(weak_form, "lhs")
    rhs_rows, rhs_cols, rhs_vals = get_pre_allocation(weak_form, "rhs")
    lhs_counts, rhs_counts = 0, 0
    for elem_id in 1:Quadrature.get_num_elements(quad_rule)
        for block_id in CartesianIndices(num_lhs_blocks)
            lhs_rows, lhs_cols, lhs_vals, lhs_counts = add_block_contributions!(
                lhs_rows,
                lhs_cols,
                lhs_vals,
                lhs_counts,
                elem_id,
                block_id,
                lhs_expressions,
                quad_rule,
                test_offsets,
                trial_offsets,
            )
        end

        for block_id in CartesianIndices(num_rhs_blocks)
            rhs_rows, rhs_cols, rhs_vals, rhs_counts = add_block_contributions!(
                rhs_rows,
                rhs_cols,
                rhs_vals,
                rhs_counts,
                elem_id,
                block_id,
                rhs_expressions,
                quad_rule,
                test_offsets,
                trial_offsets,
            )
        end
    end

    lhs_rows, lhs_cols, lhs_vals, rhs_rows, rhs_cols, rhs_vals = add_bc!(
        lhs_rows, lhs_cols, lhs_vals, rhs_rows, rhs_cols, rhs_vals, dirichlet_bcs
    )
    lhs_size = get_lhs_size(weak_form)
    rhs_size = get_rhs_size(weak_form)
    lhs = build_matrix(
        lhs_type,
        lhs_rows[1:lhs_counts],
        lhs_cols[1:lhs_counts],
        lhs_vals[1:lhs_counts],
        lhs_size,
    )
    rhs = build_matrix(
        rhs_type,
        rhs_rows[1:rhs_counts],
        rhs_cols[1:rhs_counts],
        rhs_vals[1:rhs_counts],
        rhs_size,
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
- `rows::Vector{Int}`: The pre-allocated row indices.
- `cols::Vector{Int}`: The pre-allocated column indices.
- `vals::Vector{Float64}`: The pre-allocated values.
"""
function get_pre_allocation(weak_form::WeakForm, side::String)
    nnz_elem = get_estimated_nnz_per_elem(weak_form)
    # TODO: Update `get_num_elements` to `get_num_quadrature_elements`
    if side == "lhs"
        nvals = nnz_elem[1] * get_num_elements(weak_form)
    elseif side == "rhs"
        nvals = nnz_elem[2] * get_num_elements(weak_form)
    else
        throw(ArgumentError("Invalid side: $(side). Must be 'lhs' or 'rhs'."))
    end

    rows = Vector{Int}(undef, nvals)
    if side == "rhs" && ~isnothing(get_forcing(weak_form))
        cols = ones(Int, nvals)
    else
        cols = Vector{Int}(undef, nvals)
    end

    vals = Vector{Float64}(undef, nvals)

    return rows, cols, vals
end

"""
    add_block_contributions!(
        rows::Vector{Int},
        cols::Vector{Int},
        vals::Vector{Float64},
        counts::Int,
        element_id::Int,
        block_id::CartesianIndex{2},
        expressions,
        quad_rule,
        test_offsets,
        trial_offsets,
    )

Updates the row, column, and value vectors with contributions from the specified real-valued
expression block at the element given by `element_id`.

# Arguments
- `rows::Vector{Int}`: The row indices of the matrix.
- `cols::Vector{Int}`: The column indices of the matrix.
- `vals::Vector{Float64}`: The values of the matrix.
- `counts::Int`: The current count of non-zero entries in the matrix.
- `element_id::Int`: The identifier of the element.
- `block_id::CartesianIndex{2}`: The identifier of the block in the expression.
- `expressions`: The expressions to evaluate.
- `quad_rule::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to use for the
    evaluation.
- `test_offsets::Vector{Int}`: The offsets for the test functions.
- `trial_offsets::Vector{Int}`: The offsets for the trial functions.

# Returns
- `rows::Vector{Int}`: The updated row indices of the matrix.
- `cols::Vector{Int}`: The updated column indices of the matrix.
- `vals::Vector{Float64}`: The updated values of the matrix.
- `counts::Int`: The updated count of non-zero entries in the matrix.
"""
function add_block_contributions!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{Float64},
    counts::Int,
    element_id::Int,
    block_id::CartesianIndex{2},
    expressions,
    quad_rule::Quadrature.AbstractGlobalQuadratureRule,
    test_offsets::Vector{Int},
    trial_offsets::Vector{Int},
)
    if expressions[block_id[1]][block_id[2]] != 0
        block_eval, block_indices = Forms.evaluate(
            expressions[block_id[1]][block_id[2]], element_id, quad_rule
        )
        for eval_id in eachindex(block_eval)
            counts += 1
            for (id, indices) in enumerate(block_indices)
                if id == 1
                    rows[counts] = indices[eval_id] + test_offsets[block_id[1]]
                elseif id == 2
                    cols[counts] = indices[eval_id] + trial_offsets[block_id[2]]
                end
            end

            vals[counts] = block_eval[eval_id]
        end
    end

    return rows, cols, vals, counts
end

"""
    add_bc!(
        lhs_rows::Vector{Int},
        lhs_cols::Vector{Int},
        lhs_vals::Vector{Float64},
        rhs_rows::Vector{Int},
        rhs_cols::Vector{Int},
        rhs_vals::Vector{Float64},
        dirichlet_bcs::Dict{Int, Float64},
    )

Adds Dirichlet boundary conditions to the left-hand side and right-hand side matrices.

# Arguments
- `lhs_rows::Vector{Int}`: The row indices of the left-hand side matrix.
- `lhs_cols::Vector{Int}`: The column indices of the left-hand side matrix.
- `lhs_vals::Vector{Float64}`: The values of the left-hand side matrix.
- `rhs_rows::Vector{Int}`: The row indices of the right-hand side matrix.
- `rhs_cols::Vector{Int}`: The column indices of the right-hand side matrix.
- `rhs_vals::Vector{Float64}`: The values of the right-hand side matrix.
- `dirichlet_bcs::Dict{Int, Float64}`: The Dirichlet boundary conditions, where the key is
    the index of the boundary condition and the value is the boundary condition value.

# Returns
- `lhs_rows::Vector{Int}`: The updated row indices of the left-hand side matrix.
- `lhs_cols::Vector{Int}`: The updated column indices of the left-hand side matrix.
- `lhs_vals::Vector{Float64}`: The updated values of the left-hand side matrix.
- `rhs_rows::Vector{Int}`: The updated row indices of the right-hand side matrix.
- `rhs_cols::Vector{Int}`: The updated column indices of the right-hand side matrix.
- `rhs_vals::Vector{Float64}`: The updated values of the right-hand side matrix.
"""
function add_bc!(
    lhs_rows::Vector{Int},
    lhs_cols::Vector{Int},
    lhs_vals::Vector{Float64},
    rhs_rows::Vector{Int},
    rhs_cols::Vector{Int},
    rhs_vals::Vector{Float64},
    dirichlet_bcs::Dict{Int, Float64},
)
    if ~isempty(dirichlet_bcs)
        for id in eachindex(lhs_rows, lhs_cols, lhs_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, lhs_rows[id])
                if lhs_cols[id] == lhs_rows[id]
                    # Diagonal term, set to 1.0.
                    lhs_vals[id] = 1.0
                else
                    # Non-diagoal term, set to 0.0.
                    lhs_vals[id] = 0.0
                end
            end
        end

        for id in eachindex(rhs_rows, rhs_cols, rhs_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, rhs_rows[id])
                # Set the value to the Dirichlet boundary condition value.
                rhs_vals[id] = dirichlet_bcs[rhs_rows[id]]
            end
        end
    end

    return lhs_rows, lhs_cols, lhs_vals, rhs_rows, rhs_cols, rhs_vals
end

"""
    build_matrix(
        matrix_type::Type,
        rows::Vector{Int},
        cols::Vector{Int},
        vals::Vector{Float64},
        size::Tuple{Int, Int},
    )

Returns a matrix of the specified type with the given row and column indices and values.

# Arguments
- `matrix_type::Type`: The type of matrix to build.
- `rows::Vector{Int}`: The row indices of the matrix.
- `cols::Vector{Int}`: The column indices of the matrix.
- `vals::Vector{Float64}`: The values of the matrix.
- `size::Tuple{Int, Int}`: The size of the matrix.

# Returns
- `::matrix_type`: The constructed matrix of the specified type.
"""
function build_matrix(
    matrix_type::Type,
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{Float64},
    size::Tuple{Int, Int},
)
    throw(
        ArgumentError("Assembly of matrix type `$(matrix_type)` not currently implemented.")
    )
end

function build_matrix(
    ::Type{SM},
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{Float64},
    size::Tuple{Int, Int},
) where {SM <: spa.AbstractSparseMatrix}
    return spa.sparse(rows, cols, vals, size...)
end

function build_matrix(
    ::Type{Matrix{Float64}},
    rows::Vector{Int},
    cols::Vector{Int},
    vals::Vector{Float64},
    size::Tuple{Int, Int},
)
    matrix = zeros(Float64, size)
    for (row, col, val) in zip(rows, cols, vals)
        matrix[row, col] += val
    end

    return matrix
end
