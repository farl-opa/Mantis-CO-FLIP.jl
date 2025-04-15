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
    lhs_type::Type=spa.SparseMatrixCSC{Float64, Int},
    rhs_type::Type=Matrix{Float64},
) where {Q <: Quadrature.AbstractQuadratureRule}
    num_lhs_blocks = get_num_lhs_blocks(weak_form)
    num_rhs_blocks = get_num_rhs_blocks(weak_form)
    lhs_expressions = get_lhs_expressions(weak_form)
    rhs_expressions = get_rhs_expressions(weak_form)
    test_offsets = get_test_offsets(weak_form)
    trial_offsets = get_trial_offsets(weak_form)
    lhs_row_ids, lhs_col_ids, lhs_vals = get_pre_allocation(weak_form, "lhs")
    rhs_row_ids, rhs_col_ids, rhs_vals = get_pre_allocation(weak_form, "rhs")
    lhs_counts, rhs_counts = 0, 0
    for elem_id in 1:Quadrature.get_num_elements(quad_rule)
        for block_id in CartesianIndices(num_lhs_blocks)
            lhs_row_ids, lhs_col_ids, lhs_vals, lhs_counts = add_element_block_contributions!(
                lhs_row_ids, lhs_col_ids, lhs_vals, lhs_counts, elem_id, block_id,
                lhs_expressions, quad_rule, test_offsets, trial_offsets
            )
        end

        for block_id in CartesianIndices(num_rhs_blocks)
            rhs_row_ids, rhs_col_ids, rhs_vals, rhs_counts = add_element_block_contributions!(
                rhs_row_ids, rhs_col_ids, rhs_vals, rhs_counts, elem_id, block_id,
                rhs_expressions, quad_rule, test_offsets, trial_offsets
            )
        end
    end

    lhs_row_ids, lhs_col_ids, lhs_vals, rhs_row_ids, rhs_col_ids, rhs_vals = add_boundary_conditions!(
        lhs_row_ids, lhs_col_ids, lhs_vals, rhs_row_ids, rhs_col_ids, rhs_vals,
        dirichlet_bcs
    )
    problem_size = get_problem_size(weak_form)
    lhs = build_matrix(lhs_type, lhs_row_ids[1:lhs_counts], lhs_col_ids[1:lhs_counts], lhs_vals[1:lhs_counts], problem_size)
    matrix_rhs = isnothing(get_forcing(weak_form))
    if matrix_rhs
        rhs = build_matrix(rhs_type, rhs_row_ids[1:rhs_counts], rhs_col_ids[1:rhs_counts], rhs_vals[1:rhs_counts], problem_size)
    else
        rhs = build_matrix(rhs_type, rhs_row_ids[1:rhs_counts], ones(Int, rhs_counts), rhs_vals[1:rhs_counts], (problem_size[1], 1))
    end

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

function add_element_block_contributions!(
    row_ids::Vector{Int}, col_ids::Vector{Int}, vals::Vector{Float64}, counts::Int, element_id::Int, block_id::CartesianIndex{2}, expressions, quad_rule, test_offsets, trial_offsets
)
    if expressions[block_id[1]][block_id[2]] != 0
        block_eval, block_indices = Forms.evaluate(
            expressions[block_id[1]][block_id[2]], element_id, quad_rule
        )
        for eval_id in eachindex(block_eval)
            counts += 1
            for (id, indices) in enumerate(block_indices)
                if id == 1
                    row_ids[counts] = indices[eval_id] + test_offsets[block_id[1]]
                elseif id == 2
                    col_ids[counts] = indices[eval_id] + trial_offsets[block_id[2]]
                end
            end

            vals[counts] = block_eval[eval_id]
        end
    end

    return row_ids, col_ids, vals, counts
end

function add_boundary_conditions!(lhs_row_ids::Vector{Int}, lhs_col_ids::Vector{Int}, lhs_vals::Vector{Float64}, rhs_row_ids::Vector{Int}, rhs_col_ids::Vector{Int}, rhs_vals::Vector{Float64}, dirichlet_bcs::Dict{Int, Float64})
    if ~isempty(dirichlet_bcs)
        for id in eachindex(lhs_row_ids, lhs_col_ids, lhs_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, lhs_row_ids[id])
                if lhs_col_ids[id] == lhs_row_ids[id]
                    # Diagonal term, set to 1.0.
                    lhs_vals[id] = 1.0
                else
                    # Non-diagoal term, set to 0.0.
                    lhs_vals[id] = 0.0
                end
            end
        end
        

        for id in eachindex(rhs_row_ids, rhs_col_ids, rhs_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, rhs_row_ids[id])
            # Set the value to the Dirichlet boundary condition value.
                rhs_vals[id] = dirichlet_bcs[rhs_row_ids[id]]
            end
        end

    end

    return lhs_row_ids, lhs_col_ids, lhs_vals, rhs_row_ids, rhs_col_ids, rhs_vals
end

function build_matrix(matrix_type::Type, row_ids::Vector{Int}, col_ids::Vector{Int}, vals::Vector{Float64}, size::Tuple{Int, Int})
    throw(ArgumentError("Assembly of matrix type `$(matrix_type)` not currently implemented."))
end

function build_matrix(matrix_type::Type{SM}, row_ids::Vector{Int}, col_ids::Vector{Int}, vals::Vector{Float64}, size::Tuple{Int, Int}) where {SM<: spa.AbstractSparseMatrix}
    return spa.sparse(row_ids, col_ids, vals, size...)
end

function build_matrix(matrix_type::Type{Matrix{Float64}}, row_ids::Vector{Int}, col_ids::Vector{Int}, vals::Vector{Float64}, size::Tuple{Int, Int})
    matrix = zeros(Float64, size)
    for (row, col, val) in zip(row_ids, col_ids, vals)
        matrix[row, col] += val
    end

    return matrix
end

