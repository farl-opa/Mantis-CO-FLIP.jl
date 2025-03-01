
@doc raw"""
    assemble(weak_form::F, weak_form_inputs::W, dirichlet_bcs::Dict{Int, Float64}) where {F <: Function, W <: AbstractInputs}

Assemble a continuous Galerkin problem for the given weak form and 
Dirichlet boundary conditions.

# Arguments
- `WeakForm <: Function`: The weak form to assemble.
- `WeakFormInputs <: AbstractInputs`: The inputs for the weak form.
- `dirichlet_bcs::Dict{Int, Float64}`: A dictionary containing the 
  Dirichlet boundary conditions, where the key is the index of the 
  boundary condition and the value is the boundary condition value.

# Returns
- `A::SparseMatrixCSC{Float64, Int}`: The assembled system matrix.
- `b::Vector{Float64}`: The assembled right-hand side vector.
"""
function assemble(weak_form::F, weak_form_inputs::W, dirichlet_bcs::Dict{Int, Float64}) where {F <: Function, W <: AbstractInputs}
    # Get the actual size of the problem.
    n_dofs_trial, n_dofs_test = get_problem_size(weak_form_inputs)
    
    # Pre-allocate
    nnz_elem = get_estimated_nnz_per_elem(weak_form_inputs)

    nvals_A = 9 * nnz_elem[1] * get_num_elements(weak_form_inputs)  # 4 times due to the forms being per component. This needs to be updated.
    A_column_idxs = Vector{Int}(undef, nvals_A)
    A_row_idxs = Vector{Int}(undef, nvals_A)
    A_vals = Vector{Float64}(undef, nvals_A)

    # The rhs-vector will have to be a dense vector to be able to solve, 
    # and this is also easier and more efficient when specifying 
    # boundary conditions, so it is already dense.
    b = zeros(Float64, n_dofs_test)
    
    # Keep track of the number of elements added to matrix A. Note that 
    # these can contain duplicates, so these will not indicate the final sizes.
    counts_A = 0

    # Loop over all active elements
    for elem_id in 1:get_num_elements(weak_form_inputs)
        # Volume/element contributions
        contrib_A, contrib_b = weak_form(weak_form_inputs, elem_id)

        for idx in eachindex(contrib_A...)
            counts_A += 1
            A_row_idxs[counts_A] = contrib_A[1][idx]
            A_column_idxs[counts_A] = contrib_A[2][idx]
            A_vals[counts_A] = contrib_A[3][idx]
        end

        for idx in eachindex(contrib_b...)
            b[contrib_b[1][idx]] += contrib_b[2][idx]
        end

        # Interface contributions (Internal boundaries)

        # Boundary contributions (External boundaries)

    end


    # Set Dirichlet conditions if needed.
    # WARNING: This may not work as intended if the boundary indices are 
    # duplicated in the given row and column vectors!
    if ~isempty(dirichlet_bcs)
        # Set the bc value for the rhs.
        for (bc_idx, bc_value) in pairs(dirichlet_bcs)
            b[bc_idx] = bc_value
        end
        
        # Update the A matrix to be an identity row for the given 
        # indices. As the default sparse matrices are column-based, we 
        # cannot easily set all values in a row to zero, so we have to 
        # loop over all indices and check if they are boundary indices.
        for idx in eachindex(A_row_idxs, A_column_idxs, A_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, A_row_idxs[idx])
                if A_column_idxs[idx] == A_row_idxs[idx]
                    # Diagonal term, set to 1.0.
                    A_vals[idx] = 1.0
                else
                    # Non-diagoal term, set to 0.0.
                    A_vals[idx] = 0.0
                end
            end
        end
    end

    A = spa.sparse(A_row_idxs[1:counts_A], A_column_idxs[1:counts_A], 
                    A_vals[1:counts_A], n_dofs_trial, n_dofs_test) 
    
    # We can call this function to remove the added zeros (especially 
    # from the boundary conditions), though I am not sure if this is 
    # worth it or needed. Note that this would also remove zero values 
    # that happen to be zero by computation. 
    #spa.dropzeros!(A)

    return A, b

end

############################################################################################
#                                    Maxwell Eigenvalue                                    #
############################################################################################

function assemble_eigenvalue(
    weak_form::F, weak_form_inputs::W, bcs::Dict{Int, Float64}
) where {F <: Function, W <: AbstractInputs}
    # Get the actual size of the problem.
    n_dofs_trial, n_dofs_test = get_problem_size(weak_form_inputs)

    # Pre-allocate
    nnz_elem = get_estimated_nnz_per_elem(weak_form_inputs)

    nvals = 9 * nnz_elem[1] * get_num_elements(weak_form_inputs)  # 4 times due to the forms being per component. This needs to be updated.
    A_column_idxs = Vector{Int}(undef, nvals)
    A_row_idxs = Vector{Int}(undef, nvals)
    A_vals = Vector{Float64}(undef, nvals)
    B_column_idxs = Vector{Int}(undef, nvals)
    B_row_idxs = Vector{Int}(undef, nvals)
    B_vals = Vector{Float64}(undef, nvals)

    # Keep track of the number of elements added to matrix A. Note that 
    # these can contain duplicates, so these will not indicate the final sizes.
    counts_A = 0
    counts_B = 0

    # Loop over all active elements
    for elem_id in 1:get_num_elements(weak_form_inputs)
        # Volume/element contributions
        contrib_A, contrib_B = weak_form(weak_form_inputs, elem_id)

        for idx in eachindex(contrib_A...)
            counts_A += 1
            A_row_idxs[counts_A] = contrib_A[1][idx]
            A_column_idxs[counts_A] = contrib_A[2][idx]
            A_vals[counts_A] = contrib_A[3][idx]
        end

        for idx in eachindex(contrib_B...)
            counts_B += 1
            B_row_idxs[counts_B] = contrib_B[1][idx]
            B_column_idxs[counts_B] = contrib_B[2][idx]
            B_vals[counts_B] = contrib_B[3][idx]
        end

        # Interface contributions (Internal boundaries)

        # Boundary contributions (External boundaries)

    end

    A = spa.sparse(
        A_row_idxs[1:counts_A],
        A_column_idxs[1:counts_A],
        A_vals[1:counts_A],
        n_dofs_trial,
        n_dofs_test,
    )
    B = spa.sparse(
        B_row_idxs[1:counts_B],
        B_column_idxs[1:counts_B],
        B_vals[1:counts_B],
        n_dofs_trial,
        n_dofs_test,
    )

    # Convert to Matrix type because eigen does not work with SparseMatrixCSC.
    A = Matrix(A)
    B = Matrix(B)

    non_boundary_rows_cols = setdiff(1:n_dofs_trial, keys(bcs))
    A = A[non_boundary_rows_cols, non_boundary_rows_cols]
    B = B[non_boundary_rows_cols, non_boundary_rows_cols]

    # We can call this function to remove the added zeros (especially 
    # from the boundary conditions), though I am not sure if this is 
    # worth it or needed. Note that this would also remove zero values 
    # that happen to be zero by computation. 
    #spa.dropzeros!(A)

    return A, B
end
