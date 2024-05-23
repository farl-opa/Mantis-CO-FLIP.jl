
struct Assembler <: AbstractAssemblers
    bc_left::Float64
    bc_right::Float64
    set_bcs::Bool
end

# Construction if no boundary conditions are needed. The b.c.s will be 
# set to zero, but won't be used.
function Assembler()
    return Assembler(0.0, 0.0, false)
end
# Construction if both boundary conditions are needed.
function Assembler(bc_left::Float64, bc_right::Float64)
    return Assembler(bc_left, bc_right, true)
end

@doc raw"""
    (assembler::Assembler)(bilinear_form::AbstractBilinearForms)

Assemble a continuous Galerkin problem with given bilinear form.

# Arguments
- `bilinear_form::AbstractBilinearForms`: bilinear form.
"""
function (assembler::Assembler)(bilinear_form::AbstractBilinearForms)
    # Pre-allocate
    nnz_elem = get_estimated_nnz_per_elem(bilinear_form)
    
    nvals_A = nnz_elem[1] * get_num_elements(bilinear_form)
    A_column_idxs = Vector{Int}(undef, nvals_A)
    A_row_idxs = Vector{Int}(undef, nvals_A)
    A_vals = Vector{Float64}(undef, nvals_A)

    nvals_b = nnz_elem[2] * get_num_elements(bilinear_form)
    b_idxs = Vector{Int}(undef, nvals_b)
    b_vals = Vector{Float64}(undef, nvals_b)

    # Keep track of the number of elements added to matrix A and vector 
    # b. Note that these can contain duplicates, so these will not 
    # indicate the final sizes.
    counts_A = 0
    counts_b = 0

    # Loop over all active elements
    for elem_id in 1:get_num_elements(bilinear_form)
        # Volume/element contributions
        contrib_A, contrib_b = bilinear_form(elem_id)

        for idx in eachindex(contrib_A...)
            counts_A += 1
            A_row_idxs[counts_A] = contrib_A[1][idx]
            A_column_idxs[counts_A] = contrib_A[2][idx]
            A_vals[counts_A] = contrib_A[3][idx]
        end

        for idx in eachindex(contrib_b...)
            counts_b += 1
            b_idxs[counts_b] = contrib_b[1][idx]
            b_vals[counts_b] = contrib_b[2][idx]
        end


        # Interface contributions (Internal boundaries)

        # Boundary contributions (External boundaries)

    end

    # Get the actual size of the problem.
    n_dofs_trial, n_dofs_test = get_problem_size(bilinear_form)

    # return (spa.sparse(A_row_idxs[1:counts_A], 
    #                    A_column_idxs[1:counts_A], 
    #                    A_vals[1:counts_A], 
    #                    n_dofs_trial, n_dofs_test), 
    #         spa.sparsevec(b_idxs[1:counts_b], b_vals[1:counts_b], n_dofs_test))

    if assembler.set_bcs
        Afull = spa.sparse(A_row_idxs[1:counts_A], 
                        A_column_idxs[1:counts_A], 
                        A_vals[1:counts_A], 
                        n_dofs_trial, n_dofs_test) 
        bfull = spa.sparsevec(b_idxs[1:counts_b], b_vals[1:counts_b], n_dofs_test)

        A = Afull[2:end-1,2:end-1]
        b = bfull[2:end-1] - Afull[2:end-1,1]*assembler.bc_left - Afull[2:end-1,end]*assembler.bc_right
    else
        # println(A_row_idxs[1:counts_A])
        # println(A_column_idxs[1:counts_A])
        # println(A_vals[1:counts_A])
        # println(n_dofs_trial, " ", n_dofs_test) 
        A = spa.sparse(A_row_idxs[1:counts_A], 
                       A_column_idxs[1:counts_A], 
                       A_vals[1:counts_A], 
                       n_dofs_trial, n_dofs_test) 
        b = spa.sparsevec(b_idxs[1:counts_b], b_vals[1:counts_b], n_dofs_test)
    end

    return A, b

end
