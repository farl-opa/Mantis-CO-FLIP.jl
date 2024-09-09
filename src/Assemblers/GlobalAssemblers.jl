
struct Assembler <: AbstractAssemblers
    dirichlet_bcs::Dict{Int, Float64}
end



@doc raw"""
    (self::Assembler)(bilinear_form::AbstractBilinearForms)

Assemble a continuous Galerkin problem with given bilinear form.

# Arguments
- `bilinear_form::AbstractBilinearForms`: bilinear form.
"""
function (self::Assembler)(weak_form::F, weak_form_inputs::W) where {F <: Function, W <: AbstractInputs}
    # Get the actual size of the problem.
    n_dofs_trial, n_dofs_test = get_problem_size(weak_form_inputs)
    
    # Pre-allocate
    nnz_elem = get_estimated_nnz_per_elem(weak_form_inputs)

    nvals_A = 4 * nnz_elem[1] * get_num_elements(weak_form_inputs)  # 4 times due to the forms being per component. This needs to be updated.
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
    if ~isempty(self.dirichlet_bcs)
        # Set the bc value for the rhs.
        for (bc_idx, bc_value) in pairs(self.dirichlet_bcs)
            b[bc_idx] = bc_value
        end
        
        # Update the A matrix to be an identity row for the given 
        # indices. As the default sparse matrices are column-based, we 
        # cannot easily set all values in a row to zero, so we have to 
        # loop over all indices and check if they are boundary indices.
        for idx in eachindex(A_row_idxs, A_column_idxs, A_vals)
            # Check if the row index is also a boundary index.
            if haskey(self.dirichlet_bcs, A_row_idxs[idx])
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




function _compute_square_error_per_element(computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2") where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, TF1 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, TF2 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, Q <: Quadrature.QuadratureRule{manifold_dim}}
    num_elements = Geometry.get_num_elements(Forms.get_geometry(computed_sol))
    result = Vector{Float64}(undef, num_elements)

    for elem_id in 1:1:num_elements
        difference = computed_sol - exact_sol
        if norm == "L2"
            result[elem_id] = sum(Forms.evaluate_inner_product(difference, difference, elem_id, quad_rule)[3])
        elseif norm == "Linf"
            println("WARNING: The Linf evaluation only uses the quadrature nodes as evaluation points!")
            result[elem_id] = maximum(Forms.evaluate(difference, elem_id, Quadrature.get_quadrature_nodes(quad_rule))[1][1])
        elseif norm == "H1"
            Error("Computing the H1 norm still needs to be updated.")
            d_difference = Forms.exterior_derivative(difference)
            result[elem_id] = sum(Forms.evaluate_inner_product(d_difference, d_difference, elem_id, quad_rule)[3])
        end
    end

    return result
end


function compute_error_per_element(computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2") where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, TF1 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, TF2 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, Q <: Quadrature.QuadratureRule{manifold_dim}}
    partial_result = _compute_square_error_per_element(computed_sol, exact_sol, quad_rule, norm)
    if norm == "Linf"
        return partial_result
    elseif norm == "L2" || norm == "H1"
        return sqrt.(partial_result)
    else
        throw(ArgumentError("Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."))
    end
end

function compute_error_total(computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2") where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, TF1 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, TF2 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, Q <: Quadrature.QuadratureRule{manifold_dim}}
    partial_result = _compute_square_error_per_element(computed_sol, exact_sol, quad_rule, norm)
    if norm == "Linf"
        return maximum(partial_result)
    elseif norm == "L2" || norm == "H1"
        return sqrt(sum(partial_result))
    else
        throw(ArgumentError("Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."))
    end
end
