
import .. Fields
import .. Geometry

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

    nvals_A = nnz_elem[1] * get_num_elements(weak_form_inputs)
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

    return A, b[1:n_dofs_test]

end



struct AssemblerError{n} <: AbstractAssemblers 
    quad_nodes::NTuple{n, Vector{Float64}}
    quad_weights::Vector{Float64}
end

struct AssemblerErrorPerElement{n} <: AbstractAssemblers 
    quad_nodes::NTuple{n, Vector{Float64}}
    quad_weights::Vector{Float64}
end

@doc raw"""
    (self::AssemblerError)(bilinear_form::AbstractBilinearForms)

Assemble a continuous Galerkin problem with given bilinear form.

# Arguments
- `bilinear_form::AbstractBilinearForms`: bilinear form.
"""
function (self::AssemblerError)(space, dofs, geom, exact_sol, norm="L2")
    result = 0.0

    # Loop over all active elements
    for elem_id in 1:Geometry.get_num_elements(geom)
        field = Fields.FEMField(space, dofs)
        element_sol = dropdims(Fields.evaluate(field, elem_id, self.quad_nodes), dims=2)

        _, jac_det = Geometry.metric(geom, elem_id, self.quad_nodes)

        phys_nodes = Geometry.evaluate(geom, elem_id, self.quad_nodes)
        element_exact = exact_sol.(Tuple(phys_nodes[:,i] for i in 1:1:Geometry.get_domain_dim(geom))...)

        @inbounds for point_idx in eachindex(jac_det, self.quad_weights, element_sol, element_exact)
            result += jac_det[point_idx] * self.quad_weights[point_idx] * (element_sol[point_idx] - element_exact[point_idx])^2
        end

    end

    return sqrt(result)

end

@doc raw"""
    (self::AssemblerErrorPerElement)(bilinear_form::AbstractBilinearForms)

Assemble a continuous Galerkin problem with given bilinear form.

# Arguments
- `bilinear_form::AbstractBilinearForms`: bilinear form.
"""
function (self::AssemblerErrorPerElement)(space, dofs, geom, exact_sol, norm="L2")
    result = 0.0
    num_els = Geometry.get_num_elements(geom)

    errors = Vector{Float64}(undef, num_els)

    # Loop over all active elements
    for elem_id in 1:num_els
        field = Fields.FEMField(space, dofs)
        element_sol = dropdims(Fields.evaluate(field, elem_id, self.quad_nodes), dims=2)

        _, jac_det = Geometry.metric(geom, elem_id, self.quad_nodes)

        phys_nodes = Geometry.evaluate(geom, elem_id, self.quad_nodes)
        element_exact = exact_sol.(Tuple(phys_nodes[:,i] for i in 1:1:Geometry.get_domain_dim(geom))...)

        result = 0.0
        @inbounds for point_idx in eachindex(jac_det, self.quad_weights, element_sol, element_exact)
            result += jac_det[point_idx] * self.quad_weights[point_idx] * (element_sol[point_idx] - element_exact[point_idx])^2
        end

        errors[elem_id] = result
    end

    return errors
end