
import .. Fields
import .. Geometry

struct Assembler <: AbstractAssemblers
    bc_values::Vector{Float64}
    set_bcs::Bool
end

# Construction if no boundary conditions are needed. The b.c.s will be 
# set to zero, but won't be used.
function Assembler()
    return Assembler([0.0, 0.0], false)
end
# Construction if both boundary conditions are needed.
function Assembler(bc_left::Float64, bc_right::Float64)
    return Assembler([bc_left, bc_right], true)
end
# Construction for more bcs.
function Assembler(bc_vals::Vector{Float64})
    return Assembler(bc_vals, true)
end

@doc raw"""
    (self::Assembler)(bilinear_form::AbstractBilinearForms)

Assemble a continuous Galerkin problem with given bilinear form.

# Arguments
- `bilinear_form::AbstractBilinearForms`: bilinear form.
"""
function (self::Assembler)(bilinear_form::AbstractBilinearForms)
    # Get the actual size of the problem.
    n_dofs_trial, n_dofs_test = get_problem_size(bilinear_form)
    
    # Pre-allocate
    nnz_elem = get_estimated_nnz_per_elem(bilinear_form)

    nvals_A = nnz_elem[1] * get_num_elements(bilinear_form)
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
            b[contrib_b[1][idx]] += contrib_b[2][idx]
        end

        # Interface contributions (Internal boundaries)

        # Boundary contributions (External boundaries)

    end

    

    if self.set_bcs
        # Set the bc value for the rhs.
        bc_idxs = get_boundary_dof_indices(bilinear_form)
        for idx in eachindex(bc_idxs)
            b[bc_idxs[idx]] = self.bc_values[idx]
        end
        
        # Update the A matrix to be an identity row for the given indices.
        for idx in eachindex(A_row_idxs, A_column_idxs, A_vals)
            if A_row_idxs[idx] in bc_idxs
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


