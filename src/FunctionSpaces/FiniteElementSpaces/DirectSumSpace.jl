"""
    DirectSumSpace{manifold_dim,num_components,F}

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct DirectSumSpace{manifold_dim, num_components, F} <: AbstractMultiValuedFiniteElementSpace{manifold_dim, num_components}
    component_spaces::F

    function DirectSumSpace(component_spaces::F) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFiniteElementSpace{manifold_dim}}}
        new{manifold_dim, num_components, F}(component_spaces)
    end
end

"""
    evaluate(space::DirectSumSpace{manifold_dim,num_components,F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components}

Evaluate the basis functions of the direct sum space at the points `xi` in the element with index `element_idx`. The function returns a tuple of `num_components` arrays, each containing the evaluations of the basis functions of the corresponding component space.

# Arguments
- `space::DirectSumSpace{manifold_dim,num_components,F}`: Direct sum space
- `element_idx::Int`: Index of the element
- `xi::NTuple{manifold_dim,Vector{Float64}}`: Points in the reference element
- `nderivatives::Int`: Number of derivatives to evaluate

# Returns
- `local_multivalued_basis::Vector{Matrix{Float64}}`: Vecto of matrices containing the evaluations of the basis functions of the direct sum space
- `multivalued_basis_indices::Vector{Int}`: Array containing the global indices of the basis functions
"""
function evaluate(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFiniteElementSpace{manifold_dim}}}
    # number of dofs for each space
    num_dofs_component = FunctionSpaces.get_num_basis.(space.component_spaces)
    dof_offset_component = zeros(Int, num_components)
    dof_offset_component[2:end] .= cumsum(num_dofs_component[1:(num_components-1)])  # we skip the first one because the offset is 0

    # Allocate memory for the evaluation of all basis for all components
    local_multivalued_basis = Vector{Matrix{Float64}}(undef, num_components)
    
    # first loop over the components to get the basis indices
    component_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)
    num_component_basis = length.(component_basis_indices)

    # generate the indices for all the bases
    # since each component of the direct sum has its own index, we need to offset it  
    # the bases that are nonzero for the first component have an offset of 0 
    # the bases that are nonzero for the second component have an offset equal to the
    # number of basis of the first component, etc.
    multivalued_basis_indices_per_component = map(.+, component_basis_indices, dof_offset_component)  # offset the indices
    multivalued_basis_indices = vcat(multivalued_basis_indices_per_component...)  # just place the indices in a single vector
    
    # next, loop over the spaces of each component and evaluate them
    count = 0
    for component_idx in 1:num_components
        # Evaluate the basis for the component
        local_component_basis, _ = FunctionSpaces.evaluate(space.component_spaces[component_idx], element_idx, xi)
                   
        # store the evaluations in the right place
        local_multivalued_basis[component_idx] = zeros(Float64, size(local_component_basis[1][1],1), sum(num_component_basis))  # first initialize the evaluation of the component to zeros
        local_multivalued_basis[component_idx][:, count .+ (1:num_component_basis[component_idx])] .= local_component_basis[1][1]  # then store the values in the right places
        count += num_component_basis[component_idx]
    end

    return local_multivalued_basis, multivalued_basis_indices
end