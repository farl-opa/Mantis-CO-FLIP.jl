"""
    DirectSumSpace{manifold_dim,num_components,F}

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct DirectSumSpace{manifold_dim,num_components,F} <: AbstractMultiValuedFiniteElementSpace{manifold_dim,num_components}
    component_spaces::F

    function DirectSumSpace(component_spaces::F) where {manifold_dim, num_components, F <: NTuple{num_components,AbstractFiniteElementSpace{manifold_dim}}}
        new{manifold_dim,num_components,F}(component_spaces)
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
function evaluate(space::DirectSumSpace{manifold_dim,num_components,F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components,F <: NTuple{num_components,AbstractFiniteElementSpace{manifold_dim}}}
    # number of dofs for each space
    num_dofs_component = FunctionSpaces.get_num_basis.(space.component_spaces)
    dof_offset_component = zeros(Int, num_components)
    dof_offset_component[2:end] .= cumsum(num_dofs_component[1:(num_components-1)])  # we skip the first one because the offset is 0

    # Allocate memory for the evaluation of all basis for all components
    local_multivalued_basis = Vector{Matrix{Float64}}(undef, num_components)
    
    # first loop over the components to get the basis indices
    component_basis_indices = [FunctionSpaces.get_basis_indices(space.component_spaces[component_idx], element_idx) for component_idx in 1:num_components]
    num_component_basis = length.(component_basis_indices)

    # allocate memory to store all the indices
    multivalued_basis_indices = Vector{Int}(undef, sum(num_component_basis))

    
    # next, loop over the spaces of the each component and evaluate them
    # Note that the indices of the basis for each component are updated
    # by the offset defined above.
    count = 0
    for component_idx in 1:num_components
        # Evaluate the basis for the component
        local_component_basis, _ = FunctionSpaces.evaluate(space.component_spaces[component_idx], element_idx, xi)
        
        # update the global basis indices
        multivalued_basis_indices[count .+ (1:num_component_basis[component_idx])] .= component_basis_indices[component_idx] .+ dof_offset_component[component_idx]
        
        # store the evaluations in the right place
        local_multivalued_basis[component_idx] = zeros(Float64, size(local_component_basis[1][1],1), sum(num_component_basis))
        local_multivalued_basis[component_idx][:, count .+ (1:num_component_basis[component_idx])] .= local_component_basis[1][1]
        count += num_component_basis[component_idx]
    end

    return local_multivalued_basis, multivalued_basis_indices
end