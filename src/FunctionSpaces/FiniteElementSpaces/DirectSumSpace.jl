"""
    DirectSumSpace{manifold_dim, num_components, F}

A multi-valued space that is the direct sum of `num_components` scalar function spaces. Consequently, their basis functions are evaluated independently and arranged in a block-diagonal matrix. Each scalar function space contributes to a separate component of the multi-valued space.

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
- `local_multivalued_basis::Vector{Vector{Vector{Array{Float64, 2}}}}`: Matrices containing the evaluations of the basis functions and its derivatives of the direct sum space.
    `local_multivalued_basis[i][j][k][l, m]` contains the (j-1)th-order derivative, with respect to the k-th coordinate, 
    of the m-th multivalued basis of component i evaluated at the lth-point.
    In this case the maximum order of derivative is first order. For higher order derivatives 
    we should follow a flattenned numbering using the indices of the derivatives.
- `multivalued_basis_indices::Vector{Int}`: Array containing the global indices of the basis functions
"""
function evaluate(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFiniteElementSpace{manifold_dim}}}
    # Check if only up to first derivatives is requested
    if nderivatives > 1
        throw(ArgumentError("nderivatives = $nderivatives > 1 not allowed!"))
    end

    # Number of evaluation matrices per derivative degree 
    # 1: evaluation of function 
    # 2: evaluation of first derivatives => manifold_dim evaluations, one per coordinate
    n_evaluation_matrices_per_derivative = manifold_dim * ones(Int, nderivatives + 1)
    n_evaluation_matrices_per_derivative[1] = 1
    
    # Get the indices of the basis
    # first get the indices for each of the component spaces
    multivalued_basis_indices, component_basis_indices = get_basis_indices_w_components(space, element_idx)
    num_basis_per_component = length.(component_basis_indices)
    num_multivaluedbasis = length(multivalued_basis_indices)
    
    # Allocate memory for the evaluation of all basis and their derivatives for all components
    #   local_multivalued_basis[j][i][k][l, m]
    # contains the (j-1)th-order derivative, derivative with respect to the k-th coordinate, 
    # of the m-th multivalued basis of component i evaluated at the lth-point
    # In this case the maximum order of derivative is first order. For higher order derivatives 
    # we should follow a flattenned numbering using the indices of the derivatives.
    n_evaluation_points = prod(size.(xi, 1))
    num_derivatives = num_components  # each component will have as many derivatives as the number of components
    local_multivalued_basis = [[[zeros(Float64, n_evaluation_points, num_multivaluedbasis) for _ = 1:n_evaluation_matrices] for _ = 1:num_components] for n_evaluation_matrices = n_evaluation_matrices_per_derivative]
    
    # next, loop over the spaces of each component and evaluate them
    count = 0
    for component_idx in 1:num_components
        # Evaluate the basis for the component
        local_component_basis, _ = FunctionSpaces.evaluate(space.component_spaces[component_idx], element_idx, xi, nderivatives)
        
        # Store the evaluation and the derivatives in the correct places
        for derivative_order_idx in 1:(nderivatives + 1)
            for derivative_idx in 1:n_evaluation_matrices_per_derivative[derivative_order_idx]
                # store the evaluations in the right place
                local_multivalued_basis[derivative_order_idx][component_idx][derivative_idx][:, count .+ (1:num_basis_per_component[component_idx])] .= local_component_basis[derivative_order_idx][derivative_idx]  # then store the values in the right places
            end
        end

        count += num_basis_per_component[component_idx]
    end

    return local_multivalued_basis, multivalued_basis_indices
end

"""
    get_num_basis(space::DirectSumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F}

Get the number of basis functions of the direct sum space.

# Arguments
- `space::DirectSumSpace{manifold_dim, num_components, F}`: Direct sum space

# Returns
- `num_basis::Int`: Number of basis functions
"""
get_num_basis(space::DirectSumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F} = sum(get_num_basis.(space.component_spaces))

"""
    get_num_basis(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int) where {manifold_dim, num_components, F}

Get the number of active basis functions of the direct sum space in element `element_idx`.

# Arguments
- `space::DirectSumSpace{manifold_dim, num_components, F}`: Direct sum space
- `element_idx::Int`: The element where to get the number of active basis.

# Returns
- `num_basis::Int`: Number of active basis functions in element `element_idx`
"""
get_num_basis(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int) where {manifold_dim, num_components, F} = sum(get_num_basis.(space.component_spaces, element_idx))

"""
    get_basis_indices(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int) where {manifold_dim, num_components, F}

Get the global indices of the basis functions of the direct sum space in the element with index `element_idx`.

# Arguments
- `space::DirectSumSpace{manifold_dim, num_components, F}`: Direct sum space
- `element_idx::Int`: Index of the element

# Returns
- `basis_indices::Vector{Int}`: Global indices of the basis functions
"""
function get_basis_indices(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int) where {manifold_dim, num_components, F}
    component_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)
    num_dofs_component = FunctionSpaces.get_num_basis.(space.component_spaces)
    dof_offset_component = zeros(Int, num_components)
    dof_offset_component[2:end] .= cumsum(num_dofs_component[1:(num_components-1)])
    
    return vcat(map(.+, component_basis_indices, dof_offset_component)...)
end

"""
    get_basis_indices_w_components(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int) where {manifold_dim, num_components, F}

Get the global indices of the multivalued basis functions of the direct sum space as well as the component spaces for the element with index `element_idx`.

# Arguments
- `space::DirectSumSpace{manifold_dim, num_components, F}`: Direct sum space
- `element_idx::Int`: Index of the element

# Returns
- `multivalued_basis_indices::Vector{Int}`: Global indices of the multivalued basis functions
- `component_basis_indices::Vector{Vector{Int}}`: Global indices of the basis functions of the component spaces
"""
function get_basis_indices_w_components(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int) where {manifold_dim, num_components, F}
    component_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)
    num_dofs_component = FunctionSpaces.get_num_basis.(space.component_spaces)
    dof_offset_component = zeros(Int, num_components)
    dof_offset_component[2:end] .= cumsum(num_dofs_component[1:(num_components-1)])
    
    return vcat(map(.+, component_basis_indices, dof_offset_component)...), component_basis_indices
end