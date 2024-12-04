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
    get_component_spaces(space::DirectSumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F}

Get the component spaces of the direct sum space.

# Arguments
- `space::DirectSumSpace{manifold_dim, num_components, F}`: Direct-sum space

# Returns
- `component_spaces::F`: Tuple of component spaces
"""
function get_component_spaces(space::DirectSumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F}
    return space.component_spaces
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
    `local_multivalued_basis[j][k][i][l, m]` contains the (j-1)th-order derivative, with respect to the k-th coordinate, 
    of the m-th multivalued basis of component i evaluated at the lth-point.
    In this case the maximum order of derivative is first order. For higher order derivatives 
    we should follow a flattenned numbering using the indices of the derivatives.
- `multivalued_basis_indices::Vector{Int}`: Array containing the global indices of the basis functions
"""
function evaluate(space::DirectSumSpace{manifold_dim, num_components, F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFiniteElementSpace{manifold_dim}}}
    
    # get the multi-valued basis indices
    multivalued_basis_indices, component_basis_indices = get_basis_indices_w_components(space, element_idx)
    num_basis_per_component = length.(component_basis_indices)
    num_multivaluedbasis = length(multivalued_basis_indices)
    n_evaluation_points = prod(size.(xi, 1))

    # Generate keys for all possible derivative combinations
    der_keys = integer_sums(nderivatives, manifold_dim+1)
    # Initialize storage of local basis functions and derivatives
    local_multivalued_basis = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        # number of derivatives of order j
        num_j_ders = binomial(manifold_dim + j - 1, manifold_dim - 1)
        # allocate space
        local_multivalued_basis[j + 1] = Vector{Vector{Matrix{Float64}}}(undef, num_j_ders)
        for der_idx in 1:num_j_ders
            local_multivalued_basis[j + 1][der_idx] = [zeros(n_evaluation_points,num_multivaluedbasis) for _ in 1:num_components]
        end
    end

    # loop over all components...
    count = 0
    for component_idx in 1:num_components
        # ... evaluate component spaces ...
        local_component_basis, _ = FunctionSpaces.evaluate(space.component_spaces[component_idx], element_idx, xi, nderivatives)
        # ... then store the derivatives in the right places ...
        for key in der_keys
            key = key[1:manifold_dim]
            j = sum(key) # order of derivative
            der_idx = get_derivative_idx(key) # index of derivative
            
            local_multivalued_basis[j + 1][der_idx][component_idx][:, count .+ (1:num_basis_per_component[component_idx])] .= local_component_basis[j+1][der_idx]
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

    multivalued_basis_indices = Vector{Vector{Int}}(undef, num_components)

    for component_idx in 1:num_components
        multivalued_basis_indices[component_idx] =  component_basis_indices[component_idx] .+ dof_offset_component[component_idx]
    end
    
    return reduce(vcat, multivalued_basis_indices)
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

    multivalued_basis_indices = Vector{Vector{Int}}(undef, num_components)

    for component_idx in 1:num_components
        multivalued_basis_indices[component_idx] =  component_basis_indices[component_idx] .+ dof_offset_component[component_idx]
    end
    
    return reduce(vcat, multivalued_basis_indices), component_basis_indices
end

"""
    get_max_local_dim(space::DirectSumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F}

Get the maximum local dimension of the direct sum space.

# Arguments
- `space::DirectSumSpace{manifold_dim, num_components, F}`: Direct sum space

# Returns
- `max_local_dim::Int`: Maximum local dimension
"""
function get_max_local_dim(space::DirectSumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F}
    max_local_dim = 0
    for space in space.component_spaces
        max_local_dim += get_max_local_dim(space)
    end
    return max_local_dim
end