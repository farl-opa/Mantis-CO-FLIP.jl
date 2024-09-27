"""
    CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}

A multi-valued space that is the direct sum of `num_spaces` multi-valued function spaces. Consequently, their basis functions are evaluated independently and arranged in a block-diagonal matrix. Each multi-valued function space contributes to several  component of the multi-valued space. If `num_spaces` is equal to `num_components`, then the CompositeDirectSumSpace is equivalent to a DirectSumSpace.

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F} <: AbstractMultiValuedFiniteElementSpace{manifold_dim, num_components}
    component_spaces::F
    component_ordering::NTuple{num_spaces, Vector{Int}}

    function CompositeDirectSumSpace(component_spaces::F, component_ordering::NTuple{num_spaces, Vector{Int}}) where {num_spaces, F <: NTuple{num_spaces, AbstractMultiValuedFiniteElementSpace}}
        manifold_dim = get_manifold_dim(component_spaces[1])
        num_components = 0
        for space in component_spaces
            if get_manifold_dim(space) != manifold_dim
                throw(ArgumentError("All component spaces must have the same manifold dimension"))
            end
            num_components += get_num_components(space)
        end
        tmp = vcat(component_ordering...)
        if num_components != maximum(tmp)
            throw(ArgumentError("The number of components does not correspond to the component ordering provided."))
        end
        if length(unique(tmp)) != length(temp)
            throw(ArgumentError("The component ordering must not contain duplicates."))
        end
        
        new{manifold_dim, num_spaces, num_components, F}(component_spaces, component_ordering)
    end

    function CompositeDirectSumSpace(component_spaces::F) where {num_spaces, F <: NTuple{num_spaces, AbstractMultiValuedFiniteElementSpace}}
        # number of components per space
        num_components_per_space = zeros(Int, num_spaces)
        for i in eachindex(component_spaces)
            num_components_per_space[i] = get_num_components(component_spaces[i])
        end
        # construct default component ordering
        component_offsets = cumsum([0; num_components_per_space])
        component_ordering = (collect(component_offsets[i]+1:component_offsets[i+1]) for i in 1:num_spaces)

        new{manifold_dim, num_spaces, num_components, F}(component_spaces, component_ordering)
    end
end

"""
    get_component_spaces(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}) where {manifold_dim, num_spaces, num_components, F}

Get the component spaces of the composite direct sum space.

# Arguments
- `space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}`: Direct-sum space

# Returns
- `component_spaces::F`: Tuple of component spaces
"""
function get_component_spaces(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}) where {manifold_dim, num_spaces, num_components, F}
    return space.component_spaces
end

"""
    evaluate(space::CompositeDirectSumSpace{manifold_dim,num_components,F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components}

Evaluate the basis functions of the composite direct sum space at the points `xi` in the element with index `element_idx`. The function returns a tuple of `num_components` arrays, built from the evaluations of the basis functions of the component space.

# Arguments
- `space::CompositeDirectSumSpace{manifold_dim,num_spaces,num_components,F}`: Direct sum space
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
function evaluate(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_spaces, num_components, F <: NTuple{num_spaces, AbstractMultiValuedFiniteElementSpace}}
    
    # get the multi-valued basis indices
    multivalued_basis_indices, component_space_basis_indices = get_basis_indices_w_components(space, element_idx)
    num_basis_per_space = length.(component_space_basis_indices)
    num_multivaluedbasis = length(multivalued_basis_indices)
    n_evaluation_points = prod(size.(xi, 1))
    num_components_per_space = length.(space.component_ordering)

    # Generate keys for all possible derivative combinations
    der_keys = _integer_sums(nderivatives, manifold_dim+1)
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
    for space_idx in 1:num_spaces
        # ... evaluate component spaces ...
        local_space_basis, _ = FunctionSpaces.evaluate(space.component_spaces[space_idx], element_idx, xi, nderivatives)
        # ... then store the derivatives in the right places ...
        for key in der_keys
            key = key[1:manifold_dim]
            j = sum(key) # order of derivative
            der_idx = _get_derivative_idx(key) # index of derivative
            
            for i in 1:num_components_per_space[space_idx]
                component_idx = space.component_ordering[space_idx][i]
                # the `i`-th local component is stored as the `component_idx`-th component of the multi-valued space
                local_multivalued_basis[j + 1][der_idx][component_idx][:, count .+ (1:num_basis_per_space[space_idx])] .= local_space_basis[j+1][der_idx][i]
            end
        end

        count += num_basis_per_space[space_idx]
        component_count += num_components_per_space[space_idx]
    end
    
    return local_multivalued_basis, multivalued_basis_indices
end

"""
    get_num_basis(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}) where {manifold_dim, num_spaces, num_components, F}

Get the number of basis functions of the composite direct sum space.

# Arguments
- `space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}`: Direct sum space

# Returns
- `num_basis::Int`: Number of basis functions
"""
get_num_basis(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}) where {manifold_dim, num_spaces, num_components, F} = sum(get_num_basis.(space.component_spaces))

"""
    get_num_basis(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}, element_idx::Int) where {manifold_dim, num_spaces, num_components, F}

Get the number of active basis functions of the composite direct sum space in element `element_idx`.

# Arguments
- `space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}`: Direct sum space
- `element_idx::Int`: The element where to get the number of active basis.

# Returns
- `num_basis::Int`: Number of active basis functions in element `element_idx`
"""
get_num_basis(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}, element_idx::Int) where {manifold_dim, num_spaces, num_components, F} = sum(get_num_basis.(space.component_spaces, element_idx))

"""
    get_basis_indices(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}, element_idx::Int) where {manifold_dim, num_spaces, num_components, F}

Get the global indices of the basis functions of the composite direct sum space in the element with index `element_idx`.

# Arguments
- `space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}`: Direct sum space
- `element_idx::Int`: Index of the element

# Returns
- `basis_indices::Vector{Int}`: Global indices of the basis functions for each component space
"""
function get_basis_indices(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}, element_idx::Int) where {manifold_dim, num_spaces, num_components, F}
    component_space_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)
    num_dofs_per_space = FunctionSpaces.get_num_basis.(space.component_spaces)
    dof_offset_per_space = zeros(Int, num_spaces)
    dof_offset_per_space[2:end] .= cumsum(num_dofs_per_space[1:(num_spaces-1)])
    
    return vcat(map(.+, component_space_basis_indices, dof_offset_per_space)...)
end

"""
    get_basis_indices_w_components(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}, element_idx::Int) where {manifold_dim, num_spaces, num_components, F}

Get the global indices of the multivalued basis functions of the direct sum space as well as the component spaces for the element with index `element_idx`.

# Arguments
- `space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}`: Composite direct sum space.
- `element_idx::Int`: Index of the element

# Returns
- `multivalued_basis_indices::Vector{Int}`: Global indices of the multivalued basis functions
- `component_basis_indices::Vector{Vector{Int}}`: Global indices of the basis functions of the component spaces
"""
function get_basis_indices_w_components(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}, element_idx::Int) where {manifold_dim, num_spaces, num_components, F}
    component_space_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)
    num_dofs_per_space = FunctionSpaces.get_num_basis.(space.component_spaces)
    dof_offset_per_space = zeros(Int, num_spaces)
    dof_offset_per_space[2:end] .= cumsum(num_dofs_per_space[1:(num_spaces-1)])
    
    return vcat(map(.+, component_space_basis_indices, dof_offset_per_space)...), component_space_basis_indices
end

"""
    get_max_local_dim(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}) where {manifold_dim, num_spaces, num_components, F}

Get the maximum local dimension of the composite direct sum space.

# Arguments
- `space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}`: composite direct sum space.

# Returns
- `max_local_dim::Int`: Maximum local dimension
"""
function get_max_local_dim(space::CompositeDirectSumSpace{manifold_dim, num_spaces, num_components, F}) where {manifold_dim, num_spaces, num_components, F}
    max_local_dim = 0
    for space in space.component_spaces
        max_local_dim += get_max_local_dim(space)
    end
    return max_local_dim
end