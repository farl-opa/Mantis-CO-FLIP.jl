"""
    SumSpace{manifold_dim, num_components, F}

A multi-valued space that is the sum of `num_components` input scalar function spaces. Each scalar function space contributes to each component of the multi-valued space.

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct SumSpace{manifold_dim, num_components, F} <: AbstractFESpace{manifold_dim, num_components}
    component_spaces::F
    space_dim::Int

    function SumSpace(component_spaces::F, space_dim::Int) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFESpace{manifold_dim, 1}}}
        new{manifold_dim, num_components, F}(component_spaces, space_dim)
    end
end

"""
    get_component_spaces(space::SumSpace)

Get the component spaces of the sum space.

# Arguments
- `space::SumSpace{manifold_dim, num_components, F}`: Sum space

# Returns
- `component_spaces::F`: Tuple of component spaces
"""
function get_component_spaces(space::SumSpace)
    return space.component_spaces
end

"""
    evaluate(space::SumSpace{manifold_dim, num_components, F}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components}

Evaluate the basis functions of the sum space at the points `xi` in the element with index `element_idx`. The function returns a tuple of `num_components` arrays, each containing the evaluations of the basis functions of the component spaces.

# Arguments
- `space::SumSpace{manifold_dim,num_components,F}`: Sum space
- `element_idx::Int`: Index of the element
- `xi::NTuple{manifold_dim,Vector{Float64}}`: Points in the reference element
- `nderivatives::Int`: Number of derivatives to evaluate

# Returns
- `local_multivalued_basis::Vector{Matrix{Float64}}`: Vector of matrices containing the evaluations of the basis functions of the sum space
- `multivalued_basis_indices::Vector{Int}`: Array containing the global indices of the basis functions
"""
function evaluate(space::SumSpace{manifold_dim, num_components, F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFESpace{manifold_dim, 1}}}

    # get the multi-valued basis indices
    multivalued_basis_indices, component_basis_indices = get_basis_indices_w_components(space, element_idx)
    # number of multivalued basis functions
    num_multivaluedbasis = length(multivalued_basis_indices)
    # find the local column that each component contributes to
    column_indices_per_component = [indexin(component_basis_indices[i], multivalued_basis_indices) for i in 1:num_components]
    # number of evaluation points
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
    for component_idx in 1:num_components
        # ... evaluate component spaces ...
        local_component_basis, _ = FunctionSpaces.evaluate(space.component_spaces[component_idx], element_idx, xi, nderivatives)
        # ... then store the derivatives in the right places ...
        for key in der_keys
            key = key[1:manifold_dim]
            j = sum(key) # order of derivative
            der_idx = get_derivative_idx(key) # index of derivative

            local_multivalued_basis[j + 1][der_idx][component_idx][:, column_indices_per_component[component_idx]] .= local_component_basis[j+1][der_idx]
        end
    end

    return local_multivalued_basis, multivalued_basis_indices
end

function get_basis_indices(space::SumSpace, element_idx::Int)
    component_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)

    return union(component_basis_indices...)
end

"""
    get_basis_indices_w_components(space::SumSpace, element_idx::Int)

Get the global indices of the multivalued basis functions of the sum space as well as the component spaces for the element with index `element_idx`.

# Arguments
- `space::SumSpace`: Sum space
- `element_idx::Int`: Index of the element

# Returns
- `multivalued_basis_indices::Vector{Int}`: Global indices of the multivalued basis functions
- `component_basis_indices::Vector{Vector{Int}}`: Global indices of the basis functions of the component spaces
"""
function get_basis_indices_w_components(space::SumSpace, element_idx::Int)
    component_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)

    return union(component_basis_indices...), component_basis_indices
end

get_num_basis(space::SumSpace) = space.space_dim
function get_num_basis(space::SumSpace, element_id::Int)
    return length(get_basis_indices(space, element_id))
end

function get_max_local_dim(space::SumSpace)
    max_local_dim = 0
    # A very conservative estimate, because otherwise we need to loop over all elements,
    # check the unions of all active basis functions, and return the length of the largest
    # union.
    for space in space.component_spaces
        max_local_dim += max(max_local_dim, get_max_local_dim(space))
    end
    return max_local_dim
end
