"""
    SumSpace{manifold_dim, num_components, F}

A multi-valued space that is the sum of `num_components` input scalar function spaces. Each scalar function space contributes to each component of the multi-valued space.

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
"""
struct SumSpace{manifold_dim, num_components, F} <: AbstractMultiValuedFiniteElementSpace{manifold_dim, num_components}
    component_spaces::F
    space_dim::Int

    function SumSpace(component_spaces::F, space_dim::Int) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFiniteElementSpace{manifold_dim}}}
        new{manifold_dim, num_components, F}(component_spaces, space_dim)
    end
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
function evaluate(space::SumSpace{manifold_dim, num_components, F}, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_components, F <: NTuple{num_components, AbstractFiniteElementSpace{manifold_dim}}}
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
    component_basis_indices = FunctionSpaces.get_basis_indices.(space.component_spaces, element_idx)

    # get the basis indices for the multivalued space
    multivalued_basis_indices = union(component_basis_indices...)
    # number of multivalued basis functions
    num_multivaluedbasis = length(multivalued_basis_indices) 
    # find the local column that each component contributes to
    column_indices_per_component = [indexin(component_basis_indices[i], multivalued_basis_indices) for i in 1:num_components]
    
    # Allocate memory for the evaluation of all basis and their derivatives for all components
    #   local_multivalued_basis[i][j][k][l, m]
    # contains the (j-1)th-order derivative, derivative with respect to the k-th coordinate, 
    # of the i-th multivalued basis of component i evaluated at the lth-point
    # In this case the maximum order of derivative is first order. For higher order derivatives 
    # we should follow a flattenned numbering using the indices of the derivatives.
    n_evaluation_points = prod(size.(xi, 1))
    num_derivatives = num_components  # each component will have as many derivatives as the number of components
    local_multivalued_basis = [[[zeros(Float64, n_evaluation_points, num_multivaluedbasis) for _ = 1:n_evaluation_matrices] for n_evaluation_matrices = n_evaluation_matrices_per_derivative] for _ = 1:num_components]
    
    # next, loop over the spaces of each component and evaluate them
    for component_idx in 1:num_components
        # Evaluate the basis for the component
        local_component_basis, _ = FunctionSpaces.evaluate(space.component_spaces[component_idx], element_idx, xi, nderivatives)
        
        # Store the evaluation and the derivatives in the correct places
        for derivative_order_idx in 1:(nderivatives + 1)
            for derivative_idx in 1:n_evaluation_matrices_per_derivative[derivative_order_idx]
                # store the evaluations in the right place
                local_multivalued_basis[component_idx][derivative_order_idx][derivative_idx][:, column_indices_per_component[component_idx]] .= local_component_basis[derivative_order_idx][derivative_idx]  # then store the values in the right places
            end
        end

    end

    return local_multivalued_basis, multivalued_basis_indices
end

"""
    get_num_basis(space::SumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F}

Get the number of basis functions of the sum space.

# Arguments
- `space::SumSpace{manifold_dim, num_components, F}`: Sum space

# Returns
- `num_basis::Int`: Number of basis functions
"""
get_num_basis(space::SumSpace{manifold_dim, num_components, F}) where {manifold_dim, num_components, F} = space.space_dim