
"""
    RationalFESpace{manifold_dim, F} <: AbstractFESpace{manifold_dim, 1}

A rational finite element space obtained by dividing all elements of a given function space
by a fixed element from the same function space. The latter is defined with the help of
specified weights.

# Fields
- `function_space::F`: The underlying function space.
- `weights::Vector{Float64}`: The weights associated with the basis functions of the
    function space.
"""
struct RationalFESpace{manifold_dim, F} <: AbstractFESpace{manifold_dim, 1}
    function_space::F
    weights::Vector{Float64}

    function RationalFESpace(
        function_space::F, weights::Vector{Float64}
    ) where {manifold_dim, F <: AbstractFESpace{manifold_dim, 1, 1}}
        if get_num_basis(function_space) != length(weights)
            error("Dimension mismatch")
        end

        return new{manifold_dim, F}(function_space, weights)
    end
end

function get_num_basis(space::RationalFESpace)
    return get_num_basis(space.function_space)
end
function get_num_basis(space::RationalFESpace, element_id::Int)
    return get_num_basis(space.function_space, element_id)
end

function get_basis_indices(space::RationalFESpace, element_id::Int)
    return get_basis_indices(space.function_space, element_id)
end

function get_num_elements(rat_space::RationalFESpace)
    return get_num_elements(rat_space.function_space)
end

"""
    get_polynomial_degree(rat_space::RationalFESpace, element_id::Int)

Returns the polynomial degree (or the degree of the underlying function space) of the
rational finite element space for a specific element.

# Arguments
- `rat_space::RationalFESpace`: The rational finite element space.
- `element_id::Int`: The index of the element.

# Returns
The polynomial degree (or the degree of the underlying function space) of the rational
finite element space for the specified element.
"""
function get_polynomial_degree(rat_space::RationalFESpace, element_id::Int)
    return get_polynomial_degree(rat_space.function_space, element_id)
end

function get_dof_partition(space::RationalFESpace)
    return get_dof_partition(space.function_space)
end

"""
    evaluate(
        rat_space::RationalFESpace{manifold_dim, F},
        element_id::Int,
        xi::NTuple{manifold_dim, Vector{Float64}},
        nderivatives::Int
    ) where {manifold_dim, F <: AbstractFESpace{manifold_dim, 1, 1}}

Evaluates the basis functions and their derivatives for the rational finite element space.

# Arguments
- `rat_space::RationalFESpace{manifold_dim, F}`: The rational finite
    element space.
- `element_id::Int`: The index of the element.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: The coordinates at which to evaluate the
    basis functions.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
A tuple containing:
- The basis functions and their derivatives evaluated at the specified coordinates.
- The basis indices.
"""
function evaluate(
    rat_space::RationalFESpace{manifold_dim, F},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int,
) where {manifold_dim, F <: AbstractFESpace{manifold_dim, 1, 1}}
    # Evaluate the basis functions and their derivatives for the underlying function space
    homog_basis, basis_indices = evaluate(
        rat_space.function_space, element_id, xi, nderivatives
    )
    n_eval = prod(length.(xi))

    # Compute the rational basis functions and their derivatives
    for j in 0:nderivatives
        if j == 0
            # Compute the weight
            temp = homog_basis[1][1][1] *
                LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
            weight = reshape(sum(temp; dims=2), n_eval)
            # Rationalize with the weights
            homog_basis[1][1][1] .= LinearAlgebra.Diagonal(weight) \ temp
        elseif j == 1
            der_keys = integer_sums(j, manifold_dim)
            for key in der_keys
                # Get the location where the derivative is stored
                der_idx = get_derivative_idx(key)
                # Compute the weight and its derivative
                temp = homog_basis[1][1][1] *
                    LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
                dtemp = homog_basis[j + 1][der_idx][1] *
                    LinearAlgebra.Diagonal(rat_space.weights[basis_indices])
                weight = reshape(sum(temp; dims=2), n_eval)
                dweight = reshape(sum(dtemp; dims=2), n_eval)
                # Compute the derivative of the rational basis functions
                homog_basis[j + 1][der_idx][1] .= LinearAlgebra.Diagonal(weight) \ dtemp -
                    LinearAlgebra.Diagonal(dweight ./ weight .^ 2) * temp
            end
        elseif j > 1
            error("Derivatives of rational spaces of order not implemented")
        end
    end

    return homog_basis, basis_indices
end

function get_max_local_dim(space::RationalFESpace)
    return get_max_local_dim(space.function_space)
end

function get_local_basis(
    space::RationalFESpace{manifold_dim, F},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim, F <: AbstractFESpace{manifold_dim, 1, 1}}
    return evaluate(space, element_id, xi, nderivatives)[1]
end

function get_extraction(space::RationalFESpace, element_id::Int, component_id::Int=1)
    # Get the basis indices for the underlying function space
    _, basis_indices = get_extraction(space.function_space, element_id)
    n_supp = length(basis_indices)

    return Matrix{Float64}(LinearAlgebra.I, n_supp, n_supp), basis_indices
end

"""
    get_element_size(rat_space::RationalFESpace, element_id::Int)

Returns the size of the element for the rational finite element space.

# Arguments
- `rat_space::RationalFESpace`: The rational finite element space.
- `element_id::Int`: The index of the element.

# Returns
The size of the element for the rational finite element space.
"""
function get_element_size(rat_space::RationalFESpace, element_id::Int)
    return get_element_size(rat_space.function_space, element_id)
end

function get_element_dimensions(rat_space::RationalFESpace, element_id::Int)
    return get_element_dimensions(rat_space.function_space, element_id)
end
