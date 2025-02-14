
"""
    TensorProductSpace{manifold_dim, T} <: AbstractFESpace{manifold_dim, 1}

A structure representing a tensor product space composed of multiple finite element spaces.
The functions in this space are `manifold_dim`-variate where `manifold_dim` is equal to the
sum of the manifold dimensions of the input spaces.

# Fields
- `fem_spaces::T`: A tuple of finite element spaces.
- `dof_partition::Vector{Vector{Vector{Int}}}`: A nested vector structure representing the
    degree of freedom partitioning.
- `data::Dict`: A dictionary to store additional data.

# Constructors
    TensorProductSpace(fem_spaces::T, data::Dict) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    TensorProductSpace(fem_spaces::T) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
"""
struct TensorProductSpace{manifold_dim, T} <: AbstractFESpace{manifold_dim, 1}
    fem_spaces::T
    dof_partition::Vector{Vector{Vector{Int}}}
    data::Dict

    function TensorProductSpace(fem_spaces::T, data::Dict) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        manifold_dim = sum(get_manifold_dim, fem_spaces)

        # Tensor product spaces were designed for scalar fields.
        for space in fem_spaces
            if get_num_components(space) != 1
                throw(ArgumentError("All input spaces must have only one component."))
            end
        end

        # Get dof partitions of the constituent spaces
        dof_partitions = [get_dof_partition(space) for space in fem_spaces]
        n_patches = [length(partition) for partition in dof_partitions]

        # Dimensions of constituent function spaces
        spaces_dims = map(get_num_basis, fem_spaces)

        # Allocate memory for degree of freedom partitioning
        dof_partition = Vector{Vector{Vector{Int}}}(undef, prod(n_patches))

        # Loop over all dimensions and build the appropriate index subsets
        patch_count = 1
        for patch_index ∈ CartesianIndices(Tuple(n_patches))
            partition_patches = [dof_partitions[i][patch_index[i]] for i ∈ 1:num_spaces]
            partition_lengths = map(length, partition_patches)
            dof_partition[patch_count] = Vector{Vector{Int}}(undef, prod(partition_lengths))
            partition_count = 1
            for partition_index ∈ CartesianIndices(Tuple(partition_lengths))
                current_partition_dofs = [partition_patches[i][partition_index[i]] for i ∈ 1:num_spaces]
                dof_partition[patch_count][partition_count] = Vector{Int}(undef, prod(length.(current_partition_dofs)))
                dof_count = 1
                for id in Iterators.product(current_partition_dofs...)
                    dof_partition[patch_count][partition_count][dof_count] = ordered_to_linear_index(id, spaces_dims)
                    dof_count += 1
                end
                partition_count += 1
            end
            patch_count += 1
        end

        new{manifold_dim, T}(fem_spaces, dof_partition, data)
    end

    function  TensorProductSpace(fem_spaces::T) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        return TensorProductSpace(fem_spaces, Dict())
    end
end

# Basic getters per space

_get_num_basis_per_space(tp_space::TensorProductSpace) = map(get_num_basis, tp_space.fem_spaces)

_get_num_elements_per_space(tp_space::TensorProductSpace) = map(get_num_elements, tp_space.fem_spaces)

function _get_basis_indices_per_space(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    max_ind_el = _get_num_elements_per_space(tp_space)
    ordered_index = linear_to_ordered_index(element_id, max_ind_el)

    return tuple(map(get_basis_indices, tp_space.fem_spaces, ordered_index)...)::NTuple{num_spaces, Vector{Int}}
end

function _get_num_basis_per_space(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    max_ind_el = _get_num_elements_per_space(tp_space)
    ordered_index = linear_to_ordered_index(element_id, max_ind_el)

    return tuple(map(get_num_basis, tp_space.fem_spaces, ordered_index)...)::NTuple{num_spaces, Int}
end

function _get_support_per_space(tp_space::TensorProductSpace{manifold_dim, T}, basis_id::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    max_ind_basis = _get_num_basis_per_space(tp_space)
    ordered_index = linear_to_ordered_index(basis_id, max_ind_basis)

    return tuple(map(get_support, tp_space.fem_spaces, ordered_index)...)::NTuple{num_spaces, Vector{Int}}
end

function _get_extraction_per_space(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    max_ind_el = _get_num_elements_per_space(tp_space)
    ordered_index = linear_to_ordered_index(element_id, max_ind_el)

    return tuple(map(get_extraction, tp_space.fem_spaces, ordered_index)...)::NTuple{num_spaces, Tuple{Matrix{Float64}, Vector{Int}}}
end

"""
    get_polynomial_degree_per_space(tp_space::TensorProductSpace)

Compute and return the polynomial degree for each space in a `TensorProductSpace` composed of univariate B-splines.

# Arguments
- `tp_space::TensorProductSpace`: The tensor-product space.

# Returns
- `::NTuple{manifold_dim, Int}`: A vector containing the polynomial degree for each space in the tensor-product space.
"""
function get_polynomial_degree_per_space(tp_space::TensorProductSpace)
    return map(get_polynomial_degree, tp_space.fem_spaces)
end

function _get_local_basis_per_space(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    # Get number of elements in each constituent space
    max_ind_el = _get_num_elements_per_space(tp_space)

    # Convert linear element ID to ordered index
    ordered_index = linear_to_ordered_index(element_id, max_ind_el)

    manifold_dim_per_space = map(get_manifold_dim, tp_space.fem_spaces)
    cum_manifold_dim_per_space = cumsum((0, manifold_dim_per_space...))
    # Split evaluation points for each constituent space
    num_points = size(xi,1)
    xis = [Tuple([Vector{Float64}(undef, num_points) for _ in 1:get_manifold_dim(space)]) for space in tp_space.fem_spaces]
    for i ∈ 1:num_spaces
        xis[i] = xi[cum_manifold_dim_per_space[i]+1:cum_manifold_dim_per_space[i]+manifold_dim_per_space[i]]
    end

    return tuple(map(get_local_basis, tp_space.fem_spaces, ordered_index, xis, fill(nderivatives, num_spaces))...)::NTuple{num_spaces,Vector{Vector{Matrix{Float64}}}}
end

function _get_element_vertices_per_space(tp_space::TensorProductSpace, element_id::Int)
   # Get number of elements in each constituent space
   max_ind_el = _get_num_elements_per_space(tp_space)

   # Convert linear element ID to ordered index
   ordered_index = linear_to_ordered_index(element_id, max_ind_el)

    return map(get_element_vertices, tp_space.fem_spaces, ordered_index)
end

function _get_element_dimensions_per_space(tp_space::TensorProductSpace, element_id::Int)
    # Get number of elements in each constituent space
   max_ind_el = _get_num_elements_per_space(tp_space)

   # Convert linear element ID to ordered index
   ordered_index = linear_to_ordered_index(element_id, max_ind_el)

   return map(get_element_dimensions, tp_space.fem_spaces, ordered_index)
end

# Basic getters for the tensor product space
"""
    get_space(tp_space::TensorProductSpace, space_id::Int)

Retrieve and return a specific finite element space from a `TensorProductSpace`.

# Arguments
- `tp_space::TensorProductSpace`: The tensor-product space from which to retrieve the finite element space.
- `space_id::Int`: The identifier of the finite element space to retrieve.

# Returns
- `::AbstractFESpace`: The finite element space corresponding to the specified `space_id`.
"""
function get_space(tp_space::TensorProductSpace, space_id::Int)
    return tp_space.fem_spaces[space_id]
end

function get_basis_indices(space::TensorProductSpace, element_id::Int)
    max_ind_basis = _get_num_basis_per_space(space)
    indices_per_space = _get_basis_indices_per_space(space, element_id)

    # Compute basis indices
    basis_indices = Vector{Int}(undef, prod(map(length, indices_per_space)))
    idx = 1
    for basis in Iterators.product(indices_per_space...)
        basis_indices[idx] = ordered_to_linear_index(basis, max_ind_basis)
        idx += 1
    end

    return basis_indices
end

get_num_basis(space::TensorProductSpace) = prod(_get_num_basis_per_space(space))
function get_num_basis(space::TensorProductSpace, element_id::Int)
    return prod(_get_num_basis_per_space(space, element_id))
end

get_num_elements(tp_space::TensorProductSpace) = prod(_get_num_elements_per_space(tp_space))

"""
    get_element_vertices(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int) where {manifold_dim, num_spaces, T<: NTuple{num_spaces, AbstractFESpace}}

Compute and return the vertices of a specified element within a `TensorProductSpace`.

# Arguments
- `tp_space::TensorProductSpace{manifold_dim, T}`: The tensor product space.
- `element_id::Int`: The identifier of the element.

# Returns
- `::NTuple{manifold_dim, Vector{Float64}}`: A tuple of vectors containing the vertices of the specified element per manifold dimension.
"""
function get_element_vertices(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int) where {manifold_dim, num_spaces, T<: NTuple{num_spaces, AbstractFESpace}}
    vertices_per_space = _get_element_vertices_per_space(tp_space, element_id)

    vertices = Vector{Vector{Float64}}(undef, manifold_dim)

    manifold_dim_per_space = map(get_manifold_dim, tp_space.fem_spaces)
    cum_manifold_dim_per_space = cumsum((0, manifold_dim_per_space...))

    for space_id ∈ 1:num_spaces
        for dim ∈ eachindex(vertices_per_space[space_id])
            vertices[dim+cum_manifold_dim_per_space[space_id]] = vertices_per_space[space_id][dim]
        end
    end

    return tuple(vertices...)::NTuple{manifold_dim, Vector{Float64}}
end

"""
    get_element_dimensions(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int) where {manifold_dim, num_spaces, T<: NTuple{num_spaces, AbstractFESpace}}

Compute and return the size of a specified element within a `TensorProductSpace` per manifold dim.

# Arguments
- `tp_space::TensorProductSpace`: The tensor product space.
- `element_id::Int`: The identifier of the element.

# Returns
- `::NTuple{manifold_dim, Float64}`: The size of the specified element per manifold dimension.
"""
function get_element_dimensions(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int) where {manifold_dim, num_spaces, T<: NTuple{num_spaces, AbstractFESpace}}
    element_dimensions_per_space = _get_element_dimensions_per_space(tp_space, element_id)
    element_dimensions = Vector{Float64}(undef, manifold_dim)

    manifold_dim_per_space = map(get_manifold_dim, tp_space.fem_spaces)
    cum_manifold_dim_per_space = cumsum((0, manifold_dim_per_space...))

    for space_id ∈ 1:num_spaces
        for dim ∈ eachindex(element_dimensions_per_space[space_id])
            element_dimensions[dim+cum_manifold_dim_per_space[space_id]] = element_dimensions_per_space[space_id][dim]
        end
    end

    return tuple(element_dimensions...)::NTuple{manifold_dim, Float64}
end

function get_max_local_dim(space::TensorProductSpace)
    return prod(map(get_max_local_dim, space.fem_spaces))
end

"""
    get_support(tp_space::TensorProductSpace, basis_id::Int) -> Vector{Int}

Compute and return all the indices of elements that make up the support of a given basis function within a `TensorProductSpace`.

# Arguments
- `tp_space::TensorProductSpace`: The tensor-product space.
- `basis_id::Int`: The identifier of the basis function.

# Returns
- `::Vector{Int}`: A vector containing all the indices of elements that make up the support of the specified basis function.
"""
function get_support(tp_space::TensorProductSpace, basis_id::Int)
    max_ind_els = _get_num_elements_per_space(tp_space)
    support_per_space = _get_support_per_space(tp_space, basis_id)

    support = Vector{Int}(undef, prod(map(length, support_per_space)))
    idx = 1
    for id in Iterators.product(support_per_space...)
        support[idx] = ordered_to_linear_index(id, max_ind_els)
        idx += 1
    end

    return support
end

function get_extraction(
    space::TensorProductSpace{manifold_dim, T},
    element_id::Int
) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    max_ind_basis = _get_num_basis_per_space(space)
    extraction_per_space = _get_extraction_per_space(space, element_id)

    # Compute Kronecker product of extraction coefficients
    extraction_coeffs = kron((extraction_per_space[num_spaces-i+1][1] for i ∈ 1:num_spaces)...)

    # Compute basis indices
    basis_indices = Vector{Int}(undef, prod(map(extraction -> length(extraction[2]), extraction_per_space)))
    idx = 1
    for basis in Iterators.product((extraction_per_space[i][2] for i ∈ 1:num_spaces)...)
        basis_indices[idx] = ordered_to_linear_index(basis, max_ind_basis)
        idx += 1
    end

    return extraction_coeffs, basis_indices
end

function get_local_basis(space::TensorProductSpace{manifold_dim, T}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    # Compute local basis for each constituent space
    local_basis_per_space = _get_local_basis_per_space(space, element_id, xi, nderivatives)

    # Generate keys for all possible derivative combinations
    der_keys = integer_sums(nderivatives, manifold_dim+1)
    # Initialize storage of local basis functions and derivatives
    local_basis = Vector{Vector{Matrix{Float64}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        num_j_ders = binomial(manifold_dim + j - 1, manifold_dim - 1)
        local_basis[j + 1] = Vector{Matrix{Float64}}(undef, num_j_ders)
    end

    manifold_dim_per_space = map(get_manifold_dim, space.fem_spaces)
    cum_manifold_dim_per_space = cumsum((0, manifold_dim_per_space...))
    # Split manifold dimensions for each constituent space
    keys_idx = Vector{UnitRange{Int}}(undef, num_spaces)
    for i ∈ 1:num_spaces
        keys_idx[i] = cum_manifold_dim_per_space[i]+1:cum_manifold_dim_per_space[i]+manifold_dim_per_space[i]
    end

    # Initialize an array to hold the basis functions for each space
    basis_functions_per_space = Vector{Matrix{Float64}}(undef, num_spaces)

    # Compute tensor product of constituent basis functions for each derivative combination
    for key in der_keys
        key = key[1:manifold_dim]
        j = sum(key)
        der_idx = get_derivative_idx(key)

        # Loop over each space to extract the relevant basis function
        for i in 1:num_spaces
            space_index = num_spaces - i + 1
            key_sum = sum(key[keys_idx[space_index]]) + 1
            derivative_index = get_derivative_idx(key[keys_idx[space_index]])
            basis_functions_per_space[i] = local_basis_per_space[space_index][key_sum][derivative_index]
        end

        local_basis[j + 1][der_idx] = kron(basis_functions_per_space...)
    end

    return local_basis
end

"""
    evaluate(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}

For given global element id `element_id` for a tensor product finite element space, evaluate the local basis functions and return.

# Arguments
- `space::TensorProductSpace`: finite element space.
- `element_id::Int`: global element id.
- `xi::Vector{Float64}`: vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated.
- `nderivatives::Int`: number of derivatives to evaluate.

# Returns
- `::Vector{Vector{Matrix{Float64}}}`: array of evaluated global basis and derivatives.
- `::Vector{Int}`: vector of global basis indices (size: num_funcs).

# Notes
See also [`get_derivative_idx(der_key::Vector{Int})`] to understand how evaluations are stored
"""
function evaluate(tp_space::TensorProductSpace{manifold_dim, T}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}, nderivatives::Int) where {manifold_dim, num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    # Get basis indices for the current element
    basis_indices = get_basis_indices(tp_space, element_id)

    # Get the extraction coefficients per space
    extraction_per_space = _get_extraction_per_space(tp_space, element_id)

    # Compute local basis functions per space and their derivatives
    local_basis_per_space = _get_local_basis_per_space(tp_space, element_id, xi, nderivatives)

    # Generate keys for all possible derivative combinations
    der_keys = integer_sums(nderivatives, manifold_dim+1)

    # Split manifold dimensions of each constituent space
    manifold_dim_per_space = map(get_manifold_dim, tp_space.fem_spaces)
    cum_manifold_dim_per_space = cumsum((0, manifold_dim_per_space...))
    keys_idx = Vector{UnitRange{Int}}(undef, num_spaces)
    for i ∈ 1:num_spaces
        keys_idx[i] = cum_manifold_dim_per_space[i]+1:cum_manifold_dim_per_space[i]+manifold_dim_per_space[i]
    end

    # Compute tensor product of constituent basis functions for each derivative combination
    evaluation = Vector{Vector{Matrix{Float64}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        num_j_ders = binomial(manifold_dim + j - 1, manifold_dim - 1)
        evaluation[j + 1] = Vector{Matrix{Float64}}(undef, num_j_ders)
    end

    # Initialize an array to hold the evaluation for each space
    evaluation_per_space = Vector{Matrix{Float64}}(undef, num_spaces)

    for key in der_keys
        key = key[1:manifold_dim]
        j = sum(key)
        der_idx = get_derivative_idx(key)

        # Loop over each space to extract the relevant evaluation
        for i in 1:num_spaces
            space_index = num_spaces - i + 1
            key_sum = sum(key[keys_idx[space_index]]) + 1
            derivative_index = get_derivative_idx(key[keys_idx[space_index]])
            evaluation_per_space[i] = local_basis_per_space[space_index][key_sum][derivative_index] * extraction_per_space[space_index][1]
        end

        evaluation[j + 1][der_idx] = kron(evaluation_per_space...)
    end

    return evaluation, basis_indices
end

# Methods for tensor product B-spline spaces
include("BSplines.jl")
