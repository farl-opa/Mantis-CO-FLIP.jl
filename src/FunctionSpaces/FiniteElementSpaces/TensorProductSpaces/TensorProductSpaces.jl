# TODO: rewrite this file to use Cartesian Indices
"""
    TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces} <: AbstractFESpace{manifold_dim, num_components, num_patches}

A structure representing a tensor product space composed of multiple finite element spaces.
The functions in this space are `manifold_dim`-variate where `manifold_dim` is equal to the
sum of the manifold dimensions of the input spaces. When the input spaces have multiple
components, the tensor product is computed component-wise.

# Fields
- `constituent_spaces::T`: A tuple of finite element spaces.
- `dof_partition::Vector{Vector{Vector{Int}}}`: A nested vector structure representing the
    degree of freedom partitioning.
- `data::Dict`: A dictionary to store additional data.

# Constructors
    TensorProductSpace(constituent_spaces::T, data::Dict) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    TensorProductSpace(constituent_spaces::T) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
"""
struct TensorProductSpace{
    manifold_dim, num_components, num_patches, num_spaces, T, CIE, LIE, CIB, LIB
} <: AbstractFESpace{manifold_dim, num_components, num_patches}
    constituent_spaces::T
    cart_num_elements::CIE
    lin_num_elements::LIE
    cart_num_basis::CIB
    lin_num_basis::LIB
    dof_partition::Vector{Vector{Vector{Int}}}

    # TODO!!! NTuple implies they are all the same. Not true!
    function TensorProductSpace(
        constituent_spaces::T
    ) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        if all(get_num_components.(constituent_spaces) .== 1)
            num_components = 1
        else
            throw(ArgumentError("All spaces must have only one component."))
        end

        # Tensor-product space
        manifold_dim = sum(get_manifold_dim, constituent_spaces)
        num_patches = prod(get_num_patches, constituent_spaces)
        # Pre-allocate memory for degree of freedom partitioning
        dof_partition = Vector{Vector{Vector{Int}}}(undef, num_patches)
        # Constituent spaces
        const_dof_partitions = ntuple(
            space -> get_dof_partition(constituent_spaces[space]), num_spaces
        )
        const_num_patches = ntuple(space -> length(const_dof_partitions[space]), num_spaces)
        const_num_basis = ntuple(
            space -> get_num_basis(constituent_spaces[space]), num_spaces
        )
        const_num_elements = ntuple(
            space -> get_num_elements(constituent_spaces[space]), num_spaces
        )
        cart_num_elements = CartesianIndices(const_num_elements)
        lin_num_elements = LinearIndices(const_num_elements)
        cart_num_basis = CartesianIndices(const_num_basis)
        lin_num_basis = LinearIndices(const_num_basis)
        # Loop over all spaces and build the appropriate index subsets
        for (patch_count, patch_ids) in enumerate(CartesianIndices(const_num_patches))
            const_patches = ntuple(
                space -> const_dof_partitions[space][patch_ids[space]], num_spaces
            )
            const_partition_lengths = ntuple(
                space -> length(const_patches[space]), num_spaces
            )
            dof_partition[patch_count] = Vector{Vector{Int}}(
                undef, prod(const_partition_lengths)
            )
            for (partition_count, partition_ids) in
                enumerate(CartesianIndices(const_partition_lengths))
                const_partition_dofs = ntuple(
                    space -> const_patches[space][partition_ids[space]], num_spaces
                )
                dof_partition[patch_count][partition_count] = Vector{Int}(
                    undef, prod(length, const_partition_dofs)
                )
                for (dof_count, const_basis_id) in
                    enumerate(Iterators.product(const_partition_dofs...))
                    basis_id = lin_num_basis[const_basis_id...]
                    dof_partition[patch_count][partition_count][dof_count] = basis_id
                end
            end
        end

        return new{
            manifold_dim,
            num_components,
            num_patches,
            num_spaces,
            T,
            typeof(cart_num_elements),
            typeof(lin_num_elements),
            typeof(cart_num_basis),
            typeof(lin_num_basis),
        }(
            constituent_spaces,
            cart_num_elements,
            lin_num_elements,
            cart_num_basis,
            lin_num_basis,
            dof_partition,
        )
    end
end

# Getters

get_cart_num_basis(space::TensorProductSpace) = space.cart_num_basis
get_lin_num_basis(space::TensorProductSpace) = space.lin_num_basis
get_cart_num_elements(space::TensorProductSpace) = space.cart_num_elements
get_lin_num_elements(space::TensorProductSpace) = space.lin_num_elements
get_constituent_spaces(space::TensorProductSpace) = space.constituent_spaces
get_num_basis(space::TensorProductSpace) = prod(get_constituent_num_basis(space))

function get_num_elements(space::TensorProductSpace)
    return prod(get_constituent_num_elements(space))
end

function get_num_spaces(
    ::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    return num_spaces
end

function get_constituent_space(space::TensorProductSpace, space_id::Int)
    return get_constituent_spaces(space)[space_id]
end

function get_constituent_element_id(space::TensorProductSpace, element_id::Int)
    return Tuple(get_cart_num_elements(space)[element_id])
end

function get_constituent_basis_id(space::TensorProductSpace, basis_id::Int)
    return Tuple(get_cart_num_basis(space)[basis_id])
end

function get_constituent_num_basis(space::TensorProductSpace)
    return Tuple(maximum(get_cart_num_basis(space)))
end

function get_constituent_num_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id = get_constituent_element_id(space, element_id)
    const_num_basis = ntuple(
        space -> get_num_basis(const_spaces[space], const_element_id[space]), num_spaces
    )

    return const_num_basis
end

function get_num_basis(space::TensorProductSpace, element_id::Int)
    return prod(get_constituent_num_basis(space, element_id))
end

function get_constituent_num_elements(space::TensorProductSpace)
    return Tuple(maximum(get_cart_num_elements(space)))
end

function get_constituent_manifold_dim(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)

    return ntuple(space -> get_manifold_dim(const_spaces[space]), num_spaces)
end

function get_constituent_basis_indices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id = get_constituent_element_id(space, element_id)
    const_basis_indices = ntuple(
        space -> get_basis_indices(const_spaces[space], const_element_id[space]), num_spaces
    )

    return const_basis_indices
end

function get_constituent_support(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_basis_id = get_constituent_basis_id(space, basis_id)
    const_spaces = get_constituent_spaces(space)
    const_support = ntuple(
        space -> get_support(const_spaces[space], const_basis_id[space]), num_spaces
    )

    return const_support
end

function get_constituent_extraction(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_element_id = get_constituent_element_id(space, element_id)
    const_spaces = get_constituent_spaces(space)
    const_extraction = ntuple(
        space -> get_extraction(const_spaces[space], const_element_id[space]), num_spaces
    )

    return const_extraction
end

function get_constituent_polynomial_degree(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_polynomial_degree = ntuple(
        space -> get_polynomial_degree(const_spaces[space]), num_spaces
    )

    return const_polynomial_degree
end

function get_constituent_local_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id = get_constituent_element_id(space, element_id)
    const_xi = get_constituent_evaluation_points(space, xi)
    const_local_basis = ntuple(
        space -> get_local_basis(
            const_spaces[space], const_element_id[space], const_xi[space], nderivatives
        ),
        num_spaces,
    )

    return const_local_basis
end

function get_constituent_evaluation_points(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_manifold_indices = get_constituent_manifold_indices(space)
    const_xi = ntuple(space -> xi[const_manifold_indices[space]], num_spaces)

    return const_xi
end

function get_constituent_manifold_indices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_manifold_dim = get_constituent_manifold_dim(space)
    cum_const_manifold_dim = (0, cumsum(const_manifold_dim)...)
    const_manifold_indices = ntuple(
        space -> (cum_const_manifold_dim[space] + 1):cum_const_manifold_dim[space + 1],
        num_spaces,
    )

    return const_manifold_indices
end

function get_constituent_element_vertices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id = get_constituent_element_id(space, element_id)
    const_element_vertices = ntuple(
        space -> get_element_vertices(const_spaces[space], const_element_id[space]),
        num_spaces,
    )

    return const_element_vertices
end

function get_constituent_element_lengths(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id = get_constituent_element_id(space, element_id)
    const_element_lengths = ntuple(
        space -> get_element_lengths(const_spaces[space], const_element_id[space]),
        num_spaces,
    )

    return const_element_lengths
end

function get_basis_indices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_basis_indices = get_constituent_basis_indices(space, element_id)
    lin_num_basis = get_lin_num_basis(space)
    basis_indices = Vector{Int}(undef, prod(length, const_basis_indices))
    for (basis_count, const_basis_id) in
        enumerate(Iterators.product(const_basis_indices...))
        basis_indices[basis_count] = lin_num_basis[const_basis_id...]
    end

    return basis_indices
end

function get_support(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_support = get_constituent_support(space, basis_id)
    lin_num_elements = get_lin_num_elements(space)
    support = Vector{Int}(undef, prod(length, const_support))
    for (element_count, const_element_id) in enumerate(Iterators.product(const_support...))
        support[element_count] = lin_num_elements[const_element_id...]
    end

    return support
end

function get_element_vertices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_element_vertices = get_constituent_element_vertices(space, element_id)
    const_manifold_dim = get_constituent_manifold_dim(space)
    cum_const_manifold_dim = (0, cumsum(const_manifold_dim)...)
    element_vertices = ntuple(manifold_dim) do dim
        const_space_id = findfirst(
            cum_manifold_dim -> dim ≤ cum_manifold_dim, cum_const_manifold_dim[2:end]
        )
        const_dim_id = dim - cum_const_manifold_dim[const_space_id]

        return const_element_vertices[const_space_id][const_dim_id]
    end

    return element_vertices
end

function get_element_lengths(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_element_lengths = get_constituent_element_lengths(space, element_id)
    const_manifold_dim = get_constituent_manifold_dim(space)
    cum_const_manifold_dim = (0, cumsum(const_manifold_dim)...)
    element_lengths = ntuple(manifold_dim) do dim
        const_space_id = findfirst(
            cum_manifold_dim -> dim ≤ cum_manifold_dim, cum_const_manifold_dim[2:end]
        )
        const_dim_id = dim - cum_const_manifold_dim[const_space_id]

        return const_element_lengths[const_space_id][const_dim_id]
    end

    return element_lengths
end

function get_max_local_dim(space::TensorProductSpace)
    return prod(get_max_local_dim, get_constituent_spaces(space))
end

function get_extraction(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches, num_spaces}
    extraction_per_space = get_constituent_extraction(space, element_id)
    extraction_coeffs = kron(
        (extraction_per_space[space][1] for space in num_spaces:-1:1)...
    )

    return extraction_coeffs, 1:size(extraction_coeffs, 1)
end

function get_local_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int=0,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_local_basis = get_constituent_local_basis(space, element_id, xi, nderivatives)
    const_sizes = ntuple(space -> size(const_local_basis[space][1][1][1]), num_spaces)
    num_points = prod(length, xi)
    num_basis = get_num_basis(space, element_id)
    local_basis = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = binomial(manifold_dim + der_order - 1, manifold_dim - 1)
        # We assume that there is only one component.
        local_basis[der_order + 1] = [[zeros(num_points, num_basis)] for _ in 1:num_der_ids]
    end

    der_keys = integer_sums(nderivatives, manifold_dim + 1)
    const_manifold_indices = get_constituent_manifold_indices(space)
    const_eval = [zeros(const_sizes[space]) for space in 1:num_spaces]
    for key in der_keys
        key = key[1:manifold_dim]
        der_order = sum(key)
        der_id = get_derivative_idx(key)
        for space in 1:num_spaces
            space_der_order = sum(key[const_manifold_indices[space]])
            space_der_id = get_derivative_idx(key[const_manifold_indices[space]])
            curr_eval = const_local_basis[space][space_der_order + 1][space_der_id][1]
            const_eval[space] .= curr_eval
        end

        if manifold_dim == 1
            local_basis[der_order + 1][der_id][1] .= const_eval[1]
        else
            local_basis[der_order + 1][der_id][1] .= kron(const_eval[end:-1:1]...)
        end
    end

    return local_basis
end

function evaluate(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    basis_indices = get_basis_indices(space, element_id)
    num_basis = length(basis_indices)
    const_local_basis = get_constituent_local_basis(space, element_id, xi, nderivatives)
    const_sizes = ntuple(space -> size(const_local_basis[space][1][1][1]), num_spaces)
    num_points = prod(length, xi)
    eval = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = binomial(manifold_dim + der_order - 1, manifold_dim - 1)
        # We assume that there is only one component.
        eval[der_order + 1] = [
            [Matrix{Float64}(undef, (num_points, num_basis))] for _ in 1:num_der_ids
        ]
    end

    der_keys = integer_sums(nderivatives, manifold_dim + 1)
    const_manifold_indices = get_constituent_manifold_indices(space)
    const_extraction = get_constituent_extraction(space, element_id)
    const_eval = ntuple(space -> zeros(const_sizes[space]), num_spaces)
    for key in der_keys
        key = key[1:manifold_dim]
        der_order = sum(key)
        der_id = get_derivative_idx(key)
        for space in 1:num_spaces
            space_der_order = sum(key[const_manifold_indices[space]])
            space_der_id = get_derivative_idx(key[const_manifold_indices[space]])
            LinearAlgebra.mul!(
                const_eval[space],
                const_local_basis[space][space_der_order + 1][space_der_id][1],
                const_extraction[space][1],
            )
        end

        if manifold_dim == 1
            eval[der_order + 1][der_id][1] = const_eval[1]
        else
            eval[der_order + 1][der_id][1] = kron(
                (const_eval[space] for space in num_spaces:-1:1)...
            )
        end
    end

    return eval, basis_indices
end

# Methods for tensor product B-spline spaces
include("BSplines.jl")
