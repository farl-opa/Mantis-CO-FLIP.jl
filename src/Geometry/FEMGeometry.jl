
"""
    FEMGeometry{manifold_dim, F} <: AbstractGeometry{manifold_dim}

Geometry defined from a finite element space `fem_space` and a matrix of geometric
coefficients `geometry_coeffs`.

# Fields
- `geometry_coeffs::Matrix{Float64}`: The coefficients used to linearly combine the basis
    functions in `fem_space` to generate the geometry. The size of `geometry_coeffs` is
    `(num_basis, image_dim)`, where `num_basis` corresponds to the number of basis functions
    that span `fem_space` and `image_dim` the number of dimensions in the resulting
    geometry.
- `fem_space::F`: Finite element space used to define the geometry.
- `num_elements::Int`: The number of elements in the geometry, given by the number of
    elements in the finite element space.

# Type parameters
- `manifold_dim`: Dimension of the domain in `fem_space`.
- `F <: FunctionSpaces.AbstractFESpace{manifold_dim, 1}`: Underlying finite element space of
    the geometry.

# Inner Constructors
- `FEMGeometry(fem_space::F, geometry_coeffs::Matrix{Float64})`: Constructs the FEMGeometry
    from a finite element space `fem_space` and a set of pre-defined `geometry_coeffs`,
    deducing the number of elements from `fem_space`.

# Outer Constructors
- [`compute_parametric_geometry`](@ref).
"""
struct FEMGeometry{manifold_dim, F} <: AbstractGeometry{manifold_dim}
    geometry_coeffs::Matrix{Float64}
    fem_space::F
    num_elements::Int

    function FEMGeometry(
        fem_space::F, geometry_coeffs::Matrix{Float64}
    ) where {manifold_dim, F <: FunctionSpaces.AbstractFESpace{manifold_dim, 1}}
        num_elements = FunctionSpaces.get_num_elements(fem_space)

        return new{manifold_dim, F}(geometry_coeffs, fem_space, num_elements)
    end
end

"""
    compute_parametric_geometry(fem_space::FunctionSpaces.AbstractFESpace)

Returns the parametric geometry associated with `fem_space` by computing the geometry
coefficients of the space.

# Arguments
- `fem_space::FunctionSpaces.AbstractFESpace`: Finite element space
    for which to compute the geometry.

# Returns
- `::FEMGeometry{manifold_dim, F}`: structure of the finite element geometry.
"""
function compute_parametric_geometry(fem_space::FunctionSpaces.AbstractFESpace)
    geometry_coefficients = FunctionSpaces._compute_parametric_geometry_coeffs(fem_space)

    return FEMGeometry(fem_space, geometry_coefficients)
end

function get_element_measure(geometry::FEMGeometry, element_id::Int)
    return FunctionSpaces.get_element_size(geometry.fem_space, element_id)
end

function get_element_lengths(geometry::FEMGeometry, element_id::Int)
    return FunctionSpaces.get_element_dimensions(geometry.fem_space, element_id)
end

function get_image_dim(geometry::FEMGeometry)
    return size(geometry.geometry_coeffs)[2]
end

function evaluate(
    geometry::FEMGeometry{manifold_dim, F},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, F}
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(
        geometry.fem_space, element_id, xi, 0
    )
    length
    num_eval_points = prod(size.(xi, 1))
    image_dim = get_image_dim(geometry)
    eval = zeros(Float64, num_eval_points, image_dim)

    for dim in 1:image_dim
        for cartesian_id in CartesianIndices(fem_basis[1][1][1])
            (point, basis_id) = Tuple(cartesian_id)
            eval[point, dim] +=
                fem_basis[1][1][1][point, basis_id] *
                geometry.geometry_coeffs[fem_basis_indices[basis_id], dim]
        end
    end

    return eval
end

function jacobian(
    geometry::FEMGeometry{manifold_dim, F},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, F}
    # Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(
        geometry.fem_space, element_id, xi, 1
    )

    # Generate derivatives indices. For derivative order 1, each dimension is derivated
    # once. Then, the corresponding derivative index for the given key is computed.
    der_idxs = ntuple(manifold_dim) do k
        key = zeros(Int, manifold_dim)
        key[k] = 1
        der_idx = FunctionSpaces.get_derivative_idx(key)

        return der_idx
    end

    num_eval_points = prod(size.(xi, 1))
    image_dim = get_image_dim(geometry)
    J = zeros(num_eval_points, image_dim, manifold_dim)
    for basis_id in eachindex(fem_basis_indices)
        for cartesian_idx in CartesianIndices(J)
            (point, k_im, k_mani) = Tuple(cartesian_idx)
            J[point, k_im, k_mani] +=
                fem_basis[2][der_idxs[k_mani]][1][point, basis_id] *
                geometry.geometry_coeffs[fem_basis_indices[basis_id], k_im]
        end
    end

    return J
end
