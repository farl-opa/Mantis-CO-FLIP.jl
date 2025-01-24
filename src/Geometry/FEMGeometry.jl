
struct FEMGeometry{manifold_dim, F} <: AbstractGeometry{manifold_dim}
    geometry_coeffs::Array{Float64,2}
    fem_space::F
    n_elements::Int

    function FEMGeometry(fem_space::F, geometry_coeffs::Array{Float64,2}) where {
        manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}
    }
        n_elements = FunctionSpaces.get_num_elements(fem_space)
        return new{manifold_dim, F}(geometry_coeffs, fem_space, n_elements)
    end
end

function get_num_elements(geometry::FEMGeometry)
    return geometry.n_elements
end

function get_domain_dim(::FEMGeometry{manifold_dim, F}) where {manifold_dim, F}
    return manifold_dim
end

function get_image_dim(geometry::FEMGeometry)
    return size(geometry.geometry_coeffs)[2]
end

# evaluate in each direction at the specific points in the ntuple
function evaluate(
    geometry::FEMGeometry{manifold_dim, F},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}}
) where {manifold_dim, F}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(
        geometry.fem_space, element_id, xi, 0
    )
    
    num_eval_points = prod(size.(xi, 1))
    image_dim = get_image_dim(geometry)
    eval = zeros(Float64, num_eval_points, image_dim)
    
    for cartesian_id in CartesianIndices(fem_basis[1][1])
        (point, basis_id) = Tuple(cartesian_id)
        for dim in 1:image_dim
            eval[point, dim] += fem_basis[1][1][point, basis_id] *
                geometry.geometry_coeffs[fem_basis_indices[basis_id], dim]
        end
    end

    return eval
end

function jacobian(
    geometry::FEMGeometry{manifold_dim, F},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}}
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
    for cartesian_idx in CartesianIndices(J)
        (point, k_im, k_mani) = Tuple(cartesian_idx)
        for basis_id in eachindex(fem_basis_indices)
            J[point, k_im, k_mani] += fem_basis[2][der_idxs[k_mani]][point, basis_id] *
                geometry.geometry_coeffs[fem_basis_indices[basis_id], k_im]
        end
    end

    return J
end

function _get_element_size(geometry::FEMGeometry, element_id::Int) 
    return FunctionSpaces.get_element_size(geometry.fem_space, element_id)
end

function _get_element_dimensions(geometry::FEMGeometry, element_id::Int)
    return FunctionSpaces.get_element_dimensions(geometry.fem_space, element_id)
end

@doc raw"""
    compute_parametric_geometry(fem_space::FunctionSpaces.AbstractFiniteElementSpace)

Returns the parametric geometry associated with `fem_space` by computing the geometry
coefficients of the space.

# Arguments
- 'fem_space::FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}': Finite element space
for which to compute the geometry.

# Returns
- '::FEMGeometry{manifold_dim, F}': structure of the finite element geometry.
"""
function compute_parametric_geometry(fem_space::FunctionSpaces.AbstractFiniteElementSpace)
    geometry_coefficients = FunctionSpaces._compute_parametric_geometry_coeffs(fem_space)

    return FEMGeometry(fem_space, geometry_coefficients)
end
