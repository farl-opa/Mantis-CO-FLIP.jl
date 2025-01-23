
struct FEMGeometry{manifold_dim, F} <: AbstractGeometry{manifold_dim}
    geometry_coeffs::Array{Float64,2}
    fem_space::F
    n_elements::Int

    function FEMGeometry(fem_space::F, geometry_coeffs::Array{Float64,2}) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
        n_elements = FunctionSpaces.get_num_elements(fem_space)
        return new{manifold_dim, F}(geometry_coeffs, fem_space, n_elements)
    end
end

function get_num_elements(geometry::FEMGeometry{manifold_dim, F}) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    return geometry.n_elements
end

function get_domain_dim(_::FEMGeometry{manifold_dim, F}) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    return manifold_dim
end

function get_image_dim(geometry::FEMGeometry{manifold_dim, F}) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    return size(geometry.geometry_coeffs)[2]
end

# evaluate in each direction at the specific points in the ntuple
function evaluate(geometry::FEMGeometry{manifold_dim, F}, element_id::Int, xi::NTuple{manifold_dim,Vector{Float64}}) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 0)
    
    # Combine basis functions with geometry coefficients and return
    return fem_basis[1][1] * geometry.geometry_coeffs[fem_basis_indices,:]
end

function jacobian(geometry::FEMGeometry{manifold_dim, F}, element_id::Int, xi::NTuple{manifold_dim,Vector{Float64}}) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    # Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 1)
    # combine with coefficients and return
    keys = Matrix{Int}(LinearAlgebra.I, manifold_dim, manifold_dim)
    n_eval_points = prod(length.(xi))
    range_dim = get_image_dim(geometry)
    J = zeros(n_eval_points, range_dim, manifold_dim)
    for k = 1:manifold_dim
        der_idx = FunctionSpaces.get_derivative_idx(keys[k,:])  # Get derivative index
        # Compute partial derivatives and store in Jacobian matrix
        J[:, :, k] .= fem_basis[2][der_idx] * geometry.geometry_coeffs[fem_basis_indices,:]
    end
    
    return J
end

function _get_element_size(geometry::FEMGeometry{manifold_dim, F}, element_id::Int) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    return FunctionSpaces.get_element_size(geometry.fem_space, element_id)
end

function _get_element_dimensions(geometry::FEMGeometry{manifold_dim, F}, element_id::Int) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    return FunctionSpaces.get_element_dimensions(geometry.fem_space, element_id)
end

# compute FEMGeometry for different FESpaces
@doc raw"""
    get_parametric_geometry(fem_space::F) where {F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim} where {manifold_dim}}

Returns the parametric geometry associated with `fem_space` by computing the geometry coefficients of the space.

# Arguments
- 'fem_space::FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}': Finite element space for which to compute the geometry.

# Returns
- '::FEMGeometry{manifold_dim, F}': structure of the finite element geometry.
"""
function compute_parametric_geometry(fem_space::F) where {manifold_dim, F<:FunctionSpaces.AbstractFiniteElementSpace{manifold_dim}}
    geometry_coefficients = FunctionSpaces._compute_parametric_geometry_coeffs(fem_space)

    return FEMGeometry(fem_space, geometry_coefficients)
end
