import LinearAlgebra

struct FEMGeometry{n, F} <: AbstractGeometry{n}
    geometry_coeffs::Array{Float64,2}
    fem_space::F
    n_elements::Int

    function FEMGeometry(fem_space::F, geometry_coeffs::Array{Float64,2}) where {n, F<:FunctionSpaces.AbstractFiniteElementSpace{n}}
        n_elements = FunctionSpaces.get_num_elements(fem_space)
        return new{n,F}(geometry_coeffs, fem_space, n_elements)
    end
end

function get_num_elements(geometry::FEMGeometry{n, F}) where {n, F<:FunctionSpaces.AbstractFiniteElementSpace{n}}
    return geometry.n_elements
end

function get_domain_dim(_::FEMGeometry{n, F}) where {n, F<:FunctionSpaces.AbstractFiniteElementSpace{n}}
    return n
end

function get_image_dim(geometry::FEMGeometry{n, F}) where {n, F<:FunctionSpaces.AbstractFiniteElementSpace{n}}
    return size(geometry.geometry_coeffs)[2]
end

# evaluate in each direction at the specific points in the ntuple
function evaluate(geometry::FEMGeometry{n, F}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n, F<:FunctionSpaces.AbstractFiniteElementSpace{n}}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 0)
    # combine with coefficients and return
    key = Tuple(zeros(Int,n))
    return fem_basis[key...] * geometry.geometry_coeffs[fem_basis_indices,:]
end

function jacobian(geometry::FEMGeometry{n, F}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n, F<:FunctionSpaces.AbstractFiniteElementSpace{n}}
    # Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 1)
    # combine with coefficients and return
    keys = Matrix{Int}(LinearAlgebra.I,n,n)
    n_eval_points = prod(length.(xi))
    m = get_image_dim(geometry)
    J = zeros(n_eval_points, m, n)
    for k = 1:n 
        J[:, :, k] .= fem_basis[Tuple(keys[k,:])...] * geometry.geometry_coeffs[fem_basis_indices,:]
    end
    return J #, fem_basis[Tuple(zeros(Float64,n))...] * geometry.geometry_coeffs[fem_basis_indices,:]
end

function _get_element_measure(geometry::FEMGeometry{n, F}, element_id::Int) where {n, F<:FunctionSpaces.AbstractFiniteElementSpace{n}}
    return FunctionSpaces._get_element_measure(geometry.fem_space, element_id)
end

# compute FEMGeometry for different FESpaces
@doc raw"""
    compute_geometry(fem_space::F) where {F<:FunctionSpaces.AbstractFiniteElementSpace{n} where {n}}

Returns the geometry associated with `fem_space` by computing the geometry coefficients of the space.

# Arguments
- 'fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}': Finite element space for which to compute the geometry.

# Returns
- '::FEMGeometry{n, F}': structure of the finite element geometry.
"""
function compute_geometry(fem_space::F) where {F<:FunctionSpaces.AbstractFiniteElementSpace{n} where {n}}
    geometry_coefficients = FunctionSpaces._compute_geometry_coeffs(fem_space)

    return FEMGeometry(fem_space, geometry_coefficients)
end