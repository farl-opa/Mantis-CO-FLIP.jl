import LinearAlgebra

struct FEMGeometry{n,m} <: AbstractGeometry{n, m}
    geometry_coeffs::Array{Float64,2}
    fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}
    n_elements::Int

    function FEMGeometry(fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}, geometry_coeffs::Array{Float64,2}) where {n}
        m = size(geometry_coeffs,2)
        n_elements = FunctionSpaces.get_num_elements(fem_space)
        return new{n,m}(geometry_coeffs, fem_space, n_elements)
    end
end

function get_num_elements(geometry::FEMGeometry{n,m}) where {n,m}
    return geometry.n_elements
end

function get_domain_dim(::FEMGeometry{n,m}) where {n, m}
    return n
end

function get_image_dim(::FEMGeometry{n,m}) where {n, m}
    return m
end

# evaluate in each direction at the specific points in the ntuple
function evaluate(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 0)
    # combine with coefficients and return
    return fem_basis[1][1] * geometry.geometry_coeffs[fem_basis_indices,:]
end

import LinearAlgebra

function jacobian(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # Jᵢⱼ = ∂Φⁱ\∂ξⱼ
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 1)
    # combine with coefficients and return
    keys = Matrix{Int}(LinearAlgebra.I,n,n)
    n_eval_points = prod(length.(xi))
    J = zeros(n_eval_points, m, n)
    for k = 1:n 
        der_idx = FunctionSpaces._get_derivative_idx(keys[k,:])
        J[:, :, k] .= fem_basis[2][der_idx] * geometry.geometry_coeffs[fem_basis_indices,:]
    end
    return J
end