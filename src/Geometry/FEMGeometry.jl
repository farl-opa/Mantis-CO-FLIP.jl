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
    key = Tuple(zeros(Int,n))
    return fem_basis[key...] * geometry.geometry_coeffs[fem_basis_indices,:]
end

import LinearAlgebra

function jacobian(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 1)
    # combine with coefficients and return
    keys = Matrix{Int}(LinearAlgebra.I,n,n)
    n_eval_points = prod(length.(xi))
    J = zeros(n_eval_points, m, n)
    for k = 1:n 
        J[:, :, k] .= fem_basis[Tuple(keys[k,:])...] * geometry.geometry_coeffs[fem_basis_indices,:]
    end
    return J #, fem_basis[Tuple(zeros(Float64,n))...] * geometry.geometry_coeffs[fem_basis_indices,:]
end

function metric(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    J = jacobian(geometry, element_id, xi)
    n_eval_points = prod(length.(xi))
    g = zeros(n_eval_points, n, n)
    sqrt_g = zeros(n_eval_points)
    for i = 1:n_eval_points
        g[i, :, :] .= reshape(J[i, :, :],m,n)' * reshape(J[i, :, :],m,n)
        sqrt_g[i] = LinearAlgebra.det(reshape(g[i, :, :],n,n))
    end
    return g, sqrt_g
end

function inv_metric(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    J = jacobian(geometry, element_id, xi)
    n_eval_points = prod(length.(xi))
    inv_g = zeros(n_eval_points, n, n)
    sqrt_g = zeros(n_eval_points)
    for i = 1:n_eval_points
        inv_g[i, :, :] .= inv(reshape(J[i, :, :],m,n)' * reshape(J[i, :, :],m,n))
        sqrt_g[i] = 1.0/LinearAlgebra.det(reshape(inv_g[i, :, :],n,n))
    end
    return inv_g, sqrt_g
end