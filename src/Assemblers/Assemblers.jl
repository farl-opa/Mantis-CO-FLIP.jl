"""
    module Assemblers

Contains all assembly-related structs and functions.
"""
module Assemblers

import LinearAlgebra
import SparseArrays; const spa = SparseArrays

import .. Mesh
import .. FunctionSpaces



# Some helper functions for the computation of inner products. At the 
# mooment, these work for the 1D case only. Extensions to 2D are almost 
# identical, but depend on how the functions take in their arguments.
@doc raw"""
    compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::F) where {F <: Function}

Compute the ``L^2``-inner product between f and g.

# Arguments
- `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
- `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
- `fxi<:Function`: one of the two function to compute the inner product with. To be evaluated at the quadrature nodes.
- `gxi<:Function`: one of the two function to compute the inner product with. To be evaluated at the quadrature nodes.
"""
function compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::F) where {F <: Function}
    result = 0.0
    @inbounds for node_idx in eachindex(quad_weights, quad_nodes)
        result += quad_weights[node_idx] * (fxi(quad_nodes[node_idx]) * gxi(quad_nodes[node_idx]))
    end
    return result
end

@doc raw"""
    compute_inner_product_L2(quad_nodes, quad_weights, fxi::T, gxi::T) where {T <: AbstractArray{Float64, 1}}

Compute the ``L^2``-inner product between f and g.

# Arguments
- `quad_nodes::Vector{Float64}`: (mapped) quadrature points. -- Not used ---
- `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
- `fxi<:AbstractArray{Float64, 1}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
- `gxi<:AbstractArray{Float64, 1}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
"""
function compute_inner_product_L2(quad_nodes, quad_weights, fxi::T1, gxi::T2) where {T1 <: AbstractArray{Float64, 1}, T2 <: AbstractArray{Float64, 1}}
    result = 0.0
    @inbounds for node_idx in eachindex(quad_weights, fxi, gxi)
        result += quad_weights[node_idx] * (fxi[node_idx] * gxi[node_idx])
    end
    return result
end

@doc raw"""
    compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::T) where {F <: Function, T <: AbstractArray{Float64, 1}

Compute the ``L^2``-inner product between f and g.

# Arguments
- `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
- `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
- `fxi<:Function`: one of the two function to compute the inner product with. To be evaluated at the quadrature nodes.
- `gxi<:AbstractArray{Float64, 1}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
"""
function compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::T) where {F <: Function, T <: AbstractArray{Float64, 1}}
    result = 0.0
    @inbounds for node_idx in eachindex(quad_weights, quad_nodes, gxi)
        result += quad_weights[node_idx] * (fxi(quad_nodes[node_idx, :]...) * gxi[node_idx])
    end
    return result
end

@doc raw"""
    compute_inner_product_L2(quad_nodes, quad_weights, fxi::T, gxi::F) where {F <: Function, T <: AbstractArray{Float64, 1}

Compute the ``L^2``-inner product between f and g.

# Arguments
- `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
- `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
- `fxi<:AbstractArray{Float64, 1}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
- `gxi<:Function`: one of the two function to compute the inner product with. To be evaluated at the quadrature nodes.
"""
function compute_inner_product_L2(quad_nodes, quad_weights, fxi::T, gxi::F) where {F <: Function, T <: AbstractArray{Float64, 1}}
    result = 0.0
    @inbounds for node_idx in eachindex(quad_weights, quad_nodes, fxi)
        result += quad_weights[node_idx] * (fxi[node_idx] * gxi(quad_nodes[node_idx]))
    end
    return result
end






function compute_vec_inner_product_L2(jac_det, quad_weights, metric, f, g)
    result = 0.0
    @inbounds for i in axes(metric, 3)
        for j in axes(metric, 2)
            for point_idx in axes(metric, 1)
                result += jac_det[point_idx] * quad_weights[point_idx] * f[j][point_idx] * metric[point_idx,j,i] * g[i][point_idx]
            end
        end
    end

    return result
end

# Special version for identity metric for which metric = LinearAlgebra.I
function compute_vec_inner_product_L2(jac_det, quad_weights, metric::LinearAlgebra.UniformScaling{Bool}, f, g)
    result = 0.0
    @inbounds for point_idx in eachindex(jac_det)
        result += jac_det[point_idx] * quad_weights[point_idx] * f[point_idx] * metric * g[point_idx]
    end

    return result
end


# function compute_vec_inner_product_L2(quad_nodes, quad_weights, metric, fxi, gxi)
#     result = 0.0
#     product = zeros(size(fxi[1]))
#     for i in 1:1:length(product)
#         product[i] = transpose([fxi[1][i], fxi[2][i]]) * metric[i,:,:] * [gxi[1][i], gxi[2][i]]
#     end
#     #display(product)
#     @inbounds for node_idx in eachindex(quad_weights, product)
#         result += quad_weights[node_idx] * product[node_idx]
#     end
#     return result
# end

function compute_novec_inner_product_L2(quad_nodes, quad_weights::T1, fxi::T2, gxi::T3) where {T1 <: AbstractArray{Float64, 1}, T2 <: AbstractArray{Float64, 1}, T3 <: AbstractArray{Float64, 1}}
    result = 0.0
    @inbounds for node_idx in eachindex(quad_weights, fxi, gxi)
        result += quad_weights[node_idx] * fxi[node_idx] * gxi[node_idx]
    end
    return result
end



abstract type AbstractAssemblers end
abstract type AbstractBilinearForms end
abstract type AbstractBoundaryForms end


include("BilinearForms.jl")
include("BoundaryForms.jl")
include("GlobalAssemblers.jl")


end