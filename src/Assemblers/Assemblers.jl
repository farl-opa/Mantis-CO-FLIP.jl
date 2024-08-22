"""
    module Assemblers

Contains all assembly-related structs and functions.
"""
module Assemblers

import LinearAlgebra
import SparseArrays; const spa = SparseArrays

import .. Mesh
import .. FunctionSpaces


# This commented section are the previous inner product functions which 
# will be removed once the form setup works.

# # Helper functions for the computation of the L2 inner product. These 
# # should be generalised to take in arbitrary fields or forms, and should 
# # probably be moved to a different place.
# @doc raw"""
#     compute_inner_product_L2(jac_det, quad_weights, metric, f, g)

# Compute the ``L^2``-inner product between f and g.

# # Arguments
# - `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
# - `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
# - `metric<:AbstractArray{Float64, 3}`: metric tensor per quadrature node.
# - `f<:AbstractArray{Float64, 2}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
# - `g<:AbstractArray{Float64, 2}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
# """
# function compute_inner_product_L2(jac_det, quad_weights, metric, f, g)
#     result = 0.0
#     @inbounds for i in axes(metric, 3)
#         for j in axes(metric, 2)
#             for point_idx in axes(metric, 1)
#                 result += jac_det[point_idx] * quad_weights[point_idx] * f[j][point_idx] * metric[point_idx,j,i] * g[i][point_idx]
#             end
#         end
#     end

#     return result
# end

# # Special version for identity metric for which metric = LinearAlgebra.I
# @doc raw"""
#     compute_inner_product_L2(jac_det, quad_weights, metric::LinearAlgebra.UniformScaling{Bool}, f, g)

# Compute the ``L^2``-inner product between f and g.

# # Arguments
# - `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
# - `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
# - `metric::LinearAlgebra.UniformScaling{Bool}`: identity (or multiples thereof) metric tensors
# - `f<:AbstractArray{Float64, 2}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
# - `g<:AbstractArray{Float64, 2}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
# """
# function compute_inner_product_L2(jac_det, quad_weights, metric::LinearAlgebra.UniformScaling{Bool}, f, g)
#     result = 0.0
#     @inbounds for point_idx in eachindex(jac_det)
#         result += jac_det[point_idx] * quad_weights[point_idx] * f[point_idx] * metric * g[point_idx]
#     end

#     return result
# end



abstract type AbstractAssemblers end
abstract type AbstractInputs end
abstract type AbstractBoundaryForms end


include("BoundaryForms.jl")
include("WeakFormulations.jl")
include("GlobalAssemblers.jl")


end