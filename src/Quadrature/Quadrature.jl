@doc raw"""
    Quadrature

This module provides a collection of Quadrature rules. The quadrature rules are valid on the
interval [0, 1].

Numerical quadrature is the process of approximating the integral of a function by
evaluating the function at specific points and combining the evaluations with weight. The
points are called quadrature nodes and the weights are the corresponding weights for each
node. How the nodes and weights are chosen determines the accuracy of the quadrature rule
and varies per rule.

In general, a quadrature rule can be written as:
```math
\int_{0}^{1} f(x) dx \approx \sum_{i=1}^{N} w_i f(x_i)
```
where ``N`` is the number of quadrature nodes, ``w_i`` are the weights, and ``x_i`` are the
quadrature nodes. Note that the integral is computed on the interval [0, 1].

The exported names are:
"""
module Quadrature



import FastGaussQuadrature
import FFTW




@doc raw"""
    AbstractQuadratureRule{manifold_dim}

Abstract type for a quadrature rule on a domain of dimension `manifold_dim`.

# Type parameters
- `manifold_dim`: Dimension of the domain

# Concrete types
- `QuadratureRule{manifold_dim}`: See [`QuadratureRule`](@ref).
"""
abstract type AbstractQuadratureRule{manifold_dim} end



@doc raw"""
    QuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim}

Represents a quadrature rule on a domain of dimension `manifold_dim`.

# Fields
- `nodes::NTuple{manifold_dim, Vector{Float64}}`: Quadrature nodes per dimension.
- `weights::Vector{Float64}`: Tensor product of quadrature rules. The shape is consistent
    with the output of the evaluate methods for `FunctionSpaces`.
- `rule_label::String`: Name or type of quadrature rule. Used for human verification.

# Type parameters
- `manifold_dim`: Dimension of the domain

# Inner Constructors
- `QuadratureRule(nodes::NTuple{manifold_dim, Vector{Float64}}, weights::Vector{Float64})`:
    General constructor.

# Outer Constructors
- [`gauss_lobatto`](@ref).
- [`gauss_legendre`](@ref).
- [`clenshaw_curtis`](@ref).
- [`newton_cotes`](@ref).
- [`tensor_product_rule(p::NTuple{manifold_dim, Integer}, quad_rule::F, rule_args_1d...) where
    {manifold_dim, F <: Function}`](@ref).
- [`tensor_product_rule(qrules_1d::NTuple{manifold_dim, QuadratureRule{1}}) where
    {manifold_dim}`](@ref).
"""
struct QuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim}
    nodes::NTuple{manifold_dim, Vector{Float64}}
    weights::Vector{Float64}
    rule_label::String
end

@doc raw"""
    get_quadrature_nodes(qr::QuadratureRule{manifold_dim}) where {manifold_dim}

Returns the quadrature nodes of a quadrature rule.

# Arguments
- `qr::QuadratureRule{manifold_dim}`: Rule to get the nodes from.

# Returns
- `nodes::NTuple{manifold_dim, Vector{Float64}}`: Nodes of the quadrature rule.
"""
function get_quadrature_nodes(qr::QuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.nodes
end

@doc raw"""
    get_quadrature_weights(qr::QuadratureRule{manifold_dim}) where {manifold_dim}

Returns the quadrature weights of a quadrature rule.

# Arguments
- `qr::QuadratureRule{manifold_dim}`: Rule to get the weights from.

# Returns
- `weights::Vector{Float64}`: Weights of the quadrature rule.
"""
function get_quadrature_weights(qr::QuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.weights
end

@doc raw"""
    get_quadrature_rule_label(qr::QuadratureRule{manifold_dim}) where {manifold_dim}

Returns the label of a quadrature rule.

# Arguments
- `qr::QuadratureRule{manifold_dim}`: Rule to get the label from.

# Returns
- `rule_label::String`: Label of the quadrature rule.
"""
function get_quadrature_rule_label(qr::QuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.rule_label
end



# One-dimensional quadrature rules.
# Quadrature rules on non-equally spaced nodes.
include("Gauss.jl")
include("ClenshawCurtis.jl")

# Quadrature rules on equally spaced nodes.
include("NewtonCotes.jl")



# Multi-dimensional quadrature rules. The tensor product rules are made by combining one-
# dimensional rules, so this file must be included after the one-dimensional rules.
include("TensorProduct.jl")


end
