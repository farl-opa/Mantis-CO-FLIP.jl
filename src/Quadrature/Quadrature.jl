@doc raw"""
    Quadrature

This module provides a collection of Quadrature rules. The quadrature 
rules are valid on the interval [0, 1].

Numerical quadrature is the process of approximating the integral of a
function by evaluating the function at specific points and combining the
evaluations with weight. The points are called quadrature nodes and the 
weights are the corresponding weights for each node. How the nodes and
weights are chosen determines the accuracy of the quadrature rule and 
varies per rule.

In general, a quadrature rule can be written as:
```math
\int_{0}^{1} f(x) dx \approx \sum_{i=1}^{p} w_i f(x_i)
```
where `p` is the number of quadrature nodes, `w_i` are the weights, and
`x_i` are the quadrature nodes.

The exported names are:
"""
module Quadrature



import FastGaussQuadrature
import FFTW




@doc raw"""
    AbstractQuadratureRule{domain_dim}

Abstract type for a quadrature rule on a domain of dimension `domain_dim`.

# Type parameters
- `domain_dim`: Dimension of the domain

# Concrete types
- `QuadratureRule{domain_dim}`: See [`QuadratureRule`](@ref).
"""
abstract type AbstractQuadratureRule{domain_dim} end



@doc raw"""
    QuadratureRule{domain_dim} <: AbstractQuadratureRule{domain_dim}

Represents a quadrature rule on a domain of dimension `domain_dim`.

# Fields
- `nodes::NTuple{domain_dim, Vector{Float64}}`: Quadrature nodes per 
                                                dimension.
- `weights::Vector{Float64}`: Tensor product of quadrature rules. The 
                              shape is consistent with the output of the 
                              evaluate methods for FunctionSpaces.
- `rule_type::String`: Type of quadrature rule.

# Type parameters
- `domain_dim`: Dimension of the domain

# Inner Constructors
- `QuadratureRule(nodes::NTuple{domain_dim, Vector{Float64}}, 
                  weights::Vector{Float64})`: General constructor.

# Outer Constructors
- [`gauss_lobatto`](@ref).
- [`gauss_legendre`](@ref).
- [`clenshaw_curtis`](@ref).
- [`newton_cotes`](@ref).
- [`tensor_product_rule`](@ref).
- [`tensor_product_rule`](@ref).
"""
struct QuadratureRule{domain_dim} <: AbstractQuadratureRule{domain_dim}
    nodes::NTuple{domain_dim, Vector{Float64}}
    weights::Vector{Float64}
    rule_type::String
end

@doc raw"""
    get_quadrature_nodes(qr::QuadratureRule{domain_dim}) where {domain_dim}

Returns the quadrature nodes of a quadrature rule.

# Arguments
- `qr::QuadratureRule{domain_dim}`: Rule to get the nodes from.

# Returns
- `nodes::NTuple{domain_dim, Vector{Float64}}`: Nodes of the quadrature rule.
"""
function get_quadrature_nodes(qr::QuadratureRule{domain_dim}) where {domain_dim}
    return qr.nodes
end

@doc raw"""
    get_quadrature_weights(qr::QuadratureRule{domain_dim}) where {domain_dim}

Returns the quadrature weights of a quadrature rule.

# Arguments
- `qr::QuadratureRule{domain_dim}`: Rule to get the weights from.

# Returns
- `weights::Vector{Float64}`: Weights of the quadrature rule.
"""
function get_quadrature_weights(qr::QuadratureRule{domain_dim}) where {domain_dim}
    return qr.weights
end

@doc raw"""
    get_quadrature_rule_type(qr::QuadratureRule{domain_dim}) where {domain_dim}

Returns the type of a quadrature rule.

# Arguments
- `qr::QuadratureRule{domain_dim}`: Rule to get the type from.

# Returns
- `rule_type::String`: Type of the quadrature rule.
"""
function get_quadrature_rule_type(qr::QuadratureRule{domain_dim}) where {domain_dim}
    return qr.rule_type
end



# One-dimensional quadrature rules.
include("Gauss.jl")
include("ClenshawCurtis.jl")

# Quadrature rules on equally spaced nodes.
include("NewtonCotes.jl")



# Multi-dimensional quadrature rules. The tensor product rules are made
# by combining one-dimensional rules, so this file must be included 
# after the one-dimensional rules.
include("TensorProduct.jl")


end
