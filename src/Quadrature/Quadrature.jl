module Quadrature

import FastGaussQuadrature
import FFTW

import ..Geometry

"""
    AbstractQuadratureRule{manifold_dim}

Abstract type for a quadrature rule on an entire domain of dimension `manifold_dim`.

# Type parameters
- `manifold_dim`: Dimension of the domain
"""
abstract type AbstractQuadratureRule{manifold_dim} end

"""
    AbstractElementQuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim}

A quadrature rule on a single quadrature element.
"""
abstract type AbstractElementQuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim} end

"""
    AbstractGlobalQuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim}

A quadrature rule on a global domain.
"""
abstract type AbstractGlobalQuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim} end

"""
    get_nodes(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}

Returns the quadrature nodes of a quadrature rule.

# Arguments
- `qr::AbstractElementQuadratureRule{manifold_dim}`: Rule to get the nodes from.

# Returns
- `nodes::NTuple{manifold_dim, Vector{Float64}}`: Nodes of the quadrature rule.
"""
function get_nodes(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.nodes
end

"""
    get_weights(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}

Returns the quadrature weights of a quadrature rule.

# Arguments
- `qr::AbstractElementQuadratureRule{manifold_dim}`: Rule to get the weights from.

# Returns
- `weights::Vector{Float64}`: Weights of the quadrature rule.
"""
function get_weights(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.weights
end

"""
    get_label(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}

Returns the label of a quadrature rule.

# Arguments
- `qr::AbstractElementQuadratureRule{manifold_dim}`: Rule to get the label from.

# Returns
- `rule_label::String`: Label of the quadrature rule.
"""
function get_label(qr::AbstractElementQuadratureRule{manifold_dim}) where {manifold_dim}
    return qr.rule_label
end

"""
    ElementQuadratureRule{manifold_dim} <: AbstractQuadratureRule{manifold_dim}

A quadrature rule for a specific element of dimension `manifold_dim`.
"""
struct ElementQuadratureRule{manifold_dim} <: AbstractElementQuadratureRule{manifold_dim}
    nodes::NTuple{manifold_dim, Vector{Float64}}
    weights::Vector{Float64}
    rule_label::String
end

"""
    CanonicalQuadratureRule{manifold_dim} <: ElementQuadratureRule{manifold_dim}

Represents a quadrature rule on a canonical element of dimension `manifold_dim`.

# Fields
- `nodes::NTuple{manifold_dim, Vector{Float64}}`: Quadrature nodes per dimension.
- `weights::Vector{Float64}`: Tensor product of quadrature rules. The shape is consistent
    with the output of the evaluate methods for `FunctionSpaces`.
- `rule_label::String`: Name or type of quadrature rule. Used for human verification.

# Type parameters
- `manifold_dim`: Dimension of the domain

# Inner Constructors
- `CanonicalQuadratureRule(nodes::NTuple{manifold_dim, Vector{Float64}}, weights::Vector{Float64})`:
    General constructor.

# Outer Constructors
- [`gauss_lobatto`](@ref).
- [`gauss_legendre`](@ref).
- [`clenshaw_curtis`](@ref).
- [`newton_cotes`](@ref).
- [`tensor_product_rule(p::NTuple{manifold_dim, Integer}, quad_rule::F, rule_args_1d...) where
    {manifold_dim, F <: Function}`](@ref).
- [`tensor_product_rule(qrules_1d::NTuple{manifold_dim, CanonicalQuadratureRule{1}}) where
    {manifold_dim}`](@ref).
"""
struct CanonicalQuadratureRule{manifold_dim} <: AbstractElementQuadratureRule{manifold_dim}
    nodes::NTuple{manifold_dim, Vector{Float64}}
    weights::Vector{Float64}
    rule_label::String
end


include("CanonicalQuadratureRules/CanonicalQuadratureRules.jl")
include("GlobalQuadratureRules/GlobalQuadratureRules.jl")

include("QuadratureHelpers.jl")

end
