"""
    Quadrature

This module provides a collection of (Gauss) Quadrature rules. The 
defined quadrature rules in the interval [0, 1].

The exported names are:
"""
module Quadrature



import FastGaussQuadrature
import FFTW





abstract type AbstractQuadratureRule{domain_dim} end



@doc raw"""
    QuadratureRule{domain_dim} <: AbstractQuadratureRule{domain_dim}

Represents a quadrature rule on a domain of dimension `domain_dim`.

# Fields
- `nodes::NTuple{domain_dim, Vector{Float64}}`: Quadrature nodes per dimension.
- `weights::Vector{Float64}`: Tensor product of quadrature rules. The shape is consistent with the output of the evaluate methods for Mantis.FunctionSpaces

# Type parameters
- `domain_dim`: Dimension of the domain

# Inner Constructors
- `QuadratureRule(nodes::NTuple{domain_dim, Vector{Float64}}, weights::Vector{Float64})`: General constructor.

# Outer Constructors
- [`gauss_lobatto(p::Integer)`](@ref).
- [`gauss_legendre(p::Integer)`](@ref).
- [`tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F) where {domain_dim, F <: Function}`](@ref).
- [`tensor_product_rule(qrules_1d::NTuple{domain_dim, QuadratureRule{domain_dim}}) where {domain_dim}`](@ref).
"""
struct QuadratureRule{domain_dim} <: AbstractQuadratureRule{domain_dim}
    nodes::NTuple{domain_dim, Vector{Float64}}
    weights::Vector{Float64}
end

@doc raw"""
    get_quadrature_nodes(qr::QuadratureRule{domain_dim}) where {domain_dim}

Returns the quadrature nodes from a quadrature rule

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

Returns the quadrature weights from a quadrature rule

# Arguments
- `qr::QuadratureRule{domain_dim}`: Rule to get the weights from.

# Returns
- `weights::Vector{Float64}`: Weights of the quadrature rule.
"""
function get_quadrature_weights(qr::QuadratureRule{domain_dim}) where {domain_dim}
    return qr.weights
end



# Gauss quadrature rules in 1D.
@doc raw"""
    gauss_lobatto(p::Integer)

Computes the nodes `ξ` and weights `w` of 
[Gauss-Lobatto quadrature](https://mathworld.wolfram.com/LobattoQuadrature.html).

Note that here the quadrature is valid for the interval ξ ∈ [0, 1], 
instead of ξ ∈ [-1, 1] as usual.

```math
\int_{0}^{1} f(x) dx \approx \sum_{i=1}^{p} w_i f(x_i)
```

# Arguments
- `p::Int`: Degree of the quadrature rule.

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule contains the nodes and weights.
"""
function gauss_lobatto(p::Integer)
    ξ, w = FastGaussQuadrature.gausslobatto(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return QuadratureRule{1}((ξ,), w)
end

@doc raw"""
    gauss_legendre(p::Integer)

Computes the nodes `ξ` and weights `w` of 
[Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).

Note that here the quadrature is valid for the interval ξ ∈ [0, 1], 
instead of ξ ∈ [-1, 1] as usual.

```math
\int_{0}^{1} f(x) dx \approx \sum_{i=1}^{p} w_i f(x_i)
```

# Arguments
- `p::Int`: Degree of the quadrature rule.

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule contains the nodes and weights.
"""
function gauss_legendre(p::Integer)
    ξ, w = FastGaussQuadrature.gausslegendre(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return QuadratureRule{1}((ξ,), w)
end



# Clenshaw-Curtis quadrature rules in 1D.
@doc raw"""
    clenshaw_curtis(p::Integer)
    
Find the roots and weights for Clenshaw-Curtis quadrature on the 
interval [0, 1]. The roots include the endpoints -1 and 1 and 
Clenshaw-Curtis quadrature is exact for polynomials up to degree 
`p`. However, in practise, this quadrature formula achieves results
that can be comparable to Gauss quadrature (in some cases), see [Trefethen2008](@cite).

# Arguments
- `p::Int`: Degree of the quadrature rule.

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule containing the nodes and weights.

# Notes
See [Waldvogel2006](@cite) for the algorithm.
"""
function clenshaw_curtis(p::Integer)
    if p <= 1
        throw(ArgumentError("Invalid degree: $p. The degree must be greater than 1."))
    end
    
    N = collect(1:2:p-1)
    l = length(N)
    m = p - l

    v0 = vcat(2.0./N./(N.-2), 1/N[end], zeros(m))
    v2 = -v0[1:end-1] .- v0[end:-1:2]

    g0 = -ones(p)
    g0[l+1] = g0[l+1] + p
    g0[m+1] = g0[m+1] + p
    g = g0 ./ (p^2 - 1 + p%2)

    w = real(FFTW.ifft(v2.+g))
    w = vcat(w, w[1])

    rts = cospi.(collect(Float64, range(0, p)) ./ p)
    ξ = rts[end:-1:begin]

    # Map roots and weights to the interval [0, 1].
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w

    return QuadratureRule{1}((ξ,), w)
end


# Quadrature rules on equally spaced nodes.
@doc raw"""
    newton_cotes(num_points::Integer, type::String="closed")

Computes the nodes `ξ` and weights `w` of a Newton-Cotes quadrature rule.

# Arguments
- `num_points::Int`: Number of points in the quadrature rule.
- `type::String`: Type of the Newton-Cotes rule. Valid types are "closed" and "open".

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule containing the nodes and weights.
"""
function newton_cotes(num_points::Integer, type::String="closed")
    # Compute the equally spaced nodes on the interval [-1, 1].
    if type == "closed"
        if num_points <= 1
            throw(ArgumentError("Invalid number of points: $num_points. A closed Newton-Cotes rule requires at least 2 points."))
        else
            ξ = [-1.0 + i * 2.0 / (num_points-1) for i = 0:num_points-1]
        end
    elseif type == "open"
        if num_points <= 0
            throw(ArgumentError("Invalid number of points: $num_points. An open Newton-Cotes rule requires at least 1 point."))
        else
            ξ = [-1.0 + i * 2.0 / (num_points+1) for i = 1:num_points]
        end
    else
        throw(ArgumentError("Invalid Newton-Cotes type: $type. Valid types are 'closed' and 'open'."))
    end

    # Compute the weights by integrating the Lagrange basis functions.
    # We need n Lagrange basis functions when given n points.
    ql, wl = FastGaussQuadrature.gausslegendre(num_points)

    lagrange_at_ql = zeros(Float64, num_points, length(ql))
    for i in 1:num_points
        for j in eachindex(ql)
            l_poly_i = 1.0
            for k in 1:num_points
                if k != i
                    l_poly_i *= (ql[j] - ξ[k]) / (ξ[i] - ξ[k])
                end
            end
            lagrange_at_ql[i,j] = l_poly_i
        end
    end

    w = zeros(Float64, num_points)
    for i in 1:num_points
        for j in eachindex(ql, wl)
            w[i] += wl[j] * lagrange_at_ql[i,j]
        end
        
    end

    # Map roots and weights to the interval [0, 1].
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w

    return QuadratureRule{1}((ξ,), w)
    
end



# Multi-dimensional quadrature rules.
@doc raw"""
    tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F) where {domain_dim, F <: Function}

Returns a tensor product quadrature rule of given degree and rule type.

# Arguments
- `p::NTuple{domain_dim, Int}`: Degree of the quadrature rule per dimension.
- `quad_rule::F`: The function that returns quadrature nodes and weights given an integer degree.

# Returns
- `::QuadratureRule{domain_dim}`: QuadratureRule of the new dimension.
"""
function tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F) where {domain_dim, F <: Function}
    # Compute the nodes and weights per dimensions for given rule type 
    # and degree.
    points = NTuple{domain_dim, Vector{Float64}}(get_quadrature_nodes(quad_rule(p[k]))[1] for k = 1:domain_dim)
    weights_1d = NTuple{domain_dim, Vector{Float64}}(get_quadrature_weights(quad_rule(p[k])) for k = 1:domain_dim)
    
    # Compute the tensor product of the quadrature weights.
    weights = Vector{Float64}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end
    
    return QuadratureRule{domain_dim}(points, weights)
end

@doc raw"""
    tensor_product_rule(qrules_1d::NTuple{domain_dim, QuadratureRule{domain_dim}}) where {domain_dim}

Returns a tensor product quadrature rule from the given rules.

# Arguments
- `qrules_1d::NTuple{domain_dim, QuadratureRule{domain_dim}}`: Quadrature rules per dimension.

# Returns
- `::QuadratureRule{domain_dim}`: QuadratureRule of the new dimension.
"""
function tensor_product_rule(qrules_1d::NTuple{domain_dim, QuadratureRule{domain_dim}}) where {domain_dim}
    # Compute the nodes and weights per dimensions for given rule type 
    # and degree.
    points = NTuple{domain_dim, Vector{Float64}}(get_quadrature_nodes(qrules_1d[k])[1] for k = 1:domain_dim)
    weights_1d = NTuple{domain_dim, Vector{Float64}}(get_weights_nodes(qrules_1d[k]) for k = 1:domain_dim)
    
    # Compute the tensor product of the quadrature weights.
    weights = Vector{Float64}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end
    
    return QuadratureRule{domain_dim}(points, weights)
end


end
