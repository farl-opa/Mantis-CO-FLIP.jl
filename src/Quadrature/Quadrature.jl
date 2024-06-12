"""
This (sub-)module provides a collection of Gauss Quadrature rules.

The exported names are:
"""
module Quadrature

struct QuadratureRule{n}
    ξ::NTuple{n,Vector{Float64}}
    w::Vector{Float64}
end

"""
    Quadrature

Defined quadrature rules in the interval ξ ∈ [0, 1]
"""

import FastGaussQuadrature



@doc raw"""
    gauss_lobatto(p::Integer) -> ξ, w  # nodes ∈ [0, 1], weights

Return nodes `ξ` and weights `w` of 
[Gauss-Lobatto quadrature](https://mathworld.wolfram.com/LobattoQuadrature.html).

Note that here the quadrature is valid for the interval ξ ∈ [0, 1], 
instead of ξ ∈ [-1, 1] as usual.

```math
\int_{0}^{1} f(x) dx \approx \sum_{i=1}^{p} w_i f(x_i)
```
"""
function gauss_lobatto(p::Integer)
    ξ, w = FastGaussQuadrature.gausslobatto(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return ξ, w
end

@doc raw"""
    gauss_legendre(p::Integer) -> ξ, w  # nodes ∈ [0, 1], weights

Return nodes `ξ` and weights `w` of 
[Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).

Note that here the quadrature is valid for the interval ξ ∈ [0, 1], 
instead of ξ ∈ [-1, 1] as usual.

```math
\int_{0}^{1} f(x) dx \approx \sum_{i=1}^{p} w_i f(x_i)
```
"""
function gauss_legendre(p::Integer)
    ξ, w = FastGaussQuadrature.gausslegendre(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return ξ, w
end



@doc raw"""
    tensor_product_weights(weights_1d::NTuple{n, Vector{Float64}}) where {n}

Returns a vector of the tensor product of quadrature weights.
"""
function tensor_product_weights(weights_1d::NTuple{n, Vector{Float64}}) where {n}
    result = Vector{Float64}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        result[linear_idx] = prod(weights_all)
    end
    return result
end

function tensor_product_rule(p::Vector{Int}, quad_rule::F) where {F <: Function}
    n = length(p)
    qrules_1d = collect(quad_rule(p[k]) for k = 1:n);
    points = Tuple(qrules_1d[k].points for k = 1:n);
    weights_1d = (qrules_1d[k].weights for k = 1:n);
    weights = Vector{Float64}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end
    return QuadratureRule(points, weights)
end


end