"""
    Quadrature

This module provides a collection of (Gauss) Quadrature rules. The 
defined quadrature rules in the interval [0, 1].

The exported names are:
"""
module Quadrature



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
    tensor_product_rule(p::NTuple{n, Int}, quad_rule::F) where {n, F <: Function}

Returns a tensor product quadrature rule of given degree and rule type.

# Arguments
- `p::NTuple{n, Int}`: Degree of the quadrature rule per dimension.
- `quad_rule::F`: The function that returns quadrature nodes and weights given an integer degree.

# Returns
- `points::NTuple{n, Vector{Float64}}`: Quadrature nodes per dimension.
- `weights::Vector{Float64}`: Tensor product of quadrature rules. The shape is consistent with the output of the evaluate methods for Mantis.FunctionSpaces
"""
function tensor_product_rule(p::NTuple{n, Int}, quad_rule::F) where {n, F <: Function}
    # Compute the nodes and weights per dimensions for given rule type 
    # and degree.
    qrules_1d = NTuple{n, NTuple{2, Vector{Float64}}}(quad_rule(p[k]) for k = 1:n)
    
    points = NTuple{n, Vector{Float64}}(qrules_1d[k][1] for k = 1:n)
    weights_1d = NTuple{n, Vector{Float64}}(qrules_1d[k][2] for k = 1:n)
    
    # Compute the tensor product of the quadrature weights.
    weights = Vector{Float64}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end
    
    return points, weights
end


end