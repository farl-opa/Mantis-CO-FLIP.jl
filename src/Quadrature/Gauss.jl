@doc raw"""
    gauss_lobatto(p::Int)

Computes the nodes `ξ` and weights `w` of 
[Gauss-Lobatto quadrature](https://mathworld.wolfram.com/LobattoQuadrature.html).

Note that here the quadrature rule is valid for the interval ξ ∈ [0, 1], 
instead of ξ ∈ [-1, 1] as usual.

# Arguments
- `p::Int`: Degree of the quadrature rule.

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule containing the 
                         nodes and weights. There will be `p` nodes and
                         weights.

# Notes
Uses the FastGaussQuadrature.jl package. We only linearly map the nodes
and weights to the interval [0, 1].
"""
function gauss_lobatto(p::Int)
    ξ, w = FastGaussQuadrature.gausslobatto(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return QuadratureRule{1}((ξ,), w, "Gauss-Lobatto")
end

@doc raw"""
    gauss_legendre(p::Int)

Computes the nodes `ξ` and weights `w` of 
[Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).

Note that here the quadrature rule is valid for the interval ξ ∈ [0, 1], 
instead of ξ ∈ [-1, 1] as usual.

# Arguments
- `p::Int`: Degree of the quadrature rule.

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule containing the 
                         nodes and weights. There will be `p` nodes and
                         weights.

# Notes
Uses the FastGaussQuadrature.jl package. We only linearly map the nodes
and weights to the interval [0, 1].
"""
function gauss_legendre(p::Int)
    if p <= 0
        throw(DomainError("Invalid degree: $p. The degree of the Gauss-Legendre quadrature rule must be greater than zero."))
    end
    ξ, w = FastGaussQuadrature.gausslegendre(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return QuadratureRule{1}((ξ,), w, "Gauss-Legendre")
end