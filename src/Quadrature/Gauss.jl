@doc raw"""
    gauss_lobatto(N::Integer)

Computes the nodes `ξ` and weights `w` of [Gauss-Lobatto
quadrature](https://mathworld.wolfram.com/LobattoQuadrature.html).

Note that here the quadrature rule is valid for the interval `ξ` ∈ [0, 1], instead of `ξ`
∈ [-1, 1] as usual.

# Arguments
- `N::Integer`: Number of nodes used in the quadrature rule.

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule containing the nodes and weights.
    There will be `N` nodes and weights.

# Throws
- `DomainError`: If `N` is less than or equal to 1. This is handled by the
    FastGaussQuadrature.jl package.

# Notes
Uses the FastGaussQuadrature.jl package. We only linearly map the nodes and weights to the
interval [0, 1].
"""
function gauss_lobatto(N::Integer)
    ξ, w = FastGaussQuadrature.gausslobatto(N)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w

    return QuadratureRule{1}((ξ,), w, "Gauss-Lobatto")
end

@doc raw"""
    gauss_legendre(N::Integer)

Computes the nodes `ξ` and weights `w` of [Gauss-Legendre
quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).

Note that here the quadrature rule is valid for the interval `ξ` ∈ [0, 1], instead of `ξ` ∈
[-1, 1] as usual.

# Arguments
- `N::Integer`: Number of nodes used in the quadrature rule.

# Returns
- `::QuadratureRule{1}`: 1 dimensional quadrature rule containing the nodes and weights.
    There will be `N` nodes and weights.

# Throws
- `DomainError`: If `N` is less than or equal to zero.

# Notes
Uses the FastGaussQuadrature.jl package. We only linearly map the nodes and weights to the
interval [0, 1].
"""
function gauss_legendre(N::Integer)
    if N <= 0
        throw(DomainError("""\
            Invalid number of nodes: $N. Gauss-Legendre quadrature must have at least one \
            node.\
            """
        ))
    end

    ξ, w = FastGaussQuadrature.gausslegendre(N)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w

    return QuadratureRule{1}((ξ,), w, "Gauss-Legendre")
end
