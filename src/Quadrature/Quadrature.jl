"""
This (sub-)module provides a collection of Gauss Quadrature rules.

The exported names are:
"""
module Quadrature




"""
    Quadrature

Defined quadrature rules in the interval ξ ∈ [0, 1]
"""

import FastGaussQuadrature



# """
#     GaussLobatto(n::Integer) -> ξ, w  # nodes ∈ [0, 1], weights

# Return nodes `ξ` and weights `w` of [Gauss-Lobatto quadrature](https://mathworld.wolfram.com/LobattoQuadrature.html).

# Note that here the quadrature is valid for the interval ξ ∈ [0, 1], instead of ξ ∈ [-1, 1] as usual.

# ```math
# \int_{0}^{1} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)
# ```
# """
function GaussLobatto(p::Integer)
    ξ, w = FastGaussQuadrature.gausslobatto(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return ξ, w
end

# """
#     GaussLegendre(n::Integer) -> ξ, w  # nodes ∈ [0, 1], weights

# Return nodes `ξ` and weights `w` of [Gauss-Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature).

# Note that here the quadrature is valid for the interval ξ ∈ [0, 1], instead of ξ ∈ [-1, 1] as usual.

# ```math
# \int_{0}^{1} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)
# ```
# """
function GaussLegendre(p::Integer)
    ξ, w = FastGaussQuadrature.gausslegendre(p)
    @. ξ = (ξ + 1.0)/2.0
    @. w = 0.5 * w
    return ξ, w
end


end