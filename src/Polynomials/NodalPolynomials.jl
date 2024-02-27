
import PolynomialBases

abstract type AbstractLagrangePolynomials <: AbstractPolynomials end




struct LobattoLegendre <: AbstractLagrangePolynomials
    p::Int  # Polynomial degree
    nodes::Vector{Float64}  # Polynomial grid nodes, there are p+1 nodes
    _core_polynomials::PolynomialBases.LobattoLegendre  # Lobatto-Legendre polynomial basis core structure from PolynomialBases
end

function LobattoLegendre(p::Int)
    ll_polynomials = PolynomialBases.LobattoLegendre(p)
    nodes = ll_polynomials.nodes
    return LobattoLegendre(p, nodes, ll_polynomials)
end


struct GaussLegendre <: AbstractLagrangePolynomials
    p::Int  # Polynomial degree
    nodes::Vector{Float64}  # Polynomial grid nodes, there are p+1 nodes
    _core_polynomials::PolynomialBases.GaussLegendre  # Gauss-Legendre polynomial basis core structure from PolynomialBases
end

function GaussLegendre(p::Int)
    gl_polynomials = PolynomialBases.GaussLegendre(p)
    nodes = gl_polynomials.nodes
    return GaussLegendre(p, nodes, gl_polynomials)
end


"""
    evaluate(polynomials::AbstractLagrangePolynomials, ξ::Vector{Float64})

Evaluate all `polynomials` at `ξ`.

All polynomials that are subtypes of the `AbstractLagrangePolynomials`-
type can be evaluated using the same method from the `PolynomialBases`
package.

# Arguments
- `polynomials::AbstractLagrangePolynomials`: polynomials to evaluate.
- `ξ::Vector{Float64}`: location to evaluate the polynomials at.
"""
function evaluate(polynomials::AbstractLagrangePolynomials, ξ::Vector{Float64})
    return PolynomialBases.interpolation_matrix(ξ, polynomials._core_polynomials.nodes, polynomials._core_polynomials.baryweights)
end
