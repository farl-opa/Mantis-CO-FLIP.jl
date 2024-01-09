
import PolynomialBases

abstract type NodalPolynomials <: GenericPolynomials end

struct LobattoLegendre <: NodalPolynomials
    p::Int  # Polynomial degree
    nodes::Vector{Float64}  # Polynomial grid nodes, there are p+1 nodes
    _core_polynomials::PolynomialBases.LobattoLegendre  # Lobatto-Legendre polynomial basis core structure from PolynomialBases
end

function LobattoLegendre(p::Int)
    ll_polynomials = PolynomialBases.LobattoLegendre(p)
    nodes = ll_polynomials.nodes
    return LobattoLegendre(p, nodes, ll_polynomials)
end

struct GaussLegendre <: NodalPolynomials
    p::Int  # Polynomial degree
    nodes::Vector{Float64}  # Polynomial grid nodes, there are p+1 nodes
    _core_polynomials::PolynomialBases.GaussLegendre  # Gauss-Legendre polynomial basis core structure from PolynomialBases
end

function GaussLegendre(p::Int)
    gl_polynomials = PolynomialBases.GaussLegendre(p)
    nodes = gl_polynomials.nodes
    return GaussLegendre(p, nodes, gl_polynomials)
end

function evaluate(polynomials::NodalPolynomials, ξ::Vector{Float64})
    return PolynomialBases.interpolation_matrix(ξ, polynomials._core_polynomials.nodes, polynomials._core_polynomials.baryweights)
end
