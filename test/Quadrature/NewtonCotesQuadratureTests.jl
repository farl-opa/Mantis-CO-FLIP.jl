module NewtonCotesQuadratureTests

import Mantis

import LinearAlgebra

using Test


# Defines test functions and tolerances for quadrature rules.
include("HelperQuadratureTests.jl")



# The degree used for the Newton-Cotes quadrature rule does not 
# affect how the nodes and weights are computed. So testing with a few
# different degrees should be enough.

# Closed rules ---------------------------------------------------------
for N in range(2, 12, step=3)
    quad_rule = Mantis.Quadrature.newton_cotes(N, "closed")

    ξ = Mantis.Quadrature.get_quadrature_nodes(quad_rule)[1]
    w = Mantis.Quadrature.get_quadrature_weights(quad_rule)

    
    # Check that the rule type is correct.
    @test Mantis.Quadrature.get_quadrature_rule_type(quad_rule) == "Newton-Cotes (closed)"


    # Constructor tests.
    @test typeof(quad_rule) == Mantis.Quadrature.QuadratureRule{1}


    # Invalid constructor tests.
    # Test that the constructor throws an error for invalid degrees.
    @test_throws DomainError Mantis.Quadrature.newton_cotes(1, "closed")
    @test_throws DomainError Mantis.Quadrature.newton_cotes(0, "closed")
    @test_throws DomainError Mantis.Quadrature.newton_cotes(-1, "closed")

    @test_throws ArgumentError Mantis.Quadrature.newton_cotes(1, "test")
    

    # Property tests.
    # Test that sum of weights is one.
    @test isapprox(sum(w), 1.0, atol=atol)
    # Test that the weights are symmetric.
    @test isapprox(w, reverse(w), atol=atol)

    # Test that nodes are not outside the range [0.0, 1.0].
    @test sum((ξ .< 0.0) .& (ξ .> 1.0)) == 0
    # Test that the endpoints are included.
    @test isapprox(ξ[1], 0.0, atol = atol)
    @test isapprox(ξ[end], 1.0, atol = atol)



    # Value tests on [0,1].
    # Test that the quadrature rule is exact for polynomials of degree
    # up to N-1 (with N the number of nodes).
    for degree in 0:N-1
        f = monomial(degree, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(degree, 1.0) - integrated_monomial(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)

        f = chebyshev(degree, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev(degree, 1.0) - integrated_chebyshev(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)
    end

    # Test that the quadrature rule is not exact for polynomials of
    # degree N.

    # For Newton-Cotes rules, the error for the integration of a 
    # monomial of degree N will be zero when N is odd.
    if N % 2 != 0
        f = monomial(N, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(N, 1.0) - integrated_monomial(N, 0.0)
        @test isapprox(I_num, I, atol=atol)
    else
        f = monomial(N, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(N, 1.0) - integrated_monomial(N, 0.0)
        @test !isapprox(I_num, I, atol=atol)
    end

    f = chebyshev(N+1, ξ)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_chebyshev(N+1, 1.0) - integrated_chebyshev(N+1, 0.0)
    @test !isapprox(I_num, I, atol=atol)
end


# Open rules -----------------------------------------------------------
for N in range(1, 10, step=3)
    quad_rule = Mantis.Quadrature.newton_cotes(N, "open")

    ξ = Mantis.Quadrature.get_quadrature_nodes(quad_rule)[1]
    w = Mantis.Quadrature.get_quadrature_weights(quad_rule)

    
    # Check that the rule type is correct.
    @test Mantis.Quadrature.get_quadrature_rule_type(quad_rule) == "Newton-Cotes (open)"


    # Constructor tests.
    @test typeof(quad_rule) == Mantis.Quadrature.QuadratureRule{1}


    # Invalid constructor tests.
    # Test that the constructor throws an error for invalid degrees.
    @test_throws DomainError Mantis.Quadrature.newton_cotes(0, "open")
    @test_throws DomainError Mantis.Quadrature.newton_cotes(-1, "open")

    @test_throws ArgumentError Mantis.Quadrature.newton_cotes(1, "test")
    

    # Property tests.
    # Test that sum of weights is one.
    @test isapprox(sum(w), 1.0, atol=atol)
    # Test that the weights are symmetric.
    @test isapprox(w, reverse(w), atol=atol)

    # Test that nodes are not outside the range [0.0, 1.0].
    @test sum((ξ .< 0.0) .& (ξ .> 1.0)) == 0



    # Value tests on [0,1].
    # Test that the quadrature rule is exact for polynomials of degree
    # up to N-1 (with N the number of nodes).
    for degree in 0:N-1
        f = monomial(degree, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(degree, 1.0) - integrated_monomial(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)

        f = chebyshev(degree, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_chebyshev(degree, 1.0) - integrated_chebyshev(degree, 0.0)
        @test isapprox(I_num, I, atol=atol)
    end

    # Test that the quadrature rule is not exact for polynomials of
    # degree N.

    # For Newton-Cotes rules, the error for the integration of a 
    # monomial of degree N will be zero when N is odd.
    if N % 2 != 0
        f = monomial(N, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(N, 1.0) - integrated_monomial(N, 0.0)
        @test isapprox(I_num, I, atol=atol)
    else
        f = monomial(N, ξ)
        I_num = LinearAlgebra.dot(w, f)
        I = integrated_monomial(N, 1.0) - integrated_monomial(N, 0.0)
        @test !isapprox(I_num, I, atol=atol)
    end

    f = chebyshev(N+1, ξ)
    I_num = LinearAlgebra.dot(w, f)
    I = integrated_chebyshev(N+1, 1.0) - integrated_chebyshev(N+1, 0.0)
    @test !isapprox(I_num, I, atol=atol)
end



end