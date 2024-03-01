module GaussQuadratureTests

import Mantis

using Test

# Gauss-Lobatto Quadrature --------------------------------------------------------------

# Perform the tests 
for p in range(2, 20)
  ξ, w = Mantis.Quadrature.gauss_lobatto(p)

  # Test that sum of weights is one
  @test sum(w) ≈ 1.0 atol = 1e-12

  # Test that nodes are not outside the range [0.0, 1.0]
  @test sum((ξ .< 0.0) .& (ξ .> 1.0)) == 0
end

# ---------------------------------------------------------------------------------------

# Gauss-Legendre Quadrature --------------------------------------------------------------

# Perform the tests 
for p in range(1, 20)
  ξ, w = Mantis.Quadrature.gauss_legendre(p)

  # Test that sum of weights is one
  @test sum(w) ≈ 1.0 atol = 1e-12

  # Test that nodes are not outside the range [0.0, 1.0]
  @test sum((ξ .< 0.0) .& (ξ .> 1.0)) == 0
end

# ---------------------------------------------------------------------------------------
end