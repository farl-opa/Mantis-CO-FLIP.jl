"""
Tests for the B-Spline bases. These tests are based on the 
standard properties of the B-Spline basses functions. See 
https://en.wikipedia.org/wiki/B-spline#Properties.
"""

import Mantis

using Test

# Piece-wise degree of the basis functions on which the tests are performed.
const degrees_to_test = 0:25

# x values of the functions
const nx = 21
const x = collect(range(0,1,nx))
# Number of elements
const n_els = 5

for p in degrees_to_test
    # Evaluate the  Bernstein polynomials on the reference element.
    b = Mantis.Polynomials.Bernstein(p)
    b_eval = Mantis.Polynomials.evaluate(b, x)

    # Initialize the basis functions
    N = zeros(nx, p+1)
    for k in -1:p-1
        # Extract the coefficients
        E = Mantis.ExtractionCoefficients.bezier_extraction([0.0, 1.0], n_els, p, k)
        for el in 1:n_els
            # Evaluate basis functions on the element
            N = Mantis.Bases.evalute(E, el, b_eval)
            
            # Test for partition for unity
            @test all(sum(N, dims=2) .≈ 1.0)
        end
    end
end