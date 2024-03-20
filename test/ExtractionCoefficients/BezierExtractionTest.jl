"""
Tests for the Bezier extraction. These tests are based on the 
standard properties of Bezier curves. See 
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Properties.
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
        E = Mantis.ExtractionCoefficients.extract_bezier_representation([0.0, 1.0], n_els, p, k)
        @test all(E .>= 0.0)
        @test E[1] == 1.0 && E[end] == 1.0
    end
end