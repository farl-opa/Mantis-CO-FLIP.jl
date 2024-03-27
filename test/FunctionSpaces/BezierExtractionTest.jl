"""
Tests for the Bezier extraction. These tests are based on the 
standard properties of Bezier curves. See 
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Properties.
"""

import Mantis

using Test

const KnotVector = Mantis.FunctionSpaces.KnotVector
const BSplineSpace = Mantis.FunctionSpaces.BSplineSpace

# Piece-wise degree of the basis functions on which the tests are performed.
const degrees_to_test = 0:25

# Patch used in the test
n = 10
test_brk = collect(LinRange(0.0, 1.0, n))

# Tests for the KnotVector structure construction
mult_length_test = fill(1, n+1)
mult_value_test = fill(1, n)
mult_value_test[3] = -1

@test_throws ArgumentError KnotVector(test_brk, 1, mult_length_test) # Different length test
@test_throws ArgumentError KnotVector(test_brk, 1, mult_value_test) # Negative multiplicity test

# Tests for the BSplineSpace structure construction
polynomial_degree = 3
regularity = fill(1, n)

length_regularity_test = fill(1, n+1)
higher_regularity_test = fill(polynomial_degree + 1, n)

@test_throws ArgumentError BSplineSpace(test_brk, -1, regularity) # Negative degree test
@test_throws ArgumentError BSplineSpace(test_brk, 1, fill(3, n+1)) # Different length test
@test_throws ArgumentError BSplineSpace(test_brk, 3, fill(4, n)) # Higher regularity test

# Tests for known properties of B-Spline basis extraction coefficients

for p in degrees_to_test, k in -1:p-1
    local regularity = fill(k, n)
    regularity[1] = -1
    regularity[end] = -1
    b_spline = BSplineSpace(test_brk, p, regularity)
    # Extract the coefficients
    E = Mantis.FunctionSpaces.extract_bezier_representation(b_spline.knot_vector)
    @test all(E .>= 0.0) # Test for non-negativity
    for el in 1:1:n-1
        @test all(isapprox.(sum((@view E[:,:,el]), dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity
    end
end