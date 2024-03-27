"""
Tests for the Bezier extraction. These tests are based on the 
standard properties of Bezier curves. See 
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Properties.
"""

import Mantis

using Test

const Patch = Mantis.Mesh.Patch
const KnotVector = Mantis.FunctionSpaces.KnotVector
const BSplineSpace = Mantis.FunctionSpaces.BSplineSpace

# Piece-wise degree of the basis functions on which the tests are performed.
const degrees_to_test = 0:3

# Patch used in the test
n1 = 6
n2 = 8
n3 = 12
test_brk1 = collect(LinRange(0.0, 1.0, n1))
test_brk2 = collect(LinRange(-3.0, 2.25, n2))
test_brk3 = collect(LinRange(1.0, 3.75, n3))
test_patch = Patch((test_brk1, test_brk2, test_brk3))

# Tests for the KnotVector structure construction
# Errors are expected when the number of breakpoints and multiplicity for each are different
# and also when there is non-positive multiplicity.
mult_length_test = fill(2, n2+1)
mult_value_test = fill(1, n1)
mult_value_test[3] = -1

@test_throws ArgumentError KnotVector(test_brk2, mult_length_test)
@test_throws ArgumentError KnotVector(test_brk1, mult_value_test)


# Tests for the BSplineSpace structure construction
# Errors should be thrown when the number of regularity conditions is different than
# the number of elements-1. Also, an error is expected when a given regularity is higher than
# the corresponding polynomial degree.
polynomial_degree = (2,1,5)
regularity = (fill(1, n1-2), fill(-1, n2-2), fill(3, n3-2))

negative_degree_test = (2,1,-3)
length_regularity_test = (fill(3, n1-2), fill(-1, n2), fill(3, n3-2))
higher_regularity_test = (fill(3, n1-2), fill(-1, n2-2), fill(3, n3-2))

@test_throws ArgumentError BSplineSpace(test_patch, negative_degree_test, regularity, 0)
@test_throws ArgumentError BSplineSpace(test_patch, negative_degree_test, length_regularity_test, 0)
@test_throws ArgumentError BSplineSpace(test_patch, negative_degree_test, higher_regularity_test, 0)

# Tests for known properties of B-Spline basis extraction coefficients

for p in degrees_to_test, k in -1:p-1
    local polynomial_degree = (p,p,p)
    local regularity = (fill(k, n1-2), fill(k, n2-2), fill(k, n3-2))
    b_spline = BSplineSpace(test_patch, polynomial_degree, regularity, 0)
    # Extract the coefficients
    E = Mantis.FunctionSpaces.extract_bezier_representation(b_spline)
    for d in 1:1:3
        @test all(E[d] .>= 0.0) # Test for non-negativity
        for el in size(test_patch)[d]
            @test all(isapprox.(sum((@view E[d][:,:,el]), dims=2), 1.0, atol=1e-15)) # Test for partition of unity
        end
    end
end
