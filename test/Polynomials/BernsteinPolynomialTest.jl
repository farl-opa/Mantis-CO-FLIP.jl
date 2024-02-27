"""
Tests for the Bernstein polynomials. These tests are based on the 
standard properties of the Bernstein polynomials, see 
https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties.
"""


import Mantis

using Test

import FastGaussQuadrature


# Degree of the polynomial on which the tests are performed. Since the 
# Bernstein implementation is explicitly written out for p = 0 to p = 3,
# those are all tested. A few extra for the general case are also 
# included.
degrees_to_test = [0, 1, 2, 3, 5, 7, 10, 25]

for p in degrees_to_test
    # Lobatto quadrature includes the endpoints, which can also be used
    # for additional test. The degree is more than large enough anyway.
    # The + is needed to have more than 0 roots when p = 0. Note that 
    # the standard quadrature rule is defined on [-1, 1], while we need 
    # it on [0, 1].
    x, w = FastGaussQuadrature.gausslobatto(3*p+3)
    @. x = (x + 1.0)/2.0
    @. w = 0.5 * w

    sum_all = zeros(size(x))
    sum_all2 = zeros(size(x))
    for l in 0:1:p
        b_lp = [Mantis.Polynomials.polynomial_bernstein(p, l, xi) for xi in x]

        # Positivity of the polynomials
        @test minimum(b_lp) >= 0.0

        # Constant definite integral
        @test isapprox(sum(w .* b_lp), 1.0/(p + 1))

        # Symmetry
        b_plp = [Mantis.Polynomials.polynomial_bernstein(p, p-l, 1.0-xi) for xi in x]
        @test isapprox(b_lp, b_plp)

        # Root at zero
        if l == 0
            # The zero-th polynomial is one at x=0.0
            @test isapprox(b_lp[1], 1.0)
        else
            @test isapprox(b_lp[1], 0.0)
        end

        # Root at one
        if l == p
            # The last polynomial is one at x=1.0
            @test isapprox(b_lp[end], 1.0)
        else
            @test isapprox(b_lp[end], 0.0)
        end

        sum_all .+= b_lp
        sum_all2 .+= l .* b_lp

    end

    # Partition of unity
    @test all(isapprox.(sum_all, 1.0))

    # Other
    @test all(isapprox.(sum_all2, p.*x))

end
