module GTBSplineTests

import Mantis

using Test

deg1 = 3
breakpoints = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints)
B1_univariate_bs = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, 1, -1])

breakpoints2 = [0.0, 0.5, 0.6, 1.0]
deg2 = 4
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
B2_univariate_bs = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, 1, 3, -1])

GB = Mantis.FunctionSpaces.GTBSplineSpace((B1_univariate_bs, B2_univariate_bs), [1, -1])
quad_rule3 = Mantis.Quadrature.gauss_legendre(max(deg1,deg2)+1)
x = Mantis.Quadrature.get_nodes(quad_rule3)[1]
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    # check extraction coefficients
    ex_coeffs = Mantis.FunctionSpaces.get_extraction_coefficients(GB, el)
    # Test for non-negativity
    @test all(ex_coeffs .>= 0.0)
    # Test for partition of unity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14))
    # check GTB-spline evaluation
    GB_eval, _ = Mantis.FunctionSpaces.evaluate(GB, el, (x,), 1)
end

# Test C1-smooth TrigonometricSplineSpace ----------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Mesh.Patch1D(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B,), ones(Int, 1))
nel = Mantis.FunctionSpaces.get_num_elements(GB)
for el in 1:1:nel
    # check extraction coefficients
    ex_coeffs = Mantis.FunctionSpaces.get_extraction_coefficients(GB, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14)) # Test for partition of unity
end

# interpolate a cosine and a sine
x2 = [0.5]
LHS = zeros(nel, nel)
RHS_C = zeros(nel)
RHS_S = zeros(nel)
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    GB_eval, inds = Mantis.FunctionSpaces.evaluate(GB, el, (x2,), 0)
    LHS[el,inds] = GB_eval[1][1][1]
    RHS_C[el] = cos(Wt * (x2[1] + el - 1))
    RHS_S[el] = sin(Wt * (x2[1] + el - 1))
end
coeffs_C = LHS \ RHS_C
coeffs_S = LHS \ RHS_S
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(GB)
    GB_eval, inds = Mantis.FunctionSpaces.evaluate(GB, el, (x,), 0)
    @test all(isapprox.(maximum(abs.(GB_eval[1][1][1]*coeffs_C[inds] - cos.(Wt * (x .+ (el - 1))))), 0.0, atol=1e-14))
    @test all(isapprox.(maximum(abs.(GB_eval[1][1][1]*coeffs_S[inds] - sin.(Wt * (x .+ (el - 1))))), 0.0, atol=1e-14))
end


end
