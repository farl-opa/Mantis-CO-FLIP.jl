import Mantis

using Test

nq = 7
p = 3
nlevels = 3
subdiv = 2

brk = collect(LinRange(0,1,nq))
patch = Mantis.Mesh.Patch1D(brk)
regularity = fill(p-1, nq)
regularity[1]=regularity[nq] = -1
bspline1 = Mantis.FunctionSpaces.BSplineSpace(patch, p, regularity)

two_scale_operators = Vector{Mantis.FunctionSpaces.TwoScaleOperator}(undef, nlevels-1)
bsplines = Vector{Mantis.FunctionSpaces.BSplineSpace}(undef, nlevels)
bsplines[1] = bspline1

for l in 1:nlevels-1
    ts_operator, bspline = Mantis.FunctionSpaces.subdivide_bspline(bsplines[l], subdiv)
    two_scale_operators[l] = ts_operator
    bsplines[l+1] = bspline
end

active_elements = Mantis.FunctionSpaces.HierarchicalActiveInfo(fill(1,nlevels), collect(0:nlevels))
active_functions = Mantis.FunctionSpaces.HierarchicalActiveInfo(fill(1,nlevels), collect(0:nlevels))

levels_el_test = Mantis.FunctionSpaces.HierarchicalActiveInfo(fill(1,nlevels), collect(0:nlevels+1))
levels_function_test = Mantis.FunctionSpaces.HierarchicalActiveInfo(fill(1,nlevels), collect(0:nlevels+1))

@test_throws ArgumentError Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(bsplines, two_scale_operators, levels_el_test, active_functions)
@test_throws ArgumentError Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(bsplines, two_scale_operators, active_elements, levels_function_test)
@test_throws ArgumentError Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(bsplines[1:end-1], two_scale_operators, active_elements, active_functions)
@test_throws ArgumentError Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(bsplines, two_scale_operators[1:end-1], active_elements, active_functions)

hierarchical_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(bsplines, two_scale_operators, active_elements, active_functions)

@test Mantis.FunctionSpaces.get_n_levels(hierarchical_space) == nlevels

for l in 1:nlevels
    @test Mantis.FunctionSpaces.get_space(hierarchical_space, l) == bsplines[l]
end

# Test according to the example in Fig 9. of https://doi.org/10.1007/s11831-022-09752-5
refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([1,6,3,4,5,6,7,8,9,10,7,8,9,10,11,12,13,14,15,16],[0,2,10,20])
hierarchical_space = Mantis.FunctionSpaces.get_hierarchical_space(bsplines, two_scale_operators, refined_domains, fill(2, nlevels))

# test if active elements are correct   
@test Mantis.FunctionSpaces.get_level_elements(hierarchical_space, 1)[2] == [1,6]
@test Mantis.FunctionSpaces.get_level_elements(hierarchical_space, 2)[2] == [3,9,10]
@test Mantis.FunctionSpaces.get_level_elements(hierarchical_space, 3)[2] == collect(7:16)

# test if active functions are correct   
@test Mantis.FunctionSpaces.get_level_functions(hierarchical_space, 1)[2] == [1,2,3,4,6,7,8,9]
@test Mantis.FunctionSpaces.get_level_functions(hierarchical_space, 2)[2] == [6,9,10]
@test Mantis.FunctionSpaces.get_level_functions(hierarchical_space, 3)[2] == collect(10:16)

# Tests for coefficients and evaluation
for el in 1:1:Mantis.FunctionSpaces.get_n_elements(hierarchical_space)

    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(hierarchical_space, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = Mantis.FunctionSpaces.evaluate(hierarchical_space, el, brk, 0)
    # Positivity of the basis
    @test minimum(h_eval[:,:,1]) >= 0.0
end