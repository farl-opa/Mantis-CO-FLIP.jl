import Mantis

using Test

nq = 11
p = 2
nlevels = 6
subdiv = 2

brk = collect(LinRange(0,1,nq))
patch = Mantis.Mesh.Patch1D(brk)
regularity = fill(1, nq)
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
