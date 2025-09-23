module HierarchicalMultiComponentTests

using Mantis

using Test

nq = 7
p = 3
nlevels = 3
subdiv = 2

# mesh, degree and regularity of the coarsest level
brk = collect(LinRange(0, 1, nq))
patch = Mesh.Patch1D(brk)
regularity_1 = fill(p - 1, nq)
regularity_1[1] = regularity_1[nq] = -1
regularity_2 = fill(p - 2, nq)
regularity_2[1] = regularity_2[nq] = -1

# refined domains
refined_domains = FunctionSpaces.HierarchicalActiveInfo([
    [1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8, 9, 10], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
])

##########################################
# STANDARD CONSTRUCTION
##########################################

# b-spline spaces for the two component direct sum space
bspline = [
    FunctionSpaces.BSplineSpace(patch, p, regularity_1),
    FunctionSpaces.BSplineSpace(patch, p, regularity_2)
]
num_components = length(bspline)

# two scale operators
two_scale_operators = [
    Vector{FunctionSpaces.TwoScaleOperator}(undef, nlevels - 1) for _ in 1:num_components
]
bsplines = [
    Vector{FunctionSpaces.BSplineSpace}(undef, nlevels) for _ in 1:num_components
]
for c in 1:num_components
    bsplines[c][1] = bspline[c]
    for l in 1:(nlevels - 1)
        ts_operator, bspline_l = FunctionSpaces.build_two_scale_operator(bsplines[c][l], subdiv)
        two_scale_operators[c][l] = ts_operator
        bsplines[c][l + 1] = bspline_l
    end
end

# hierarchical space built component wise
hier_space = [
    FunctionSpaces.HierarchicalFiniteElementSpace(
        tuple(bsplines[c]...), tuple(two_scale_operators[c]...), refined_domains, (subdiv,)
    ) for c in 1:num_components
]
hier_space = FunctionSpaces.DirectSumSpace(tuple(hier_space...))

##########################################
# NEW CONSTRUCTION
##########################################

# b-spline spaces for the two component direct sum space
bspline_DS = FunctionSpaces.DirectSumSpace(tuple(bspline...))

# two scale operators
two_scale_operators_DS = Vector{FunctionSpaces.TwoScaleOperator}(undef, nlevels - 1)
bsplines_DS = Vector{FunctionSpaces.DirectSumSpace}(undef, nlevels)
bsplines_DS[1] = bspline_DS
for l in 1:(nlevels - 1)
    ts_operator, bspline_l = FunctionSpaces.build_two_scale_operator(bsplines_DS[l], subdiv)
    two_scale_operators_DS[l] = ts_operator
    bsplines_DS[l + 1] = bspline_l
end

# hierarchical space built component wise
hier_space_DS = FunctionSpaces.HierarchicalFiniteElementSpace(
    tuple(bsplines_DS...), tuple(two_scale_operators_DS...), refined_domains, (subdiv,)
)

##########################################
# TESTS COMPARING hier_space AND hier_space_DS
##########################################

# loop over elements: check that evaluations, supported basis ids and local_subdiv_matrices are the same
for elem in 1:FunctionSpaces.get_num_elements(hier_space)

end

# loop over basis: check that basis supports are the same
@test FunctionSpaces.get_num_basis(hier_space) == FunctionSpaces.get_num_basis(hier_space_DS)
for i in 1:FunctionSpaces.get_num_basis(hier_space)
    supp = FunctionSpaces.get_support(hier_space, i)
    supp_DS = FunctionSpaces.get_support(hier_space_DS, i)
    @test supp == supp_DS
end

end
