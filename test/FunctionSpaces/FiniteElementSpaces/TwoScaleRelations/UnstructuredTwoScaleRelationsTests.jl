module UnstructuredTwoScaleRelationsTests

import Mantis

using Test

# Univariate test ------------------------------

# Test parameters
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.8, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

###
### Univariate test 1
###
deg1 = 2
deg2 = deg1

B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, deg2-1, -1])

nsub1 = 2
nsub2 = nsub1

ts1, B1_fine = Mantis.FunctionSpaces.build_two_scale_operator(B1, nsub1)
ts2, B2_fine = Mantis.FunctionSpaces.build_two_scale_operator(B2, nsub2)

coarse_GTB = Mantis.FunctionSpaces.GTBSplineSpace((B1, B2), [1, -1])
fine_GTB = Mantis.FunctionSpaces.GTBSplineSpace((B1_fine, B2_fine), [1, -1])

ts_GTB, _ = Mantis.FunctionSpaces.build_two_scale_operator(coarse_GTB, fine_GTB, ((nsub1,),(nsub2,)))

breakpoints = vcat(breakpoints1, breakpoints2[2:end].+breakpoints1[end])
patch = Mantis.Mesh.Patch1D(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, deg1, [-1, deg1-1, 1, deg1-1, -1])
ts, B_fine = Mantis.FunctionSpaces.build_two_scale_operator(B, nsub1)

@test(isapprox(maximum(abs.(ts.global_subdiv_matrix-ts_GTB.global_subdiv_matrix)), 0.0, atol=1e-13))

###
### Univariate test 2
###
deg1 = 2
deg2 = 3

B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, deg2-1, -1])

nsub1 = 2
nsub2 = 3

ts1, B1_fine = Mantis.FunctionSpaces.build_two_scale_operator(B1, nsub1)
ts2, B2_fine = Mantis.FunctionSpaces.build_two_scale_operator(B2, nsub2)

coarse_GTB = Mantis.FunctionSpaces.GTBSplineSpace((B1, B2), [1, -1])
fine_GTB = Mantis.FunctionSpaces.GTBSplineSpace((B1_fine, B2_fine), [1, -1])

ts_GTB, _ = Mantis.FunctionSpaces.build_two_scale_operator(coarse_GTB, fine_GTB, ((nsub1,),(nsub2,)))

# randomly chosen coarse GTB dofs
coarse_dofs_GTB = rand(Mantis.FunctionSpaces.get_num_basis(coarse_GTB))
# fine GTB dofs implied by two-scale relation
fine_dofs_GTB = ts_GTB.global_subdiv_matrix*coarse_dofs_GTB

# global extraction operators for coarse and fine spaces
coarse_ex_op = Mantis.FunctionSpaces.assemble_global_extraction_matrix(coarse_GTB)
fine_ex_op = Mantis.FunctionSpaces.assemble_global_extraction_matrix(fine_GTB)

# coarse GTB dofs extracted to B1 and B2
coarse_dofs_discont = coarse_ex_op * coarse_dofs_GTB
coarse_dofs_B1 = coarse_dofs_discont[1:Mantis.FunctionSpaces.get_num_basis(B1)]
coarse_dofs_B2 = coarse_dofs_discont[Mantis.FunctionSpaces.get_num_basis(B1)+1:end]

# fine B1 and B2 dofs implied by B-spline two-scale relations
fine_dofs_B1 = ts1.global_subdiv_matrix * coarse_dofs_B1
fine_dofs_B2 = ts2.global_subdiv_matrix * coarse_dofs_B2

# fine B1 and B2 dofs computed in two different ways
fine_dofs_discont_1 = vcat(fine_dofs_B1, fine_dofs_B2)
fine_dofs_discont_2 = fine_ex_op * fine_dofs_GTB

# check approximate equality
@test(isapprox(maximum(abs.(fine_dofs_discont_1-fine_dofs_discont_2)), 0.0, atol=1e-13))

# --------------------------------------------------
#
end
