import Mantis

using Test

# Univariate test ------------------------------

# Test parameters
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)


# Univariate test 1
deg1 = 2
deg2 = 2

B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, deg2-1, -1])

nsub1 = 2
nsub2 = 2

ts1 = Mantis.FunctionSpaces.build_two_scale_operator(B1, nsub1)
ts2 = Mantis.FunctionSpaces.build_two_scale_operator(B2, nsub2)

coarse_GTB = Mantis.FunctionSpaces.GTBSplineSpace((B1, B2), [1, -1])
fine_GTB = Mantis.FunctionSpaces.GTBSplineSpace((ts1.fine_space, ts2.fine_space), [1, -1])

ts = Mantis.FunctionSpaces.build_two_scale_operator(coarse_GTB, fine_GTB, ((nsub1,),(nsub2,)))