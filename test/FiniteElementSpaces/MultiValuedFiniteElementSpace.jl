import Mantis

breakpoints = [0.0, 1.0]
patch = Mantis.Mesh.Patch1D(breakpoints);
B1 = Mantis.FunctionSpaces.BSplineSpace(patch, 2, [-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch, 3, [-1, -1])
D = Mantis.FunctionSpaces.DirectSumSpace((B1,B2))
D_eval, D_ind = Mantis.FunctionSpaces.evaluate(D, 1, ([0.0, 0.5, 1.0],), 0)
display(D_eval[1])
display(D_eval[2])

# 3×7 Matrix{Float64}:
#  1.0   0.0  0.0   0.0  0.0  0.0  0.0
#  0.25  0.5  0.25  0.0  0.0  0.0  0.0
#  0.0   0.0  1.0   0.0  0.0  0.0  0.0
# 3×7 Matrix{Float64}:
#  0.0  0.0  0.0  1.0    0.0    0.0    0.0
#  0.0  0.0  0.0  0.125  0.375  0.375  0.125
#  0.0  0.0  0.0  0.0    0.0    0.0    1.0