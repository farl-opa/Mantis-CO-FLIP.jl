module SumSpaceTests

import Mantis
import LinearAlgebra
using Test

breakpoints = [0.0, 1.0]
patch = Mantis.Mesh.Patch1D(breakpoints);
B1 = Mantis.FunctionSpaces.BSplineSpace(patch, 2, [-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch, 3, [-1, -1])

# Example of a DirectSumSpace ------------------------------------------------------------
D = Mantis.FunctionSpaces.DirectSumSpace((B1, B2))
D_eval, D_ind = Mantis.FunctionSpaces.evaluate(D, 1, ([0.0, 0.5, 1.0],), 1)

#display(D_eval[1][1][1])
#display(D_eval[2][1][1])

# D = Mantis.FunctionSpaces.DirectSumSpace((TP, TP))
# D_eval, D_ind = Mantis.FunctionSpaces.evaluate(D, 1, ([0.0, 1.0], [0.5, 1.0]), 1)

# 3×7 Matrix{Float64}:
#  1.0   0.0  0.0   0.0  0.0  0.0  0.0
#  0.25  0.5  0.25  0.0  0.0  0.0  0.0
#  0.0   0.0  1.0   0.0  0.0  0.0  0.0
# 3×7 Matrix{Float64}:
#  0.0  0.0  0.0  1.0    0.0    0.0    0.0
#  0.0  0.0  0.0  0.125  0.375  0.375  0.125
#  0.0  0.0  0.0  0.0    0.0    0.0    1.0

# Example of SumSpace that is equivalent to the above DirectSumSpace ------------------------------------------------------------
# create fake unstructured spaces from the two spaces
extraction_operator_1 = Mantis.FunctionSpaces.ExtractionOperator([Matrix(LinearAlgebra.I(3))], [[1, 2, 3]], 1, 3)
U1 = Mantis.FunctionSpaces.AbstractMultiPatchFESpace((B1,), extraction_operator_1, Dict())
extraction_operator_2 = Mantis.FunctionSpaces.ExtractionOperator([Matrix(LinearAlgebra.I(4))], [[4, 5, 6, 7]], 1, 4)
U2 = Mantis.FunctionSpaces.AbstractMultiPatchFESpace((B2,), extraction_operator_2, Dict())

S = Mantis.FunctionSpaces.SumSpace((U1, U2), 7)
S_eval, S_ind = Mantis.FunctionSpaces.evaluate(S, 1, ([0.0, 0.5, 1.0],), 1)

#display(S_eval[1][1][1])
#display(S_eval[2][1][1])

@test S_eval[1][1][1] == D_eval[1][1][1] && S_eval[2][1][1] == D_eval[2][1][1]

@test Mantis.FunctionSpaces.get_basis_indices(S, 1) == [1, 2, 3, 4, 5, 6, 7]

# Example of a different SumSpace that mixes the components ------------------------------------------------------------
# create fake unstructured spaces from the two spaces
extraction_operator_1 = Mantis.FunctionSpaces.ExtractionOperator([Matrix(LinearAlgebra.I(3))], [[1, 2, 3]], 1, 3)
U1 = Mantis.FunctionSpaces.AbstractMultiPatchFESpace((B1,), extraction_operator_1, Dict())
extraction_operator_2 = Mantis.FunctionSpaces.ExtractionOperator([Matrix(LinearAlgebra.I(4))], [[1, 4, 5, 6]], 1, 4)
U2 = Mantis.FunctionSpaces.AbstractMultiPatchFESpace((B2,), extraction_operator_2, Dict())

S = Mantis.FunctionSpaces.SumSpace((U1, U2), 6)
S_eval, S_ind = Mantis.FunctionSpaces.evaluate(S, 1, ([0.0, 0.5, 1.0],), 1)

@test Mantis.FunctionSpaces.get_basis_indices(S, 1) == [1, 2, 3, 4, 5, 6]

#display(S_eval[1][1][1])
#display(S_eval[2][1][1])

# 3×6 Matrix{Float64}:
#  1.0   0.0  0.0   0.0  0.0  0.0
#  0.25  0.5  0.25  0.0  0.0  0.0
#  0.0   0.0  1.0   0.0  0.0  0.0
# 3×6 Matrix{Float64}:
#  1.0    0.0  0.0  0.0    0.0    0.0
#  0.125  0.0  0.0  0.375  0.375  0.125
#  0.0    0.0  0.0  0.0    0.0    1.0
#
end
