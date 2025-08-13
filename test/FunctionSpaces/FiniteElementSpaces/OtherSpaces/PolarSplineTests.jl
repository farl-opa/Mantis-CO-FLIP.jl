module PolarSplineTests

import Mantis
import LinearAlgebra
using Test

# 1D spaces in radial and poloidal directions
breakpoints_p = [0.0, 0.5, 1.0, 1.5, 2.0]
breakpoints_r = [0.0, 0.5, 1.0]
num_elements_p = length(breakpoints_p) - 1
num_elements_r = length(breakpoints_r) - 1
patch_p = Mantis.Mesh.Patch1D(breakpoints_p)
patch_r = Mantis.Mesh.Patch1D(breakpoints_r)
deg_p = 2
deg_r = 3
regularity_p = vcat(-1, fill(deg_p-1, length(breakpoints_p) - 2), -1)
regularity_r = vcat(-1, fill(deg_r-1, length(breakpoints_r) - 2), -1)
B_p = Mantis.FunctionSpaces.BSplineSpace(patch_p, deg_p, regularity_p)
B_r = Mantis.FunctionSpaces.BSplineSpace(patch_r, deg_r, regularity_r)
n_p = Mantis.FunctionSpaces.get_num_basis(B_p)
n_r = Mantis.FunctionSpaces.get_num_basis(B_r)

# degenerate control points for the polar spline space
geom_coeffs_tp, _, _ = Mantis.FunctionSpaces._build_standard_degenerate_control_points(n_p, n_r, 1.0)


#################################################################
# ScalarPolarSplineSpace
#################################################################

# build scalar polar spline space
P_scalar = Mantis.FunctionSpaces.ScalarPolarSplineSpace(
    B_p,
    B_r,
    (geom_coeffs_tp[:, 1, :], geom_coeffs_tp[:, 2, :])
)

@test Mantis.FunctionSpaces.get_num_basis(P_scalar) == n_p * (n_r - 2) + 3
@test Mantis.FunctionSpaces.get_num_elements(P_scalar) == num_elements_p * num_elements_r
@test Mantis.FunctionSpaces.get_num_elements_per_patch(P_scalar)[1] == num_elements_p * num_elements_r

# evaluate basis functions
xi = ([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.33, 0.66, 1.0])
for element_id in 1:Mantis.FunctionSpaces.get_num_elements(P_scalar)
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(P_scalar, element_id, 1)
    @test all(isapprox.(sum(ex_coeffs, dims=2) .- 1.0, 0.0, atol=1e-14))

    evaluations, basis_indices = Mantis.FunctionSpaces.evaluate(
        P_scalar, element_id, xi
    )
    @test size(evaluations[1][1][1]) == (prod(length.(xi)), length(basis_indices))
    @test all(evaluations[1][1][1] .>= 0.0)
    @test all(isapprox.(sum(evaluations[1][1][1], dims=2) .- 1.0, 0.0, atol=1e-14))
end

#################################################################
# VectorPolarSplineSpace
#################################################################

# build vector polar spline space
P_vector = Mantis.FunctionSpaces.VectorPolarSplineSpace(
    B_p,
    B_r,
    (geom_coeffs_tp[:, 1, :], geom_coeffs_tp[:, 2, :]),
    Mantis.FunctionSpaces.get_derivative_space(B_p),
    Mantis.FunctionSpaces.get_derivative_space(B_r)
)

@test Mantis.FunctionSpaces.get_num_basis(P_vector) == 2 * n_p * (n_r - 2) + 2
@test Mantis.FunctionSpaces.get_num_elements(P_vector) == num_elements_p * num_elements_r
@test Mantis.FunctionSpaces.get_num_elements_per_patch(P_vector)[1] == num_elements_p * num_elements_r

end
