module TensorProductTwoScaleRelationsTests

import Mantis

using Test

# Test parameters
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
breakpoints3 = [0.0, 1.0]
patch3 = Mantis.Mesh.Patch1D(breakpoints3)

deg1 = 2
deg2 = 3
deg3 = 1

B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
B3 = Mantis.FunctionSpaces.BSplineSpace(patch3, deg3, [-1, -1])

nsub1 = 3
nsub2 = 2
nsub3 = 4

ts1, BF1 = Mantis.FunctionSpaces.build_two_scale_operator(B1, nsub1)
ts2, BF2 = Mantis.FunctionSpaces.build_two_scale_operator(B2, nsub2)
ts3, BF3 = Mantis.FunctionSpaces.build_two_scale_operator(B3, nsub3)

coarse_space_1 = Mantis.FunctionSpaces.TensorProductSpace((Mantis.FunctionSpaces.TensorProductSpace((B1, B2)),B3)) 
fine_space_1 = Mantis.FunctionSpaces.TensorProductSpace((Mantis.FunctionSpaces.TensorProductSpace((BF1, BF2)),BF3))
coarse_space_2 = Mantis.FunctionSpaces.TensorProductSpace((B1, Mantis.FunctionSpaces.TensorProductSpace((B2, B3))))
fine_space_2 = Mantis.FunctionSpaces.TensorProductSpace((BF1, Mantis.FunctionSpaces.TensorProductSpace((BF2, BF3))))
coarse_space_3 = Mantis.FunctionSpaces.TensorProductSpace((B1, B2, B3))
fine_space_3 = Mantis.FunctionSpaces.TensorProductSpace((BF1, BF2, BF3))
ts_operators_1 = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(Mantis.FunctionSpaces.TensorProductSpace((B1, B2)), Mantis.FunctionSpaces.TensorProductSpace((BF1, BF2)), (ts1, ts2))
ts_operators_2 = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(Mantis.FunctionSpaces.TensorProductSpace((B2, B3)), Mantis.FunctionSpaces.TensorProductSpace((BF2, BF3)), (ts2, ts3))
ts_operators_3 = (ts1, ts2, ts3)

tts_1 = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(coarse_space_1, fine_space_1, (ts_operators_1, ts3))
tts_2 = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(coarse_space_2, fine_space_2, (ts1, ts_operators_2))
tts_3 = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(coarse_space_3, fine_space_3, ts_operators_3)

# Tests for TensorTwoScaleOperator

# Check if global subdivision matrices are equal
@test isapprox(Mantis.FunctionSpaces.get_global_subdiv_matrix(tts_1), Mantis.FunctionSpaces.get_global_subdiv_matrix(tts_2), atol=1e-15)
@test isapprox(Mantis.FunctionSpaces.get_global_subdiv_matrix(tts_2), Mantis.FunctionSpaces.get_global_subdiv_matrix(tts_3), atol=1e-15)
# Check total number of coarse elements
@test Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_coarse_space(tts_1)) == Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_coarse_space(tts_2))
@test Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_coarse_space(tts_2)) == Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_coarse_space(tts_3))
@test Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_fine_space(tts_1)) == Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_fine_space(tts_2))
@test Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_fine_space(tts_2)) == Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_fine_space(tts_3))

# Check if local subdivision matrices are the same.
for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(tts_1.fine_space) 
    # Check if parent elements are the same
    coarse_el_1 = Mantis.FunctionSpaces.get_element_parent(tts_1, el)
    coarse_el_2 = Mantis.FunctionSpaces.get_element_parent(tts_2, el)
    coarse_el_3 = Mantis.FunctionSpaces.get_element_parent(tts_3, el)
    @test coarse_el_1 == coarse_el_2
    @test coarse_el_2 == coarse_el_3
    @test isapprox(Mantis.FunctionSpaces.get_local_subdiv_matrix(tts_1, coarse_el_1, el), Mantis.FunctionSpaces.get_local_subdiv_matrix(tts_2, coarse_el_2, el), atol=1e-15)
    @test isapprox(Mantis.FunctionSpaces.get_local_subdiv_matrix(tts_2, coarse_el_2, el), Mantis.FunctionSpaces.get_local_subdiv_matrix(tts_3, coarse_el_3, el), atol=1e-15)
end

# Check if finer elements are the same.
for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(Mantis.FunctionSpaces.get_coarse_space(tts_1)) 
    @test Mantis.FunctionSpaces.get_element_children(tts_1, el) == Mantis.FunctionSpaces.get_element_children(tts_2, el) 
    @test Mantis.FunctionSpaces.get_element_children(tts_2, el) == Mantis.FunctionSpaces.get_element_children(tts_3, el) 
end

# Check if total number of basis functions are the same
@test Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_coarse_space(tts_1)) == Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_coarse_space(tts_2))
@test Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_coarse_space(tts_2)) == Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_coarse_space(tts_3))
@test Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_fine_space(tts_1)) == Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_fine_space(tts_2))
@test Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_fine_space(tts_2)) == Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_fine_space(tts_3))

# Check if basis parents are the same
for basis_id ∈ 1:Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_fine_space(tts_1)) 
    @test Mantis.FunctionSpaces.get_basis_parents(tts_1, basis_id) == Mantis.FunctionSpaces.get_basis_parents(tts_2, basis_id)
    @test Mantis.FunctionSpaces.get_basis_parents(tts_2, basis_id) == Mantis.FunctionSpaces.get_basis_parents(tts_3, basis_id)
end

# Check if basis children are the same
for basis_id ∈ 1:Mantis.FunctionSpaces.get_num_basis(Mantis.FunctionSpaces.get_coarse_space(tts_1)) 
    @test Mantis.FunctionSpaces.get_basis_children(tts_1, basis_id) == Mantis.FunctionSpaces.get_basis_children(tts_2, basis_id)
    @test Mantis.FunctionSpaces.get_basis_children(tts_2, basis_id) == Mantis.FunctionSpaces.get_basis_children(tts_3, basis_id)
end

end
