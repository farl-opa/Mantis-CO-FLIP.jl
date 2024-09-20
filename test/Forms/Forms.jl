import Mantis

using Test


# 2D tests --------------------------------------------------------------------

# Setup the geometry 

# First the 2D rectangular domain parameters
starting_point_2d = (0.0, 0.0)
box_size_2d = (1.0, 1.0)
num_elements_2d = (2, 3)
degree_2d = (2, 2)
regularity_2d = degree_2d .- 1

# Generate the 2D rectangular straight geometry
geo_2d_cart = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, num_elements_2d)

# Setup the form spaces

# Generate a tensor product space (single valued space)
TP_Space_2d = Mantis.FunctionSpaces.create_bspline_space(starting_point_2d, box_size_2d, num_elements_2d, degree_2d, regularity_2d)

# Generate the multivalued FEMSpaces (DirectSumSpace)
dsTP_0_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_2d,))  # direct sum space
dsTP_1_form_2d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_2d, TP_Space_2d))  # direct sum space
dsTP_top_form_2d = dsTP_0_form_2d # direct sum space

# Generate the form spaces
zero_form_space_2d = Mantis.Forms.FormSpace(0, geo_2d_cart, dsTP_0_form_2d, "ν")
one_form_space_2d = Mantis.Forms.FormSpace(1, geo_2d_cart, dsTP_1_form_2d, "η")
top_form_space_2d = Mantis.Forms.FormSpace(2, geo_2d_cart, dsTP_top_form_2d, "σ")

# Checks if incorrect arguments throw errors 
@test_throws ArgumentError Mantis.Forms.FormSpace(1, geo_2d_cart, dsTP_0_form_2d, "ν")
@test_throws ArgumentError Mantis.Forms.FormSpace(0, geo_2d_cart, dsTP_1_form_2d, "ν")
@test_throws ArgumentError Mantis.Forms.FormSpace(2, geo_2d_cart, dsTP_1_form_2d, "ν")

# Evaluate forms

# Evaluation points
ξ_1_evaluation = [0.0, 1.0]
ξ_2_evaluation = [0.0, 1.0]
ξ_evaluation = (ξ_1_evaluation, ξ_2_evaluation)
n_evaluation_points = prod(size.(ξ_evaluation, 1))

# Element where to evaluate
element_idx = 1

# 0-forms
zero_form_space_eval, zero_form_space_idx = Mantis.Forms.evaluate(zero_form_space_2d, element_idx, ξ_evaluation)

# Test output 
n_active_basis_0_form = Mantis.FunctionSpaces.get_num_basis(dsTP_0_form_2d, element_idx)
 
@test size(zero_form_space_eval) == (1,)  # check that we get one components
@test size(zero_form_space_eval[1]) == (n_evaluation_points, n_active_basis_0_form)   # check that we get a matrix with the correct dimensions for the single component
@test size(zero_form_space_idx) == (n_active_basis_0_form, ) # check that we get one single vector with as many indices as the active basis in the element

# 1-forms
one_form_space_eval, one_form_space_idx = Mantis.Forms.evaluate(one_form_space_2d, element_idx, ξ_evaluation)

# Test output 
n_active_basis_1_form = Mantis.FunctionSpaces.get_num_basis(dsTP_1_form_2d, element_idx)
n_nonzero_basis_in_xi_1 = Mantis.FunctionSpaces.get_num_basis(TP_Space_2d, element_idx)
n_nonzero_basis_in_xi_2 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum

@test size(one_form_space_eval) == (2,)  # check that we get two components
@test all([size(form_component) == (n_evaluation_points, n_active_basis_1_form) for form_component in one_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test size(one_form_space_idx) == (n_active_basis_1_form, )  # check that we get one single vector with as many indices as the active basis in the element

@test sum(abs.(one_form_space_eval[1][:, (n_nonzero_basis_in_xi_1 + 1):end])) == 0.0  # check that for the first component the second set of basis are all zero (because the underlying space is a direct sum space)
@test sum(abs.(one_form_space_eval[2][:, 1:n_nonzero_basis_in_xi_1])) == 0.0  # check that for the second component the first set of basis are all zero (because the underlying space is a direct sum space) 

# top-forms
top_form_space_eval, top_form_space_idx = Mantis.Forms.evaluate(top_form_space_2d, element_idx, ξ_evaluation)

# Test output 
n_active_basis_top_form = Mantis.FunctionSpaces.get_num_basis(dsTP_top_form_2d, element_idx)
 
@test size(top_form_space_eval) == (1,)  # check that we get one components
@test size(top_form_space_eval[1]) == (n_evaluation_points, n_active_basis_top_form)   # check that we get a matrix with the correct dimensions for the single component
@test size(top_form_space_idx) == (n_active_basis_top_form, ) # check that we get one single vector with as many indices as the active basis in the element

# Exterior derivative 

# Exterior derivative 

# 0-forms 
d_zero_form_space_eval, d_zero_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(zero_form_space_2d, element_idx, ξ_evaluation)

# Test output
n_active_basis_0_form = Mantis.FunctionSpaces.get_num_basis(dsTP_0_form_2d, element_idx)
 
@test size(d_zero_form_space_eval) == (2,)  # check that we get three components (3D)
@test size(d_zero_form_space_idx) == (n_active_basis_0_form, )  # check that we get one single vector with as many indices as the active basis in the element
@test all([size(form_component) == (n_evaluation_points, n_active_basis_0_form) for form_component in d_zero_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12) 

# 1-forms 
d_one_form_space_eval, d_one_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(one_form_space_2d, element_idx, ξ_evaluation)

# Test output
n_active_basis_1_form = Mantis.FunctionSpaces.get_num_basis(dsTP_1_form_2d, element_idx)
 
@test size(d_one_form_space_eval) == (1,)  # check that we get three components (3D)
@test size(d_one_form_space_idx) == (n_active_basis_1_form, )  # check that we get one single vector with as many indices as the active basis in the element
@test all([size(form_component) == (n_evaluation_points, n_active_basis_1_form) for form_component in d_one_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12) 

# top-forms 
@test_throws ArgumentError d_top_form_space_eval, d_top_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(top_form_space_2d, element_idx, ξ_evaluation)


# hodge_zero_form_space_eval, hodge_zero_form_space_idx = Mantis.Forms.evaluate_hodge_star(zero_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

# d_one_form_space_eval, d_one_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
# hodge_one_form_space_eval, hodge_zero_form_space_idx = Mantis.Forms.evaluate_hodge_star(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

# top_form_space_eval, top_form_space_idx = Mantis.Forms.evaluate(top_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
# hodge_top_form_space_eval, hodge_top_form_space_idx = Mantis.Forms.evaluate_hodge_star(top_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))


# -----------------------------------------------------------------------------


# 3D tests --------------------------------------------------------------------

# Setup the geometry 

# First the 3D rectangular cuboid domain parameters
starting_point_3d = (0.0, 0.0, 0.0)
box_size_3d = (1.0, 1.0, 1.0)
num_elements_3d = (2, 3, 2)
degree_3d = (2, 2, 3)
regularity_3d = degree_3d .- 1

# Generate the 3D rectangular cuboid straight geometry
geo_3d_cart = Mantis.Geometry.create_cartesian_geometry(starting_point_3d, box_size_3d, num_elements_3d)

# Setup the form spaces

# Generate a tensor product space (single valued space)
TP_Space_3d = Mantis.FunctionSpaces.create_bspline_space(starting_point_3d, box_size_3d, num_elements_3d, degree_3d, regularity_3d)

# Generate the multivalued FEMSpaces (DirectSumSpace)
dsTP_0_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d,))  # direct sum space
dsTP_1_form_3d = Mantis.FunctionSpaces.DirectSumSpace((TP_Space_3d, TP_Space_3d, TP_Space_3d))  # direct sum space
dsTP_2_form_3d = dsTP_1_form_3d
dsTP_top_form_3d = dsTP_0_form_3d

# Generate the form spaces
zero_form_space_3d = Mantis.Forms.FormSpace(0, geo_3d_cart, dsTP_0_form_3d, "ν")
one_form_space_3d = Mantis.Forms.FormSpace(1, geo_3d_cart, dsTP_1_form_3d, "η")
two_form_space_3d = Mantis.Forms.FormSpace(2, geo_3d_cart, dsTP_2_form_3d, "ψ")
top_form_space_3d = Mantis.Forms.FormSpace(3, geo_3d_cart, dsTP_top_form_3d, "σ")

# Checks if incorrect arguments throw errors 
@test_throws ArgumentError Mantis.Forms.FormSpace(1, geo_3d_cart, dsTP_0_form_3d, "ν")
@test_throws ArgumentError Mantis.Forms.FormSpace(2, geo_3d_cart, dsTP_0_form_3d, "ν")
@test_throws ArgumentError Mantis.Forms.FormSpace(0, geo_3d_cart, dsTP_1_form_3d, "ν")
@test_throws ArgumentError Mantis.Forms.FormSpace(3, geo_3d_cart, dsTP_1_form_3d, "ν")
@test_throws MethodError Mantis.Forms.FormSpace(3, geo_2d_cart, dsTP_top_form_3d, "ν")  # incompatible manifold size (2D) for the components
@test_throws MethodError Mantis.Forms.FormSpace(1, geo_2d_cart, dsTP_1_form_3d, "ν")
@test_throws MethodError Mantis.Forms.FormSpace(2, geo_2d_cart, dsTP_2_form_3d, "ν")

# Evaluate forms

# Evaluation points
ξ_1_evaluation = [0.0, 1.0]
ξ_2_evaluation = [0.0, 1.0]
ξ_3_evaluation = [0.0, 1.0]
ξ_evaluation = (ξ_1_evaluation, ξ_2_evaluation, ξ_3_evaluation)
n_evaluation_points = prod(size.(ξ_evaluation, 1))

# Element where to evaluate 
element_idx = 1

# 0-forms
zero_form_space_eval, zero_form_space_idx = Mantis.Forms.evaluate(zero_form_space_3d, element_idx, ξ_evaluation)

# Test output 
n_active_basis_0_form = Mantis.FunctionSpaces.get_num_basis(dsTP_0_form_3d, element_idx)
 
@test size(zero_form_space_eval) == (1,)  # check that we get one components
@test size(zero_form_space_eval[1]) == (n_evaluation_points, n_active_basis_0_form)   # check that we get a matrix with the correct dimensions for the single component
@test size(zero_form_space_idx) == (n_active_basis_0_form, ) # check that we get one single vector with as many indices as the active basis in the element

# 1-forms
one_form_space_eval, one_form_space_idx = Mantis.Forms.evaluate(one_form_space_3d, element_idx, ξ_evaluation)

# Test output 
n_active_basis_1_form = Mantis.FunctionSpaces.get_num_basis(dsTP_1_form_3d, element_idx)
n_nonzero_basis_in_xi_1 = Mantis.FunctionSpaces.get_num_basis(TP_Space_3d, element_idx)
n_nonzero_basis_in_xi_2 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum
n_nonzero_basis_in_xi_3 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum

@test size(one_form_space_eval) == (3,)  # check that we get three components (3D)
@test all([size(form_component) == (n_evaluation_points, n_active_basis_1_form) for form_component in one_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test size(one_form_space_idx) == (n_active_basis_1_form, )  # check that we get one single vector with as many indices as the active basis in the element

@test sum(abs.(one_form_space_eval[1][:, (n_nonzero_basis_in_xi_1 + 1):end])) == 0.0  # check that for the first component the second  and third sets of basis are all zero (because the underlying space is a direct sum space)
@test sum(abs.(one_form_space_eval[2][:, 1:n_nonzero_basis_in_xi_1])) == 0.0  # check that for the second component the first and third sets of basis are all zero (because the underlying space is a direct sum space) 
@test sum(abs.(one_form_space_eval[2][:, (2*n_nonzero_basis_in_xi_1 + 1):end])) == 0.0
@test sum(abs.(one_form_space_eval[3][:, 1:(2*n_nonzero_basis_in_xi_1)])) == 0.0  # check that for the third component the first  and second sets of basis are all zero (because the underlying space is a direct sum space)

# 2-forms
two_form_space_eval, two_form_space_idx = Mantis.Forms.evaluate(two_form_space_3d, element_idx, ξ_evaluation)

# Test output 
n_active_basis_2_form = Mantis.FunctionSpaces.get_num_basis(dsTP_2_form_3d, element_idx)
n_nonzero_basis_in_xi_1 = Mantis.FunctionSpaces.get_num_basis(TP_Space_3d, element_idx)
n_nonzero_basis_in_xi_2 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum
n_nonzero_basis_in_xi_3 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum

@test size(two_form_space_eval) == (3,)  # check that we get three components (3D)
@test all([size(form_component) == (n_evaluation_points, n_active_basis_1_form) for form_component in two_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test size(two_form_space_idx) == (n_active_basis_2_form, )  # check that we get one single vector with as many indices as the active basis in the element

@test sum(abs.(two_form_space_eval[1][:, (n_nonzero_basis_in_xi_1 + 1):end])) == 0.0  # check that for the first component the second  and third sets of basis are all zero (because the underlying space is a direct sum space)
@test sum(abs.(two_form_space_eval[2][:, 1:n_nonzero_basis_in_xi_1])) == 0.0  # check that for the second component the first and third sets of basis are all zero (because the underlying space is a direct sum space) 
@test sum(abs.(two_form_space_eval[2][:, (2*n_nonzero_basis_in_xi_1 + 1):end])) == 0.0
@test sum(abs.(two_form_space_eval[3][:, 1:(2*n_nonzero_basis_in_xi_1)])) == 0.0  # check that for the third component the first  and second sets of basis are all zero (because the underlying space is a direct sum space)


# top-forms
top_form_space_eval, top_form_space_idx = Mantis.Forms.evaluate(top_form_space_3d, element_idx, ξ_evaluation)

# Test output 
n_active_basis_top_form = Mantis.FunctionSpaces.get_num_basis(dsTP_top_form_3d, element_idx)
 
@test size(top_form_space_eval) == (1,)  # check that we get one components
@test size(top_form_space_eval[1]) == (n_evaluation_points, n_active_basis_top_form)   # check that we get a matrix with the correct dimensions for the single component
@test size(top_form_space_idx) == (n_active_basis_top_form, ) # check that we get one single vector with as many indices as the active basis in the element

# Exterior derivative 

# 0-forms 
d_zero_form_space_eval, d_zero_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(zero_form_space_3d, element_idx, ξ_evaluation)

# Test output
n_active_basis_0_form = Mantis.FunctionSpaces.get_num_basis(dsTP_0_form_3d, element_idx)
 
@test size(d_zero_form_space_eval) == (3,)  # check that we get three components (3D)
@test size(d_zero_form_space_idx) == (n_active_basis_0_form, )  # check that we get one single vector with as many indices as the active basis in the element
@test all([size(form_component) == (n_evaluation_points, n_active_basis_0_form) for form_component in d_zero_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12) 


# 1-forms 
d_one_form_space_eval, d_one_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(one_form_space_3d, element_idx, ξ_evaluation)

# Test output
n_active_basis_1_form = Mantis.FunctionSpaces.get_num_basis(dsTP_1_form_3d, element_idx)
 
@test size(d_one_form_space_eval) == (3,)  # check that we get three components (3D)
@test size(d_one_form_space_idx) == (n_active_basis_1_form, )  # check that we get one single vector with as many indices as the active basis in the element
@test all([size(form_component) == (n_evaluation_points, n_active_basis_1_form) for form_component in d_one_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12) 

# 2-forms 
d_two_form_space_eval, d_two_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(two_form_space_3d, element_idx, ξ_evaluation)

# Test output
n_active_basis_two_form = Mantis.FunctionSpaces.get_num_basis(dsTP_2_form_3d, element_idx)
 
@test size(d_two_form_space_eval) == (1,)  # check that we get three components (3D)
@test size(d_two_form_space_idx) == (n_active_basis_2_form, )  # check that we get one single vector with as many indices as the active basis in the element
@test all([size(form_component) == (n_evaluation_points, n_active_basis_2_form) for form_component in d_two_form_space_eval])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_two_form_space_eval)), 0.0, atol=1e-12) 

# top-forms 
@test_throws ArgumentError d_top_form_space_eval, d_top_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(top_form_space_2d, element_idx, ξ_evaluation)


# # Generate the form expressions
# α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
# ξ¹ = Mantis.Forms.FormField(one_form_space, "ξ")
# β² = Mantis.Forms.FormField(top_form_space, "β")
# θ² = Mantis.Forms.FormExpression((α⁰, β²), 2, "∧")
# ζ² = Mantis.Forms.FormExpression((α⁰, θ²), 2, "∧")

# print(θ².label)
# print("\n")
# ζ² = Mantis.Forms.FormExpression((α⁰, θ²), 2, "∧")
# print(ζ².label)


# # function (∧)(form_1::F_1, form_2::F_2) where {F_1 <: Mantis.Forms.AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: Mantis.Forms.AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
# #     return Mantis.Forms.FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
# # end

# # function (⋆)(form::F) where {F <: Mantis.Forms.AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
# #     return Mantis.Forms.FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
# # end

# print("\nComputing the wedge operator\n")
# γ = Mantis.Forms.wedge(α⁰, β²)
# print(γ.label)

# γ2 = Mantis.Forms.wedge(γ, β²)
# print("\n")
# print(γ2.label)
# print("\n")


α⁰.coefficients .= 1.0
ξ¹.coefficients .= 1.0
β².coefficients .= 1.0

# dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
# dξ¹ = Mantis.Forms.exterior_derivative(ξ¹)
# ★α⁰ = Mantis.Forms.hodge(α⁰)
# ★ξ¹ = Mantis.Forms.hodge(ξ¹)
# ★β² = Mantis.Forms.hodge(β²)

# α⁰_eval = Mantis.Forms.evaluate(α⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
# dα⁰_eval = Mantis.Forms.evaluate(dα⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
# ★α⁰_eval = Mantis.Forms.evaluate(★α⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
# ξ¹_eval = Mantis.Forms.evaluate(ξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))
# dξ¹_eval = Mantis.Forms.evaluate(dξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))
# ★ξ¹_eval = Mantis.Forms.evaluate(★ξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))
# β²_eval = Mantis.Forms.evaluate(β², 1, ([0.0, 1.0], [0.0, 1.0]))
# ★β²_eval = Mantis.Forms.evaluate(★β², 1, ([0.0, 1.0], [0.0, 1.0]))


println()
geo_2d_cart2 = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, num_elements_2d .+ 1) # Used to test if different geometries break, as they should.
zero_form_space_cart = Mantis.Forms.FormSpace(0, geo_2d_cart, (TP_Space,), "ν")
zero_form_space_cart2 = Mantis.Forms.FormSpace(0, geo_2d_cart2, (TP_Space,), "ν")
d_zero_form_space_cart = Mantis.Forms.exterior_derivative(zero_form_space)
one_form_space_cart = Mantis.Forms.FormSpace(1, geo_2d_cart, (TP_Space, TP_Space), "η")
top_form_space_cart = Mantis.Forms.FormSpace(2, geo_2d_cart, (TP_Space,), "σ")

# # Generate the form expressions
# α⁰ = Mantis.Forms.FormField(zero_form_space_cart, "α")
# α⁰.coefficients .= 1.0
# β⁰ = Mantis.Forms.FormField(zero_form_space_cart2, "β")
# β⁰.coefficients .= 1.0
# ζ¹ = Mantis.Forms.FormField(one_form_space_cart, "ζ")
# ζ¹.coefficients .= 1.0
# dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
# γ² = Mantis.Forms.FormField(top_form_space_cart, "γ")
# γ².coefficients .= 1.0
# dζ¹ = Mantis.Forms.exterior_derivative(ζ¹)

q_rule = Mantis.Quadrature.tensor_product_rule(degree_2d.+1, Mantis.Quadrature.gauss_legendre)
#display(Mantis.Forms.evaluate_inner_product(zero_form_space_cart, zero_form_space_cart, 1, q_rule))
println("Starting inner product computations")
# Note that we cannot do mixed inner products

# for elem_id in 1:1:Mantis.Geometry.get_num_elements(geo_2d_cart)
#     ordered_idx = Tuple(geo_2d_cart.cartesian_idxs[elem_id])
#     elem_area = 1.0
#     for dim_idx in range(1, 2)
#         start_breakpoint_idx = ordered_idx[dim_idx]
#         end_breakpoint_idx = ordered_idx[dim_idx] + 1
#         elem_area *= geo_2d_cart.breakpoints[dim_idx][end_breakpoint_idx] - geo_2d_cart.breakpoints[dim_idx][start_breakpoint_idx]
#     end

#     g, det_g = Mantis.Geometry.metric(geo_2d_cart, elem_id, Mantis.Quadrature.get_quadrature_nodes(q_rule))
#     # 0-forms
#     @test isapprox(Mantis.Forms.evaluate_inner_product(α⁰, α⁰, elem_id, q_rule)[3][1], elem_area, atol=1e-12)
#     @test_throws ArgumentError Mantis.Forms.evaluate_inner_product(α⁰, β⁰, elem_id, q_rule)
    
#     # 1-forms
#     println(Mantis.Forms.evaluate_inner_product(ζ¹, ζ¹, elem_id, q_rule)[3])
#     @test isapprox(sum(Mantis.Forms.evaluate_inner_product(ζ¹, ζ¹, elem_id, q_rule)[3]), sum((g./det_g)[1,:,:]), atol=1e-12)
#     @test isapprox(Mantis.Forms.evaluate_inner_product(dα⁰, dα⁰, elem_id, q_rule)[3][1], 0.0, atol=1e-12)
    
#     # n-forms
#     @test isapprox(Mantis.Forms.evaluate_inner_product(γ², γ², elem_id, q_rule)[3][1], 1/elem_area, atol=1e-12)
#     @test isapprox(Mantis.Forms.evaluate_inner_product(dζ¹, dζ¹, elem_id, q_rule)[3][1], 0.0, atol=1e-12)
# end

# # 3d
# geo_3d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2, breakpoints1))
# zero_form_space_cart = Mantis.Forms.FormSpace(0, geo_3d_cart, (TP_Space3d,), "ν")
# one_form_space_cart = Mantis.Forms.FormSpace(1, geo_3d_cart, (TP_Space3d, TP_Space3d, TP_Space3d), "θ")
# two_form_space_cart = Mantis.Forms.FormSpace(2, geo_3d_cart, (TP_Space3d, TP_Space3d, TP_Space3d), "η")
# top_form_space_cart = Mantis.Forms.FormSpace(3, geo_3d_cart, (TP_Space3d,), "σ")

# # Generate the form expressions
# α⁰ = Mantis.Forms.FormField(zero_form_space_cart, "α")
# α⁰.coefficients .= 1.0
# θ¹ = Mantis.Forms.FormField(one_form_space_cart, "θ")
# θ¹.coefficients .= 1.0
# ζ² = Mantis.Forms.FormField(two_form_space_cart, "ζ")
# ζ².coefficients .= 1.0
# dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
# γ³ = Mantis.Forms.FormField(top_form_space_cart, "γ")
# γ³.coefficients .= 1.0

# q_rule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1, deg1+1), Mantis.Quadrature.gauss_legendre)
# # Note that we cannot do mixed inner products

# for elem_id in 1:1:Mantis.Geometry.get_num_elements(geo_3d_cart)
#     ordered_idx = Tuple(geo_3d_cart.cartesian_idxs[elem_id])
#     elem_vol = 1.0
#     for dim_idx in range(1, 3)
#         start_breakpoint_idx = ordered_idx[dim_idx]
#         end_breakpoint_idx = ordered_idx[dim_idx] + 1
#         elem_vol *= geo_3d_cart.breakpoints[dim_idx][end_breakpoint_idx] - geo_3d_cart.breakpoints[dim_idx][start_breakpoint_idx]
#     end

#     g_inv, g, det_g = Mantis.Geometry.inv_metric(geo_3d_cart, elem_id, Mantis.Quadrature.get_quadrature_nodes(q_rule))

#     # 0-forms
#     @test isapprox(Mantis.Forms.evaluate_inner_product(α⁰, α⁰, elem_id, q_rule)[3][1], elem_vol, atol=1e-12)

#     # 1-forms
#     @test isapprox(sum(Mantis.Forms.evaluate_inner_product(θ¹, θ¹, elem_id, q_rule)[3]), sum((g_inv.*det_g)[1,:,:]), atol=1e-12)

#     # 2-forms
#     @test isapprox(sum(Mantis.Forms.evaluate_inner_product(ζ², ζ², elem_id, q_rule)[3]), sum((g./det_g)[1,:,:]), atol=1e-12)

#     # n-forms
#     @test isapprox(Mantis.Forms.evaluate_inner_product(γ³, γ³, elem_id, q_rule)[3][1], 1/elem_vol, atol=1e-12)
# end

# ★α⁰ = Mantis.Forms.hodge(α⁰)
# ★θ¹ = Mantis.Forms.hodge(θ¹)
# ★ζ² = Mantis.Forms.hodge(ζ²)
# ★γ³ = Mantis.Forms.hodge(γ³)

# ★α⁰_eval = Mantis.Forms.evaluate(★α⁰, 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
# ★θ¹_eval = Mantis.Forms.evaluate(★θ¹, 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
# ★ζ²_eval = Mantis.Forms.evaluate(★ζ², 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
# ★γ³_eval = Mantis.Forms.evaluate(★γ³, 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))


# Test AnalyticalFormField
function one_form_function(x::Matrix{Float64})
    return [x[:, 1], x[:, 2]]
end

function zero_form_function(x::Matrix{Float64})
    return [x[:, 2]]
end

function volume_form_function(x::Matrix{Float64})
    return [ones(size(x, 1))]
end

β⁰ = Mantis.Forms.AnalyticalFormField(0, zero_form_function, geo_2d_cart, "β")
σ² = Mantis.Forms.AnalyticalFormField(2, volume_form_function, geo_2d_cart, "σ")
α¹ = Mantis.Forms.AnalyticalFormField(1, one_form_function, geo_2d_cart, "α")

β⁰_eval = Mantis.Forms.evaluate(β⁰, 1, ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]))
σ²_eval = Mantis.Forms.evaluate(σ², 1, ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]))

using SparseArrays

q_rule_high = Mantis.Quadrature.tensor_product_rule(degree_2d .+ 10, Mantis.Quadrature.gauss_legendre)
alpha1_l2_norm_square = 0.0
for elem_id in 1:1:Mantis.Geometry.get_num_elements(geo_2d_cart)
    global alpha1_l2_norm_square += Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(α¹, α¹, elem_id, q_rule_high)...))[1,1]
end
@test isapprox(alpha1_l2_norm_square, 2/3, atol=1e-12)

Lleft = 0.0
Lright = 1.0
Lbottom = 0.0
Ltop = 1.0

crazy_c = 0.2
function mapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [x[1] + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x[2] + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
end
function dmapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [1.0 + pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/(Ltop-Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new); ((Ltop-Lbottom)/(Lright-Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0 + pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
end
dimension = (2, 2)
curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
geom_crazy = Mantis.Geometry.MappedGeometry(geo_2d_cart, curved_mapping)
α¹ = Mantis.Forms.AnalyticalFormField(1, one_form_function, geom_crazy, "α")
alpha1_l2_norm_square = 0.0
for elem_id in 1:1:Mantis.Geometry.get_num_elements(geom_crazy)
    global alpha1_l2_norm_square += Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(α¹, α¹, elem_id, q_rule_high)...))[1,1]
end
@test isapprox(alpha1_l2_norm_square, 2/3, atol=1e-12)

# Test "error" computation -------------
zero_form_space = Mantis.Forms.FormSpace(0, geo_2d_cart, (TP_Space,), "ν")
α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
α⁰.coefficients .= 1.0

function zero_form_function(x::Matrix{Float64})
    return [x[:, 2]]
end

# Generate an analytical form field
β⁰ = Mantis.Forms.AnalyticalFormField(0, zero_form_function, geo_2d_cart, "β")

# Define error
e = α⁰ - β⁰

# compute the L^2 norm of the differences
using SparseArrays
error_L2 = 0.0
q_rule = Mantis.Quadrature.tensor_product_rule(degree_2d .+ 1, Mantis.Quadrature.gauss_legendre)
for elem_id in 1:1:Mantis.Geometry.get_num_elements(geo_2d_cart)
    global error_L2 += Matrix(SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(e, e, elem_id, q_rule)...))[1,1]
end
error_L2 = sqrt(error_L2)
@test(isapprox(error_L2^2, 1/3, atol=1e-10))