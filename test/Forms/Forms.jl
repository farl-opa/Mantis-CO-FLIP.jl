import Mantis

using Test

# Setup the form spaces
# First the FEM spaces
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

# first B-spline patch
deg1 = 2
deg2 = 2
B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
# second B-spline patch
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
# tensor-product B-spline patch
TP_Space = Mantis.FunctionSpaces.TensorProductSpace(B1, B2)
TP_Space3d = Mantis.FunctionSpaces.TensorProductSpace(TP_Space, B1)

# Then the geometry 
# Line 1
line_1_geo = Mantis.Geometry.CartesianGeometry((breakpoints1,))

# Line 2
line_2_geo = Mantis.Geometry.CartesianGeometry((breakpoints2,))

# Tensor product geometry 
tensor_prod_geo = Mantis.Geometry.TensorProductGeometry(line_1_geo, line_2_geo)

# This seems to be a bug:
#println(Mantis.Geometry.jacobian(tensor_prod_geo, 1, ([0.0, 0.0], [0.0, 0.0])))

# Then the form space 
zero_form_space = Mantis.Forms.FormSpace(0, tensor_prod_geo, (TP_Space,), "ν")
d_zero_form_space = Mantis.Forms.exterior_derivative(zero_form_space)
one_form_space = Mantis.Forms.FormSpace(1, tensor_prod_geo, (TP_Space, TP_Space), "η")
top_form_space = Mantis.Forms.FormSpace(2, tensor_prod_geo, (TP_Space,), "σ")

# Generate the form expressions
α⁰ = Mantis.Forms.FormField(zero_form_space, "α")
ξ¹ = Mantis.Forms.FormField(one_form_space, "ξ")
β² = Mantis.Forms.FormField(top_form_space, "β")
θ² = Mantis.Forms.FormExpression((α⁰, β²), 2, "∧")
ζ² = Mantis.Forms.FormExpression((α⁰, θ²), 2, "∧")

print(θ².label)
print("\n")
ζ² = Mantis.Forms.FormExpression((α⁰, θ²), 2, "∧")
print(ζ².label)


# function (∧)(form_1::F_1, form_2::F_2) where {F_1 <: Mantis.Forms.AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: Mantis.Forms.AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
#     return Mantis.Forms.FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
# end

# function (⋆)(form::F) where {F <: Mantis.Forms.AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
#     return Mantis.Forms.FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
# end

print("\nComputing the wedge operator\n")
γ = Mantis.Forms.wedge(α⁰, β²)
print(γ.label)

γ2 = Mantis.Forms.wedge(γ, β²)
print("\n")
print(γ2.label)
print("\n")

zero_form_space_eval, zero_form_space_idx = Mantis.Forms.evaluate(zero_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
d_zero_form_space_eval, d_zero_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(zero_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
hodge_zero_form_space_eval, hodge_zero_form_space_idx = Mantis.Forms.evaluate_hodge_star(zero_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

one_form_space_eval, one_form_space_idx = Mantis.Forms.evaluate(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
d_one_form_space_eval, d_one_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
hodge_one_form_space_eval, hodge_zero_form_space_idx = Mantis.Forms.evaluate_hodge_star(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

top_form_space_eval, top_form_space_idx = Mantis.Forms.evaluate(top_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
hodge_top_form_space_eval, hodge_top_form_space_idx = Mantis.Forms.evaluate_hodge_star(top_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

α⁰.coefficients .= 1.0
ξ¹.coefficients .= 1.0
β².coefficients .= 1.0

dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
dξ¹ = Mantis.Forms.exterior_derivative(ξ¹)
🟉α⁰ = Mantis.Forms.hodge(α⁰)
🟉ξ¹ = Mantis.Forms.hodge(ξ¹)
🟉β² = Mantis.Forms.hodge(β²)

α⁰_eval = Mantis.Forms.evaluate(α⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
dα⁰_eval = Mantis.Forms.evaluate(dα⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
🟉α⁰_eval = Mantis.Forms.evaluate(🟉α⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
ξ¹_eval = Mantis.Forms.evaluate(ξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))
dξ¹_eval = Mantis.Forms.evaluate(dξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))
🟉ξ¹_eval = Mantis.Forms.evaluate(🟉ξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))
β²_eval = Mantis.Forms.evaluate(β², 1, ([0.0, 1.0], [0.0, 1.0]))
🟉β²_eval = Mantis.Forms.evaluate(🟉β², 1, ([0.0, 1.0], [0.0, 1.0]))


println()
geo_2d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2))
geo_2d_cart2 = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints1))
zero_form_space_cart = Mantis.Forms.FormSpace(0, geo_2d_cart, (TP_Space,), "ν")
zero_form_space_cart2 = Mantis.Forms.FormSpace(0, geo_2d_cart2, (TP_Space,), "ν")
d_zero_form_space_cart = Mantis.Forms.exterior_derivative(zero_form_space)
one_form_space_cart = Mantis.Forms.FormSpace(1, geo_2d_cart, (TP_Space, TP_Space), "η")
top_form_space_cart = Mantis.Forms.FormSpace(2, geo_2d_cart, (TP_Space,), "σ")

# Generate the form expressions
α⁰ = Mantis.Forms.FormField(zero_form_space_cart, "α")
α⁰.coefficients .= 1.0
β⁰ = Mantis.Forms.FormField(zero_form_space_cart2, "β")
β⁰.coefficients .= 1.0
ζ¹ = Mantis.Forms.FormField(one_form_space_cart, "ζ")
ζ¹.coefficients .= 1.0
dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
γ² = Mantis.Forms.FormField(top_form_space_cart, "γ")
γ².coefficients .= 1.0
dζ¹ = Mantis.Forms.exterior_derivative(ζ¹)

q_rule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)
#display(Mantis.Forms.inner_product(zero_form_space_cart, zero_form_space_cart, 1, q_rule))
println("Starting inner product computations")
# Note that we cannot do mixed inner products
for elem_id in 1:1:Mantis.Geometry.get_num_elements(geo_2d_cart)
    ordered_idx = Tuple(geo_2d_cart.cartesian_idxs[elem_id])
    elem_area = 1.0
    for dim_idx in range(1, 2)
        start_breakpoint_idx = ordered_idx[dim_idx]
        end_breakpoint_idx = ordered_idx[dim_idx] + 1
        elem_area *= geo_2d_cart.breakpoints[dim_idx][end_breakpoint_idx] - geo_2d_cart.breakpoints[dim_idx][start_breakpoint_idx]
    end

    g, det_g = Mantis.Geometry.metric(geo_2d_cart, elem_id, Mantis.Quadrature.get_quadrature_nodes(q_rule))

    # 0-forms
    @test isapprox(Mantis.Forms.inner_product(α⁰, α⁰, elem_id, q_rule)[3][1], elem_area, atol=1e-12)
    @test_throws ArgumentError Mantis.Forms.inner_product(α⁰, β⁰, elem_id, q_rule)

    # 1-forms
    @test isapprox(Mantis.Forms.inner_product(ζ¹, ζ¹, elem_id, q_rule)[3][1,1,:,:], (g./det_g)[1,:,:], atol=1e-12)
    @test isapprox(Mantis.Forms.inner_product(dα⁰, dα⁰, elem_id, q_rule)[3][1,1,:,:], [0.0 0.0; 0.0 0.0], atol=1e-12)

    # n-forms
    @test isapprox(Mantis.Forms.inner_product(γ², γ², elem_id, q_rule)[3][1], 1/elem_area, atol=1e-12)
    @test isapprox(Mantis.Forms.inner_product(dζ¹, dζ¹, elem_id, q_rule)[3][1], 0.0, atol=1e-12)
end

# 3d
geo_3d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2, breakpoints1))
zero_form_space_cart = Mantis.Forms.FormSpace(0, geo_3d_cart, (TP_Space3d,), "ν")
one_form_space_cart = Mantis.Forms.FormSpace(1, geo_3d_cart, (TP_Space3d, TP_Space3d, TP_Space3d), "θ")
two_form_space_cart = Mantis.Forms.FormSpace(2, geo_3d_cart, (TP_Space3d, TP_Space3d, TP_Space3d), "η")
top_form_space_cart = Mantis.Forms.FormSpace(3, geo_3d_cart, (TP_Space3d,), "σ")

# Generate the form expressions
α⁰ = Mantis.Forms.FormField(zero_form_space_cart, "α")
α⁰.coefficients .= 1.0
θ¹ = Mantis.Forms.FormField(one_form_space_cart, "θ")
θ¹.coefficients .= 1.0
ζ² = Mantis.Forms.FormField(two_form_space_cart, "ζ")
ζ².coefficients .= 1.0
dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
γ³ = Mantis.Forms.FormField(top_form_space_cart, "γ")
γ³.coefficients .= 1.0

q_rule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1, deg1+1), Mantis.Quadrature.gauss_legendre)
# Note that we cannot do mixed inner products
for elem_id in 1:1:Mantis.Geometry.get_num_elements(geo_3d_cart)
    ordered_idx = Tuple(geo_3d_cart.cartesian_idxs[elem_id])
    elem_vol = 1.0
    for dim_idx in range(1, 3)
        start_breakpoint_idx = ordered_idx[dim_idx]
        end_breakpoint_idx = ordered_idx[dim_idx] + 1
        elem_vol *= geo_3d_cart.breakpoints[dim_idx][end_breakpoint_idx] - geo_3d_cart.breakpoints[dim_idx][start_breakpoint_idx]
    end

    g_inv, g, det_g = Mantis.Geometry.inv_metric(geo_3d_cart, elem_id, Mantis.Quadrature.get_quadrature_nodes(q_rule))

    # 0-forms
    @test isapprox(Mantis.Forms.inner_product(α⁰, α⁰, elem_id, q_rule)[3][1], elem_vol, atol=1e-12)

    # 1-forms
    @test isapprox(Mantis.Forms.inner_product(θ¹, θ¹, elem_id, q_rule)[3][1,1,:,:], (g_inv.*det_g)[1,:,:], atol=1e-12)

    # 2-forms
    @test isapprox(Mantis.Forms.inner_product(ζ², ζ², elem_id, q_rule)[3][1,1,:,:], (g./det_g)[1,:,:], atol=1e-12)

    # n-forms
    @test isapprox(Mantis.Forms.inner_product(γ³, γ³, elem_id, q_rule)[3][1], 1/elem_vol, atol=1e-12)
end

🟉α⁰ = Mantis.Forms.hodge(α⁰)
🟉θ¹ = Mantis.Forms.hodge(θ¹)
🟉ζ² = Mantis.Forms.hodge(ζ²)
🟉γ³ = Mantis.Forms.hodge(γ³)

🟉α⁰_eval = Mantis.Forms.evaluate(🟉α⁰, 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
🟉θ¹_eval = Mantis.Forms.evaluate(🟉θ¹, 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
🟉ζ²_eval = Mantis.Forms.evaluate(🟉ζ², 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
🟉γ³_eval = Mantis.Forms.evaluate(🟉γ³, 1, ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))