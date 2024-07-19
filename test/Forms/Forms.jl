import Mantis

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

one_form_space_eval, one_form_space_idx = Mantis.Forms.evaluate(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
d_one_form_space_eval, d_one_form_space_idx = Mantis.Forms.evaluate_exterior_derivative(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

α⁰.coefficients .= 1.0
ξ¹.coefficients .= 1.0

dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
dξ¹ = Mantis.Forms.exterior_derivative(ξ¹)

α⁰_eval = Mantis.Forms.evaluate(α⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
dα⁰_eval = Mantis.Forms.evaluate(dα⁰, 1, ([0.0, 1.0], [0.0, 1.0]))
ξ¹_eval = Mantis.Forms.evaluate(ξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))
dξ¹_eval = Mantis.Forms.evaluate(dξ¹, 1, ([0.0, 1.0], [0.0, 1.0]))

println()
geo_2d_cart = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2))
zero_form_space_cart = Mantis.Forms.FormSpace(0, geo_2d_cart, (TP_Space,), "ν")
d_zero_form_space_cart = Mantis.Forms.exterior_derivative(zero_form_space)
one_form_space_cart = Mantis.Forms.FormSpace(1, geo_2d_cart, (TP_Space, TP_Space), "η")
top_form_space_cart = Mantis.Forms.FormSpace(2, geo_2d_cart, (TP_Space,), "σ")

# Generate the form expressions
α⁰ = Mantis.Forms.FormField(zero_form_space_cart, "α")
α⁰.coefficients .= 1.0
β⁰ = Mantis.Forms.FormField(zero_form_space_cart, "β")
β⁰.coefficients .= 1.0
println(α⁰.coefficients)
dα⁰ = Mantis.Forms.exterior_derivative(α⁰)
println("Starting inner product computation")
q_rule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)
println(Mantis.Forms.inner_product(α⁰, β⁰, 1, q_rule))


# abstract type AbstractMyStruct{n} end
# abstract type AbstractOtherStruct{n} end

# struct OtherStruct{k} <: AbstractOtherStruct{k}
#     a::Int
# end

#  struct MyStruct{k, F} <: AbstractMyStruct{k}
#     fem_space::F

#     # 0- and n-form constructor
#     function MyStruct(form_rank::Int, fem_space::F) where {F <: AbstractOtherStruct{k}} where {k}
#         new{k, F}(fem_space)
#     end
# end